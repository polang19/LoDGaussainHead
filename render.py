#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import copy

from gaussian_renderer import render, gumbel_select_render_flame
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh,
               compression_ratio=1.0, use_semantic_weights=False, unbind_selection=False,
               scale_expansion=False, scale_expansion_mode="volume", use_gumbel=False,
               interpolate_frames=False, frames_per_timestep=1):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    
    # Build output directory name with compression configuration
    iter_name = f"ours_{iteration}"
    if compression_ratio < 1.0:
        # Add compression suffix to distinguish different configurations
        suffix_parts = [f"comp{compression_ratio:.2f}".replace(".", "")]
        if use_semantic_weights:
            suffix_parts.append("semantic")
        if unbind_selection:
            suffix_parts.append("unbind")
        if scale_expansion:
            suffix_parts.append(f"scale_{scale_expansion_mode}")
        iter_name = f"{iter_name}_{'_'.join(suffix_parts)}"
    
    iter_path = Path(dataset.model_path) / name / iter_name
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Determine if we should use Gumbel network for selection
    use_gumbel_for_rendering = False
    if compression_ratio < 1.0:
        # Check if Gumbel network is available and enabled
        if use_gumbel and hasattr(gaussians, 'gumbel_net') and gaussians.gumbel_net is not None:
            use_gumbel_for_rendering = True
            print(f"\n[Gumbel Rendering] Using trained Gumbel network for point selection (ratio={compression_ratio:.2f})")
        else:
            # Fall back to fixed importance selection (Phase A)
            print(f"\n[Compression] Using fixed importance selection (ratio={compression_ratio:.2f})")
            if use_gumbel:
                print("[Warning] Gumbel network requested but not available, falling back to fixed selection")
    
    # Pre-compute selection mask if compression is enabled and NOT using Gumbel (compute once, reuse for all frames)
    selected_mask = None
    if compression_ratio < 1.0 and not use_gumbel_for_rendering:
        from utils.importance_utils import fixed_importance_selection
        
        selected_mask = fixed_importance_selection(
            gaussians,
            ratio=compression_ratio,
            use_semantic_weights=use_semantic_weights,
            enforce_binding_constraint=not unbind_selection,
            enable_debug=True
        )
        
        selected_count = selected_mask.sum().item()
        total_count = len(gaussians._xyz)
        actual_ratio = selected_count / total_count
        print(f"[Compression] Selected {selected_count}/{total_count} points ({actual_ratio*100:.2f}%)")
        
        if gaussians.binding is not None and not unbind_selection:
            covered_faces = torch.unique(gaussians.binding[selected_mask]).numel()
            total_faces = len(gaussians.binding_counter)
            print(f"[Compression] Covered {covered_faces}/{total_faces} faces")
        
        if scale_expansion:
            print(f"[Compression] Scale expansion enabled (mode: {scale_expansion_mode})")

    views = sorted(views, key=lambda v: (
        v.timestep if v.timestep is not None else -1,
        getattr(v, 'camera_id', -1) if hasattr(v, 'camera_id') else -1
    ))

    if interpolate_frames and frames_per_timestep > 0 and hasattr(gaussians, 'interpolate_mesh_by_timesteps'):
        views_by_camera_timestep = {}
        for view in views:
            ts = view.timestep if view.timestep is not None else -1
            cam_id = getattr(view, 'camera_id', -1) if hasattr(view, 'camera_id') else -1
            key = (cam_id, ts)
            if key not in views_by_camera_timestep:
                views_by_camera_timestep[key] = []
            views_by_camera_timestep[key].append(view)

        unique_cameras = sorted(set(k[0] for k in views_by_camera_timestep.keys()))
        all_timesteps = sorted(set(k[1] for k in views_by_camera_timestep.keys() if k[1] != -1))

        if len(all_timesteps) > 1:
            interpolated_views = []
            for cam_id in unique_cameras:
                camera_timesteps = sorted(set(
                    k[1] for k in views_by_camera_timestep.keys() if k[0] == cam_id and k[1] != -1
                ))
                if len(camera_timesteps) > 1:
                    for i in range(len(camera_timesteps)):
                        ts = camera_timesteps[i]
                        key = (cam_id, ts)
                        if key in views_by_camera_timestep:
                            interpolated_views.extend(views_by_camera_timestep[key])
                        if i < len(camera_timesteps) - 1:
                            ts_next = camera_timesteps[i + 1]
                            base_view = None
                            if key in views_by_camera_timestep and len(views_by_camera_timestep[key]) > 0:
                                base_view = views_by_camera_timestep[key][0]
                            if base_view is not None:
                                for frame_idx in range(1, frames_per_timestep + 1):
                                    alpha = frame_idx / (frames_per_timestep + 1)
                                    interp_view = copy.deepcopy(base_view)
                                    interp_view._is_interpolated = True
                                    interp_view._interp_ts1 = ts
                                    interp_view._interp_ts2 = ts_next
                                    interp_view._interp_alpha = alpha
                                    interp_view.timestep = ts
                                    interpolated_views.append(interp_view)
            views = interpolated_views
            views = sorted(views, key=lambda v: (
                v.timestep if v.timestep is not None else -1,
                getattr(v, 'camera_id', -1) if hasattr(v, 'camera_id') else -1
            ))

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=0 if interpolate_frames else 8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        # Only call select_mesh_by_timestep if it's a FlameGaussianModel
        # GaussianModel doesn't support this method (raises NotImplementedError)
        if gaussians.binding != None and hasattr(gaussians, 'flame_model'):
            if hasattr(view, '_is_interpolated') and view._is_interpolated and hasattr(gaussians, 'interpolate_mesh_by_timesteps'):
                gaussians.interpolate_mesh_by_timesteps(view._interp_ts1, view._interp_ts2, view._interp_alpha)
            else:
                gaussians.select_mesh_by_timestep(view.timestep)
        
        # Use Gumbel network rendering if enabled, otherwise use standard rendering
        if use_gumbel_for_rendering:
            # Enable debug output for first frame only
            if idx == 0:
                gaussians._debug_print = True
            else:
                gaussians._debug_print = False
            
            # Use Gumbel network for point selection (learnable, supports arbitrary ratios)
            render_pkg = gumbel_select_render_flame(
                view, gaussians, pipeline, background,
                ratio=compression_ratio,
                cur_tau=0.1,  # Low temperature for sharper selection during inference
                enforce_binding_constraint=not unbind_selection
            )
            rendering = render_pkg["render"]
            
            # Log actual selection ratio for first frame
            if idx == 0:
                if "gumbel_hard_mask" in render_pkg and render_pkg["gumbel_hard_mask"] is not None:
                    actual_selected = (render_pkg["gumbel_hard_mask"] > 0.5).sum().item()
                    total_points = len(render_pkg["gumbel_hard_mask"])
                    actual_ratio = actual_selected / total_points
                    print(f"\n[Gumbel Rendering Summary]")
                    print(f"  Target ratio: {compression_ratio:.4f}")
                    print(f"  Actual ratio: {actual_ratio:.4f}")
                    print(f"  Selected points: {actual_selected}/{total_points}")
                    if gaussians.binding is not None:
                        min_ratio = len(gaussians.binding_counter) / total_points
                        print(f"  Min ratio (binding constraint): {min_ratio:.4f}")
                        if actual_ratio >= min_ratio and compression_ratio < min_ratio:
                            print(f"  ⚠️  WARNING: Actual ratio ({actual_ratio:.4f}) >= min ratio ({min_ratio:.4f})")
                            print(f"     This means binding constraint is forcing selection!")
                            print(f"     For ratios < {min_ratio:.4f}, results will be similar.")
                    print()
        else:
            # Use standard rendering with pre-computed selection mask
            rendering = render(view, gaussians, pipeline, background,
                              selected_mask=selected_mask,
                              scale_expansion=scale_expansion,
                              scale_expansion_mode=scale_expansion_mode)["render"]
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool,
                compression_ratio=1.0, use_semantic_weights=False, unbind_selection=False,
                scale_expansion=False, scale_expansion_mode="volume", use_gumbel=False,
                interpolate_frames=False, frames_per_timestep=1):
    with torch.no_grad():
        # Determine model type: try to use FlameGaussianModel if FLAME model file exists
        # or if dataset.bind_to_mesh is True (indicates FLAME model was used during training)
        use_flame_model = False
        
        # Check if FLAME model file exists (try multiple possible paths)
        flame_model_paths = [
            "flame_model/assets/flame/flame2023.pkl",  # Default path
            "flame_model/assets/flame/generic_model.pkl",  # FLAME 2020
            os.path.join(os.path.dirname(__file__), "flame_model/assets/flame/flame2023.pkl"),  # Absolute path
        ]
        flame_model_path = None
        flame_model_exists = False
        for path in flame_model_paths:
            if os.path.exists(path):
                flame_model_path = path
                flame_model_exists = True
                break
        
        if dataset.bind_to_mesh:
            # Check if flame_param.npz exists (indicates FlameGaussianModel was used during training)
            if iteration == -1:
                from utils.system_utils import searchForMaxIteration
                iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
            
            iteration_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
            flame_param_path = os.path.join(iteration_dir, "flame_param.npz")
            
            if os.path.exists(flame_param_path):
                # Model has FLAME parameters, must use FlameGaussianModel
                # (GaussianModel.get_xyz will fail with NotImplementedError if binding exists)
                use_flame_model = True
                if flame_model_exists:
                    print(f"[Info] Found flame_param.npz and FLAME model file, using FlameGaussianModel")
                else:
                    print(f"[Warning] Found flame_param.npz but FLAME model file check failed")
                    print(f"[Warning] Attempting to use FlameGaussianModel (will fail with clear error if file missing)")
                    print(f"[Warning] Expected path: flame_model/assets/flame/flame2023.pkl")
            else:
                # No flame_param.npz, but if bind_to_mesh is True, still try FlameGaussianModel if file exists
                if flame_model_exists:
                    use_flame_model = True
                    print(f"[Info] bind_to_mesh=True and FLAME model file exists, using FlameGaussianModel")
        
        # Check if Gumbel network was used during training
        # This is determined by checking if the model was trained with --is_gumbel flag
        # We can infer this from the checkpoint or try to load with is_gumbel=True
        use_gumbel_model = use_gumbel  # Use command-line flag if provided
        
        # Try to detect if Gumbel network exists in checkpoint
        if iteration == -1:
            from utils.system_utils import searchForMaxIteration
            iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        
        # Checkpoint is saved in model_path root, not in iteration_dir
        checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
        
        # If use_gumbel is True, we'll try to load with Gumbel network
        # The restore() method will handle loading the Gumbel state if it exists
        
        # Try to create the appropriate model
        if use_flame_model:
            try:
                # Create model with Gumbel network if requested
                gaussians = FlameGaussianModel(dataset.sh_degree, is_gumbel=use_gumbel_model)
                if use_gumbel_model:
                    print(f"[Info] Created FlameGaussianModel with Gumbel network support")
                else:
                    print(f"[Info] Successfully created FlameGaussianModel")
            except FileNotFoundError as e:
                if "flame2023.pkl" in str(e) or "flame" in str(e).lower():
                    print(f"[Error] FLAME model file not found: {e}")
                    print(f"[Error] Please ensure flame2023.pkl is in flame_model/assets/flame/")
                    print(f"[Error] Or download from https://flame.is.tue.mpg.de/download.php")
                    raise
                else:
                    raise
            except Exception as e:
                print(f"[Error] Failed to create FlameGaussianModel: {e}")
                raise
        else:
            # Use GaussianModel (no binding support, but won't fail)
            gaussians = GaussianModel(dataset.sh_degree)
            print(f"[Info] Using GaussianModel (no FLAME binding support)")
        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # After loading PLY, manually compute binding_counter if binding exists and not already set
        # This is needed for importance_utils functions
        if gaussians.binding is not None:
            if not hasattr(gaussians, 'binding_counter') or gaussians.binding_counter is None:
                max_face_id = gaussians.binding.max().item()
                gaussians.binding_counter = torch.bincount(
                    gaussians.binding,
                    minlength=max_face_id + 1
                ).cuda()
                print(f"[Info] Computed binding_counter: {len(gaussians.binding_counter)} faces")
        
        # Load checkpoint to restore Gumbel network state if available
        if use_gumbel_model and os.path.exists(checkpoint_path):
            try:
                print(f"[Info] Loading checkpoint from {checkpoint_path} to restore Gumbel network...")
                checkpoint_data = torch.load(checkpoint_path)
                model_params, checkpoint_iter = checkpoint_data
                
                # Restore model state (including Gumbel network if present)
                # Create a dummy training_args object for restore()
                from arguments import OptimizationParams
                dummy_parser = ArgumentParser()
                dummy_opt = OptimizationParams(dummy_parser)
                gaussians.restore(model_params, dummy_opt.extract(Namespace()))
                
                print(f"[Info] Checkpoint loaded successfully (iteration {checkpoint_iter})")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint: {e}")
                print(f"[Warning] Gumbel network may not be properly initialized")
        
        # Check if Gumbel network was successfully loaded
        if use_gumbel_model:
            if hasattr(gaussians, 'gumbel_net') and gaussians.gumbel_net is not None:
                print(f"[Info] Gumbel network loaded successfully")
                # Verify network has parameters (not just initialized)
                num_params = sum(p.numel() for p in gaussians.gumbel_net.parameters())
                if num_params > 0:
                    print(f"[Info] Gumbel network has {num_params:,} parameters")
                else:
                    print(f"[Warning] Gumbel network has no parameters, may not be properly loaded")
            else:
                print(f"[Warning] Gumbel network requested but not available after loading")
                use_gumbel_model = False  # Disable Gumbel rendering

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh,
                       compression_ratio=compression_ratio, use_semantic_weights=use_semantic_weights, unbind_selection=unbind_selection,
                       scale_expansion=scale_expansion, scale_expansion_mode=scale_expansion_mode, use_gumbel=use_gumbel_model,
                       interpolate_frames=interpolate_frames, frames_per_timestep=frames_per_timestep)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh,
                          compression_ratio=compression_ratio, use_semantic_weights=use_semantic_weights, unbind_selection=unbind_selection,
                          scale_expansion=scale_expansion, scale_expansion_mode=scale_expansion_mode, use_gumbel=use_gumbel_model,
                          interpolate_frames=interpolate_frames, frames_per_timestep=frames_per_timestep)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh,
                          compression_ratio=compression_ratio, use_semantic_weights=use_semantic_weights, unbind_selection=unbind_selection,
                          scale_expansion=scale_expansion, scale_expansion_mode=scale_expansion_mode, use_gumbel=use_gumbel_model,
                          interpolate_frames=interpolate_frames, frames_per_timestep=frames_per_timestep)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh,
                          compression_ratio=compression_ratio, use_semantic_weights=use_semantic_weights, unbind_selection=unbind_selection,
                          scale_expansion=scale_expansion, scale_expansion_mode=scale_expansion_mode, use_gumbel=use_gumbel_model,
                          interpolate_frames=interpolate_frames, frames_per_timestep=frames_per_timestep)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    
    # Compression-related arguments
    parser.add_argument("--compression_ratio", type=float, default=1.0,
                       help="Compression ratio (0.0-1.0), 1.0 means no compression (default: 1.0)")
    parser.add_argument("--use_semantic_weights", action="store_true",
                       help="Use semantic weights for eyes/mouth regions in importance calculation")
    parser.add_argument("--unbind_selection", action="store_true",
                       help="Use unbind selection mode (allow arbitrary compression ratio, may break binding constraints)")
    parser.add_argument("--scale_expansion", action="store_true",
                       help="Enable scale expansion to compensate for compression (improves visual quality)")
    parser.add_argument("--scale_expansion_mode", type=str, default="volume",
                       choices=["volume", "linear"],
                       help="Scale expansion mode: 'volume' (volume-preserving) or 'linear' (default: volume)")
    parser.add_argument("--use_gumbel", action="store_true",
                       help="Use trained Gumbel network for point selection (requires model trained with --is_gumbel)")
    parser.add_argument("--interpolate_frames", action="store_true")
    parser.add_argument("--frames_per_timestep", type=int, default=1)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    if args.compression_ratio < 1.0:
        print(f"[Compression] Compression ratio: {args.compression_ratio:.2f}")
        if args.use_gumbel:
            print("[Compression] Mode: Gumbel network (learnable selection, supports arbitrary ratios)")
        elif args.unbind_selection:
            print("[Compression] Mode: Unbind (arbitrary ratio allowed)")
        else:
            print("[Compression] Mode: Bind (ensures all faces covered)")
        if args.scale_expansion:
            print(f"[Compression] Scale expansion: {args.scale_expansion_mode} mode")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh,
                compression_ratio=args.compression_ratio,
                use_semantic_weights=args.use_semantic_weights,
                unbind_selection=args.unbind_selection,
                scale_expansion=args.scale_expansion,
                scale_expansion_mode=args.scale_expansion_mode,
                use_gumbel=args.use_gumbel,
                interpolate_frames=args.interpolate_frames,
                frames_per_timestep=args.frames_per_timestep)