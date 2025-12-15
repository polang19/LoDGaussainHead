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

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, gumbel_select_render_flame
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Determine if Gumbel training is enabled
    is_gumbel = getattr(opt, 'is_gumbel', False)
    
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(
            dataset.sh_degree, 
            dataset.disable_flame_static_offset, 
            dataset.not_finetune_flame_params,
            is_gumbel=is_gumbel
        )
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    # Initialize importance scores for Gumbel training (if enabled)
    imp_list = None
    importance_computer = None
    if is_gumbel and hasattr(gaussians, 'gumbel_net') and gaussians.gumbel_net is not None:
        # Initialize importance list (will be updated during training)
        imp_list = None
        select_interval = getattr(opt, 'select_interval', 1000)  # Default 1000 to match FlexGS
        time_ratios = getattr(opt, 'time_ratios', [0.3, 0.5, 0.7])
        # Initialize async importance computer to avoid blocking GPU
        # Automatically uses GPU acceleration (faiss) if available, falls back to CPU otherwise
        from utils.importance_async import AsyncImportanceComputer
        importance_computer = AsyncImportanceComputer(use_gpu=True)  # Try GPU first, auto-fallback to CPU
        print(f"[Gumbel Training] Enabled with ratios: {time_ratios}, select_interval: {select_interval}")
        print(f"[Gumbel Training] Using async importance computation (GPU-accelerated if faiss available)")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Optimize data loading: increase prefetch for better GPU utilization
    loader_camera_train = DataLoader(
        scene.getTrainCameras(), 
        batch_size=None, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4  # Prefetch more batches to reduce GPU waiting
    )
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        if iteration == first_iter:
            print(f"[DEBUG] Before Gumbel check, iteration={iteration}")
        
        # Gumbel training: use gumbel_select_render_flame if enabled
        # Only enable Gumbel training after gumbel_start_iter
        gumbel_start_iter = getattr(opt, 'gumbel_start_iter', 0)
        use_gumbel = (is_gumbel and 
                     iteration >= gumbel_start_iter and
                     hasattr(gaussians, 'gumbel_net') and 
                     gaussians.gumbel_net is not None)
        
        if iteration == first_iter:
            print(f"[DEBUG] use_gumbel={use_gumbel}, iteration={iteration}")
        if use_gumbel:
            time_ratios = getattr(opt, 'time_ratios', [0.3, 0.5, 0.7])
            ratio_weights = getattr(opt, 'ratio_weights', None)
            if ratio_weights is None or len(ratio_weights) != len(time_ratios):
                ratio_weights = [1.0 / len(time_ratios)] * len(time_ratios)
            ratio_weights = np.array(ratio_weights, dtype=np.float64)
            ratio_weights = ratio_weights / ratio_weights.sum()
            ratio = float(np.random.choice(time_ratios, p=ratio_weights))
            
            # Update importance scores periodically using async computation
            # Note: FlexGS uses select_interval=1000 for better performance
            # We use 1000 as default to match FlexGS performance
            # Also delay importance computation for early iterations to speed up training
            select_interval = getattr(opt, 'select_interval', 1000)
            delay_iterations = getattr(opt, 'delay_iterations', 10000)
            
            # Check if we should start/update importance computation
            # Skip computation on first iteration to avoid blocking
            should_compute_importance = (
                (imp_list is None and iteration > first_iter) or 
                (iteration >= delay_iterations and iteration % select_interval == 0)
            )
            
            if should_compute_importance and hasattr(gaussians, '_xyz') and gaussians._xyz.shape[0] > 0:
                # Start async computation (non-blocking)
                if importance_computer is not None:
                    if not importance_computer.is_busy():
                        importance_computer.start_computation(gaussians._xyz, k=10)
            
            # Check for completed computation (non-blocking)
            if importance_computer is not None:
                new_imp_list = importance_computer.get_result(
                    device=gaussians._xyz.device,
                    dtype=gaussians._xyz.dtype
                )
                if new_imp_list is not None:
                    imp_list = new_imp_list
                    print(f"[ITER {iteration}] Updated importance scores (async)")
            
            # Fallback: if no async result and imp_list is None, use simple uniform importance
            if imp_list is None:
                # Use uniform importance as fallback (fast, no computation)
                imp_list = torch.ones(
                    gaussians._xyz.shape[0],
                    device=gaussians._xyz.device,
                    dtype=gaussians._xyz.dtype
                )
            
            # Gumbel render
            render_pkg = gumbel_select_render_flame(
                viewpoint_cam, gaussians, pipe, background, 
                ratio=ratio, cur_tau=1.0, enforce_binding_constraint=True
            )
            selected_image = render_pkg["render"]  # Selected render (using Gumbel-selected points)
            full_image = render_pkg.get("full_render", selected_image)  # Full render (all points)
            image = selected_image  # Use selected render for main loss
            gumbel_soft_mask = render_pkg.get("gumbel_soft_mask")
            gumbel_hard_mask = render_pkg.get("gumbel_hard_mask")
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
        else:
            # Standard render (no Gumbel)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image = render_pkg["render"]
            selected_image = None
            full_image = None
            gumbel_soft_mask = None
            gumbel_hard_mask = None
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

        # Loss
        # Optimize: check if already on GPU to avoid unnecessary transfer
        if viewpoint_cam.original_image.is_cuda:
            gt_image = viewpoint_cam.original_image
        else:
            gt_image = viewpoint_cam.original_image.cuda()

        losses = {}
        # Render loss computation
        if use_gumbel and full_image is not None and selected_image is not None:
            # Gumbel training: combine full and selected render losses
            # Performance optimization: if full_image == selected_image (ratio >= 1.0), only compute once
            gumbel_weight = getattr(opt, 'gumbel_weight', 1.0)
            full_weight = 0.5  # Weight for combining full and selected losses
            
            # Check if full and selected are the same (ratio >= 1.0 case)
            # Use torch.allclose for efficient comparison
            if torch.allclose(full_image, selected_image, atol=1e-6):
                # Same image, only compute loss once
                losses['l1'] = l1_loss(selected_image, gt_image) * (1.0 - opt.lambda_dssim)
                losses['ssim'] = (1.0 - ssim(selected_image, gt_image)) * opt.lambda_dssim
            else:
                # Different images, compute both
                losses['l1'] = (l1_loss(full_image, gt_image) * (1.0 - opt.lambda_dssim) * full_weight + 
                               l1_loss(selected_image, gt_image) * (1.0 - opt.lambda_dssim) * full_weight)
                losses['ssim'] = ((1.0 - ssim(full_image, gt_image)) * opt.lambda_dssim * full_weight + 
                                (1.0 - ssim(selected_image, gt_image)) * opt.lambda_dssim * full_weight)
        else:
            # Standard training: use single render
            losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
            losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim

        if gaussians.binding != None:
            # Critical: Check if visibility_filter size matches current point count
            # Point count may change due to densification/pruning after rendering
            current_point_count = gaussians._xyz.shape[0]
            visibility_filter_size = visibility_filter.shape[0] if visibility_filter is not None else 0
            
            if visibility_filter_size != current_point_count:
                # visibility_filter size doesn't match, skip xyz loss for this iteration
                if iteration % 100 == 0:
                    print(f"[WARNING] Iter {iteration}: visibility_filter size ({visibility_filter_size}) != point count ({current_point_count}), skipping xyz loss")
            else:
                if opt.metric_xyz:
                    losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
                else:
                    # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                    losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                # Check visibility_filter size (already checked above, but check again for safety)
                if visibility_filter_size == current_point_count:
                    if opt.metric_scale:
                        losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                    else:
                        # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                        losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        
        # Gumbel loss (if enabled)
        if use_gumbel and gumbel_soft_mask is not None and gumbel_hard_mask is not None and imp_list is not None:
            # Critical: Check if mask size matches current point count
            # Point count may change due to densification/pruning after rendering
            current_point_count = gaussians._xyz.shape[0]
            mask_size = gumbel_soft_mask.shape[0]
            
            if mask_size != current_point_count:
                # Mask size doesn't match current point count (densification/pruning happened)
                # Skip Gumbel loss for this iteration to avoid IndexError
                if iteration % 100 == 0:
                    print(f"[WARNING] Iter {iteration}: Gumbel mask size ({mask_size}) != point count ({current_point_count}), skipping Gumbel loss")
            else:
                # Mask size matches, proceed with normal Gumbel loss calculation
                # Also check if imp_list size matches (may be stale after densification/pruning)
                if imp_list.shape[0] != current_point_count:
                    # imp_list is stale, skip Gumbel loss for this iteration
                    if iteration % 100 == 0:
                        print(f"[WARNING] Iter {iteration}: imp_list size ({imp_list.shape[0]}) != point count ({current_point_count}), skipping Gumbel loss")
                else:
                    # All sizes match, proceed with Gumbel loss calculation
                    gumbel_weight = getattr(opt, 'gumbel_weight', 1.0)
                    time_ratios = getattr(opt, 'time_ratios', [0.3, 0.5, 0.7])
                    ratio_weights = getattr(opt, 'ratio_weights', None)
                    if ratio_weights is None or len(ratio_weights) != len(time_ratios):
                        ratio_weights = [1.0 / len(time_ratios)] * len(time_ratios)
                    ratio_weights = np.array(ratio_weights, dtype=np.float64)
                    ratio_weights = ratio_weights / ratio_weights.sum()
                    ratio = float(np.random.choice(time_ratios, p=ratio_weights))
                    
                    # Generate pseudo-label from importance scores
                    N = imp_list.shape[0]
                    k = max(1, min(int((1 - ratio) * N), N))
                    value_nth_percentile, _ = torch.kthvalue(imp_list, k, dim=0)
                    pseudo_gt_mask = (imp_list > value_nth_percentile).float()
                    
                    # L_hard: Match pseudo-label
                    if gumbel_hard_mask.shape[0] == pseudo_gt_mask.shape[0]:
                        losses['gumbel_hard'] = F.l1_loss(gumbel_hard_mask.float(), pseudo_gt_mask) * gumbel_weight
                    
                    # L_hard_1: Constrain selection ratio
                    actual_ratio = gumbel_hard_mask.sum() / len(gumbel_hard_mask)
                    target_ratio = torch.tensor(ratio, device=gumbel_hard_mask.device)
                    # Increased weight from 0.01 to 1.0 to enforce ratio constraint
                    # This is critical for the network to learn ratio-dependent selection
                    losses['gumbel_ratio'] = F.l1_loss(actual_ratio.unsqueeze(0), target_ratio.unsqueeze(0)) * opt.gumbel_ratio_weight
                    
                    # Debug: log actual vs target ratio (every 100 iterations)
                    if iteration % 100 == 0:
                        print(f"[ITER {iteration}] Gumbel ratio: actual={actual_ratio.item():.4f}, target={target_ratio.item():.4f}, loss={losses['gumbel_ratio'].item():.7f}")
        
        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                if 'gumbel_hard' in losses:
                    postfix["gumbel_hard"] = f"{losses['gumbel_hard']:.{7}f}"
                if 'gumbel_ratio' in losses:
                    # Use more precision to show small values
                    gumbel_ratio_val = losses['gumbel_ratio'].item()
                    if gumbel_ratio_val < 0.0000001:
                        postfix["gumbel_ratio"] = f"{gumbel_ratio_val:.9f}"
                    else:
                        postfix["gumbel_ratio"] = f"{gumbel_ratio_val:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Critical: Check if all tensors size match current point count before indexing
                # Note: All these tensors are from render result, size may not match if densification/pruning happened
                current_point_count_densify = gaussians._xyz.shape[0]
                visibility_filter_size_densify = visibility_filter.shape[0] if visibility_filter is not None else 0
                viewspace_point_tensor_size = viewspace_point_tensor.shape[0] if viewspace_point_tensor is not None else 0
                radii_size = radii.shape[0] if radii is not None else 0
                
                if (visibility_filter_size_densify == current_point_count_densify and
                    viewspace_point_tensor_size == current_point_count_densify and
                    radii_size == current_point_count_densify):
                    # All sizes match, proceed with densification stats
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    # Size mismatch, skip densification stats for this iteration
                    if iteration % 100 == 0:
                        print(f"[WARNING] Iter {iteration}: Size mismatch - visibility_filter: {visibility_filter_size_densify}, viewspace_point_tensor: {viewspace_point_tensor_size}, radii: {radii_size}, point_count: {current_point_count_densify}, skipping densification stats")

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        if 'gumbel_hard' in losses:
            tb_writer.add_scalar('train_loss_patches/gumbel_hard', losses['gumbel_hard'].item(), iteration)
        if 'gumbel_ratio' in losses:
            tb_writer.add_scalar('train_loss_patches/gumbel_ratio', losses['gumbel_ratio'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        # Only clear cache periodically to avoid performance impact
        if iteration % 1000 == 0:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
