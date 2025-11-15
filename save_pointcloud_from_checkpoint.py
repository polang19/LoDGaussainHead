#!/usr/bin/env python3
"""
Script to save point cloud from checkpoint.
This is useful when training completed but point cloud was not saved.
"""

import torch
import os
import sys
from argparse import ArgumentParser, Namespace
from scene import Scene
from scene.flame_gaussian_model import FlameGaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

def save_pointcloud_from_checkpoint(model_path, iteration, source_path, bind_to_mesh=True, white_background=False):
    """
    Load checkpoint and save point cloud.
    
    Args:
        model_path: Path to model directory
        iteration: Iteration number to load
        source_path: Path to source dataset (required for Scene initialization)
        bind_to_mesh: Whether to use FLAME binding
        white_background: Whether to use white background
    """
    print(f"Loading checkpoint from {model_path} at iteration {iteration}")
    
    # Create model
    if bind_to_mesh:
        gaussians = FlameGaussianModel(sh_degree=3, is_gumbel=True)  # Enable Gumbel if checkpoint has it
    else:
        from scene.gaussian_model import GaussianModel
        gaussians = GaussianModel(sh_degree=3)
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path)
    model_params, checkpoint_iter = checkpoint_data
    
    # Create dummy training args for restore
    parser = ArgumentParser()
    opt = OptimizationParams(parser)
    dummy_args = Namespace()
    for key, value in vars(opt).items():
        setattr(dummy_args, key, value)
    training_args = opt.extract(dummy_args)
    
    # Restore model state
    gaussians.restore(model_params, training_args)
    print(f"Model restored from checkpoint (iteration {checkpoint_iter})")
    
    # Create dataset args for Scene
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    dummy_args = Namespace(
        source_path=source_path,
        model_path=model_path,
        images="images",
        resolution=-1,
        white_background=white_background,
        eval=False,
        bind_to_mesh=bind_to_mesh,
        disable_flame_static_offset=False,
        not_finetune_flame_params=False,
        select_camera_id=-1,
        target_path="",
        sh_degree=3
    )
    dataset = model_params.extract(dummy_args)
    
    # Create scene (this will load meshes if needed)
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    
    # Manually set the point cloud data from checkpoint
    # The checkpoint already has all the point cloud data in gaussians
    
    # Save point cloud
    print(f"Saving point cloud to iteration_{iteration}...")
    scene.save(iteration)
    
    print(f"Point cloud saved successfully!")
    print(f"Location: {os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')}")
    
    return True

if __name__ == "__main__":
    parser = ArgumentParser(description="Save point cloud from checkpoint")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--iteration", type=int, required=True,
                       help="Iteration number to load from checkpoint")
    parser.add_argument("--source_path", "-s", type=str, required=True,
                       help="Path to source dataset")
    parser.add_argument("--bind_to_mesh", action="store_true", default=True,
                       help="Use FLAME binding (default: True)")
    parser.add_argument("--white_background", action="store_true",
                       help="Use white background")
    
    args = parser.parse_args()
    
    success = save_pointcloud_from_checkpoint(
        args.model_path,
        args.iteration,
        args.source_path,
        bind_to_mesh=args.bind_to_mesh,
        white_background=args.white_background
    )
    
    if success:
        print("\n✓ Point cloud saved successfully!")
        print(f"You can now use render.py with --iteration {args.iteration}")
    else:
        print("\n✗ Failed to save point cloud")
        sys.exit(1)

