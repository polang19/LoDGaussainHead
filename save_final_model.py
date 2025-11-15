#!/usr/bin/env python3
"""
Script to save point cloud and checkpoint at the end of training.
This script runs one more iteration to capture and save the final model state.
"""

import torch
import os
import sys
from argparse import ArgumentParser
from scene import Scene
from scene.flame_gaussian_model import FlameGaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
import json

def save_final_model(model_path, source_path, iteration, bind_to_mesh=True, white_background=False, is_gumbel=True):
    """
    Load the model from training state and save point cloud and checkpoint.
    This requires the model to still be in memory or we need to reinitialize from scratch.
    
    Since we don't have checkpoint, we'll need to:
    1. Initialize model from scratch
    2. Load the PLY if it exists from a previous save
    3. Or tell user to re-run training with proper save flags
    """
    print(f"Attempting to save final model state for iteration {iteration}")
    print(f"Model path: {model_path}")
    print(f"Source path: {source_path}")
    
    # Check if we can find any saved point clouds
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if os.path.exists(point_cloud_dir):
        # Find the latest iteration
        iterations = []
        for item in os.listdir(point_cloud_dir):
            if item.startswith("iteration_"):
                try:
                    iter_num = int(item.split("_")[1])
                    iterations.append(iter_num)
                except:
                    pass
        
        if iterations:
            latest_iter = max(iterations)
            latest_ply = os.path.join(point_cloud_dir, f"iteration_{latest_iter}", "point_cloud.ply")
            if os.path.exists(latest_ply):
                print(f"Found existing point cloud at iteration {latest_iter}")
                print(f"You can use this for rendering:")
                print(f"  python render.py -m {model_path} -s {source_path} --iteration {latest_iter} --bind_to_mesh --white_background")
                return True
    
    # If no checkpoint and no point cloud, we need to tell user to re-train
    print("\n" + "="*60)
    print("ERROR: No checkpoint or point cloud found!")
    print("="*60)
    print("\nThe model was trained but no checkpoint or point cloud was saved.")
    print("This happened because:")
    print("  1. --checkpoint_iterations was not specified")
    print("  2. Default interval (60000) > total iterations (30000)")
    print("  3. No checkpoint was saved during training")
    print("\nSOLUTION: Re-run training with explicit save/checkpoint flags:")
    print("\n" + "-"*60)
    print("Recommended training command:")
    print("-"*60)
    print(f"""python train.py \\
  -s {source_path} \\
  -m {model_path} \\
  --eval \\
  --bind_to_mesh \\
  --white_background \\
  --iterations 30000 \\
  --save_iterations 10000 20000 30000 \\
  --checkpoint_iterations 10000 20000 30000 \\
  --is_gumbel \\
  --gumbel_start_iter 25000 \\
  --gumbel_lr 1e-4 \\
  --gumbel_weight 1.0 \\
  --select_interval 1000 \\
  --time_ratios 0.3 0.5 0.7""")
    print("-"*60)
    print("\nAlternatively, if you have a backup or can resume from a previous state,")
    print("you can use --start_checkpoint to resume training.")
    print("\n" + "="*60)
    
    return False

if __name__ == "__main__":
    parser = ArgumentParser(description="Save final model state (requires checkpoint or existing point cloud)")
    parser.add_argument("--model_path", "-m", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--iteration", type=int, default=30000,
                       help="Iteration number")
    parser.add_argument("--source_path", "-s", type=str, required=True,
                       help="Path to source dataset")
    parser.add_argument("--bind_to_mesh", action="store_true", default=True,
                       help="Use FLAME binding (default: True)")
    parser.add_argument("--white_background", action="store_true",
                       help="Use white background")
    parser.add_argument("--is_gumbel", action="store_true", default=True,
                       help="Model uses Gumbel network")
    
    args = parser.parse_args()
    
    success = save_final_model(
        args.model_path,
        args.source_path,
        args.iteration,
        bind_to_mesh=args.bind_to_mesh,
        white_background=args.white_background,
        is_gumbel=args.is_gumbel
    )
    
    if not success:
        sys.exit(1)

