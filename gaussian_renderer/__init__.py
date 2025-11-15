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
import math
from typing import Union, Optional
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           selected_mask = None, scale_expansion = False, scale_expansion_mode = "volume"):
    """
    Render the scene with optional compression support.
    
    Background tensor (bg_color) must be on GPU!
    
    Args:
        selected_mask: Boolean mask indicating which points to render (None = all points)
        scale_expansion: If True, expand scales of selected points to compensate compression
        scale_expansion_mode: "volume" or "linear" expansion mode
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Apply selection mask if provided
    if selected_mask is not None:
        # Calculate compression ratio for scale expansion
        compression_ratio = selected_mask.sum().float() / len(selected_mask)
        
        # Calculate scale multiplier if expansion is enabled
        if scale_expansion:
            if scale_expansion_mode == "volume":
                # Volume-preserving expansion: scale_multiplier = (1/ratio)^(1/3)
                scale_multiplier = (1.0 / compression_ratio) ** (1.0 / 3.0)
            elif scale_expansion_mode == "linear":
                # Linear expansion: scale_multiplier = 1 + (1 - ratio) * 0.5
                scale_multiplier = 1.0 + (1.0 - compression_ratio) * 0.5
            else:
                # Default to volume mode
                scale_multiplier = (1.0 / compression_ratio) ** (1.0 / 3.0)
        else:
            scale_multiplier = 1.0
        
        # Apply mask to all point attributes
        means3D = pc.get_xyz[selected_mask]
        means2D = screenspace_points[selected_mask]
        opacity = pc.get_opacity[selected_mask]
        
        # Handle scales and rotations
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)[selected_mask]
            if scale_expansion:
                # For precomputed covariance, scale by scale_multiplier^2
                cov3D_precomp = cov3D_precomp * (scale_multiplier ** 2)
        else:
            scales = pc.get_scaling[selected_mask]
            rotations = pc.get_rotation[selected_mask]
            if scale_expansion:
                scales = scales * scale_multiplier
        
        # Handle colors/SHs
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                features_selected = pc.get_features[selected_mask]
                shs_view = features_selected.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features[selected_mask]
        else:
            colors_precomp = override_color[selected_mask] if override_color.shape[0] == len(selected_mask) else override_color
        
        # Adjust scaling_modifier if scale expansion is enabled
        if scale_expansion:
            scaling_modifier = scaling_modifier * scale_multiplier
    else:
        # No compression, use all points
        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def gumbel_select_render_flame(
    viewpoint_camera, 
    pc : FlameGaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None,
    ratio = 1.0,
    cur_tau = 1.0,
    enforce_binding_constraint = True
):
    """
    Render the scene using Gumbel network for learnable point selection.
    
    This function is similar to FlexGS's gumbel_select_render, but adapted for
    FlameGaussianModel (no deformation network, uses FLAME model instead).
    
    Args:
        viewpoint_camera: Camera viewpoint
        pc: FlameGaussianModel instance (must have gumbel_net)
        pipe: Pipeline parameters
        bg_color: Background color tensor (must be on GPU)
        scaling_modifier: Scaling modifier for Gaussians
        override_color: Override color (optional)
        ratio: Compression ratio (0.0-1.0), where 1.0 = no compression
        cur_tau: Temperature parameter for Gumbel-Softmax
        enforce_binding_constraint: If True, ensure each face has at least one point
    
    Returns:
        Dictionary containing:
        - render: Selected render (using Gumbel-selected points)
        - full_render: Full render (using all points)
        - gumbel_soft_mask: Soft selection mask (differentiable)
        - gumbel_hard_mask: Hard selection mask (discrete)
        - visibility_filter: Visibility filter for selected points
        - radii: Radii for selected points
        - full_radii: Radii for all points
        - viewspace_points: Viewspace points
    """
    # Check if Gumbel network is available
    if not hasattr(pc, 'gumbel_net') or pc.gumbel_net is None:
        raise ValueError("FlameGaussianModel must have gumbel_net initialized (set is_gumbel=True)")
    
    # Create zero tensor for gradients
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Get point attributes
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    # Handle colors/SHs
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Performance optimization: Only render full image if ratio >= 1.0 or if explicitly needed
    # For ratio < 1.0, we skip full render to save computation time
    full_rendered_image = None
    full_radii = None
    selected_rendered_image = None
    selected_radii = None
    gumbel_soft_mask = None
    gumbel_hard_mask = None
    
    # Use Gumbel network to select points if ratio < 1.0
    if ratio < 1.0:
        # Prepare inputs for Gumbel network
        N = means3D.shape[0]
        ratio_tensor = torch.ones(N, 1, dtype=torch.float32, device=means3D.device) * ratio
        
        # Debug: Verify ratio is being passed correctly
        if hasattr(pc, '_debug_print') and pc._debug_print:
            print(f"[Gumbel Debug] Input ratio: {ratio:.4f}, ratio_tensor mean: {ratio_tensor.mean().item():.4f}")
        
        # Get binding information if available
        binding = pc.binding if hasattr(pc, 'binding') and pc.binding is not None else None
        
        # Forward pass through Gumbel network
        _, hard_output, soft1 = pc.gumbel_net(
            means3D, rotations, scales, ratio_tensor, binding, cur_tau=cur_tau
        )
        
        # Debug: Check network output statistics and logits
        if hasattr(pc, '_debug_print') and pc._debug_print:
            # Get raw logits to check if ratio affects network output
            with torch.no_grad():
                pos_emd = pc.gumbel_net.pos_emd(means3D)
                rotation_emd = pc.gumbel_net.rotation_emd(rotations)
                scale_emd = pc.gumbel_net.scale_emd(scales)
                ratio_emd = pc.gumbel_net.ratio_emd(ratio_tensor)
                if binding is not None and pc.gumbel_net.use_binding_hint:
                    binding_max = binding.max()
                    binding_normalized = binding.float() / torch.clamp(binding_max, min=1.0)
                    binding_emd = pc.gumbel_net.binding_emd(binding_normalized.unsqueeze(-1))
                    gumbel_input = torch.cat((pos_emd, rotation_emd, scale_emd, ratio_emd, binding_emd), dim=1)
                else:
                    gumbel_input = torch.cat((pos_emd, rotation_emd, scale_emd, ratio_emd), dim=1)
                raw_logits = pc.gumbel_net.soft_net(gumbel_input)
            
            print(f"[Gumbel Debug] Network output - hard_output range: [{hard_output.min().item():.4f}, {hard_output.max().item():.4f}], mean: {hard_output.mean().item():.4f}")
            print(f"[Gumbel Debug] Network output - soft1 range: [{soft1.min().item():.4f}, {soft1.max().item():.4f}], mean: {soft1.mean().item():.4f}")
            print(f"[Gumbel Debug] Raw logits - not_select: {raw_logits[:, 0].mean().item():.4f} ± {raw_logits[:, 0].std().item():.4f}, select: {raw_logits[:, 1].mean().item():.4f} ± {raw_logits[:, 1].std().item():.4f}")
            print(f"[Gumbel Debug] Ratio encoder output mean: {ratio_emd.mean().item():.4f}, std: {ratio_emd.std().item():.4f}")
            # Check if ratio encoder output is constant (would indicate it's not working)
            if ratio_emd.std().item() < 1e-6:
                print(f"[Gumbel Debug] ⚠️  WARNING: Ratio encoder output is constant! This means ratio is not affecting network output.")
        
        # Create selection masks
        gumbel_soft_mask = soft1  # Differentiable mask (for training)
        gumbel_hard_mask = hard_output  # Discrete mask (for inference)
        selected_mask_hard = gumbel_hard_mask > 0.5  # Boolean mask
        # Store original Gumbel output for reference
        original_gumbel_hard_mask = gumbel_hard_mask.clone()
        
        # Debug: Log initial selection before binding constraint
        initial_selected = selected_mask_hard.sum().item()
        total_points = len(selected_mask_hard)
        initial_ratio = initial_selected / total_points
        
        # FlexGS-style threshold fallback: if Gumbel output doesn't match target ratio well,
        # use threshold selection based on soft mask
        # This handles cases where the network hasn't learned ratio-dependent selection
        # Use relative tolerance: 20% of target ratio, with minimum 5% absolute tolerance
        # This ensures fallback triggers for both small and large ratios
        ratio_tolerance_relative = 0.20  # 20% relative deviation
        ratio_tolerance_absolute = 0.05  # 5% absolute minimum
        ratio_tolerance = max(ratio * ratio_tolerance_relative, ratio_tolerance_absolute)
        
        if abs(initial_ratio - ratio) > ratio_tolerance:
            if hasattr(pc, '_debug_print') and pc._debug_print:
                print(f"[Gumbel Debug] ⚠️  Gumbel output ({initial_ratio:.4f}) doesn't match target ({ratio:.4f}), using threshold fallback")
            # Use threshold selection based on soft mask (similar to FlexGS)
            sorted_imp_list, _ = torch.sort(gumbel_soft_mask, dim=0)
            index_nth_percentile = int((1 - ratio) * (sorted_imp_list.shape[0] - 1))
            value_nth_percentile = sorted_imp_list[index_nth_percentile]
            selected_mask_hard = gumbel_soft_mask > value_nth_percentile
            # Update gumbel_hard_mask to reflect fallback selection (for return value)
            gumbel_hard_mask = selected_mask_hard.float()
            # Recalculate initial_ratio after fallback
            initial_selected = selected_mask_hard.sum().item()
            initial_ratio = initial_selected / total_points
            if hasattr(pc, '_debug_print') and pc._debug_print:
                print(f"[Gumbel Debug] After threshold fallback: {initial_selected}/{total_points} ({initial_ratio:.4f})")
        
        if hasattr(pc, '_debug_print') and pc._debug_print:
            print(f"[Gumbel Debug] Initial selection: {initial_selected}/{total_points} ({initial_ratio:.4f}), target ratio: {ratio:.4f}")
        
        # Enforce binding constraint if requested
        if enforce_binding_constraint and binding is not None:
            # Ensure each face has at least one point
            # Performance optimization: use vectorized operations instead of CPU loop
            unique_faces = torch.unique(binding[selected_mask_hard])
            # Performance optimization: avoid .item() to prevent CPU-GPU synchronization
            if hasattr(pc, 'binding_counter') and pc.binding_counter is not None:
                total_faces = len(pc.binding_counter)
            else:
                # Fallback: use binding.max() + 1, but try to avoid sync
                # Note: This is only called once per render, not in loop
                binding_max_gpu = binding.max()
                # Use torch operations to compute total_faces on GPU, then convert only once
                total_faces = int(binding_max_gpu.item()) + 1  # Sync only once, outside loop
            
            # Find uncovered faces (vectorized)
            all_faces = torch.arange(total_faces, device=binding.device)
            covered_faces = torch.zeros(total_faces, dtype=torch.bool, device=binding.device)
            covered_faces[unique_faces] = True
            uncovered_faces = all_faces[~covered_faces]
            
            # Vectorized optimization: process all uncovered faces at once
            if len(uncovered_faces) > 0:
                # Get all points for uncovered faces
                uncovered_mask = torch.isin(binding, uncovered_faces)
                uncovered_indices = torch.where(uncovered_mask)[0]
                
                if len(uncovered_indices) > 0:
                    # Group by face_id and find max for each face (vectorized)
                    uncovered_bindings = binding[uncovered_indices]
                    uncovered_soft_values = gumbel_soft_mask[uncovered_indices]
                    
                    # Use scatter operations for better performance
                    # For each uncovered face, find the point with max soft value
                    for face_id in uncovered_faces:
                        # Vectorized: find all points for this face
                        face_mask = (uncovered_bindings == face_id)
                        if face_mask.any():
                            face_local_indices = torch.where(face_mask)[0]
                            face_global_indices = uncovered_indices[face_local_indices]
                            face_values = uncovered_soft_values[face_local_indices]
                            
                            # Find max value index
                            max_local_idx = face_values.argmax()
                            max_global_idx = face_global_indices[max_local_idx]
                            selected_mask_hard[max_global_idx] = True
            
            # Debug: Log final selection after binding constraint
            final_selected = selected_mask_hard.sum().item()
            final_ratio = final_selected / total_points
            if hasattr(pc, '_debug_print') and pc._debug_print:
                added_by_constraint = final_selected - initial_selected
                min_ratio_needed = total_faces / total_points
                print(f"[Gumbel Debug] After binding constraint: {final_selected}/{total_points} ({final_ratio:.4f}), added {added_by_constraint} points")
                print(f"[Gumbel Debug] Min ratio needed (one point per face): {min_ratio_needed:.4f}")
                if final_ratio >= min_ratio_needed:
                    print(f"[Gumbel Debug] ⚠️  Final ratio ({final_ratio:.4f}) >= min ratio ({min_ratio_needed:.4f})")
                    print(f"[Gumbel Debug]    This means binding constraint is dominating the selection!")
                    print(f"[Gumbel Debug]    For ratios < {min_ratio_needed:.4f}, results will be similar due to binding constraint.")
        
        # Apply selection mask
        selected_means3D = means3D[selected_mask_hard]
        selected_means2D = means2D[selected_mask_hard]
        selected_opacity = opacity[selected_mask_hard]
        selected_scales = scales[selected_mask_hard]
        selected_rotations = rotations[selected_mask_hard]
        
        # Handle colors/SHs for selected points
        selected_shs = None
        selected_colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                selected_colors_precomp = colors_precomp[selected_mask_hard] if colors_precomp is not None else None
            else:
                selected_shs = shs[selected_mask_hard] if shs is not None else None
        else:
            selected_colors_precomp = override_color[selected_mask_hard] if override_color.shape[0] == len(selected_mask_hard) else override_color
        
        # Render selected points
        selected_rendered_image, selected_radii = rasterizer(
            means3D=selected_means3D,
            means2D=selected_means2D,
            shs=selected_shs,
            colors_precomp=selected_colors_precomp,
            opacities=selected_opacity,
            scales=selected_scales,
            rotations=selected_rotations,
            cov3D_precomp=None
        )
    else:
        # ratio >= 1.0: render all points (no selection)
        selected_rendered_image, selected_radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        # For ratio >= 1.0, full_render is same as selected_render
        full_rendered_image = selected_rendered_image
        full_radii = selected_radii
    
    # Only compute full_render if not already computed (for loss computation)
    # This is optional and can be disabled for better performance
    if full_rendered_image is None:
        # Skip full render for better performance (can be enabled if needed for loss)
        full_rendered_image = selected_rendered_image
        full_radii = selected_radii
    
    return {
        "render": selected_rendered_image,
        "full_render": full_rendered_image,
        "gumbel_soft_mask": gumbel_soft_mask,
        "gumbel_hard_mask": gumbel_hard_mask,
        "visibility_filter": selected_radii > 0,
        "radii": selected_radii,
        "full_radii": full_radii,
        "viewspace_points": screenspace_points,
    }
