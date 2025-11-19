#
# Importance calculation and selection utilities for Gaussian compression
# Part of Phase A: Fixed importance selection (non-differentiable)
#
# This module provides functions for computing importance scores and selecting
# Gaussians based on spatial density and FLAME face binding information.
# All functions are designed to be independent and can be used flexibly.
#

import torch
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional


def compute_spatial_density(xyz: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute spatial density for each point based on k-nearest neighbors.
    
    Density = 1 / (mean_k_nearest_distance + eps)
    Higher density means the point is in a more crowded region.
    
    This function is independent and can be used without binding information.
    
    Args:
        xyz: Point positions, shape (N, 3), on GPU or CPU
        k: Number of nearest neighbors to consider (default: 10)
    
    Returns:
        density: Density scores, shape (N,), on the same device as xyz
    
    Example:
        >>> xyz = torch.randn(1000, 3).cuda()
        >>> density = compute_spatial_density(xyz, k=10)
        >>> print(density.shape)  # torch.Size([1000])
    """
    device = xyz.device
    dtype = xyz.dtype
    
    # Convert to CPU numpy for scipy (scipy doesn't support GPU)
    xyz_np = xyz.detach().cpu().numpy()
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(xyz_np)
    
    # Query k+1 neighbors (including the point itself)
    distances, _ = tree.query(xyz_np, k=k+1)
    
    # Exclude the point itself (distance=0) and compute mean distance
    # distances shape: (N, k+1), we take columns 1 to k+1
    mean_distances = distances[:, 1:].mean(axis=1)
    
    # Compute density: higher density for smaller distances
    # Add small epsilon to avoid division by zero
    density = 1.0 / (mean_distances + 1e-6)
    
    # Convert back to original device and dtype
    return torch.tensor(density, device=device, dtype=dtype)


def compute_semantic_face_weights(
    pc, 
    eye_weight: float = 2.0,
    mouth_weight: float = 1.5,
    default_weight: float = 1.0,
    enable_debug: bool = False
) -> Optional[torch.Tensor]:
    """
    Compute semantic face weights based on spatial location.
    
    Uses heuristics to identify eye and mouth regions based on point positions:
    - Eye regions: high y/z, x on sides (left/right)
    - Mouth region: low y/z, x near center
    
    This function requires binding information. If pc.binding is None, returns None.
    
    Args:
        pc: GaussianModel or FlameGaussianModel with binding information
        eye_weight: Weight for eye regions (default: 2.0)
        mouth_weight: Weight for mouth region (default: 1.5)
        default_weight: Weight for other regions (default: 1.0)
        enable_debug: If True, print debug information (default: False)
    
    Returns:
        face_weights: Semantic weights for each face, shape (num_faces,)
                     Returns None if pc.binding is None
    
    Example:
        >>> # Only works with FlameGaussianModel that has binding
        >>> if gaussians.binding is not None:
        >>>     face_weights = compute_semantic_face_weights(gaussians, enable_debug=True)
    """
    if pc.binding is None:
        return None
    
    num_faces = len(pc.binding_counter)
    device = pc._xyz.device
    face_weights = torch.ones(num_faces, device=device, dtype=pc._xyz.dtype) * default_weight
    
    # Compute center position for each face (average of bound points)
    face_centers = torch.zeros(num_faces, 3, device=device)
    face_point_counts = torch.zeros(num_faces, device=device, dtype=torch.long)
    
    for face_id in range(num_faces):
        face_point_mask = (pc.binding == face_id)
        face_point_indices = face_point_mask.nonzero(as_tuple=False).squeeze(-1)
        
        if len(face_point_indices) > 0:
            if face_point_indices.dim() == 0:
                face_point_indices = face_point_indices.unsqueeze(0)
            
            face_points = pc._xyz[face_point_indices]
            face_centers[face_id] = face_points.mean(dim=0)
            face_point_counts[face_id] = len(face_point_indices)
    
    # Normalize positions (assuming face is roughly centered and normalized)
    # Compute bounding box
    all_centers = face_centers[face_point_counts > 0]
    if len(all_centers) == 0:
        return face_weights
    
    center_min = all_centers.min(dim=0)[0]
    center_max = all_centers.max(dim=0)[0]
    center_range = center_max - center_min
    center_range = torch.clamp(center_range, min=1e-6)
    
    # Normalize to [-1, 1] range
    normalized_centers = 2.0 * (face_centers - center_min.unsqueeze(0)) / center_range.unsqueeze(0) - 1.0
    
    # Determine which axis is height by checking range
    # The axis with largest range is likely the height axis
    xyz_ranges = [
        (center_range[0].item(), 0),  # x
        (center_range[1].item(), 1),  # y
        (center_range[2].item(), 2),  # z
    ]
    height_axis = max(xyz_ranges)[1]
    
    # Heuristic: Identify eye and mouth regions
    # Try both y and z as height axis, use the one with larger range
    if height_axis == 2:  # z is height
        # Eye regions: z > 0.0, |x| > 0.1 (high and on sides)
        # Mouth region: z < -0.1, |x| < 0.5 (low and near center)
        eye_mask = (normalized_centers[:, 2] > 0.0) & (torch.abs(normalized_centers[:, 0]) > 0.1)
        mouth_mask = (normalized_centers[:, 2] < -0.1) & (torch.abs(normalized_centers[:, 0]) < 0.5)
    else:  # y is height (default)
        # Eye regions: y > 0.0, |x| > 0.1 (high and on sides)
        # Mouth region: y < -0.1, |x| < 0.5 (low and near center)
        eye_mask = (normalized_centers[:, 1] > 0.0) & (torch.abs(normalized_centers[:, 0]) > 0.1)
        mouth_mask = (normalized_centers[:, 1] < -0.1) & (torch.abs(normalized_centers[:, 0]) < 0.5)
    
    # Apply weights
    face_weights[eye_mask] = eye_weight
    face_weights[mouth_mask] = mouth_weight
    
    # Debug output: check if regions are identified
    eye_count = eye_mask.sum().item()
    mouth_count = mouth_mask.sum().item()
    
    # Fallback: if still no regions identified, use even more lenient thresholds
    if eye_count == 0 and mouth_count == 0:
        if height_axis == 2:
            # Very lenient: z > -0.2, |x| > 0.0 (almost any high point on sides)
            eye_mask = (normalized_centers[:, 2] > -0.2) & (torch.abs(normalized_centers[:, 0]) > 0.0)
            mouth_mask = (normalized_centers[:, 2] < 0.0) & (torch.abs(normalized_centers[:, 0]) < 0.6)
        else:
            eye_mask = (normalized_centers[:, 1] > -0.2) & (torch.abs(normalized_centers[:, 0]) > 0.0)
            mouth_mask = (normalized_centers[:, 1] < 0.0) & (torch.abs(normalized_centers[:, 0]) < 0.6)
        face_weights[eye_mask] = eye_weight
        face_weights[mouth_mask] = mouth_weight
        
        # Update counts after fallback
        eye_count = eye_mask.sum().item()
        mouth_count = mouth_mask.sum().item()
    
    # Print debug info if enabled
    if enable_debug:
        print(f"[Semantic Weights] Height axis: {'z' if height_axis == 2 else 'y'}")
        print(f"[Semantic Weights] Eye faces identified: {eye_count} ({eye_count/num_faces*100:.2f}%)")
        print(f"[Semantic Weights] Mouth faces identified: {mouth_count} ({mouth_count/num_faces*100:.2f}%)")
        if eye_count == 0 and mouth_count == 0:
            print(f"[Semantic Weights] WARNING: No key regions identified! Semantic weights may be ineffective.")
    
    return face_weights


def compute_binding_importance(
    pc, 
    face_weights: Optional[torch.Tensor] = None,
    use_semantic_weights: bool = False,
    eye_weight: float = 2.0,
    mouth_weight: float = 1.5,
    enable_debug: bool = False
) -> torch.Tensor:
    """
    Compute importance based on FLAME face binding.
    
    Can use uniform weights, provided weights, or semantic weights (eyes/mouth regions).
    If pc.binding is None, returns uniform importance for all points.
    
    Args:
        pc: GaussianModel or FlameGaussianModel with binding information
        face_weights: Optional weights for each face, shape (num_faces,)
                     If None and use_semantic_weights=False, uses uniform weights
        use_semantic_weights: If True, compute semantic weights based on spatial location
        eye_weight: Weight for eye regions (default: 2.0, only used if use_semantic_weights=True)
        mouth_weight: Weight for mouth region (default: 1.5, only used if use_semantic_weights=True)
        enable_debug: If True, enable debug output in semantic weight computation (default: False)
    
    Returns:
        importance: Importance scores for each point, shape (num_points,)
    
    Example:
        >>> # With binding (FlameGaussianModel)
        >>> binding_imp = compute_binding_importance(gaussians, use_semantic_weights=True)
        >>> 
        >>> # Without binding (GaussianModel)
        >>> binding_imp = compute_binding_importance(gaussians)  # Returns uniform importance
    """
    if pc.binding is None:
        # No binding information, return uniform importance
        return torch.ones(len(pc._xyz), device=pc._xyz.device, dtype=pc._xyz.dtype)
    
    num_faces = len(pc.binding_counter)
    
    if face_weights is None:
        if use_semantic_weights:
            # Compute semantic weights based on spatial location
            face_weights = compute_semantic_face_weights(
                pc, 
                eye_weight=eye_weight, 
                mouth_weight=mouth_weight,
                enable_debug=enable_debug
            )
            if face_weights is None:
                # Fallback to uniform weights if semantic weights failed
                face_weights = torch.ones(num_faces, device=pc._xyz.device, dtype=pc._xyz.dtype)
        else:
            # Uniform weights for all faces
            face_weights = torch.ones(num_faces, device=pc._xyz.device, dtype=pc._xyz.dtype)
    else:
        # Ensure face_weights is on the correct device
        face_weights = face_weights.to(pc._xyz.device)
        if len(face_weights) != num_faces:
            raise ValueError(f"face_weights length ({len(face_weights)}) must match number of faces ({num_faces})")
    
    # Map face weights to points via binding
    point_importance = face_weights[pc.binding]
    
    return point_importance


def select_with_binding_constraint(
    pc, 
    importance_scores: torch.Tensor, 
    ratio: float
) -> torch.Tensor:
    """
    Select points based on importance scores while ensuring binding constraints.
    
    Strategy:
    1. First, ensure each face has at least one point (select the most important point per face)
    2. Then, fill remaining quota based on importance scores
    
    This function requires binding information. If pc.binding is None, performs simple top-k selection.
    
    Args:
        pc: GaussianModel or FlameGaussianModel with binding information
        importance_scores: Importance scores for each point, shape (num_points,)
        ratio: Compression ratio (0.0-1.0), where 1.0 means no compression
    
    Returns:
        selected_mask: Boolean mask indicating selected points, shape (num_points,)
    
    Example:
        >>> importance = compute_binding_importance(gaussians)
        >>> selected = select_with_binding_constraint(gaussians, importance, ratio=0.5)
        >>> print(f"Selected {selected.sum().item()} out of {len(selected)} points")
    """
    total_points = len(pc._xyz)
    target_points = int(total_points * ratio)
    
    device = pc._xyz.device
    selected_mask = torch.zeros(total_points, dtype=torch.bool, device=device)
    
    # Handle case where binding is None (no constraints)
    if pc.binding is None:
        # Simple selection: just take top-k by importance
        sorted_indices = torch.argsort(importance_scores, descending=True)
        selected_mask[sorted_indices[:target_points]] = True
        return selected_mask
    
    num_faces = len(pc.binding_counter)
    
    # Step 1: Ensure each face has at least one point
    # Select the most important point for each face
    for face_id in range(num_faces):
        # Find all points bound to this face
        face_point_mask = (pc.binding == face_id)
        face_point_indices = face_point_mask.nonzero(as_tuple=False)
        
        if len(face_point_indices) > 0:
            # Get importance scores for points in this face
            face_indices = face_point_indices.squeeze(-1)
            if face_indices.dim() == 0:
                # Single point case
                face_indices = face_indices.unsqueeze(0)
            
            face_importance = importance_scores[face_indices]
            
            # Select the point with highest importance in this face
            best_local_idx = face_importance.argmax()
            best_global_idx = face_indices[best_local_idx]
            
            selected_mask[best_global_idx] = True
    
    # Step 2: Fill remaining quota based on importance
    remaining_quota = target_points - selected_mask.sum().item()
    
    if remaining_quota > 0:
        # Get unselected points
        unselected_mask = ~selected_mask
        unselected_indices = unselected_mask.nonzero(as_tuple=False).squeeze(-1)
        
        if unselected_indices.dim() == 0:
            # Edge case: only one unselected point
            unselected_indices = unselected_indices.unsqueeze(0)
        
        if len(unselected_indices) > 0:
            if len(unselected_indices) == 1:
                # Only one unselected point, select it if quota allows
                if remaining_quota > 0:
                    selected_mask[unselected_indices] = True
            else:
                # Get importance scores for unselected points
                unselected_importance = importance_scores[unselected_indices]
                
                # Sort by importance (descending)
                sorted_local_indices = torch.argsort(unselected_importance, descending=True)
                sorted_global_indices = unselected_indices[sorted_local_indices]
                
                # Select top remaining_quota points
                num_to_select = min(remaining_quota, len(sorted_global_indices))
                selected_mask[sorted_global_indices[:num_to_select]] = True
    elif remaining_quota < 0:
        # Edge case: number of faces already exceeds target_points
        # In this case, we've already selected one point per face
        # We need to reduce to target_points by keeping only the most important points
        # This is a low compression ratio scenario
        selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
        if selected_indices.dim() == 0:
            selected_indices = selected_indices.unsqueeze(0)
        
        selected_importance = importance_scores[selected_indices]
        sorted_indices = torch.argsort(selected_importance, descending=True)
        
        # Keep only top target_points
        selected_mask.fill_(False)
        selected_mask[selected_indices[sorted_indices[:target_points]]] = True
    
    return selected_mask


def select_without_binding_constraint(
    pc, 
    importance_scores: torch.Tensor, 
    ratio: float
) -> torch.Tensor:
    """
    Select points based on importance scores WITHOUT binding constraints.
    
    This function performs simple top-k selection based on importance scores,
    without enforcing the "at least one point per face" constraint.
    The binding information is preserved for animation purposes, but selection
    is not constrained by it.
    
    This function works with or without binding information.
    
    Args:
        pc: GaussianModel or FlameGaussianModel (binding info preserved but not used for selection)
        importance_scores: Importance scores for each point, shape (num_points,)
        ratio: Compression ratio (0.0-1.0), where 1.0 means no compression
    
    Returns:
        selected_mask: Boolean mask indicating selected points, shape (num_points,)
    
    Example:
        >>> importance = compute_spatial_density(gaussians._xyz)
        >>> selected = select_without_binding_constraint(gaussians, importance, ratio=0.1)
        >>> print(f"Selected {selected.sum().item()} points (10% compression)")
    """
    total_points = len(pc._xyz)
    target_points = int(total_points * ratio)
    
    # Handle edge cases
    if target_points <= 0:
        return torch.zeros(total_points, dtype=torch.bool, device=pc._xyz.device)
    if target_points >= total_points:
        return torch.ones(total_points, dtype=torch.bool, device=pc._xyz.device)
    
    # Simple top-k selection based on importance scores
    sorted_indices = torch.argsort(importance_scores, descending=True)
    selected_mask = torch.zeros(total_points, dtype=torch.bool, device=pc._xyz.device)
    selected_mask[sorted_indices[:target_points]] = True
    
    return selected_mask


def fixed_importance_selection(
    pc, 
    ratio: float, 
    use_binding: bool = True,
    density_weight: float = 0.6,
    binding_weight: float = 0.4,
    use_semantic_weights: bool = False,
    eye_weight: float = 2.0,
    mouth_weight: float = 1.5,
    enforce_binding_constraint: bool = True,
    enable_debug: bool = False
) -> torch.Tensor:
    """
    Fixed importance selection (non-differentiable).
    
    Combines spatial density and binding importance to compute a combined
    importance score, then selects points based on this score.
    
    This is the main entry point for Phase A compression. All parameters
    can be controlled to enable/disable specific features.
    
    Args:
        pc: GaussianModel or FlameGaussianModel
        ratio: Compression ratio (0.0-1.0), where 1.0 means no compression
        use_binding: Whether to use binding importance (default: True)
                     If False, only uses spatial density
        density_weight: Weight for spatial density (default: 0.6)
        binding_weight: Weight for binding importance (default: 0.4)
        use_semantic_weights: If True, use semantic weights for eyes/mouth regions (default: False)
        eye_weight: Weight for eye regions (default: 2.0, only used if use_semantic_weights=True)
        mouth_weight: Weight for mouth region (default: 1.5, only used if use_semantic_weights=True)
        enforce_binding_constraint: If True, ensures each face has at least one point (default: True)
                                   If False, performs simple top-k selection without binding constraints
                                   Note: binding information is preserved for animation even if constraint is disabled
        enable_debug: If True, print debug information (default: False)
    
    Returns:
        selected_mask: Boolean mask indicating selected points
    
    Example:
        >>> # Basic usage with binding constraint
        >>> selected = fixed_importance_selection(gaussians, ratio=0.5)
        >>> 
        >>> # With semantic weights
        >>> selected = fixed_importance_selection(
        ...     gaussians, ratio=0.3, 
        ...     use_semantic_weights=True, enable_debug=True
        ... )
        >>> 
        >>> # Unbind mode (no binding constraint)
        >>> selected = fixed_importance_selection(
        ...     gaussians, ratio=0.1,
        ...     enforce_binding_constraint=False
        ... )
    """
    # Validate ratio
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"Compression ratio must be in (0.0, 1.0], got {ratio}")
    
    # Normalize weights
    total_weight = density_weight + (binding_weight if use_binding else 0.0)
    if total_weight > 0:
        density_weight = density_weight / total_weight
        if use_binding:
            binding_weight = binding_weight / total_weight
    else:
        # Fallback: use uniform weights if both are zero
        density_weight = 1.0
        binding_weight = 0.0
    
    # Step 1: Compute spatial density importance
    density_importance = compute_spatial_density(pc._xyz)
    
    # Step 2: Compute binding importance (if enabled and binding exists)
    if use_binding and pc.binding is not None:
        binding_importance = compute_binding_importance(
            pc, 
            use_semantic_weights=use_semantic_weights,
            eye_weight=eye_weight,
            mouth_weight=mouth_weight,
            enable_debug=enable_debug
        )
        
        # Normalize both importance scores before combining
        # This ensures fair combination and preserves semantic weight differences
        def normalize_importance(imp):
            """Normalize importance scores to [0, 1] range"""
            imp_min = imp.min()
            imp_max = imp.max()
            imp_range = imp_max - imp_min
            if imp_range > 1e-6:
                return (imp - imp_min) / imp_range
            else:
                # All points have same importance, return uniform
                return torch.ones_like(imp)
        
        # Normalize density and binding importance separately
        density_importance_norm = normalize_importance(density_importance)
        binding_importance_norm = normalize_importance(binding_importance)
        
        # Combine normalized importance scores
        combined_importance = density_weight * density_importance_norm + binding_weight * binding_importance_norm
    else:
        # No binding, just normalize density importance
        density_min = density_importance.min()
        density_max = density_importance.max()
        density_range = density_max - density_min
        if density_range > 1e-6:
            combined_importance = (density_importance - density_min) / density_range
        else:
            combined_importance = torch.ones_like(density_importance)
    
    # Step 3: Final normalization (should be minimal now since both components are already normalized)
    # This step ensures the final scores are in [0, 1] range for consistency
    importance_min = combined_importance.min()
    importance_max = combined_importance.max()
    importance_range = importance_max - importance_min
    
    if importance_range > 1e-6:
        combined_importance = (combined_importance - importance_min) / importance_range
    else:
        # All points have same importance, use uniform
        combined_importance = torch.ones_like(combined_importance)
    
    # Step 4: Select points based on constraint setting
    if enforce_binding_constraint and pc.binding is not None:
        # Use binding constraint: ensure each face has at least one point
        selected_mask = select_with_binding_constraint(pc, combined_importance, ratio)
    else:
        # No binding constraint: simple top-k selection
        # Note: binding information is preserved for animation, but not used for selection
        selected_mask = select_without_binding_constraint(pc, combined_importance, ratio)
    
    if enable_debug:
        total_points = len(pc._xyz)
        selected_points = selected_mask.sum().item()
        actual_ratio = selected_points / total_points
        print(f"[Fixed Importance Selection] Target ratio: {ratio:.4f}, Actual ratio: {actual_ratio:.4f}")
        if pc.binding is not None:
            num_faces = len(pc.binding_counter)
            if enforce_binding_constraint:
                # Check binding integrity
                covered_faces = torch.unique(pc.binding[selected_mask]).numel()
                print(f"[Fixed Importance Selection] Covered faces: {covered_faces}/{num_faces}")
    
    return selected_mask


def compute_min_compression_ratio(pc) -> float:
    """
    Compute minimum compression ratio based on binding constraints.
    
    If binding is enabled, the minimum ratio is num_faces / total_points,
    ensuring each face has at least one point.
    
    Args:
        pc: GaussianModel or FlameGaussianModel
    
    Returns:
        min_ratio: Minimum compression ratio (0.0 if no binding)
    
    Example:
        >>> min_ratio = compute_min_compression_ratio(gaussians)
        >>> print(f"Minimum compression ratio: {min_ratio:.4f}")
        >>> # Use this to validate compression ratio
        >>> if target_ratio < min_ratio:
        >>>     print(f"Warning: ratio {target_ratio} is below minimum {min_ratio}")
    """
    if pc.binding is None:
        return 0.0  # No constraint
    
    num_faces = len(pc.binding_counter)
    total_points = len(pc._xyz)
    min_ratio = num_faces / total_points
    
    return min_ratio

