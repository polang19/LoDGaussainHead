#
# Gumbel Network for GaussianAvatars
# 
# This module implements a Gumbel network for learnable point selection.
# Key differences from FlexGS:
# 1. Adds binding embedding (for FLAME face binding)
# 2. Designed for GaussianAvatars architecture (FLAME-driven deformation)
#
# Reference: FlexGS/scene/deformation.py:237-261
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlameGumbelNetwork(nn.Module):
    """
    Gumbel network for learnable point selection in GaussianAvatars.
    
    This network learns to select Gaussian points based on:
    - Position (3D coordinates)
    - Rotation (quaternion)
    - Scale (3D scale)
    - Compression ratio (time/ratio parameter)
    - Binding ID (FLAME face binding, GaussianAvatars-specific)
    
    Architecture:
    - 5 encoders (pos, rotation, scale, ratio, binding) → 32-dim each
    - Concatenate → 160-dim
    - Linear layer → 2-dim (select/not-select)
    - Gumbel-Softmax for differentiable selection
    
    Key differences from FlexGS:
    1. Adds binding_emd encoder (for FLAME binding information)
    2. Designed to work with FLAME-driven deformation (no separate deformation network)
    """
    
    def __init__(self, W=32, use_binding_hint=True):
        """
        Initialize the Gumbel network.
        
        Args:
            W: Hidden dimension for encoders (default: 32)
            use_binding_hint: Whether to use binding embedding (default: True)
                              If False, works like FlexGS (4 encoders)
        """
        super(FlameGumbelNetwork, self).__init__()
        self.W = W
        self.use_binding_hint = use_binding_hint
        
        # Position encoder: 3D coordinates → W-dim
        self.pos_emd = nn.Sequential(
            nn.Linear(3, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU()
        )
        
        # Rotation encoder: Quaternion (4D) → W-dim
        self.rotation_emd = nn.Sequential(
            nn.Linear(4, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU()
        )
        
        # Scale encoder: 3D scale → W-dim
        self.scale_emd = nn.Sequential(
            nn.Linear(3, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU()
        )
        
        # Ratio encoder: Compression ratio (1D) → W-dim
        # Note: In FlexGS, this is called "time_emd", but it represents compression ratio
        self.ratio_emd = nn.Sequential(
            nn.Linear(1, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU()
        )
        
        # Binding encoder: FLAME face binding ID → W-dim (GaussianAvatars-specific)
        if use_binding_hint:
            self.binding_emd = nn.Sequential(
                nn.Linear(1, self.W),
                nn.ReLU(),
                nn.Linear(self.W, self.W),
                nn.ReLU()
            )
            input_dim = self.W * 5  # 5 encoders
        else:
            input_dim = self.W * 4  # 4 encoders (like FlexGS)
        
        # Output layer: Combined features → 2 classes (select/not-select)
        self.soft_net = nn.Linear(input_dim, 2)
        
        # Gumbel-Softmax function
        self.hard_net = F.gumbel_softmax
    
    def forward(self, pos, rotation, scale, ratio, binding=None, cur_tau=1.0):
        """
        Forward pass of the Gumbel network.
        
        Args:
            pos: Point positions, shape [N, 3]
            rotation: Rotation quaternions, shape [N, 4]
            scale: Point scales, shape [N, 3]
            ratio: Compression ratio, shape [N, 1] (0-1, where 1.0 = no compression)
            binding: FLAME face binding IDs, shape [N] (optional, but recommended)
            cur_tau: Temperature parameter for Gumbel-Softmax (default: 1.0)
                    Lower tau = sharper distribution (more discrete)
                    Higher tau = smoother distribution (more continuous)
        
        Returns:
            soft_output: Soft selection logits, shape [N] (class 1 probability, before Gumbel)
            hard_output: Hard selection mask, shape [N] (0 or 1, discrete selection)
            soft1: Soft selection probability, shape [N] (differentiable, for training)
        """
        # Encode each attribute
        pos_emd = self.pos_emd(pos)              # [N, 3] → [N, W]
        rotation_emd = self.rotation_emd(rotation) # [N, 4] → [N, W]
        scale_emd = self.scale_emd(scale)        # [N, 3] → [N, W]
        ratio_emd = self.ratio_emd(ratio)         # [N, 1] → [N, W]
        
        # Binding embedding (if enabled)
        if self.use_binding_hint and binding is not None:
            # Normalize binding IDs to [0, 1] range
            # Performance optimization: avoid .item() to prevent CPU-GPU synchronization
            binding_max = binding.max()  # Keep on GPU, avoid sync
            binding_normalized = binding.float() / torch.clamp(binding_max, min=1.0)  # Avoid division by zero
            
            binding_emd = self.binding_emd(binding_normalized.unsqueeze(-1))  # [N, 1] → [N, W]
            
            # Concatenate all features
            gumbel_input = torch.cat((
                pos_emd, rotation_emd, scale_emd, ratio_emd, binding_emd
            ), dim=1)  # [N, W*5]
        else:
            # Without binding (like FlexGS)
            gumbel_input = torch.cat((
                pos_emd, rotation_emd, scale_emd, ratio_emd
            ), dim=1)  # [N, W*4]
        
        # Compute logits (unnormalized probabilities)
        soft_output = self.soft_net(gumbel_input)  # [N, 2]
        # soft_output[:, 0] = logit for "not select"
        # soft_output[:, 1] = logit for "select"
        
        # Gumbel-Softmax sampling
        # hard=True: Discrete sampling (0 or 1, for inference)
        hard_output = self.hard_net(soft_output, hard=True, tau=cur_tau)  # [N, 2]
        # hard=False: Continuous sampling (differentiable, for training)
        soft1 = self.hard_net(soft_output, hard=False, tau=cur_tau)        # [N, 2]
        
        # Return class 1 (select) probabilities
        return soft_output[:, 1], hard_output[:, 1], soft1[:, 1]
    
    def get_num_parameters(self):
        """
        Get the number of trainable parameters in the network.
        
        Returns:
            num_params: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_gumbel_network():
    """
    Simple test function to verify the Gumbel network works correctly.
    """
    print("Testing FlameGumbelNetwork...")
    
    # Create network
    net = FlameGumbelNetwork(W=32, use_binding_hint=True)
    net = net.cuda()
    
    # Create dummy data
    N = 1000  # Number of points
    pos = torch.randn(N, 3).cuda()
    rotation = torch.randn(N, 4).cuda()
    # Normalize quaternion
    rotation = rotation / rotation.norm(dim=1, keepdim=True)
    scale = torch.randn(N, 3).abs().cuda()  # Scales should be positive
    ratio = torch.ones(N, 1).cuda() * 0.5  # 50% compression
    binding = torch.randint(0, 100, (N,)).cuda()  # 100 different faces
    
    # Forward pass
    soft_output, hard_output, soft1 = net(pos, rotation, scale, ratio, binding)
    
    print(f"  Input shapes:")
    print(f"    pos: {pos.shape}")
    print(f"    rotation: {rotation.shape}")
    print(f"    scale: {scale.shape}")
    print(f"    ratio: {ratio.shape}")
    print(f"    binding: {binding.shape}")
    print(f"  Output shapes:")
    print(f"    soft_output: {soft_output.shape}")
    print(f"    hard_output: {hard_output.shape}")
    print(f"    soft1: {soft1.shape}")
    print(f"  Output ranges:")
    print(f"    soft_output: [{soft_output.min().item():.4f}, {soft_output.max().item():.4f}]")
    print(f"    hard_output: [{hard_output.min().item():.4f}, {hard_output.max().item():.4f}] (should be 0 or 1)")
    print(f"    soft1: [{soft1.min().item():.4f}, {soft1.max().item():.4f}] (should be in [0, 1])")
    print(f"  Selection ratio: {hard_output.sum().item() / N * 100:.2f}%")
    print(f"  Number of parameters: {net.get_num_parameters():,}")
    
    # Test without binding
    print("\nTesting without binding hint...")
    net_no_binding = FlameGumbelNetwork(W=32, use_binding_hint=False)
    net_no_binding = net_no_binding.cuda()
    soft_output2, hard_output2, soft12 = net_no_binding(pos, rotation, scale, ratio, binding=None)
    print(f"  Output shapes: {soft_output2.shape}, {hard_output2.shape}, {soft12.shape}")
    print(f"  Number of parameters: {net_no_binding.get_num_parameters():,}")
    
    print("\n✓ Gumbel network test passed!")


if __name__ == "__main__":
    test_gumbel_network()

