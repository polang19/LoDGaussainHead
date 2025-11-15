#!/usr/bin/env python3
#
# 压缩指标计算工具
# 实现绑定完整性检查和压缩比例计算
#

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import json


def compute_compression_ratio(
    selected_mask: torch.Tensor,
    total_count: Optional[int] = None
) -> float:
    """
    计算实际压缩比例
    
    Args:
        selected_mask: 选择mask（bool或float tensor）
        total_count: 总点数（如果为None，使用mask的长度）
    
    Returns:
        实际压缩比例（0.0 ~ 1.0）
    """
    if total_count is None:
        total_count = len(selected_mask)
    
    selected_count = selected_mask.sum().item() if isinstance(selected_mask, torch.Tensor) else sum(selected_mask)
    return selected_count / total_count


def compute_ratio_error(
    actual_ratio: float,
    target_ratio: float
) -> float:
    """
    计算压缩比例误差
    
    Args:
        actual_ratio: 实际压缩比例
        target_ratio: 目标压缩比例
    
    Returns:
        相对误差（0.0 ~ 1.0）
    """
    if target_ratio == 0:
        return 1.0 if actual_ratio > 0 else 0.0
    
    return abs(actual_ratio - target_ratio) / target_ratio


def compute_binding_integrity(
    pc,
    selected_mask: torch.Tensor
) -> Dict[str, float]:
    """
    计算绑定完整性指标
    
    Args:
        pc: GaussianAvatars模型（需要binding属性）
        selected_mask: 选择mask
    
    Returns:
        包含完整性指标的字典
    """
    if pc.binding is None:
        return {
            "coverage": 1.0,
            "total_faces": 0,
            "covered_faces": 0,
            "min_points_per_face": 0,
            "max_points_per_face": 0,
            "mean_points_per_face": 0,
        }
    
    # 获取选中的绑定ID
    selected_bindings = pc.binding[selected_mask]
    
    # 统计每个面片的点数
    total_faces = len(pc.binding_counter)
    face_counts = torch.bincount(
        selected_bindings,
        minlength=total_faces
    )
    
    # 计算覆盖率
    covered_faces = (face_counts > 0).sum().item()
    coverage = covered_faces / total_faces if total_faces > 0 else 0.0
    
    # 统计信息
    non_zero_counts = face_counts[face_counts > 0]
    min_points = non_zero_counts.min().item() if len(non_zero_counts) > 0 else 0
    max_points = face_counts.max().item()
    mean_points = face_counts.float().mean().item()
    
    return {
        "coverage": coverage,
        "total_faces": total_faces,
        "covered_faces": covered_faces,
        "min_points_per_face": min_points,
        "max_points_per_face": max_points,
        "mean_points_per_face": mean_points,
        "face_counts": face_counts.cpu().numpy().tolist(),  # 保存完整统计
    }


def compute_face_coverage(
    pc,
    selected_mask: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    计算面片覆盖率
    
    Args:
        pc: GaussianAvatars模型
        selected_mask: 选择mask
    
    Returns:
        (覆盖率, 每个面片的点数统计)
    """
    if pc.binding is None:
        return 1.0, torch.tensor([])
    
    selected_bindings = pc.binding[selected_mask]
    total_faces = len(pc.binding_counter)
    face_counts = torch.bincount(
        selected_bindings,
        minlength=total_faces
    )
    
    coverage = (face_counts > 0).float().mean().item()
    return coverage, face_counts


def compute_selection_consistency(
    mask1: torch.Tensor,
    mask2: torch.Tensor
) -> Dict[str, float]:
    """
    计算两个选择mask的一致性
    
    Args:
        mask1: 第一个选择mask
        mask2: 第二个选择mask
    
    Returns:
        包含一致性指标的字典
    """
    # 转换为bool类型
    mask1_bool = mask1.bool() if mask1.dtype != torch.bool else mask1
    mask2_bool = mask2.bool() if mask2.dtype != torch.bool else mask2
    
    # 计算交集和并集
    intersection = (mask1_bool & mask2_bool).sum().item()
    union = (mask1_bool | mask2_bool).sum().item()
    
    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0
    
    # 精确度（Precision）
    precision = intersection / mask2_bool.sum().item() if mask2_bool.sum() > 0 else 0.0
    
    # 召回率（Recall）
    recall = intersection / mask1_bool.sum().item() if mask1_bool.sum() > 0 else 0.0
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "intersection": intersection,
        "union": union,
    }


def evaluate_compression(
    pc,
    selected_mask: torch.Tensor,
    target_ratio: float
) -> Dict:
    """
    综合评估压缩效果
    
    Args:
        pc: GaussianAvatars模型
        selected_mask: 选择mask
        target_ratio: 目标压缩比例
    
    Returns:
        包含所有评估指标的字典
    """
    # 压缩比例
    actual_ratio = compute_compression_ratio(selected_mask)
    ratio_error = compute_ratio_error(actual_ratio, target_ratio)
    
    # 绑定完整性
    binding_stats = compute_binding_integrity(pc, selected_mask)
    
    # 综合评估
    evaluation = {
        "compression": {
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "ratio_error": ratio_error,
            "selected_count": selected_mask.sum().item(),
            "total_count": len(selected_mask),
        },
        "binding": binding_stats,
    }
    
    return evaluation


def save_evaluation_report(
    evaluation: Dict,
    output_path: Path
):
    """
    保存评估报告到JSON文件
    
    Args:
        evaluation: 评估结果字典
        output_path: 输出文件路径
    """
    # 转换numpy数组为列表
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_eval = convert_to_serializable(evaluation)
    
    # 保存到文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_eval, f, indent=2, ensure_ascii=False)
    
    print(f"评估报告已保存到: {output_path}")


def print_evaluation_summary(evaluation: Dict):
    """
    打印评估摘要
    
    Args:
        evaluation: 评估结果字典
    """
    print("\n" + "=" * 70)
    print("压缩评估摘要")
    print("=" * 70)
    
    # 压缩比例信息
    comp = evaluation["compression"]
    print(f"\n压缩比例:")
    print(f"  目标: {comp['target_ratio']:.2%}")
    print(f"  实际: {comp['actual_ratio']:.2%}")
    print(f"  误差: {comp['ratio_error']:.2%}")
    print(f"  选中点数: {comp['selected_count']}/{comp['total_count']}")
    
    # 绑定完整性信息
    binding = evaluation["binding"]
    print(f"\n绑定完整性:")
    print(f"  覆盖率: {binding['coverage']:.2%}")
    print(f"  覆盖面片: {binding['covered_faces']}/{binding['total_faces']}")
    print(f"  每面片点数: 最小={binding['min_points_per_face']}, "
          f"最大={binding['max_points_per_face']}, "
          f"平均={binding['mean_points_per_face']:.2f}")


if __name__ == "__main__":
    # 示例用法
    print("压缩指标计算工具")
    print("使用方法:")
    print("  from utils.compression_metrics import evaluate_compression")
    print("  evaluation = evaluate_compression(pc, selected_mask, target_ratio)")

