"""
웹 표시를 위한 변환 유틸리티
원본 모델 출력을 웹에서 표시 가능한 형식으로 변환
"""

import logging
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import torch
from pathlib import Path

from agent.utils.box_transforms import cxcywh_to_xyxy

logger = logging.getLogger(__name__)


def convert_dino_boxes_for_display(
    boxes: torch.Tensor,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    DINO 원본 박스를 웹 표시용 형식으로 변환
    
    Args:
        boxes: (N, 4) torch.Tensor [cx, cy, w, h] (정규화 0-1) - DINO 원본 출력
        image_width: 이미지 너비 (픽셀)
        image_height: 이미지 높이 (픽셀)
    
    Returns:
        (N, 4) numpy array [x1, y1, x2, y2] (픽셀 좌표)
    """
    if len(boxes) == 0:
        return np.array([]).reshape(0, 4)
    
    # [cx, cy, w, h] (정규화) -> [x1, y1, x2, y2] (픽셀)
    boxes_xyxy = cxcywh_to_xyxy(
        boxes=boxes,
        image_width=image_width,
        image_height=image_height,
        normalized=True,
    )
    
    # numpy로 변환
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xyxy = boxes_xyxy.cpu().numpy()
    
    return boxes_xyxy


def convert_sam_masks_for_display(
    masks: torch.Tensor,
) -> List[np.ndarray]:
    """
    SAM 원본 마스크를 웹 표시용 형식으로 변환
    
    Args:
        masks: (N, 1, H, W) torch.Tensor - SAM 원본 출력
    
    Returns:
        List of (H, W) binary numpy arrays
    """
    if masks.shape[0] == 0:
        return []
    
    # CPU로 이동하고 numpy로 변환
    masks_np = masks.cpu().numpy()  # (N, 1, H, W)
    
    # List of (H, W)로 변환
    masks_list = [masks_np[i, 0] for i in range(masks_np.shape[0])]
    
    return masks_list


def convert_scores_for_display(
    scores: torch.Tensor,
) -> np.ndarray:
    """
    DINO 원본 점수를 웹 표시용 형식으로 변환
    
    Args:
        scores: (N,) torch.Tensor - DINO 원본 출력
    
    Returns:
        (N,) numpy array
    """
    if len(scores) == 0:
        return np.array([])
    
    if isinstance(scores, torch.Tensor):
        return scores.cpu().numpy()
    return np.array(scores)


def convert_result_for_web_display(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    labels: List[str],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """
    LabelingResult의 원본 출력을 웹 표시용 형식으로 변환
    
    Args:
        boxes: (N, 4) torch.Tensor [cx, cy, w, h] (정규화) - DINO 원본
        scores: (N,) torch.Tensor - DINO 원본
        masks: (N, 1, H, W) torch.Tensor - SAM 원본
        labels: List[str] 클래스 레이블
        image_width: 이미지 너비 (픽셀)
        image_height: 이미지 높이 (픽셀)
    
    Returns:
        웹 표시용 딕셔너리:
        {
            "boxes": (N, 4) numpy array [x1, y1, x2, y2] (픽셀),
            "scores": (N,) numpy array,
            "masks": List of (H, W) numpy arrays,
            "labels": List[str],
            "num_objects": int
        }
    """
    boxes_xyxy = convert_dino_boxes_for_display(boxes, image_width, image_height)
    scores_np = convert_scores_for_display(scores)
    masks_list = convert_sam_masks_for_display(masks)
    
    return {
        "boxes": boxes_xyxy.tolist() if len(boxes_xyxy) > 0 else [],
        "scores": scores_np.tolist() if len(scores_np) > 0 else [],
        "masks": [mask.tolist() for mask in masks_list],
        "labels": labels,
        "num_objects": len(boxes_xyxy),
    }


def convert_result_for_json(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    labels: List[str],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """
    LabelingResult의 원본 출력을 JSON 직렬화 가능한 형식으로 변환
    
    Args:
        boxes: (N, 4) torch.Tensor [cx, cy, w, h] (정규화) - DINO 원본
        scores: (N,) torch.Tensor - DINO 원본
        masks: (N, 1, H, W) torch.Tensor - SAM 원본
        labels: List[str] 클래스 레이블
        image_width: 이미지 너비 (픽셀)
        image_height: 이미지 높이 (픽셀)
    
    Returns:
        JSON 직렬화 가능한 딕셔너리
    """
    boxes_xyxy = convert_dino_boxes_for_display(boxes, image_width, image_height)
    scores_np = convert_scores_for_display(scores)
    masks_list = convert_sam_masks_for_display(masks)
    
    # 마스크를 base64나 RLE로 인코딩할 수도 있지만, 여기서는 간단히 리스트로 변환
    # 큰 마스크의 경우 RLE 인코딩을 고려해야 함
    
    return {
        "boxes": boxes_xyxy.tolist() if len(boxes_xyxy) > 0 else [],
        "scores": scores_np.tolist() if len(scores_np) > 0 else [],
        "masks": [mask.tolist() for mask in masks_list],
        "labels": labels,
        "num_objects": len(boxes_xyxy),
        "image_width": image_width,
        "image_height": image_height,
    }
