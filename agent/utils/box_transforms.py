"""
Bounding Box 좌표 변환 유틸리티
"""

import numpy as np
import torch
from typing import Union, Tuple


def cxcywh_to_xyxy(
    boxes: Union[np.ndarray, torch.Tensor],
    image_width: int,
    image_height: int,
    normalized: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    [cx, cy, w, h] 형식을 [x1, y1, x2, y2] 형식으로 변환
    
    Args:
        boxes: (N, 4) array [cx, cy, w, h]
        image_width: 이미지 너비 (픽셀)
        image_height: 이미지 높이 (픽셀)
        normalized: boxes가 정규화된 좌표(0-1)인지 여부
    
    Returns:
        (N, 4) array [x1, y1, x2, y2] (픽셀 좌표)
    """
    if len(boxes) == 0:
        if isinstance(boxes, torch.Tensor):
            return boxes.reshape(0, 4)
        return np.array([]).reshape(0, 4)
    
    is_tensor = isinstance(boxes, torch.Tensor)
    
    if is_tensor:
        device = boxes.device
        # 정규화된 좌표를 픽셀 좌표로 변환
        if normalized:
            scale_fct = torch.tensor([image_width, image_height, image_width, image_height], device=device)
            boxes_scaled = boxes * scale_fct
        else:
            boxes_scaled = boxes
        
        # cxcywh -> x1y1x2y2 변환
        x1y1 = boxes_scaled[:, :2] - boxes_scaled[:, 2:] / 2
        x2y2 = boxes_scaled[:, :2] + boxes_scaled[:, 2:] / 2
        boxes_xyxy = torch.cat([x1y1, x2y2], dim=-1)
        
        # 클리핑 (이미지 경계 밖으로 나가는 것 방지)
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, image_width)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, image_height)
        
        return boxes_xyxy
    else:
        # NumPy 버전
        if normalized:
            scale_fct = np.array([image_width, image_height, image_width, image_height])
            boxes_scaled = boxes * scale_fct
        else:
            boxes_scaled = boxes
        
        # cxcywh -> x1y1x2y2 변환
        x1y1 = boxes_scaled[:, :2] - boxes_scaled[:, 2:] / 2
        x2y2 = boxes_scaled[:, :2] + boxes_scaled[:, 2:] / 2
        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=-1)
        
        # 클리핑
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, image_width)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, image_height)
        
        return boxes_xyxy


def xyxy_to_cxcywh(
    boxes: Union[np.ndarray, torch.Tensor],
    image_width: int,
    image_height: int,
    normalize: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    [x1, y1, x2, y2] 형식을 [cx, cy, w, h] 형식으로 변환
    
    Args:
        boxes: (N, 4) array [x1, y1, x2, y2] (픽셀 좌표)
        image_width: 이미지 너비 (픽셀)
        image_height: 이미지 높이 (픽셀)
        normalize: 정규화된 좌표(0-1)로 반환할지 여부
    
    Returns:
        (N, 4) array [cx, cy, w, h]
    """
    if len(boxes) == 0:
        if isinstance(boxes, torch.Tensor):
            return boxes.reshape(0, 4)
        return np.array([]).reshape(0, 4)
    
    is_tensor = isinstance(boxes, torch.Tensor)
    
    if is_tensor:
        device = boxes.device
        
        # x1y1x2y2 -> cxcywh 변환
        cx_cy = (boxes[:, :2] + boxes[:, 2:]) / 2
        w_h = boxes[:, 2:] - boxes[:, :2]
        boxes_cxcywh = torch.cat([cx_cy, w_h], dim=-1)
        
        # 정규화
        if normalize:
            scale_fct = torch.tensor([image_width, image_height, image_width, image_height], device=device)
            boxes_cxcywh = boxes_cxcywh / scale_fct
        
        return boxes_cxcywh
    else:
        # NumPy 버전
        cx_cy = (boxes[:, :2] + boxes[:, 2:]) / 2
        w_h = boxes[:, 2:] - boxes[:, :2]
        boxes_cxcywh = np.concatenate([cx_cy, w_h], axis=-1)
        
        # 정규화
        if normalize:
            scale_fct = np.array([image_width, image_height, image_width, image_height])
            boxes_cxcywh = boxes_cxcywh / scale_fct
        
        return boxes_cxcywh


def normalize_boxes(
    boxes: Union[np.ndarray, torch.Tensor],
    image_width: int,
    image_height: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    픽셀 좌표를 정규화된 좌표(0-1)로 변환
    
    Args:
        boxes: (N, 4) array (픽셀 좌표)
        image_width: 이미지 너비
        image_height: 이미지 높이
    
    Returns:
        (N, 4) array (정규화된 좌표)
    """
    if len(boxes) == 0:
        return boxes
    
    is_tensor = isinstance(boxes, torch.Tensor)
    
    if is_tensor:
        device = boxes.device
        scale_fct = torch.tensor([image_width, image_height, image_width, image_height], device=device)
        return boxes / scale_fct
    else:
        scale_fct = np.array([image_width, image_height, image_width, image_height])
        return boxes / scale_fct


def denormalize_boxes(
    boxes: Union[np.ndarray, torch.Tensor],
    image_width: int,
    image_height: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    정규화된 좌표(0-1)를 픽셀 좌표로 변환
    
    Args:
        boxes: (N, 4) array (정규화된 좌표)
        image_width: 이미지 너비
        image_height: 이미지 높이
    
    Returns:
        (N, 4) array (픽셀 좌표)
    """
    if len(boxes) == 0:
        return boxes
    
    is_tensor = isinstance(boxes, torch.Tensor)
    
    if is_tensor:
        device = boxes.device
        scale_fct = torch.tensor([image_width, image_height, image_width, image_height], device=device)
        return boxes * scale_fct
    else:
        scale_fct = np.array([image_width, image_height, image_width, image_height])
        return boxes * scale_fct
