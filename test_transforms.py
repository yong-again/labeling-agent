"""
Test script for coordinate transformation refactoring
"""

import torch
import numpy as np
from agent.utils.box_transforms import cxcywh_to_xyxy, xyxy_to_cxcywh

def test_coordinate_transforms():
    """Test coordinate transformation functions"""
    
    print("=" * 60)
    print("좌표 변환 테스트")
    print("=" * 60)
    
    # 테스트 데이터
    image_width = 1000
    image_height = 800
    
    # 정규화된 [cx, cy, w, h] (DINO 출력 형식)
    boxes_cxcywh_normalized = torch.tensor([
        [0.5, 0.5, 0.4, 0.3],  # 중앙 박스
        [0.25, 0.25, 0.2, 0.2], # 좌상단 박스
    ])
    
    print(f"\n1. 입력 (정규화된 [cx, cy, w, h]):")
    print(boxes_cxcywh_normalized)
    
    # [cx, cy, w, h] -> [x1, y1, x2, y2] 변환
    boxes_xyxy = cxcywh_to_xyxy(
        boxes_cxcywh_normalized,
        image_width=image_width,
        image_height=image_height,
        normalized=True
    )
    
    print(f"\n2. 변환 결과 (픽셀 [x1, y1, x2, y2]):")
    print(boxes_xyxy)
    
    # 역변환 테스트
    boxes_cxcywh_back = xyxy_to_cxcywh(
        boxes_xyxy,
        image_width=image_width,
        image_height=image_height,
        normalize=True
    )
    
    print(f"\n3. 역변환 결과 (정규화된 [cx, cy, w, h]):")
    print(boxes_cxcywh_back)
    
    # 차이 확인
    diff = torch.abs(boxes_cxcywh_normalized - boxes_cxcywh_back)
    print(f"\n4. 변환 전후 차이:")
    print(diff)
    print(f"   최대 차이: {diff.max().item():.10f}")
    
    # NumPy 버전 테스트
    print("\n" + "=" * 60)
    print("NumPy 버전 테스트")
    print("=" * 60)
    
    boxes_np = boxes_cxcywh_normalized.numpy()
    print(f"\n1. 입력 (NumPy):")
    print(boxes_np)
    
    boxes_xyxy_np = cxcywh_to_xyxy(
        boxes_np,
        image_width=image_width,
        image_height=image_height,
        normalized=True
    )
    
    print(f"\n2. 변환 결과 (NumPy [x1, y1, x2, y2]):")
    print(boxes_xyxy_np)
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_coordinate_transforms()
