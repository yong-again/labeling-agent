"""
마스크 결과 비교 테스트
pipeline.py의 sam.py 수정 후 segment_anything lib와 결과가 동일한지 확인
"""

import numpy as np
from agent.config import Config
from agent.pipeline import LabelingPipeline

# Config 생성
config = Config()

# Pipeline 초기화
pipeline = LabelingPipeline(config)

# 테스트 이미지 경로
IMAGE_PATH = "/workspace/labeling-agent/uploads/a52cbc65-01ac-4a76-b964-19cc6c166680.png"
TEXT_PROMPT = "cat"

# Pipeline으로 처리
print("=" * 80)
print("Pipeline.py를 통한 마스크 생성 테스트")
print("=" * 80)

result = pipeline.process_image(
    image_path=IMAGE_PATH,
    text_prompt=TEXT_PROMPT,
    confidence_threshold=0.35,
)

print(f"\n검출된 객체 수: {result.num_objects}")
print(f"Masks shape: {result.masks.shape}")

if result.has_masks:
    # 마지막 마스크 추출 (temp.py와 동일하게)
    masks_array = result.masks.cpu().numpy()  # (N, 1, H, W)
    last_mask = masks_array[-1, 0]  # (H, W) - 마지막 마스크의 2D 형태
    
    # 좌표 추출
    coords = np.where(last_mask > 0)
    
    print(f"\n마지막 마스크 정보:")
    print(f"  Shape: {last_mask.shape}")
    print(f"  True 픽셀 개수: {np.sum(last_mask > 0)}")
    print(f"  좌표 배열 개수: {len(coords[0])}")
    print(f"\n좌표 샘플 (처음 10개):")
    print(f"  Y 좌표: {coords[0][:10]}")
    print(f"  X 좌표: {coords[1][:10]}")
    print(f"\n좌표 샘플 (마지막 10개):")
    print(f"  Y 좌표: {coords[0][-10:]}")
    print(f"  X 좌표: {coords[1][-10:]}")
    
    # 전체 인덱스 결과 출력
    print(f"\n전체 인덱스 결과:")
    print(f"  (array(shape=({len(coords[0])},)), array(shape=({len(coords[1])},)))")
    print(f"  Y range: [{coords[0].min()}, {coords[0].max()}]")
    print(f"  X range: [{coords[1].min()}, {coords[1].max()}]")
    
    # 박스 좌표도 출력
    print(f"\n검출된 박스 좌표 (첫 번째):")
    print(f"  CXCYWH (norm): {result.boxes[0]}")
    from agent.utils.box_transforms import cxcywh_to_xyxy
    boxes_xyxy = cxcywh_to_xyxy(result.boxes, result.image_width, result.image_height, normalized=True)
    print(f"  XYXY (pixel): {boxes_xyxy[0]}")
    
    print(f"\n검출된 박스 좌표 (마지막):")
    print(f"  CXCYWH (norm): {result.boxes[-1]}")
    print(f"  XYXY (pixel): {boxes_xyxy[-1]}")
    
    # segment_anything lib 직접 사용 결과와 비교하기 위한 정보 출력
    print("\n" + "=" * 80)
    print("비교를 위해 temp.py에서 동일한 출력을 확인하세요:")
    print("  - 픽셀 개수가 37,979개 정도여야 정상입니다")
    print("  - Y 좌표 범위가 [112, 435] 정도여야 정상입니다")
    print("  - X 좌표 범위가 [276, 304] 정도여야 정상입니다")
    print("=" * 80)
else:
    print("마스크가 생성되지 않았습니다!")
