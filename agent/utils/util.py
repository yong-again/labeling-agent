import logging
logger = logging.getLogger(__name__)

def get_mask_cordinates(masks, boxes):
        if len(masks) > 0:
            import numpy as np
            # 마지막 마스크 추출
            masks_array = masks.cpu().numpy()  # (N, 1, H, W)
            last_mask = masks_array[-1, 0]  # (H, W) - 마지막 마스크의 2D 형태
            
            # 좌표 추출
            coords = np.where(last_mask > 0)
            
            logger.info(f"\n마지막 마스크 정보:")
            logger.info(f"  Shape: {last_mask.shape}")
            logger.info(f"  True 픽셀 개수: {np.sum(last_mask > 0)}")
            logger.info(f"  좌표 배열 개수: {len(coords[0])}")
            logger.info(f"\n좌표 샘플 (처음 10개):")
            logger.info(f"  Y 좌표: {coords[0][:10]}")
            logger.info(f"  X 좌표: {coords[1][:10]}")
            logger.info(f"\n좌표 샘플 (마지막 10개):")
            logger.info(f"  Y 좌표: {coords[0][-10:]}")
            logger.info(f"  X 좌표: {coords[1][-10:]}")
            
            # 전체 인덱스 결과 출력
            logger.info(f"\n전체 인덱스 결과:")
            logger.info(f"  (array(shape=({len(coords[0])},)), array(shape=({len(coords[1])},)))")
            logger.info(f"  Y range: [{coords[0].min()}, {coords[0].max()}]")
            logger.info(f"  X range: [{coords[1].min()}, {coords[1].max()}]")
            logger.info("=" * 80)

            # 박스 좌표 확인
            logger.info(f"\n변환된 박스 좌표 (첫 번째):")
            logger.info(f"  {boxes[0]}")
            logger.info(f"\n변환된 박스 좌표 (마지막):")
            logger.info(f"  {boxes[-1]}")