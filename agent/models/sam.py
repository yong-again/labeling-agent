"""
SAM (Segment Anything Model) 래퍼
Bounding box를 마스크로 변환
"""

import logging
from typing import List, Tuple, Optional
import torch
import numpy as np
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "SAM이 설치되지 않았습니다.\n"
        "설치: pip install git+https://github.com/facebookresearch/segment-anything.git"
    )

logger = logging.getLogger(__name__)


class SAM:
    """SAM 모델 래퍼"""
    
    # 모델 타입별 체크포인트 URL 매핑
    MODEL_CHECKPOINTS = {
        "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_type: 모델 타입 ('default', 'vit_h', 'vit_l', 'vit_b' 또는 'sam_vit_h', 'sam_vit_l', 'sam_vit_b')
            checkpoint_path: 체크포인트 파일 경로. None이면 자동 다운로드
            device: 디바이스 ('cuda' or 'cpu'). None이면 자동 선택
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "SAM이 설치되지 않았습니다.\n"
                "설치: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        logger.info(f"SAM 모델 로드 중: {self.model_type}")
        
        try:
            # 모델 로드
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            logger.info("모델 로드 완료")
        
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def predict_from_boxes(
        self,
        image_source: np.ndarray,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Bounding box로부터 마스크 생성
        
        Args:
            image_source: numpy array (H, W, 3) - BGR 형식 (ImageLoader에서 로드)
            boxes: (N, 4) numpy array [x1, y1, x2, y2] (픽셀 좌표)
        
        Returns:    
            masks: List of (H, W) binary numpy arrays
        """
        if self.predictor is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        if len(boxes) == 0:
            return []
        
        # BGR to RGB 변환 (image_source는 BGR 형식)
        import cv2
        image_np = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        
        # SAM에 이미지 설정
        self.predictor.set_image(image_np)
        
        # 이미지 크기
        h, w = image_np.shape[:2]
        
        # boxes는 이미 픽셀 좌표 [x1, y1, x2, y2] (DINO에서 denormalize된 상태)
        # SAM predict_torch는 boxes를 torch.Tensor로 받음
        # 형식: (N, 4) where each row is [x1, y1, x2, y2] in pixel coordinates
        
        # transform.apply_boxes_torch를 사용하여 SAM 입력 형식으로 변환
        h, w = image_np.shape[:2]
        box_tensor = torch.tensor(boxes, device=self.device, dtype=torch.float32)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(box_tensor, (h, w))
        
        # 마스크 예측
        # predict_torch는 배치 처리를 위한 메서드
        # boxes가 (N, 4) 형식이면 각 박스에 대해 마스크 생성
        try:
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        except (AttributeError, TypeError) as e:
            # predict_torch가 없는 경우 predict 사용 (순차 처리)
            logger.warning(f"predict_torch를 사용할 수 없습니다 ({e}). predict로 대체합니다.")
            masks_list = []
            for box in boxes:
                # SAM predict는 box를 numpy array로 받음 [x1, y1, x2, y2] (픽셀 좌표)
                mask, score, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,  # numpy array [x1, y1, x2, y2]
                    multimask_output=False,
                )
                masks_list.append(mask[0])  # 첫 번째 마스크만 사용
            return masks_list
        
        # CPU로 이동하고 numpy로 변환
        masks_np = masks.cpu().numpy()  # (N, 1, H, W)
        masks_list = [masks_np[i, 0] for i in range(masks_np.shape[0])]  # List of (H, W)
        
        logger.debug(f"{len(masks_list)}개 마스크 생성됨")
        
        # 디버그: 마스크 생성 결과 상세 정보
        if len(masks_list) > 0:
            image_h, image_w = image_np.shape[:2]
            logger.debug(f"[SAM 상세] 이미지 크기: {image_w}x{image_h}")
            for i, mask in enumerate(masks_list):
                mask_area = int(mask.sum())
                mask_ratio = mask_area / (image_w * image_h) * 100
                mask_h, mask_w = mask.shape
                logger.debug(f"  [{i+1}] 마스크 크기: {mask_w}x{mask_h}, "
                           f"영역: {mask_area} 픽셀 ({mask_ratio:.2f}%), "
                           f"True 픽셀: {mask_area}, False 픽셀: {mask_w*mask_h - mask_area}")
        
        return masks_list
    
    def predict_from_boxes_batch(
        self,
        image_sources: List[np.ndarray],
        boxes_list: List[np.ndarray],
    ) -> List[List[np.ndarray]]:
        """
        배치 예측 (현재는 순차 처리)
        
        Args:
            image_sources: 로드된 이미지 리스트 [numpy array (H, W, 3), ...]
            boxes_list: 박스 리스트 [(N, 4) numpy array, ...]
        """
        results = []
        for image_source, boxes in zip(image_sources, boxes_list):
            masks = self.predict_from_boxes(image_source, boxes)
            results.append(masks)
        return results
