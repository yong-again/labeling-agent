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
            sam.eval()  # 평가 모드로 설정 (중요!)
            self.sam_model = sam  # 모델 자체를 저장
            self.predictor = SamPredictor(sam)
            logger.info(f"모델 로드 완료 (training mode: {sam.training})")
        
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def predict_from_boxes(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
        multimask_output: bool = False,
    ) -> torch.Tensor:
        """
        Bounding box로부터 마스크 생성
        
        Args:
            image: numpy array (H, W, 3) - RGB 형식 (load_image_for_sam에서 로드)
            boxes: (N, 4) torch.Tensor [x1, y1, x2, y2] (픽셀 좌표)
            image_width: 이미지 너비
            image_height: 이미지 높이
            multimask_output: True면 각 박스당 3개의 마스크 후보 반환 (N, 3, H, W), 
                            False면 1개만 반환 (N, 1, H, W)
        
        Returns:    
            masks: torch.Tensor 
                - multimask_output=False: (N, 1, H, W)
                - multimask_output=True: (N, 3, H, W)
        """
        if self.predictor is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.bool).reshape(0, 1, 0, 0)
        
        # 디버그: 이미지 정보 출력 (첫 픽셀값 추가)
        first_pixel = image[0, 0, :] if len(image.shape) == 3 else image[0, 0]
        logger.info(f"SAM input image - shape: {image.shape}, dtype: {image.dtype}, "
                   f"min: {image.min():.3f}, max: {image.max():.3f}, "
                   f"mean: {image.mean():.3f}, first_pixel: {first_pixel}")
        
        # SAM에 이미지 설정 (RGB 형식, image_format 생략하여 기본값 사용)
        self.predictor.set_image(image)
        
        #SAM input은 1024x1024 이므로 box도 이미지 해상도에 맞게끔 transform
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        transformed_boxes = transformed_boxes.to(self.predictor.device)
        
        #logger.info(f"transformed_boxes:{transformed_boxes} in sam.py")
        
        # 마스크 예측
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=multimask_output,
        )
        
        logger.info(f"masks shape in sam.py: {masks.shape}")
        
        return masks
    
    def predict_from_boxes_batch(
        self,
        image_sources: List[np.ndarray],
        boxes_list: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """
        배치 예측 (현재는 순차 처리)
        
        Args:
            image_sources: 로드된 이미지 리스트 [numpy array (H, W, 3), ...]
            boxes_list: 박스 리스트 [(N, 4) numpy array, ...]
        
        Returns:
            List of (N, 1, H, W) torch.Tensor - 원본 SAM 출력
        """
        results = []
        for image_source, boxes in zip(image_sources, boxes_list):
            masks = self.predict_from_boxes(image_source, boxes)
            results.append(masks)
        return results
