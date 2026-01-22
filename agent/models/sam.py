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
        image: np.ndarray,
        boxes: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> torch.Tensor:
        """
        Bounding box로부터 마스크 생성
        
        Args:
            image: numpy array (H, W, 3) - BGR 형식 (ImageLoader에서 로드)
            boxes: (N, 4) numpy array [x1, y1, x2, y2] (픽셀 좌표)
        
        Returns:    
            masks: (N, 1, H, W) torch.Tensor - 원본 SAM 출력 (변환 없음)
        """
        if self.predictor is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.bool).reshape(0, 1, 0, 0)
        
        # SAM에 이미지 설정
        self.predictor.set_image(image)
        
        # transform.apply_boxes_torch를 사용하여 SAM 입력 형식으로 변환
        boxes = boxes.to(self.predictor.device)
        
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, (image_height, image_width))
        print(transformed_boxes)
        
        # 마스크 예측
        # predict_torch는 배치 처리를 위한 메서드
        # boxes가 (N, 4) 형식이면 각 박스에 대해 마스크 생성
        # try:
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # except (AttributeError, TypeError) as e:
        #     # predict_torch가 없는 경우 predict 사용 (순차 처리)
        #     logger.warning(f"predict_torch를 사용할 수 없습니다 ({e}). predict로 대체합니다.")
        #     masks_list = []
        #     for box in boxes:
        #         # SAM predict는 box를 numpy array로 받음 [x1, y1, x2, y2] (픽셀 좌표)
        #         mask, score, _ = self.predictor.predict(
        #             point_coords=None,
        #             point_labels=None,
        #             box=box,  # numpy array [x1, y1, x2, y2]
        #             multimask_output=False,
        #         )
        #         # numpy array를 torch tensor로 변환하고 차원 추가
        #         mask_tensor = torch.from_numpy(mask[0]).to(self.device).unsqueeze(0)  # (1, H, W)
        #         masks_list.append(mask_tensor)
        
        # logger.debug(f"{masks.shape[0]}개 마스크 생성됨 (shape: {masks.shape})")
        
        # # 원본 출력 반환 (변환 없음)
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
