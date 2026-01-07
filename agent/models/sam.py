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
        model_type: str = "sam_vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_type: 모델 타입 ('sam_vit_h', 'sam_vit_l', 'sam_vit_b')
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
        logger.info(f"SAM 모델 로드 중: {self.model_type} (device: {self.device})")
        
        try:
            # 체크포인트 경로 확인
            if self.checkpoint_path is None:
                checkpoint_url = self.MODEL_CHECKPOINTS.get(self.model_type)
                if checkpoint_url:
                    import os
                    import urllib.request
                    checkpoint_dir = os.path.expanduser("~/.cache/sam")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.checkpoint_path = os.path.join(
                        checkpoint_dir, f"{self.model_type}.pth"
                    )
                    
                    if not os.path.exists(self.checkpoint_path):
                        logger.info(f"체크포인트 다운로드 중: {checkpoint_url}")
                        urllib.request.urlretrieve(checkpoint_url, self.checkpoint_path)
                        logger.info("다운로드 완료")
                else:
                    raise ValueError(f"알 수 없는 모델 타입: {self.model_type}")
            
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
        image: Image.Image,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Bounding box로부터 마스크 생성
        
        Args:
            image: PIL Image
            boxes: (N, 4) numpy array [x1, y1, x2, y2] (정규화 좌표 0-1)
        
        Returns:
            masks: List of (H, W) binary numpy arrays
        """
        if self.predictor is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        if len(boxes) == 0:
            return []
        
        # 이미지를 numpy array로 변환
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # Grayscale
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = image_np[:, :, :3]
        
        # SAM에 이미지 설정
        self.predictor.set_image(image_np)
        
        # 이미지 크기
        h, w = image_np.shape[:2]
        
        # 박스를 픽셀 좌표로 변환
        # boxes는 정규화 좌표 [x1, y1, x2, y2] (0-1 범위)
        boxes_pixel = boxes.copy()
        boxes_pixel[:, [0, 2]] *= w  # x 좌표
        boxes_pixel[:, [1, 3]] *= h  # y 좌표
        
        # SAM 형식으로 변환: (N, 4) -> (N, 4) [x1, y1, x2, y2]
        # SAM predict_torch는 boxes를 torch.Tensor로 받음
        # 형식: (N, 4) where each row is [x1, y1, x2, y2] in pixel coordinates
        sam_boxes = torch.from_numpy(boxes_pixel).to(self.device)
        
        # 마스크 예측
        # predict_torch는 배치 처리를 위한 메서드
        # boxes가 (N, 4) 형식이면 각 박스에 대해 마스크 생성
        try:
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=sam_boxes,
                multimask_output=False,
            )
        except (AttributeError, TypeError) as e:
            # predict_torch가 없는 경우 predict 사용 (순차 처리)
            logger.warning(f"predict_torch를 사용할 수 없습니다 ({e}). predict로 대체합니다.")
            masks_list = []
            for box in boxes_pixel:
                # SAM predict는 box를 numpy array로 받음 [x1, y1, x2, y2]
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
        
        return masks_list
    
    def predict_from_boxes_batch(
        self,
        images: List[Image.Image],
        boxes_list: List[np.ndarray],
    ) -> List[List[np.ndarray]]:
        """배치 예측 (현재는 순차 처리)"""
        results = []
        for image, boxes in zip(images, boxes_list):
            masks = self.predict_from_boxes(image, boxes)
            results.append(masks)
        return results
