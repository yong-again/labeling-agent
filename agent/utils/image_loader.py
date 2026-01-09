"""
Image Loader 모듈
DINO, SAM 등 여러 모델에서 공통으로 사용할 수 있는 이미지 로딩 기능 제공
"""

import logging
from typing import Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch

try:
    from groundingdino.util.inference import load_image as dino_load_image
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageLoader:
    """이미지 로더 클래스 - 여러 모델에서 사용할 수 있는 표준화된 이미지 로딩"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, torch.Tensor, Image.Image]:
        """
        이미지 로드 - DINO와 SAM에서 공통으로 사용할 수 있는 형식으로 반환
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            image_source: numpy array (H, W, 3) - BGR 형식 (OpenCV 호환)
            image_transformed: torch.Tensor - DINO 모델 입력용 전처리된 이미지
            pil_image: PIL.Image - SAM 등에서 사용
        """
        image_path = str(image_path)
        
        if not GROUNDING_DINO_AVAILABLE:
            raise ImportError(
                "Grounding DINO가 설치되지 않았습니다.\n"
                "이미지 로딩을 위해 설치가 필요합니다: pip install groundingdino-py"
            )
        
        # DINO load_image 사용 (image_source: np.ndarray BGR, image: torch.Tensor)
        image_source, image_transformed = dino_load_image(image_path)
        
        # PIL Image도 생성 (SAM 및 기타 용도)
        pil_image = Image.open(image_path).convert("RGB")
        
        logger.debug(f"이미지 로드 완료: {image_path}, 크기: {pil_image.size}")
        
        return image_source, image_transformed, pil_image
    
    @staticmethod
    def load_image_for_sam(image_source: np.ndarray) -> np.ndarray:
        """
        SAM 모델용 이미지 전처리
        
        Args:
            image_source: numpy array (H, W, 3) - BGR 형식
            
        Returns:
            image_np: numpy array (H, W, 3) - RGB 형식
        """
        # BGR to RGB 변환 (image_source는 BGR)
        import cv2
        image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    @staticmethod
    def get_image_size(image_source: np.ndarray) -> Tuple[int, int]:
        """
        이미지 크기 반환
        
        Args:
            image_source: numpy array (H, W, 3)
            
        Returns:
            (width, height) 튜플
        """
        h, w = image_source.shape[:2]
        return w, h


def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, torch.Tensor, Image.Image]:
    """
    편의 함수: ImageLoader.load_image의 래퍼
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        image_source: numpy array (H, W, 3)
        image_transformed: torch.Tensor - DINO 모델 입력용
        pil_image: PIL.Image
    """
    return ImageLoader.load_image(image_path)
