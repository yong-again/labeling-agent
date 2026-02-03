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
import cv2

try:
    import groundingdino.datasets.transforms as T
except ImportError:
    raise ImportError(
        "groundingdino.datasets.transforms이 설치되지 않았습니다.\n"
        "설치: pip install groundingdino-py"
    )

logger = logging.getLogger(__name__)


class ImageLoader:
    """이미지 로더 클래스 - DINO와 SAM용 이미지 로딩 분리"""
    
    @staticmethod
    def load_image_for_dino(image_path: Union[str, Path]) -> Tuple[np.ndarray, torch.Tensor]:
        """
        DINO 모델용 이미지 로드
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            image: numpy array (H, W, 3) - RGB 형식
            image_transformed: torch.Tensor - DINO 모델 입력용 전처리된 이미지
        """
        image_path = str(image_path)
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)  # RGB numpy array
        image_transformed, _ = transform(image_source, None)
        
        logger.debug(f"DINO 이미지 로드 완료: {image_path}, 크기: {image.shape}")
        
        return image, image_transformed
    
    @staticmethod
    def load_image_for_sam(image_path: Union[str, Path]) -> np.ndarray:
        """
        SAM 모델용 이미지 로드 (OpenCV 사용, RGB 반환)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            image_rgb: numpy array (H, W, 3) - RGB 형식
        """
        image_path = str(image_path)
        
        # OpenCV로 로드 (BGR)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # BGR to RGB 변환
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        logger.debug(f"SAM 이미지 로드 완료: {image_path}, 크기: {image_rgb.shape}")
        
        return image_rgb
    
    @staticmethod
    def get_image_size(image_source: np.ndarray) -> Tuple[int, int]:
        """
        이미지 크기 반환
        
        Args:
            image_source: numpy array (H, W, 3)
            
        Returns:
            (height, width) 튜플
        """
        h, w = image_source.shape[:2]
        return h, w


def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, torch.Tensor]:
    """
    DINO 모델용 이미지 로드 (편의 함수)
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        image: numpy array (H, W, 3) - RGB 형식
        image_transformed: torch.Tensor - DINO 모델 입력용
    """
    return ImageLoader.load_image_for_dino(image_path)


def load_image_for_sam(image_path: Union[str, Path]) -> np.ndarray:
    """
    SAM 모델용 이미지 로드 (편의 함수)
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        image_rgb: numpy array (H, W, 3) - RGB 형식
    """
    return ImageLoader.load_image_for_sam(image_path)


def get_image_size(image_source: np.ndarray) -> Tuple[int, int]:
    """
    이미지 크기 반환
    
    Args:
        image_source: numpy array (H, W, 3)
        
    Returns:
        (height, width) 튜플
    """
    h, w = image_source.shape[:2]
    return h, w