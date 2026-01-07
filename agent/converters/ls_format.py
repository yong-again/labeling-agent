"""
Label Studio 포맷 변환기
DINO + SAM 결과를 Label Studio Prediction 포맷으로 변환
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LabelStudioConverter:
    """Label Studio 포맷 변환기"""
    
    def __init__(self, output_format: str = "polygonlabels"):
        """
        Args:
            output_format: 출력 포맷 ("polygonlabels" or "rectanglelabels")
        """
        if output_format not in ["polygonlabels", "rectanglelabels"]:
            raise ValueError(
                f"지원하지 않는 포맷: {output_format}. "
                "지원 포맷: 'polygonlabels', 'rectanglelabels'"
            )
        self.output_format = output_format
        logger.info(f"출력 포맷: {output_format}")
    
    def boxes_to_rectanglelabels(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: List[str],
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Bounding box를 rectanglelabels 포맷으로 변환
        
        Args:
            boxes: (N, 4) [x1, y1, x2, y2] (정규화 좌표 0-1)
            scores: (N,) confidence scores
            labels: List[str] 클래스 레이블
            image_width: 이미지 너비
            image_height: 이미지 높이
        
        Returns:
            Label Studio rectanglelabels 포맷 리스트
        """
        results = []
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # 정규화 좌표를 픽셀 좌표로 변환
            x1 = box[0] * image_width
            y1 = box[1] * image_height
            x2 = box[2] * image_width
            y2 = box[3] * image_height
            
            # Label Studio 포맷: 퍼센트 좌표 (0-100)
            result = {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": float(x1 / image_width * 100),
                    "y": float(y1 / image_height * 100),
                    "width": float((x2 - x1) / image_width * 100),
                    "height": float((y2 - y1) / image_height * 100),
                    "rectanglelabels": [label],
                },
                "score": float(score),
            }
            results.append(result)
        
        return results
    
    def masks_to_polygonlabels(
        self,
        masks: List[np.ndarray],
        scores: np.ndarray,
        labels: List[str],
        image_width: int,
        image_height: int,
        min_area: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        마스크를 polygonlabels 포맷으로 변환
        
        Args:
            masks: List of (H, W) binary numpy arrays
            scores: (N,) confidence scores
            labels: List[str] 클래스 레이블
            image_width: 이미지 너비
            image_height: 이미지 높이
            min_area: 최소 영역 (픽셀 단위, 이보다 작으면 제외)
        
        Returns:
            Label Studio polygonlabels 포맷 리스트
        """
        results = []
        
        for mask, score, label in zip(masks, scores, labels):
            # 마스크를 컨투어로 변환
            contours = self._mask_to_contours(mask, min_area=min_area)
            
            if not contours:
                logger.debug(f"마스크 {label}의 컨투어가 없거나 너무 작음 (min_area={min_area})")
                continue
            
            # 가장 큰 컨투어 선택 (또는 모든 컨투어 병합 가능)
            # 여기서는 가장 큰 컨투어만 사용
            largest_contour = max(contours, key=lambda c: len(c))
            
            # 컨투어를 픽셀 좌표로 변환
            points = []
            for point in largest_contour:
                x, y = point[0], point[1]
                # Label Studio 포맷: 퍼센트 좌표 (0-100)
                points.append(float(x / image_width * 100))
                points.append(float(y / image_height * 100))
            
            result = {
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "value": {
                    "points": points,
                    "polygonlabels": [label],
                },
                "score": float(score),
            }
            results.append(result)
        
        return results
    
    def _mask_to_contours(
        self,
        mask: np.ndarray,
        min_area: int = 100,
    ) -> List[np.ndarray]:
        """
        마스크를 컨투어로 변환
        
        Args:
            mask: (H, W) binary numpy array
            min_area: 최소 영역
        
        Returns:
            List of contours (각 contour는 (N, 2) numpy array)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV가 필요합니다. 설치: pip install opencv-python"
            )
        
        # 마스크를 uint8로 변환
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        # 최소 영역 필터링
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # (N, 1, 2) -> (N, 2)
                contour_2d = contour.reshape(-1, 2)
                filtered_contours.append(contour_2d)
        
        return filtered_contours
    
    def convert(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: List[str],
        masks: Optional[List[np.ndarray]] = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> List[Dict[str, Any]]:
        """
        DINO + SAM 결과를 Label Studio 포맷으로 변환
        
        Args:
            boxes: (N, 4) bounding boxes [x1, y1, x2, y2] (정규화 좌표)
            scores: (N,) confidence scores
            labels: List[str] 클래스 레이블
            masks: Optional[List[np.ndarray]] 마스크 리스트
            image_width: 이미지 너비
            image_height: 이미지 높이
        
        Returns:
            Label Studio prediction 포맷 리스트
        """
        if self.output_format == "rectanglelabels":
            return self.boxes_to_rectanglelabels(
                boxes, scores, labels, image_width, image_height
            )
        elif self.output_format == "polygonlabels":
            if masks is None or len(masks) == 0:
                logger.warning(
                    "polygonlabels 포맷을 사용하지만 마스크가 없습니다. "
                    "rectanglelabels로 대체합니다."
                )
                return self.boxes_to_rectanglelabels(
                    boxes, scores, labels, image_width, image_height
                )
            return self.masks_to_polygonlabels(
                masks, scores, labels, image_width, image_height
            )
        else:
            raise ValueError(f"알 수 없는 포맷: {self.output_format}")

