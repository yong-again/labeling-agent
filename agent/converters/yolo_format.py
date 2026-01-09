"""
YOLO 포맷 변환기
DINO + SAM 결과를 YOLO format으로 변환
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class YOLOConverter:
    """YOLO format 변환기"""
    
    def __init__(self, class_mapping: Optional[Dict[str, int]] = None):
        """
        Args:
            class_mapping: 클래스 이름 -> 클래스 ID 매핑
                          None이면 자동으로 0부터 할당 (YOLO는 0-indexed)
        """
        self.class_mapping = class_mapping or {}
        self._auto_class_id = 0
        
    def _get_class_id(self, label: str) -> int:
        """클래스 이름에 해당하는 클래스 ID 반환 (0-indexed)"""
        if label not in self.class_mapping:
            self.class_mapping[label] = self._auto_class_id
            self._auto_class_id += 1
        return self.class_mapping[label]
    
    def _box_to_yolo_format(
        self, 
        box: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple:
        """
        픽셀 박스 좌표를 YOLO format으로 변환
        
        Args:
            box: [x1, y1, x2, y2] (픽셀 좌표)
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            (x_center, y_center, width, height) (정규화 좌표 0-1)
        """
        x1, y1, x2, y2 = box
        
        # 픽셀 좌표를 정규화 좌표로 변환
        x1_norm = x1 / image_width
        y1_norm = y1 / image_height
        x2_norm = x2 / image_width
        y2_norm = y2 / image_height
        
        x_center = (x1_norm + x2_norm) / 2
        y_center = (y1_norm + y2_norm) / 2
        width = x2_norm - x1_norm
        height = y2_norm - y1_norm
        
        return float(x_center), float(y_center), float(width), float(height)
    
    def _mask_to_yolo_segmentation(
        self, 
        mask: np.ndarray,
        image_width: int,
        image_height: int,
        simplify: bool = True,
        epsilon_ratio: float = 0.001,
    ) -> List[float]:
        """
        바이너리 마스크를 YOLO segmentation format으로 변환
        
        Args:
            mask: (H, W) binary numpy array
            image_width: 이미지 너비
            image_height: 이미지 높이
            simplify: 폴리곤 단순화 여부
            epsilon_ratio: 단순화 정도 (낮을수록 정밀)
            
        Returns:
            [x1, y1, x2, y2, ...] (정규화 좌표 0-1)
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV가 없어 segmentation 변환을 건너뜁니다")
            return []
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 폴리곤 단순화
        if simplify:
            epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(largest_contour) < 3:
            return []
        
        # 정규화 좌표로 변환
        points = []
        for point in largest_contour:
            x = point[0][0] / image_width
            y = point[0][1] / image_height
            points.extend([float(x), float(y)])
        
        return points
    
    def convert_detection(
        self,
        boxes: np.ndarray,
        labels: List[str],
        image_width: int,
        image_height: int,
    ) -> str:
        """
        Detection 결과를 YOLO format 문자열로 변환
        
        Args:
            boxes: (N, 4) bounding boxes [x1, y1, x2, y2] (픽셀 좌표)
            labels: List[str] 클래스 레이블
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            YOLO format 문자열 (한 줄에 하나의 객체)
            format: <class_id> <x_center> <y_center> <width> <height>
        """
        lines = []
        
        for box, label in zip(boxes, labels):
            class_id = self._get_class_id(label)
            x_center, y_center, width, height = self._box_to_yolo_format(box, image_width, image_height)
            
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def convert_segmentation(
        self,
        masks: List[np.ndarray],
        labels: List[str],
        image_width: int,
        image_height: int,
    ) -> str:
        """
        Segmentation 결과를 YOLO format 문자열로 변환
        
        Args:
            masks: List of (H, W) binary numpy arrays
            labels: List[str] 클래스 레이블
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            YOLO format 문자열 (한 줄에 하나의 객체)
            format: <class_id> <x1> <y1> <x2> <y2> ...
        """
        lines = []
        
        for mask, label in zip(masks, labels):
            class_id = self._get_class_id(label)
            points = self._mask_to_yolo_segmentation(mask, image_width, image_height)
            
            if not points:
                continue
            
            points_str = " ".join(f"{p:.6f}" for p in points)
            line = f"{class_id} {points_str}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def convert(
        self,
        boxes: np.ndarray,
        labels: List[str],
        masks: Optional[List[np.ndarray]] = None,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> str:
        """
        DINO + SAM 결과를 YOLO format으로 변환
        
        Args:
            boxes: (N, 4) bounding boxes [x1, y1, x2, y2] (픽셀 좌표)
            labels: List[str] 클래스 레이블
            masks: Optional[List[np.ndarray]] 마스크 리스트
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            YOLO format 문자열
        """
        if masks:
            result = self.convert_segmentation(masks, labels, image_width, image_height)
            logger.info(f"YOLO segmentation 변환 완료: {len(masks)}개 객체")
        else:
            result = self.convert_detection(boxes, labels, image_width, image_height)
            logger.info(f"YOLO detection 변환 완료: {len(boxes)}개 객체")
        
        return result
    
    def save(self, annotation: str, output_path: str) -> None:
        """YOLO annotation 파일 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(annotation)
        logger.info(f"YOLO 파일 저장: {output_path}")
    
    def save_classes(self, output_path: str) -> None:
        """classes.txt 파일 저장"""
        # ID 순으로 정렬
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
        
        logger.info(f"Classes 파일 저장: {output_path}")
    
    def save_yaml(self, output_path: str, train_path: str = "train/images", 
                  val_path: str = "val/images") -> None:
        """YOLOv5/v8 data.yaml 파일 저장"""
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        class_names = [name for name, _ in sorted_classes]
        
        yaml_content = f"""# Auto-generated by DINO-SAM Labeling Agent
path: .
train: {train_path}
val: {val_path}

nc: {len(class_names)}
names: {class_names}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"YAML 파일 저장: {output_path}")
