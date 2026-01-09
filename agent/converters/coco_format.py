"""
COCO 포맷 변환기
DINO + SAM 결과를 COCO format으로 변환
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class COCOConverter:
    """COCO format 변환기"""
    
    def __init__(self, class_mapping: Optional[Dict[str, int]] = None):
        """
        Args:
            class_mapping: 클래스 이름 -> 카테고리 ID 매핑
                          None이면 자동으로 1부터 할당
        """
        self.class_mapping = class_mapping or {}
        self._auto_category_id = 1
        
    def _get_category_id(self, label: str) -> int:
        """클래스 이름에 해당하는 카테고리 ID 반환"""
        if label not in self.class_mapping:
            self.class_mapping[label] = self._auto_category_id
            self._auto_category_id += 1
        return self.class_mapping[label]
    
    def _mask_to_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        바이너리 마스크를 RLE (Run-Length Encoding)로 변환
        
        Args:
            mask: (H, W) binary numpy array
            
        Returns:
            RLE 형식 딕셔너리
        """
        pixels = mask.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        return {
            "counts": runs.tolist(),
            "size": list(mask.shape)
        }
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        바이너리 마스크를 polygon 좌표로 변환
        
        Args:
            mask: (H, W) binary numpy array
            
        Returns:
            List of polygon segments (각 segment는 [x1, y1, x2, y2, ...] 형식)
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV가 없어 polygon 변환을 건너뜁니다")
            return []
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # 최소 3개 점 (6개 좌표)
                polygons.append(contour)
        
        return polygons
    
    def _box_to_coco_bbox(
        self, 
        box: np.ndarray, 
        image_width: int, 
        image_height: int
    ) -> List[float]:
        """
        박스 좌표를 COCO bbox로 변환
        
        Args:
            box: [x1, y1, x2, y2] (픽셀 좌표)
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            [x, y, width, height] (픽셀 좌표)
        """
        x1, y1, x2, y2 = box
        
        return [
            float(x1),
            float(y1),
            float(x2 - x1),  # width
            float(y2 - y1)   # height
        ]
    
    def _compute_area(self, mask: Optional[np.ndarray], bbox: List[float]) -> float:
        """영역 계산 (마스크 우선, 없으면 bbox 사용)"""
        if mask is not None:
            return float(mask.sum())
        return bbox[2] * bbox[3]  # width * height
    
    def convert(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: List[str],
        masks: Optional[List[np.ndarray]] = None,
        image_id: int = 1,
        image_width: int = 1920,
        image_height: int = 1080,
        image_filename: str = "image.jpg",
        use_rle: bool = False,
    ) -> Dict[str, Any]:
        """
        DINO + SAM 결과를 COCO format으로 변환
        
        Args:
            boxes: (N, 4) bounding boxes [x1, y1, x2, y2] (픽셀 좌표)
            scores: (N,) confidence scores
            labels: List[str] 클래스 레이블
            masks: Optional[List[np.ndarray]] 마스크 리스트
            image_id: 이미지 ID
            image_width: 이미지 너비
            image_height: 이미지 높이
            image_filename: 이미지 파일명
            use_rle: RLE 대신 polygon 사용 여부
            
        Returns:
            COCO format 딕셔너리
        """
        annotations = []
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            mask = masks[i] if masks and i < len(masks) else None
            bbox = self._box_to_coco_bbox(box, image_width, image_height)
            
            annotation = {
                "id": i + 1,
                "image_id": image_id,
                "category_id": self._get_category_id(label),
                "bbox": bbox,
                "area": self._compute_area(mask, bbox),
                "iscrowd": 0,
                "score": float(score),
            }
            
            # 세그멘테이션 추가
            if mask is not None:
                if use_rle:
                    annotation["segmentation"] = self._mask_to_rle(mask)
                else:
                    annotation["segmentation"] = self._mask_to_polygon(mask)
            
            annotations.append(annotation)
        
        # 카테고리 정보 생성
        categories = [
            {"id": cat_id, "name": name, "supercategory": "object"}
            for name, cat_id in self.class_mapping.items()
        ]
        
        # COCO format 구성
        coco_result = {
            "info": {
                "description": "Auto-labeled by DINO-SAM",
                "date_created": datetime.now().isoformat(),
                "version": "1.0",
            },
            "licenses": [],
            "images": [{
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height,
            }],
            "annotations": annotations,
            "categories": categories,
        }
        
        logger.info(f"COCO 변환 완료: {len(annotations)}개 annotation 생성")
        return coco_result
    
    def merge(self, coco_datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        여러 COCO 데이터셋 병합
        
        Args:
            coco_datasets: COCO format 딕셔너리 리스트
            
        Returns:
            병합된 COCO format 딕셔너리
        """
        if not coco_datasets:
            return {"info": {}, "images": [], "annotations": [], "categories": []}
        
        merged = {
            "info": coco_datasets[0].get("info", {}),
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        
        category_map = {}  # name -> new_id
        next_category_id = 1
        annotation_id = 1
        image_id_offset = 0
        
        for dataset in coco_datasets:
            # 카테고리 매핑
            old_to_new_category = {}
            for cat in dataset.get("categories", []):
                if cat["name"] not in category_map:
                    category_map[cat["name"]] = next_category_id
                    merged["categories"].append({
                        "id": next_category_id,
                        "name": cat["name"],
                        "supercategory": cat.get("supercategory", "object")
                    })
                    next_category_id += 1
                old_to_new_category[cat["id"]] = category_map[cat["name"]]
            
            # 이미지 ID 매핑
            old_to_new_image = {}
            for img in dataset.get("images", []):
                new_img_id = img["id"] + image_id_offset
                old_to_new_image[img["id"]] = new_img_id
                merged_img = img.copy()
                merged_img["id"] = new_img_id
                merged["images"].append(merged_img)
            
            # Annotations 추가
            for ann in dataset.get("annotations", []):
                new_ann = ann.copy()
                new_ann["id"] = annotation_id
                new_ann["image_id"] = old_to_new_image.get(ann["image_id"], ann["image_id"])
                new_ann["category_id"] = old_to_new_category.get(ann["category_id"], ann["category_id"])
                merged["annotations"].append(new_ann)
                annotation_id += 1
            
            # 다음 데이터셋을 위한 offset 업데이트
            if dataset.get("images"):
                image_id_offset = max(img["id"] for img in merged["images"])
        
        return merged
    
    def save(self, coco_data: Dict[str, Any], output_path: str) -> None:
        """COCO JSON 파일 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        logger.info(f"COCO 파일 저장: {output_path}")
