"""
학습 데이터 수집기
HITL 피드백을 기반으로 Continuous Learning용 학습 데이터 생성
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

from agent.feedback import FeedbackManager, FeedbackStatus

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataset:
    """학습 데이터셋"""
    images: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    output_dir: str


class DataCollector:
    """HITL 피드백 기반 학습 데이터 수집기"""
    
    def __init__(self, feedback_db_path: str = "./feedback.db"):
        """
        Args:
            feedback_db_path: 피드백 DB 경로
        """
        self.feedback_manager = FeedbackManager(feedback_db_path)
        logger.info("DataCollector 초기화 완료")
    
    def collect_approved_data(
        self,
        output_dir: str,
        format: Literal["coco", "yolo"] = "coco",
        include_corrected: bool = True,
        copy_images: bool = True,
    ) -> TrainingDataset:
        """
        승인된 피드백 데이터를 학습 데이터셋으로 변환
        
        Args:
            output_dir: 출력 디렉토리
            format: 출력 포맷 ("coco" or "yolo")
            include_corrected: 수정된 데이터 포함 여부
            copy_images: 이미지 파일 복사 여부
            
        Returns:
            TrainingDataset 객체
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 디렉토리
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # 승인된 데이터 수집
        training_data = self.feedback_manager.export_training_data(
            status=None,  # approved + corrected
            include_corrections=include_corrected,
        )
        
        if not training_data:
            logger.warning("학습 데이터가 없습니다")
            return TrainingDataset(
                images=[],
                annotations=[],
                metadata={"count": 0},
                output_dir=str(output_path),
            )
        
        logger.info(f"{len(training_data)}개 학습 샘플 수집됨")
        
        images = []
        annotations = []
        
        for i, item in enumerate(training_data):
            image_path = Path(item["image_path"])
            
            if not image_path.exists():
                logger.warning(f"이미지 파일 없음: {image_path}")
                continue
            
            # 이미지 복사
            if copy_images:
                new_image_path = images_dir / image_path.name
                if not new_image_path.exists():
                    shutil.copy(image_path, new_image_path)
            else:
                new_image_path = image_path
            
            # 이미지 정보
            labels_data = item["labels"]
            image_info = {
                "id": i + 1,
                "file_name": image_path.name,
                "width": labels_data.get("image_width", 1920),
                "height": labels_data.get("image_height", 1080),
                "prompt": item["prompt"],
            }
            images.append(image_info)
            
            # 라벨 변환
            if format == "yolo":
                self._save_yolo_labels(
                    labels_data,
                    labels_dir / f"{image_path.stem}.txt"
                )
            else:
                # COCO annotations
                for j, label in enumerate(labels_data.get("labels", [])):
                    boxes = labels_data.get("boxes", [])
                    scores = labels_data.get("scores", [])
                    
                    if j < len(boxes):
                        box = boxes[j]
                        # 정규화 좌표를 픽셀 좌표로 변환
                        x1 = box[0] * image_info["width"]
                        y1 = box[1] * image_info["height"]
                        w = (box[2] - box[0]) * image_info["width"]
                        h = (box[3] - box[1]) * image_info["height"]
                        
                        annotation = {
                            "id": len(annotations) + 1,
                            "image_id": i + 1,
                            "category_id": j + 1,  # 간단히 순서대로
                            "category_name": label,
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                        annotations.append(annotation)
        
        metadata = {
            "format": format,
            "count": len(images),
            "created_at": datetime.now().isoformat(),
            "include_corrected": include_corrected,
        }
        
        # 메타데이터 저장
        if format == "coco":
            self._save_coco_format(images, annotations, output_path / "annotations.json")
        
        logger.info(f"학습 데이터셋 생성 완료: {len(images)}개 이미지, {len(annotations)}개 annotation")
        
        return TrainingDataset(
            images=images,
            annotations=annotations,
            metadata=metadata,
            output_dir=str(output_path),
        )
    
    def _save_yolo_labels(self, labels_data: Dict[str, Any], output_path: Path):
        """YOLO 형식 라벨 저장"""
        lines = []
        
        labels = labels_data.get("labels", [])
        boxes = labels_data.get("boxes", [])
        
        # 클래스 이름 -> ID 매핑 (간단히 순서대로)
        class_map = {name: i for i, name in enumerate(set(labels))}
        
        for i, (label, box) in enumerate(zip(labels, boxes)):
            if len(box) < 4:
                continue
            
            class_id = class_map.get(label, 0)
            # 정규화 좌표 -> YOLO 중심 좌표
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
    
    def _save_coco_format(
        self,
        images: List[Dict],
        annotations: List[Dict],
        output_path: Path
    ):
        """COCO 포맷 저장"""
        # 카테고리 추출
        category_names = set()
        for ann in annotations:
            if "category_name" in ann:
                category_names.add(ann["category_name"])
        
        categories = [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(sorted(category_names))
        ]
        
        # 카테고리 ID 재매핑
        name_to_id = {cat["name"]: cat["id"] for cat in categories}
        for ann in annotations:
            if "category_name" in ann:
                ann["category_id"] = name_to_id.get(ann["category_name"], 1)
                del ann["category_name"]
        
        coco_data = {
            "info": {
                "description": "Training data from DINO-SAM Labeling Agent",
                "date_created": datetime.now().isoformat(),
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 가능한 데이터 통계"""
        stats = self.feedback_manager.get_stats()
        
        approved = stats.get("by_status", {}).get("approved", 0)
        corrected = stats.get("by_status", {}).get("corrected", 0)
        
        return {
            "total_approved": approved,
            "total_corrected": corrected,
            "available_for_training": approved + corrected,
            "pending_review": stats.get("by_status", {}).get("pending", 0),
            "rejected": stats.get("by_status", {}).get("rejected", 0),
        }
