"""
파이프라인 오케스트레이션
DINO -> SAM -> Export (Label Studio 제거됨)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from agent.config import Config
from agent.models.dino import GroundingDINO
from agent.models.sam import SAM
from agent.converters.coco_format import COCOConverter
from agent.converters.yolo_format import YOLOConverter
from agent.utils.visualize import draw_bounding_boxes, draw_segmentation_masks, draw_dino_and_sam, save_visualization
from agent.utils.image_loader import load_image
from agent.utils.box_transforms import cxcywh_to_xyxy

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """라벨링 결과 데이터 클래스 (원본 모델 출력 저장)"""
    image_path: str
    image_width: int
    image_height: int
    boxes: torch.Tensor  # (N, 4) [cx, cy, w, h] (정규화 0-1) - DINO 원본 출력
    scores: torch.Tensor  # (N,) - DINO 원본 출력
    labels: List[str]
    masks: torch.Tensor  # (N, 1, H, W) - SAM 원본 출력
    prompt: str
    
    @property
    def num_objects(self) -> int:
        return len(self.boxes)
    
    @property
    def has_masks(self) -> bool:
        return self.masks.shape[0] > 0 if len(self.masks.shape) > 0 else False
    
    def to_dict(self) -> dict:
        """JSON 직렬화 가능한 딕셔너리로 변환 (원본 데이터)"""
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "num_objects": self.num_objects,
            "labels": self.labels,
            "scores": self.scores.cpu().numpy().tolist() if len(self.scores) > 0 else [],
            "boxes": self.boxes.cpu().numpy().tolist() if len(self.boxes) > 0 else [],
            "masks_shape": list(self.masks.shape) if self.has_masks else [],
            "prompt": self.prompt,
        }


class LabelingPipeline:
    """DINO-SAM 라벨링 파이프라인"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # 모델 초기화
        self.dino = GroundingDINO(
            model_name=config.dino_model_name,
            config_path=config.dino_config_path,
            checkpoint_path=config.dino_checkpoint_path,
            device=config.device,
        )
        self.sam = SAM(
            model_type=config.sam_model_name,
            checkpoint_path=config.sam_checkpoint_path,
            device=config.device,
        )
        
        # Converters
        self.coco_converter = COCOConverter(config.class_mapping.copy())
        self.yolo_converter = YOLOConverter(config.class_mapping.copy())
        
        logger.info("파이프라인 초기화 완료")
    
    def process_image(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
    ) -> LabelingResult:
        """
        단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            text_prompt: 텍스트 프롬프트
            confidence_threshold: Confidence threshold (None이면 config 사용)
        
        Returns:
            LabelingResult 객체
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold
        
        # 이미지 경로 문자열로 변환
        image_path = str(image_path)
        
        # 이미지 로드
        image_source, image_transformed, pil_image = load_image(image_path)
        image_width, image_height = pil_image.size
        
        print(image_width, image_height)
        
        # DINO: Bounding box 검출 (원본 출력 반환)
        logger.info(f"DINO 검출 실행: prompt='{text_prompt}', threshold={confidence_threshold}")
        boxes_cxcywh, scores, labels = self.dino.predict(
            image=image_transformed,
            text_prompt=text_prompt,
            box_threshold=confidence_threshold,
        )
        
        if len(boxes_cxcywh) == 0:
            logger.warning("검출된 박스가 없습니다")
            return LabelingResult(
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                boxes=torch.tensor([], dtype=torch.float32).reshape(0, 4),
                scores=torch.tensor([], dtype=torch.float32),
                labels=labels,
                masks=torch.tensor([], dtype=torch.bool).reshape(0, 1, 0, 0),
                prompt=text_prompt,
            )
        
        # SAM 입력을 위해 박스를 픽셀 좌표로 변환 (SAM 내부에서만 사용)
        logger.debug(f"SAM 입력을 위한 좌표 변환: [cx, cy, w, h] (정규화) -> [x1, y1, x2, y2] (픽셀)")
        boxes_xyxy = cxcywh_to_xyxy(
            boxes=boxes_cxcywh,
            image_width=image_width,
            image_height=image_height,
            normalized=True,
        )
        
        print(boxes_xyxy)
        
        # SAM: Box -> Mask 변환 (원본 출력 반환)
        logger.info(f"SAM 마스크 생성: {len(boxes_xyxy)}개 박스")
        masks = self.sam.predict_from_boxes(image=image_source, 
                                            image_width=image_width,
                                            image_height=image_height,
                                            boxes=boxes_xyxy)
        
        logger.info(f"라벨링 완료: {len(boxes_cxcywh)}개 객체 검출 (원본 출력 저장)")
        
        # 원본 출력 저장 (변환 없음)
        return LabelingResult(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            boxes=boxes_cxcywh,  # 원본 DINO 출력 [cx, cy, w, h] (정규화)
            scores=scores,  # 원본 DINO 출력
            labels=labels,
            masks=masks,  # 원본 SAM 출력 (N, 1, H, W)
            prompt=text_prompt,
        )
    
    def export_coco(
        self,
        result: LabelingResult,
        output_path: str,
        image_id: int = 1,
        use_rle: bool = False,
    ) -> dict:
        """
        COCO 포맷으로 내보내기
        
        Args:
            result: LabelingResult 객체 (원본 출력 저장)
            output_path: 출력 파일 경로
            image_id: 이미지 ID
            use_rle: RLE 사용 여부 (False면 polygon)
            
        Returns:
            COCO format 딕셔너리
        """
        # 원본 출력을 COCO 형식으로 변환
        boxes_xyxy = cxcywh_to_xyxy(
            boxes=result.boxes,
            image_width=result.image_width,
            image_height=result.image_height,
            normalized=True,
        )
        boxes_np = boxes_xyxy.cpu().numpy() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy
        scores_np = result.scores.cpu().numpy() if isinstance(result.scores, torch.Tensor) else result.scores
        
        # 마스크 변환: (N, 1, H, W) -> List of (H, W)
        masks_list = None
        if result.has_masks:
            masks_np = result.masks.cpu().numpy()  # (N, 1, H, W)
            masks_list = [masks_np[i, 0] for i in range(masks_np.shape[0])]
        
        coco_data = self.coco_converter.convert(
            boxes=boxes_np,
            scores=scores_np,
            labels=result.labels,
            masks=masks_list,
            image_id=image_id,
            image_width=result.image_width,
            image_height=result.image_height,
            image_filename=Path(result.image_path).name,
            use_rle=use_rle,
        )
        
        self.coco_converter.save(coco_data, output_path)
        return coco_data
    
    def export_yolo(
        self,
        result: LabelingResult,
        output_path: str,
        save_classes: bool = True,
    ) -> str:
        """
        YOLO 포맷으로 내보내기
        
        Args:
            result: LabelingResult 객체 (원본 출력 저장)
            output_path: 출력 파일 경로
            save_classes: classes.txt 저장 여부
            
        Returns:
            YOLO format 문자열
        """
        # 원본 출력을 YOLO 형식으로 변환
        # YOLO는 정규화된 [cx, cy, w, h] 형식을 사용하므로 원본 DINO 출력을 그대로 사용 가능
        boxes_np = result.boxes.cpu().numpy() if isinstance(result.boxes, torch.Tensor) else result.boxes
        
        # 마스크 변환: (N, 1, H, W) -> List of (H, W)
        masks_list = None
        if result.has_masks:
            masks_np = result.masks.cpu().numpy()  # (N, 1, H, W)
            masks_list = [masks_np[i, 0] for i in range(masks_np.shape[0])]
        
        yolo_data = self.yolo_converter.convert(
            boxes=boxes_np,
            labels=result.labels,
            masks=masks_list,
            image_width=result.image_width,
            image_height=result.image_height,
        )
        
        self.yolo_converter.save(yolo_data, output_path)
        
        # classes.txt 저장
        if save_classes:
            classes_path = Path(output_path).parent / "classes.txt"
            self.yolo_converter.save_classes(str(classes_path))
        
        return yolo_data
    
    def export(
        self,
        result: LabelingResult,
        output_path: str,
        format: Optional[str] = None,
    ) -> Union[dict, str]:
        """
        설정된 포맷으로 내보내기
        
        Args:
            result: LabelingResult 객체
            output_path: 출력 파일 경로
            format: 출력 포맷 ("coco" or "yolo"). None이면 config 사용
            
        Returns:
            내보내기 결과 (COCO: dict, YOLO: str)
        """
        if format is None:
            format = self.config.output_format
        
        format = format.lower()
        
        if format == "coco":
            return self.export_coco(result, output_path)
        elif format == "yolo":
            return self.export_yolo(result, output_path)
        else:
            raise ValueError(f"지원하지 않는 포맷: {format}. 지원 포맷: coco, yolo")
    
    def visualize(
        self,
        result: LabelingResult,
        output_dir: Optional[str] = None,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        라벨링 결과 시각화
        
        Args:
            result: LabelingResult 객체 (원본 출력 저장)
            output_dir: 시각화 이미지 저장 디렉터리 (None이면 저장 안함)
            
        Returns:
            (dino_result, sam_result, combined_result) 튜플
            - dino_result: DINO Bounding Box만 그린 이미지
            - sam_result: SAM Segmentation Mask만 그린 이미지  
            - combined_result: 둘 다 그린 이미지
        """
        # 원본 출력을 시각화용 형식으로 변환
        from agent.utils.web_display_converter import (
            convert_dino_boxes_for_display,
            convert_sam_masks_for_display,
            convert_scores_for_display,
        )
        
        boxes_xyxy = convert_dino_boxes_for_display(
            result.boxes, result.image_width, result.image_height
        )
        scores_np = convert_scores_for_display(result.scores)
        masks_list = convert_sam_masks_for_display(result.masks)
        
        dino_result, sam_result, combined_result = draw_dino_and_sam(
            image=result.image_path,
            boxes=boxes_xyxy,
            labels=result.labels,
            masks=masks_list,
            scores=scores_np,
            normalized=False,  # 픽셀 좌표
        )
        
        # 저장
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_stem = Path(result.image_path).stem
            
            save_visualization(dino_result, output_path / f"{image_stem}_dino_bbox.jpg")
            save_visualization(sam_result, output_path / f"{image_stem}_sam_mask.jpg")
            save_visualization(combined_result, output_path / f"{image_stem}_combined.jpg")
            
            logger.info(f"시각화 저장 완료: {output_path}")
        
        return dino_result, sam_result, combined_result
    
    def visualize_dino(
        self,
        result: LabelingResult,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        DINO Bounding Box만 시각화
        
        Args:
            result: LabelingResult 객체 (원본 출력 저장)
            output_path: 저장 경로 (선택사항)
            
        Returns:
            Bounding Box가 그려진 PIL Image
        """
        # 원본 출력을 시각화용 형식으로 변환
        from agent.utils.web_display_converter import (
            convert_dino_boxes_for_display,
            convert_scores_for_display,
        )
        
        boxes_xyxy = convert_dino_boxes_for_display(
            result.boxes, result.image_width, result.image_height
        )
        scores_np = convert_scores_for_display(result.scores)
        
        dino_result = draw_bounding_boxes(
            image=result.image_path,
            boxes=boxes_xyxy,
            labels=result.labels,
            scores=scores_np,
            normalized=False,  # 픽셀 좌표
        )
        
        if output_path:
            save_visualization(dino_result, output_path)
        
        return dino_result
    
    def visualize_sam(
        self,
        result: LabelingResult,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        SAM Segmentation Mask만 시각화
        
        Args:
            result: LabelingResult 객체 (원본 출력 저장)
            output_path: 저장 경로 (선택사항)
            
        Returns:
            Segmentation Mask가 그려진 PIL Image
        """
        # 원본 출력을 시각화용 형식으로 변환
        from agent.utils.web_display_converter import convert_sam_masks_for_display
        
        masks_list = convert_sam_masks_for_display(result.masks)
        
        sam_result = draw_segmentation_masks(
            image=result.image_path,
            masks=masks_list,
            labels=result.labels,
        )
        
        if output_path:
            save_visualization(sam_result, output_path)
        
        return sam_result
    
    def process_directory(
        self,
        image_dir: str,
        text_prompt: str,
        output_dir: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        format: Optional[str] = None,
    ) -> List[LabelingResult]:
        """
        디렉터리의 모든 이미지 처리
        
        Args:
            image_dir: 이미지 디렉터리 경로
            text_prompt: 텍스트 프롬프트
            output_dir: 출력 디렉터리 (None이면 config 사용)
            confidence_threshold: Confidence threshold
            format: 출력 포맷
        
        Returns:
            처리 결과 리스트
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        if format is None:
            format = self.config.output_format
        
        # 출력 디렉터리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 찾기
        image_dir_path = Path(image_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in image_dir_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {image_dir}")
        
        logger.info(f"{len(image_files)}개 이미지 파일 발견")
        
        results = []
        coco_datasets = []
        
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"[{i+1}/{len(image_files)}] 처리 중: {image_file.name}")
                
                result = self.process_image(
                    image_file,
                    text_prompt,
                    confidence_threshold,
                )
                results.append(result)
                
                # 포맷에 따라 저장
                if format == "yolo":
                    label_path = output_path / f"{image_file.stem}.txt"
                    self.export_yolo(result, str(label_path), save_classes=(i == 0))
                elif format == "coco":
                    # 원본 출력을 COCO 형식으로 변환
                    boxes_xyxy = cxcywh_to_xyxy(
                        boxes=result.boxes,
                        image_width=result.image_width,
                        image_height=result.image_height,
                        normalized=True,
                    )
                    boxes_np = boxes_xyxy.cpu().numpy() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy
                    scores_np = result.scores.cpu().numpy() if isinstance(result.scores, torch.Tensor) else result.scores
                    
                    # 마스크 변환: (N, 1, H, W) -> List of (H, W)
                    masks_list = None
                    if result.has_masks:
                        masks_np = result.masks.cpu().numpy()  # (N, 1, H, W)
                        masks_list = [masks_np[j, 0] for j in range(masks_np.shape[0])]
                    
                    coco_data = self.coco_converter.convert(
                        boxes=boxes_np,
                        scores=scores_np,
                        labels=result.labels,
                        masks=masks_list,
                        image_id=i + 1,
                        image_width=result.image_width,
                        image_height=result.image_height,
                        image_filename=image_file.name,
                    )
                    coco_datasets.append(coco_data)
                
            except Exception as e:
                logger.error(f"이미지 처리 실패 ({image_file}): {e}")
        
        # COCO의 경우 하나의 파일로 병합
        if format == "coco" and coco_datasets:
            merged_coco = self.coco_converter.merge(coco_datasets)
            self.coco_converter.save(merged_coco, str(output_path / "annotations.json"))
        
        logger.info(f"처리 완료: {len(results)}/{len(image_files)}개 이미지")
        return results
