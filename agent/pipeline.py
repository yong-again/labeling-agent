"""
파이프라인 오케스트레이션
DINO -> SAM -> Label Studio 변환 및 업로드
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from agent.config import Config
from agent.ls_client import LabelStudioClient
from agent.models.dino import GroundingDINO
from agent.models.sam import SAM
from agent.converters.ls_format import LabelStudioConverter

logger = logging.getLogger(__name__)


class LabelingPipeline:
    """라벨링 파이프라인"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        
        # 클라이언트 및 모델 초기화
        self.ls_client = LabelStudioClient(config.ls_url, config.ls_api_token)
        self.dino = GroundingDINO(
            model_name=config.dino_model_name,
            device=config.device,
        )
        self.sam = SAM(
            model_type=config.sam_model_name,
            checkpoint_path=config.sam_checkpoint_path,
            device=config.device,
        )
        self.converter = LabelStudioConverter(output_format=config.output_format)
        
        logger.info("파이프라인 초기화 완료")
    
    def process_image(
        self,
        image_path: str,
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray]]:
        """
        단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            text_prompt: 텍스트 프롬프트
            confidence_threshold: Confidence threshold (None이면 config 사용)
        
        Returns:
            boxes, scores, labels, masks
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold
        
        # 이미지 로드
        logger.info(f"이미지 로드: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        
        # DINO: Bounding box 검출
        logger.info(f"DINO 검출 실행: prompt='{text_prompt}'")
        boxes, scores, labels = self.dino.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=confidence_threshold,
        )
        
        if len(boxes) == 0:
            logger.warning("검출된 박스가 없습니다")
            return boxes, scores, labels, []
        
        # SAM: Box -> Mask 변환
        logger.info(f"SAM 마스크 생성: {len(boxes)}개 박스")
        masks = self.sam.predict_from_boxes(image, boxes)
        
        return boxes, scores, labels, masks
    
    def process_image_to_ls_format(
        self,
        image_path: str,
        text_prompt: str,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        이미지를 처리하여 Label Studio 포맷으로 변환
        
        Args:
            image_path: 이미지 파일 경로
            text_prompt: 텍스트 프롬프트
            confidence_threshold: Confidence threshold
        
        Returns:
            Label Studio prediction 포맷 리스트
        """
        # 이미지 로드 (크기 확인용)
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        
        # 파이프라인 실행
        boxes, scores, labels, masks = self.process_image(
            image_path, text_prompt, confidence_threshold
        )
        
        if len(boxes) == 0:
            return []
        
        # Label Studio 포맷으로 변환
        results = self.converter.convert(
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks=masks if self.config.output_format == "polygonlabels" else None,
            image_width=image_width,
            image_height=image_height,
        )
        
        return results
    
    def process_directory(
        self,
        image_dir: str,
        text_prompt: str,
        project_id: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        디렉터리의 모든 이미지 처리 및 Label Studio 업로드
        
        Args:
            image_dir: 이미지 디렉터리 경로
            text_prompt: 텍스트 프롬프트
            project_id: Label Studio 프로젝트 ID (None이면 config 사용)
            confidence_threshold: Confidence threshold
            batch_size: 배치 크기 (None이면 config 사용)
        
        Returns:
            처리 결과 통계
        """
        if project_id is None:
            project_id = self.config.ls_project_id
            if project_id is None:
                raise ValueError("project_id가 필요합니다 (인자 또는 환경변수 LS_PROJECT_ID)")
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
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
        
        # 통계
        stats = {
            "total": len(image_files),
            "processed": 0,
            "failed": 0,
            "tasks_created": 0,
            "predictions_uploaded": 0,
        }
        
        # 배치 처리
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            logger.info(f"배치 {i//batch_size + 1} 처리 중: {len(batch_files)}개 파일")
            
            # 태스크 생성
            tasks = []
            for image_file in batch_files:
                try:
                    # 로컬 파일 경로를 Label Studio가 접근할 수 있는 형태로 변환
                    # 실제 환경에서는 서버에 마운트된 경로나 URL 사용 필요
                    task_data = {
                        "data": {
                            "image": str(image_file.absolute())
                        }
                    }
                    tasks.append(task_data)
                except Exception as e:
                    logger.error(f"태스크 생성 실패 ({image_file}): {e}")
                    stats["failed"] += 1
            
            if not tasks:
                continue
            
            # Label Studio에 태스크 업로드
            try:
                created_tasks = self.ls_client.create_tasks(project_id, tasks)
                stats["tasks_created"] += len(created_tasks)
                logger.info(f"{len(created_tasks)}개 태스크 생성됨")
            except Exception as e:
                logger.error(f"태스크 업로드 실패: {e}")
                stats["failed"] += len(tasks)
                continue
            
            # 각 이미지 처리 및 예측 업로드
            predictions = []
            for image_file, task in zip(batch_files, created_tasks):
                try:
                    # 이미지 처리
                    results = self.process_image_to_ls_format(
                        str(image_file),
                        text_prompt,
                        confidence_threshold,
                    )
                    
                    if results:
                        prediction = {
                            "task": task["id"],
                            "result": results,
                            "score": float(np.mean([r.get("score", 0.5) for r in results])),
                        }
                        predictions.append(prediction)
                        stats["processed"] += 1
                    else:
                        logger.warning(f"검출 결과 없음: {image_file}")
                        stats["processed"] += 1  # 처리됨으로 간주
                
                except Exception as e:
                    logger.error(f"이미지 처리 실패 ({image_file}): {e}")
                    stats["failed"] += 1
            
            # 예측 업로드
            if predictions:
                try:
                    uploaded = self.ls_client.upload_predictions(project_id, predictions)
                    stats["predictions_uploaded"] += len(uploaded)
                    logger.info(f"{len(uploaded)}개 예측 업로드됨")
                except Exception as e:
                    logger.error(f"예측 업로드 실패: {e}")
        
        logger.info(f"처리 완료: {stats}")
        return stats
