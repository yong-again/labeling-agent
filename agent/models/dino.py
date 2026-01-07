"""
Grounding DINO 모델 래퍼
오브젝트 검출 (bounding box 생성)
"""

import logging
from typing import List, Tuple, Optional
import torch
import numpy as np
from PIL import Image

try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Grounding DINO가 설치되지 않았습니다.\n"
        "설치: pip install groundingdino-py"
    )

logger = logging.getLogger(__name__)


class GroundingDINO:
    """Grounding DINO 모델 래퍼"""
    
    def __init__(
        self,
        model_name: str = "groundingdino/groundingdino_swinb_cogcoor",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: 모델 이름 또는 경로
            device: 디바이스 ('cuda' or 'cpu'). None이면 자동 선택
        """
        if not GROUNDING_DINO_AVAILABLE:
            raise ImportError(
                "Grounding DINO가 설치되지 않았습니다.\n"
                "설치: pip install groundingdino-py"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        logger.info(f"Grounding DINO 모델 로드 중: {self.model_name} (device: {self.device})")
        try:
            self.model = load_model(
                model_config_path=None,  # 자동으로 설정 파일 찾음
                model_checkpoint_path=None,  # 자동으로 체크포인트 찾음
                device=self.device,
            )
            self.model.eval()
            logger.info("모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def predict(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        이미지에서 오브젝트 검출
        
        Args:
            image: PIL Image
            text_prompt: 텍스트 프롬프트 (예: "phone. screen. crack")
            box_threshold: 박스 confidence threshold
            text_threshold: 텍스트 매칭 threshold
        
        Returns:
            boxes: (N, 4) numpy array [x1, y1, x2, y2] (정규화 좌표 0-1)
            scores: (N,) numpy array confidence scores
            labels: List[str] 클래스 레이블
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        logger.debug(f"검출 실행: prompt='{text_prompt}', threshold={box_threshold}")
        
        # 이미지 전처리
        image_source, _ = load_image(image)
        
        # 예측
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_source,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        # 결과 후처리
        if len(boxes) == 0:
            logger.debug("검출된 박스가 없습니다")
            return np.array([]).reshape(0, 4), np.array([]), []
        
        # boxes는 이미 [x1, y1, x2, y2] 형식 (정규화 좌표)
        boxes_np = boxes.cpu().numpy()
        scores_np = logits.cpu().numpy()
        
        # phrases에서 클래스 이름 추출
        # phrases 형식: "phone", "screen" 등
        labels = [phrase.strip() for phrase in phrases]
        
        logger.info(f"{len(boxes_np)}개 박스 검출됨")
        
        return boxes_np, scores_np, labels
    
    def predict_batch(
        self,
        images: List[Image.Image],
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """배치 예측 (현재는 순차 처리)"""
        results = []
        for image in images:
            result = self.predict(image, text_prompt, box_threshold, text_threshold)
            results.append(result)
        return results
