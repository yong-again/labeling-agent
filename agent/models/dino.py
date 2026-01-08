"""
Grounding DINO 모델 래퍼
오브젝트 검출 (bounding box 생성)
"""

import logging
from typing import List, Tuple, Optional
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    import groundingdino
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
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: 모델 이름 (config_path가 None일 때 자동으로 config 파일 찾는데 사용)
            config_path: Config 파일 경로 (None이면 자동으로 찾음)
            checkpoint_path: Checkpoint 파일 경로 (None이면 자동으로 찾음)
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
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()
    
    def _get_config_path(self) -> str:
        """설정 파일 경로 반환 (예제 패턴: groundingdino/config/GroundingDINO_SwinT_OGC.py)"""
        # 1. config_path가 명시적으로 제공된 경우 사용
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"지정된 Config 경로 사용: {self.config_path}")
            return self.config_path
        
        # 2. 자동으로 찾기 (모델 이름 기반)
        model_lower = self.model_name.lower()
        
        # 모델 이름에 따라 config 파일 선택
        if "swinb" in model_lower or "swin_b" in model_lower:
            config_file = "GroundingDINO_SwinB_cfg.py"
        elif "swint" in model_lower or "swin_t" in model_lower:
            config_file = "GroundingDINO_SwinT_OGC.py"
        else:
            # 기본값: SwinB
            config_file = "GroundingDINO_SwinB_cfg.py"
            logger.warning(f"알 수 없는 모델 이름: {self.model_name}, SwinB config 사용")
        
        # GroundingDINO 패키지의 config 디렉터리 찾기
        groundingdino_dir = os.path.dirname(groundingdino.__file__)
        config_path = os.path.join(groundingdino_dir, "config", config_file)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
        
        return config_path
    
    def _get_checkpoint_path(self) -> str:
        """체크포인트 파일 경로 반환 """
        # 1. checkpoint_path가 명시적으로 제공된 경우 사용
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"지정된 Checkpoint 경로 사용: {self.checkpoint_path}")
            return self.checkpoint_path
        
        # 2. 자동으로 찾기 (프로젝트의 weights 디렉터리에서)
        project_root = Path(__file__).parent.parent.parent
        weights_dir = project_root / "weights"
        
        # 모델 이름에 따라 체크포인트 파일명 추론
        model_lower = self.model_name.lower()
        
        if "swinb" in model_lower or "swin_b" in model_lower:
            checkpoint_name = "groundingdino_swinb_cogcoor.pth"
        elif "swint" in model_lower or "swin_t" in model_lower:
            checkpoint_name = "groundingdino_swint_ogc.pth"
        else:
            checkpoint_name = "groundingdino_swinb_cogcoor.pth"
        
        # weights 디렉터리에서 찾기
        local_checkpoint = weights_dir / checkpoint_name
        if local_checkpoint.exists():
            logger.info(f"로컬 체크포인트 사용: {local_checkpoint}")
            return str(local_checkpoint)
        
        # 대체 파일명으로 재시도
        for alt_name in ["groundingdino_swinb_cogcoor.pth", "groundingdino_swint_ogc.pth"]:
            if alt_name == checkpoint_name:
                continue
            alt_path = weights_dir / alt_name
            if alt_path.exists():
                logger.info(f"대체 체크포인트 사용: {alt_path}")
                return str(alt_path)
        
        raise FileNotFoundError(
            f"체크포인트 파일을 찾을 수 없습니다.\n"
            f"다음 위치에서 확인하세요: {weights_dir}/{checkpoint_name}"
        )
    
    def _load_model(self):
        """모델 로드"""
        logger.info(f"Grounding DINO 모델 로드 중: {self.model_name} (device: {self.device})")
        try:
            config_path = self._get_config_path()
            checkpoint_path = self._get_checkpoint_path()
            
            logger.info(f"Config 경로: {config_path}")
            logger.info(f"Checkpoint 경로: {checkpoint_path}")
            
            self.model = load_model(
                model_config_path=config_path,
                model_checkpoint_path=checkpoint_path,
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
