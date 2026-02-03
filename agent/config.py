"""
설정 관리 모듈
DINO-SAM 자동 라벨링 설정 (.env 파일 지원)
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    .env 파일 로드
    
    Args:
        env_path: .env 파일 경로. None이면 프로젝트 루트에서 자동 탐색
    """
    if not DOTENV_AVAILABLE:
        return
    
    if env_path:
        load_dotenv(env_path)
    else:
        current_dir = Path(__file__).parent.parent
        env_file = current_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()


@dataclass
class Config:
    """전역 설정 클래스"""
    
    # 모델 설정
    dino_model_name: str = "groundingdino_swint_ogc"
    dino_config_path: Optional[str] = '/workspace/labeling-agent/agent/model_config/GroundingDINO_SwinT_OGC_cfg.py'
    dino_checkpoint_path: Optional[str] = "/workspace/labeling-agent/weights/groundingdino_swint_ogc.pth"
    sam_model_name: str = "vit_h"
    sam_checkpoint_path: Optional[str] = "/workspace/labeling-agent/weights/sam_vit_h_4b8939.pth"
    
    # 파이프라인 설정
    confidence_threshold: float = 0.35
    device: Optional[str] = None
    batch_size: int = 1
    
    # 출력 설정 (COCO or YOLO)
    output_format: Literal["coco", "yolo"] = "coco"
    output_dir: str = "./output"
    
    # HITL 설정
    hitl_enabled: bool = True
    feedback_db_path: str = "./feedback.db"
    
    # 웹 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000
    upload_dir: str = "./uploads"
    
    # 클래스 매핑 (프롬프트 -> 클래스 ID)
    class_mapping: dict = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """
        환경변수에서 설정 로드 (.env 파일 지원)
        """
        load_env_file(env_path)
        
        # GPU 자동 선택
        device = os.getenv("DEVICE")
        if not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam_checkpoint = os.getenv("SAM_CHECKPOINT_PATH")
        dino_config = os.getenv("DINO_CONFIG_PATH")
        dino_checkpoint = os.getenv("DINO_CHECKPOINT_PATH")
        
        return cls(
            device=device,
            sam_checkpoint_path=sam_checkpoint or cls.sam_checkpoint_path,
            dino_config_path=dino_config or cls.dino_config_path,
            dino_checkpoint_path=dino_checkpoint or cls.dino_checkpoint_path,
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
            batch_size=int(os.getenv("BATCH_SIZE", "1")),
            output_format=os.getenv("OUTPUT_FORMAT", "coco"),
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
            hitl_enabled=os.getenv("HITL_ENABLED", "true").lower() == "true",
            feedback_db_path=os.getenv("FEEDBACK_DB_PATH", "./feedback.db"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
        )
