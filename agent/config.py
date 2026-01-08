"""
설정 관리 모듈
환경변수 및 기본 설정값 관리 (.env 파일 지원)
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

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
        # 프로젝트 루트 디렉터리에서 .env 파일 찾기
        current_dir = Path(__file__).parent.parent
        env_file = current_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        else:
            # 현재 작업 디렉터리에서도 찾기
            load_dotenv()


@dataclass
class Config:
    """전역 설정 클래스"""
    
    # Label Studio 설정
    ls_url: str
    ls_api_token: str
    ls_project_id: Optional[int] = None
    
    # 모델 설정
    dino_model_name: str = "groundingdino_swint_ogc"
    dino_config_path: Optional[str] = '/workspace/labeling-agent/agent/model_config/GroundingDINO_SwinT_OGC.py'
    dino_checkpoint_path: Optional[str] = "/workspace/labeling-agent/weights/groundingdino_swint_ogc.pth"
    sam_model_name: str = "default"  # default, vit_h, vit_l, vit_b 또는 sam_vit_h, sam_vit_l, sam_vit_b
    sam_checkpoint_path: Optional[str] = None
    
    # 파이프라인 설정
    confidence_threshold: float = 0.35
    device: Optional[str] = None  # None이면 자동 선택
    batch_size: int = 1
    
    # 출력 설정
    output_format: str = "polygonlabels"  # "polygonlabels" or "rectanglelabels"
    
    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """
        환경변수에서 설정 로드 (.env 파일 지원)
        
        Args:
            env_path: .env 파일 경로. None이면 자동 탐색
        
        환경변수 우선순위:
        1. 시스템 환경변수
        2. .env 파일
        """
        # .env 파일 로드 (시스템 환경변수보다 우선순위 낮음)
        load_env_file(env_path)
        
        ls_url = os.getenv("LS_URL")
        ls_api_token = os.getenv("LS_API_TOKEN")
        
        if not ls_url or not ls_api_token:
            raise ValueError(
                "환경변수 LS_URL과 LS_API_TOKEN이 필요합니다.\n"
                "다음 중 하나의 방법으로 설정하세요:\n"
                "1. .env 파일 생성 (권장):\n"
                "   LS_URL=http://localhost:8080\n"
                "   LS_API_TOKEN=your_token_here\n"
                "2. 환경변수로 설정:\n"
                "   export LS_URL=http://localhost:8080\n"
                "   export LS_API_TOKEN=your_token_here"
            )
        
        ls_project_id = os.getenv("LS_PROJECT_ID")
        if ls_project_id:
            ls_project_id = int(ls_project_id)
        
        # GPU 자동 선택
        device = os.getenv("DEVICE")
        if not device:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam_checkpoint = os.getenv("SAM_CHECKPOINT_PATH")
        dino_config = os.getenv("DINO_CONFIG_PATH")
        dino_checkpoint = os.getenv("DINO_CHECKPOINT_PATH")
        
        return cls(
            ls_url=ls_url.rstrip("/"),
            ls_api_token=ls_api_token,
            ls_project_id=ls_project_id,
            device=device,
            sam_checkpoint_path=sam_checkpoint,
            dino_config_path=dino_config,
            dino_checkpoint_path=dino_checkpoint,
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
            batch_size=int(os.getenv("BATCH_SIZE", "1")),
            output_format=os.getenv("OUTPUT_FORMAT", "polygonlabels"),
        )

