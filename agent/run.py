"""
CLI 실행 스크립트
"""

import argparse
import logging
import sys
from pathlib import Path

from agent.config import Config
from agent.pipeline import LabelingPipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="Label Studio + DINO + SAM 기반 자동 프리라벨링 파이프라인"
    )
    
    # 필수 인자
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 이미지 디렉터리 경로",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="텍스트 프롬프트 (예: 'phone, screen, crack')",
    )
    
    # 선택 인자
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio 프로젝트 ID (환경변수 LS_PROJECT_ID 우선)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Confidence threshold (기본값: 0.35)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (기본값: 1)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["polygonlabels", "rectanglelabels"],
        default=None,
        help="출력 포맷 (기본값: polygonlabels)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="디바이스 (기본값: 자동 선택)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 조정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 설정 로드
        config = Config.from_env()
        
        # CLI 인자로 오버라이드
        if args.project_id is not None:
            config.ls_project_id = args.project_id
        if args.threshold is not None:
            config.confidence_threshold = args.threshold
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.output_format is not None:
            config.output_format = args.output_format
        if args.device is not None:
            config.device = args.device
        
        # 프로젝트 ID 확인
        if config.ls_project_id is None:
            logger.error("프로젝트 ID가 필요합니다 (--project-id 또는 환경변수 LS_PROJECT_ID)")
            sys.exit(1)
        
        # 입력 디렉터리 확인
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"입력 디렉터리가 존재하지 않습니다: {args.input}")
            sys.exit(1)
        if not input_path.is_dir():
            logger.error(f"입력 경로가 디렉터리가 아닙니다: {args.input}")
            sys.exit(1)
        
        # 파이프라인 실행
        logger.info("=" * 60)
        logger.info("Labeling Pipeline Agent 시작")
        logger.info(f"입력 디렉터리: {args.input}")
        logger.info(f"프롬프트: {args.prompt}")
        logger.info(f"프로젝트 ID: {config.ls_project_id}")
        logger.info(f"Confidence Threshold: {config.confidence_threshold}")
        logger.info(f"출력 포맷: {config.output_format}")
        logger.info(f"디바이스: {config.device}")
        logger.info("=" * 60)
        
        pipeline = LabelingPipeline(config)
        stats = pipeline.process_directory(
            image_dir=str(input_path),
            text_prompt=args.prompt,
            project_id=config.ls_project_id,
            confidence_threshold=config.confidence_threshold,
            batch_size=config.batch_size,
        )
        
        # 결과 출력
        logger.info("=" * 60)
        logger.info("처리 완료")
        logger.info(f"전체: {stats['total']}개")
        logger.info(f"성공: {stats['processed']}개")
        logger.info(f"실패: {stats['failed']}개")
        logger.info(f"태스크 생성: {stats['tasks_created']}개")
        logger.info(f"예측 업로드: {stats['predictions_uploaded']}개")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
