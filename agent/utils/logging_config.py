"""
로깅 설정 모듈
- 파일 저장 (logs/*.log)
- 콘솔 출력 시 위험도별 색상 (DEBUG=파랑, INFO=초록, WARNING=노랑, ERROR=빨강, CRITICAL=굵은 빨강)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# ANSI 색상 코드 (대부분 터미널 지원)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# 위험도별 색상
_COLORS = {
    logging.DEBUG: "\033[36m",      # Cyan (디버그)
    logging.INFO: "\033[32m",       # Green (정보)
    logging.WARNING: "\033[33m",    # Yellow (경고)
    logging.ERROR: "\033[31m",      # Red (오류)
    logging.CRITICAL: "\033[1;31m", # Bold Red (심각)
}


class ColoredFormatter(logging.Formatter):
    """콘솔용: 레벨별 색상을 적용하는 Formatter (record 원본 유지)"""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, _RESET)
        original = record.levelname
        record.levelname = f"{color}{original}{_RESET}"
        result = super().format(record)
        record.levelname = original  # 다른 핸들러를 위해 복원
        return result


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_file: bool = True,
    enable_console_color: bool = True,
) -> None:
    """
    로깅 설정: 파일 저장 + 콘솔 출력 (위험도별 색상)

    Args:
        level: 로그 레벨 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 디렉터리 (기본: ./logs)
        log_file: 로그 파일명 (기본: labeling_agent.log)
        enable_file: 파일 저장 여부
        enable_console_color: 콘솔 색상 사용 여부
    """
    log_dir = Path(log_dir or "./logs")
    log_file = log_file or "labeling_agent.log"
    log_path = log_dir / log_file

    # 기본 포맷
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # 루트 로거 설정
    root = logging.getLogger()
    root.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # 1) 콘솔 핸들러 (색상)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if enable_console_color:
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt=date_fmt))
    else:
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(console_handler)

    # 2) 파일 핸들러 (색상 없음 - 파일에는 plain text)
    if enable_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        root.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.debug(f"로깅 설정 완료: level={logging.getLevelName(level)}, file={log_path if enable_file else 'off'}")


def get_log_file_path(log_dir: str = "./logs", log_file: str = "labeling_agent.log") -> Path:
    """현재 사용 중인 로그 파일 경로 반환"""
    return Path(log_dir) / log_file
