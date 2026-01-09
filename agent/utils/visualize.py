"""
시각화 유틸리티 모듈
- DINO: Bounding Box 시각화
- SAM: Segmentation Mask 시각화
"""

import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 클래스별 색상 팔레트
DEFAULT_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky Blue
    (255, 0, 128),    # Pink
]


def get_color(index: int) -> Tuple[int, int, int]:
    """인덱스에 따른 색상 반환"""
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]


def draw_bounding_boxes(
    image: Union[str, Path, Image.Image],
    boxes: np.ndarray,
    labels: List[str],
    scores: Optional[np.ndarray] = None,
    line_width: int = 3,
    font_size: int = 16,
    normalized: bool = True,
) -> Image.Image:
    """
    DINO Bounding Box 시각화
    
    Args:
        image: 이미지 경로 또는 PIL Image
        boxes: (N, 4) [x1, y1, x2, y2] 박스 좌표
        labels: 클래스 레이블 리스트
        scores: Confidence 점수 (선택사항)
        line_width: 박스 선 두께
        font_size: 레이블 폰트 크기
        normalized: 좌표가 정규화(0-1)되어 있는지 여부
        
    Returns:
        박스가 그려진 PIL Image
    """
    # 이미지 로드
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.copy().convert("RGB")
    
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # 클래스별 색상 매핑
    unique_labels = list(set(labels))
    label_to_color = {label: get_color(i) for i, label in enumerate(unique_labels)}
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # 정규화 좌표인 경우 픽셀 좌표로 변환
        if normalized:
            x1, y1, x2, y2 = box
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        else:
            x1, y1, x2, y2 = box
        
        color = label_to_color[label]
        
        # 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # 레이블 텍스트 생성
        if scores is not None and i < len(scores):
            text = f"{label}: {scores[i]:.2f}"
        else:
            text = label
        
        # 텍스트 배경
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 텍스트 배경 박스
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        # 텍스트 그리기
        draw.text((x1 + 2, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)
    
    logger.debug(f"Bounding box {len(boxes)}개 시각화 완료")
    return pil_image


def draw_segmentation_masks(
    image: Union[str, Path, Image.Image],
    masks: List[np.ndarray],
    labels: Optional[List[str]] = None,
    alpha: float = 0.5,
    draw_contours: bool = True,
    contour_width: int = 2,
) -> Image.Image:
    """
    SAM Segmentation Mask 시각화
    
    Args:
        image: 이미지 경로 또는 PIL Image
        masks: 마스크 리스트 (각각 H x W binary array)
        labels: 클래스 레이블 리스트 (색상 결정용)
        alpha: 마스크 투명도 (0-1)
        draw_contours: 마스크 윤곽선 그리기 여부
        contour_width: 윤곽선 두께
        
    Returns:
        마스크가 그려진 PIL Image
    """
    # 이미지 로드
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.copy().convert("RGB")
    
    if not masks:
        return pil_image
    
    width, height = pil_image.size
    
    # 마스크 오버레이 생성
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    # 클래스별 색상 매핑
    if labels:
        unique_labels = list(set(labels))
        label_to_color = {label: get_color(i) for i, label in enumerate(unique_labels)}
    
    for i, mask in enumerate(masks):
        # 마스크 크기 조정
        if mask.shape != (height, width):
            mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height)))
            mask = (mask_resized > 127).astype(np.uint8)
        
        # 색상 결정
        if labels and i < len(labels):
            color = label_to_color[labels[i]]
        else:
            color = get_color(i)
        
        # 마스크 영역 오버레이
        mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = (*color, int(255 * alpha))
        
        mask_image = Image.fromarray(mask_rgba, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_image)
        
        # 윤곽선 그리기
        if draw_contours:
            try:
                import cv2
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                contour_overlay = np.zeros((height, width, 4), dtype=np.uint8)
                cv2.drawContours(contour_overlay, contours, -1, (*color, 255), contour_width)
                contour_image = Image.fromarray(contour_overlay, mode="RGBA")
                overlay = Image.alpha_composite(overlay, contour_image)
            except ImportError:
                logger.warning("OpenCV가 설치되지 않아 윤곽선을 그릴 수 없습니다")
    
    # 원본 이미지와 합성
    pil_image = pil_image.convert("RGBA")
    result = Image.alpha_composite(pil_image, overlay)
    
    logger.debug(f"Segmentation mask {len(masks)}개 시각화 완료")
    return result.convert("RGB")


def draw_dino_and_sam(
    image: Union[str, Path, Image.Image],
    boxes: np.ndarray,
    labels: List[str],
    masks: List[np.ndarray],
    scores: Optional[np.ndarray] = None,
    normalized: bool = True,
    box_line_width: int = 3,
    mask_alpha: float = 0.4,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    DINO Bounding Box와 SAM Mask를 각각 시각화
    
    Args:
        image: 이미지 경로 또는 PIL Image
        boxes: (N, 4) [x1, y1, x2, y2] 박스 좌표
        labels: 클래스 레이블 리스트
        masks: 마스크 리스트
        scores: Confidence 점수 (선택사항)
        normalized: 박스 좌표가 정규화(0-1)되어 있는지 여부
        box_line_width: 박스 선 두께
        mask_alpha: 마스크 투명도
        
    Returns:
        (dino_result, sam_result, combined_result) 튜플
        - dino_result: Bounding Box만 그린 이미지
        - sam_result: Segmentation Mask만 그린 이미지
        - combined_result: 둘 다 그린 이미지
    """
    # 이미지 로드
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = image.copy().convert("RGB")
    
    # DINO: Bounding Box
    dino_result = draw_bounding_boxes(
        pil_image,
        boxes,
        labels,
        scores,
        line_width=box_line_width,
        normalized=normalized,
    )
    
    # SAM: Segmentation Mask
    sam_result = draw_segmentation_masks(
        pil_image,
        masks,
        labels,
        alpha=mask_alpha,
    )
    
    # Combined: Mask + Box
    combined_result = draw_segmentation_masks(
        pil_image,
        masks,
        labels,
        alpha=mask_alpha,
    )
    combined_result = draw_bounding_boxes(
        combined_result,
        boxes,
        labels,
        scores,
        line_width=box_line_width,
        normalized=normalized,
    )
    
    return dino_result, sam_result, combined_result


def save_visualization(
    image: Image.Image,
    output_path: Union[str, Path],
    quality: int = 95,
) -> str:
    """
    시각화 이미지 저장
    
    Args:
        image: PIL Image
        output_path: 저장 경로
        quality: JPEG 품질 (1-100)
        
    Returns:
        저장된 파일 경로
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 포맷에 따라 저장
    if output_path.suffix.lower() in [".jpg", ".jpeg"]:
        image.save(output_path, quality=quality)
    else:
        image.save(output_path)
    
    logger.info(f"시각화 이미지 저장: {output_path}")
    return str(output_path)
