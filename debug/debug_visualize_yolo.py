"""
YOLO export 디버깅 시각화 스크립트
yolo_*/ 디렉터리의 .txt 어노테이션과 업로드된 이미지를 오버레이로 저장합니다.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.utils.colors import color_for_index


def _load_classes(classes_path: Path) -> List[str]:
    """classes.txt에서 클래스 이름 로드"""
    if not classes_path.exists():
        return []
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_yolo_line(line: str) -> Optional[Tuple[int, List[float]]]:
    """
    YOLO segmentation 한 줄 파싱
    format: class_id x1 y1 x2 y2 ... (정규화 0-1)

    Returns:
        (class_id, [x1,y1,x2,y2,...]) or None
    """
    parts = line.strip().split()
    if len(parts) < 7:  # class_id + 최소 3점(6값)
        return None
    try:
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        if len(coords) % 2 != 0:
            return None
        return class_id, coords
    except (ValueError, IndexError):
        return None


def _parse_yolo_txt(txt_path: Path) -> List[Tuple[int, List[float]]]:
    """YOLO .txt 파일 파싱"""
    annotations = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_yolo_line(line)
        if parsed:
            annotations.append(parsed)
    return annotations


def _find_image(uploads_dir: Path, base_name: str) -> Optional[Path]:
    """이미지 파일 찾기 (확장자 시도)"""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = uploads_dir / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


def _norm_to_pixel(points: List[float], width: int, height: int) -> List[Tuple[float, float]]:
    """정규화 좌표(0-1)를 픽셀 좌표로 변환"""
    result = []
    for i in range(0, len(points), 2):
        x = points[i] * width
        y = points[i + 1] * height
        result.append((x, y))
    return result


def _overlay_polygon(
    base: Image.Image,
    points_px: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    color_weight: float = 0.3,
) -> Image.Image:
    """Polygon을 color로 fill (color_weight 비율로 블렌드)"""
    if len(points_px) < 3:
        return base
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(points_px, fill=255)
    alpha = int(255 * color_weight)
    mask_arr = (np.array(mask) > 0).astype(np.uint8) * alpha
    mask_img = Image.fromarray(mask_arr, mode="L")
    color_img = Image.new("RGBA", base.size, color + (0,))
    color_img.putalpha(mask_img)
    return Image.alpha_composite(base.convert("RGBA"), color_img)


def _draw_class_text(
    draw: ImageDraw.ImageDraw,
    points_px: List[Tuple[float, float]],
    text: str,
    color: Tuple[int, int, int],
    image_width: int,
    image_height: int,
) -> None:
    """클래스 텍스트 표시"""
    if not text or not points_px:
        return
    xs = [p[0] for p in points_px]
    ys = [p[1] for p in points_px]
    x, y = min(xs), min(ys)
    try:
        bbox_result = draw.textbbox((0, 0), text)
        tw = bbox_result[2] - bbox_result[0]
        th = bbox_result[3] - bbox_result[1]
    except AttributeError:
        tw, th = draw.textsize(text)
    pad = 2
    tx = max(0, min(x, image_width - tw - pad * 2))
    ty = max(0, min(y - th - pad * 2, image_height - th - pad * 2))
    if ty < 0:
        ty = y + pad
    bg = (0, 0, 0)
    draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=bg)
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255))


def visualize_yolo(
    yolo_dir: Path,
    uploads_dir: Path,
    image_base: Optional[str],
    output_path: Optional[Path],
    color_weight: float = 0.3,
) -> None:
    """
    YOLO export 시각화

    Args:
        yolo_dir: yolo_xxx 디렉터리 경로
        uploads_dir: 업로드 이미지 디렉터리
        image_base: 특정 이미지만 처리 (base name without ext). None이면 전체
        output_path: 결과 저장 경로
        color_weight: 마스크 색상 가중치 (0~1)
    """
    classes_path = yolo_dir / "classes.txt"
    class_names = _load_classes(classes_path)
    if not class_names:
        raise ValueError(f"classes.txt를 찾을 수 없거나 비어 있습니다: {classes_path}")

    txt_files = [p for p in yolo_dir.iterdir() if p.suffix == ".txt" and p.name != "classes.txt"]
    if not txt_files:
        raise ValueError(f"어노테이션 .txt 파일이 없습니다: {yolo_dir}")

    if image_base:
        txt_files = [p for p in txt_files if p.stem == image_base]
        if not txt_files:
            raise ValueError(f"지정한 이미지에 대한 .txt가 없습니다: {image_base}")

    for txt_path in txt_files:
        base_name = txt_path.stem
        image_path = _find_image(uploads_dir, base_name)
        if not image_path:
            print(f"[skip] 이미지를 찾을 수 없음: {base_name}")
            continue

        annotations = _parse_yolo_txt(txt_path)
        if not annotations:
            print(f"[skip] 파싱된 어노테이션 없음: {txt_path}")
            continue

        image = Image.open(image_path).convert("RGBA")
        w_img, h_img = image.size

        # 1) Segment mask color fill
        for idx, (class_id, points_norm) in enumerate(annotations):
            color = color_for_index(idx)
            points_px = _norm_to_pixel(points_norm, w_img, h_img)
            image = _overlay_polygon(image, points_px, color, color_weight)

        draw = ImageDraw.Draw(image)

        # 2) bbox outline 및 class text
        for idx, (class_id, points_norm) in enumerate(annotations):
            color = color_for_index(idx)
            points_px = _norm_to_pixel(points_norm, w_img, h_img)
            draw.polygon(points_px, outline=color, width=2)
            cat_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
            _draw_class_text(draw, points_px, cat_name, color, w_img, h_img)

        out = output_path if output_path and len(txt_files) == 1 else yolo_dir / f"debug_{base_name}.png"
        image.convert("RGB").save(out)
        print(f"saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="YOLO export 디버깅 시각화")
    parser.add_argument("--yolo-dir", required=True, help="yolo_xxx 디렉터리 경로")
    parser.add_argument("--uploads-dir", default="../uploads", help="업로드 이미지 디렉터리")
    parser.add_argument("--image", default=None, help="특정 이미지만 처리 (base name without ext)")
    parser.add_argument("--output", default=None, help="결과 이미지 저장 경로 (단일 이미지일 때)")
    parser.add_argument("--color-weight", type=float, default=0.3, help="마스크 색상 가중치 (0~1)")
    args = parser.parse_args()

    visualize_yolo(
        yolo_dir=Path(args.yolo_dir),
        uploads_dir=Path(args.uploads_dir),
        image_base=args.image,
        output_path=Path(args.output) if args.output else None,
        color_weight=args.color_weight,
    )


if __name__ == "__main__":
    main()
