"""
COCO export 디버깅 시각화 스크립트
export_*.json과 업로드된 이미지를 오버레이로 저장합니다.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.utils.colors import color_for_index


def _decode_rle_string(counts_str: str) -> List[int]:
    counts: List[int] = []
    m = 0
    x = 0
    for ch in counts_str:
        c = ord(ch) - 48
        x |= (c & 0x1F) << (5 * m)
        if c & 0x20:
            m += 1
        else:
            if c & 0x10:
                x |= -1 << (5 * m)
            counts.append(x)
            m = 0
            x = 0
    return counts


def _decode_rle_counts(rle: Dict[str, Any]) -> Tuple[List[int], str, Optional[str]]:
    counts = rle.get("counts", [])
    counts_type = type(counts).__name__
    decoded_from: Optional[str] = None
    if isinstance(counts, (bytes, bytearray)):
        try:
            counts = counts.decode("ascii")
        except Exception:
            return [], counts_type, counts_type
        decoded_from = counts_type
    if isinstance(counts, str):
        decoded_from = counts_type if decoded_from is None else decoded_from
        return _decode_rle_string(counts), counts_type, decoded_from
    if isinstance(counts, list):
        return counts, counts_type, decoded_from
    return [], counts_type, decoded_from


def _decode_rle(rle: Dict[str, Any]) -> Tuple[np.ndarray, str, int, Optional[str]]:
    size = rle.get("size", [])
    if len(size) != 2:
        return np.zeros((0, 0), dtype=np.uint8), "unknown", 0, None

    counts, counts_type, decoded_from = _decode_rle_counts(rle)
    if not counts:
        return np.zeros((0, 0), dtype=np.uint8), counts_type, 0, decoded_from

    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        if run > 0:
            flat[idx:idx + run] = val
        idx += run
        val = 1 - val

    return flat.reshape((height, width), order="F"), counts_type, len(counts), decoded_from


def _debug_rle(
    ann: Dict[str, Any],
    rle: Dict[str, Any],
    mask: np.ndarray,
    counts_type: str,
    counts_len: int,
    decoded_from: Optional[str],
):
    ann_id = ann.get("id")
    size = rle.get("size", [])
    expected_pixels = int(size[0]) * int(size[1]) if len(size) == 2 else 0
    decoded_pixels = int(mask.size)
    foreground = int(mask.sum()) if mask.size else 0
    decoded_info = f" decoded_from={decoded_from}" if decoded_from else ""
    print(
        "[rle-debug] "
        f"ann_id={ann_id} size={size} counts_type={counts_type} "
        f"counts_len={counts_len} expected_pixels={expected_pixels} "
        f"decoded_pixels={decoded_pixels} foreground={foreground}{decoded_info}"
    )


def _get_image_entry(
    coco: Dict[str, Any],
    image_id: Optional[int],
    file_name: Optional[str],
) -> Dict[str, Any]:
    images = coco.get("images", [])
    if not images:
        raise ValueError("COCO json에 images가 없습니다.")

    if image_id is not None:
        for img in images:
            if img.get("id") == image_id:
                return img
        raise ValueError(f"image_id={image_id}를 찾을 수 없습니다.")

    if file_name is not None:
        for img in images:
            if img.get("file_name") == file_name:
                return img
        raise ValueError(f"file_name={file_name}을 찾을 수 없습니다.")

    return images[0]


def _get_annotations(coco: Dict[str, Any], image_id: int) -> List[Dict[str, Any]]:
    return [ann for ann in coco.get("annotations", []) if ann.get("image_id") == image_id]


def _get_category_name(coco: Dict[str, Any], category_id: int) -> str:
    for cat in coco.get("categories", []):
        if cat.get("id") == category_id:
            return cat.get("name", "")
    return str(category_id)


def _overlay_polygon(
    base: Image.Image,
    polygon: List[float],
    color: Tuple[int, int, int],
    color_weight: float = 0.3,
) -> Image.Image:
    """Polygon을 color로 fill (color_weight 비율로 블렌드)"""
    if len(polygon) < 6:
        return base
    points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
    w, h = base.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    alpha = int(255 * color_weight)
    mask_arr = np.array(mask)
    mask_arr = (mask_arr > 0).astype(np.uint8) * alpha
    mask_img = Image.fromarray(mask_arr, mode="L")
    color_img = Image.new("RGBA", base.size, color + (0,))
    color_img.putalpha(mask_img)
    return Image.alpha_composite(base.convert("RGBA"), color_img)


def _overlay_mask(
    base: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    color_weight: float = 0.3,
) -> Image.Image:
    """Segment mask를 color로 fill (color_weight 비율로 블렌드)"""
    if mask.size == 0:
        return base
    alpha = int(255 * color_weight)
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * alpha, mode="L")
    color_img = Image.new("RGBA", base.size, color + (0,))
    color_img.putalpha(mask_img)
    return Image.alpha_composite(base.convert("RGBA"), color_img)


def _draw_class_text(
    draw: ImageDraw.ImageDraw,
    bbox: List[float],
    text: str,
    color: Tuple[int, int, int],
    image_width: int,
    image_height: int,
) -> None:
    if not text or len(bbox) != 4:
        return
    x, y, w, h = bbox
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
    bg = (0, 0, 0)  # text 배경
    draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=bg)
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255))  # 흰 글씨로 가독성


def visualize_export(
    coco_path: Path,
    uploads_dir: Path,
    image_id: Optional[int],
    file_name: Optional[str],
    output_path: Optional[Path],
    color_weight: float = 0.3,
):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_entry = _get_image_entry(coco, image_id, file_name)
    image_id = image_entry.get("id")
    image_file = uploads_dir / image_entry.get("file_name", "")
    if not image_file.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_file}")

    image = Image.open(image_file).convert("RGBA")
    w_img, h_img = image.size

    annotations = _get_annotations(coco, image_id)

    # 1) Segment mask를 color로 fill (color_weight 비율 블렌드)
    for idx, ann in enumerate(annotations):
        color = color_for_index(idx)
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            for polygon in seg:
                image = _overlay_polygon(image, polygon, color, color_weight)
        elif isinstance(seg, dict):
            mask, counts_type, counts_len, decoded_from = _decode_rle(seg)
            _debug_rle(ann, seg, mask, counts_type, counts_len, decoded_from)
            image = _overlay_mask(image, mask, color, color_weight)

    draw = ImageDraw.Draw(image)

    # 2) bbox와 class text 표시
    for idx, ann in enumerate(annotations):
        color = color_for_index(idx)
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        cat_name = _get_category_name(coco, ann.get("category_id", 0))
        if bbox and cat_name:
            _draw_class_text(draw, bbox, cat_name, color, w_img, h_img)

    if output_path is None:
        output_path = coco_path.parent / f"debug_{image_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    print(f"saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="COCO export 디버깅 시각화")
    parser.add_argument("--json", required=True, help="export_*.json 경로")
    parser.add_argument("--uploads-dir", default="../uploads", help="업로드 이미지 디렉터리")
    parser.add_argument("--image-id", type=int, default=None, help="이미지 ID (옵션)")
    parser.add_argument("--file-name", default=None, help="이미지 파일명 (옵션)")
    parser.add_argument("--output", default=None, help="결과 이미지 저장 경로 (옵션)")
    parser.add_argument("--color-weight", type=float, default=0.3, help="마스크 색상 가중치 (0~1, 기본 0.3)")
    args = parser.parse_args()

    visualize_export(
        coco_path=Path(args.json),
        uploads_dir=Path(args.uploads_dir),
        image_id=args.image_id,
        file_name=args.file_name,
        output_path=Path(args.output) if args.output else None,
        color_weight=args.color_weight,
    )


if __name__ == "__main__":
    main()
