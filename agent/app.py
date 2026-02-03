"""
FastAPI 웹 서버
DINO-SAM 라벨링 Web UI
"""

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import torch
import cv2

from agent.config import Config
from agent.pipeline import LabelingPipeline, LabelingResult
from agent.converters.coco_format import COCOConverter
from agent.converters.yolo_format import YOLOConverter
from agent.utils.box_transforms import cxcywh_to_xyxy, denormalize_boxes
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="DINO-SAM Labeling Agent",
    description="Auto-labeling with Grounding DINO + SAM",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 객체 (lazy initialization)
config: Optional[Config] = None
pipeline: Optional[LabelingPipeline] = None

# 라벨링 결과 캐시 (Export 시 사용, image_id -> LabelingResult)
labeling_results_cache: dict = {}


def get_config() -> Config:
    global config
    if config is None:
        config = Config.from_env()
    return config


def get_pipeline() -> LabelingPipeline:
    """파이프라인 반환 (한 번 로드 후 재사용)"""
    global pipeline
    if pipeline is None:
        pipeline = LabelingPipeline(get_config())
        logger.info("파이프라인 로드 완료 (캐시)")
    return pipeline


# 변환기 캐시 (Export 전용, 모델 로드 불필요)
_coco_converter: Optional[COCOConverter] = None
_yolo_converter: Optional[YOLOConverter] = None


def _run_labeling_and_overlay(
    image_path: str,
    prompt: str,
    confidence_threshold: float,
    image_id: str,
    output_overlay_path: Path,
) -> Tuple[dict, LabelingResult]:
    """라벨링 + 오버레이 생성 (블로킹, 스레드풀에서 실행)"""
    pipe = get_pipeline()
    result = pipe.process_image(image_path, prompt, confidence_threshold)
    result_dict = result.to_dict()

    boxes_xyxy = cxcywh_to_xyxy(
        result.boxes,
        result.image_width,
        result.image_height,
        normalized=True,
    )
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xyxy = boxes_xyxy.cpu().numpy()

    result_dict["boxes_percent"] = []
    if len(boxes_xyxy) > 0:
        for box in boxes_xyxy:
            result_dict["boxes_percent"].append({
                "x": float(box[0] / result.image_width * 100),
                "y": float(box[1] / result.image_height * 100),
                "width": float((box[2] - box[0]) / result.image_width * 100),
                "height": float((box[3] - box[1]) / result.image_height * 100),
            })

    if result.has_masks:
        image_bgr = cv2.imread(image_path)
        if image_bgr is not None:
            masks_np = result.masks.cpu().numpy()
            overlay = image_bgr.copy()
            palette = [
                (0, 99, 255), (255, 144, 30), (113, 179, 60),
                (0, 215, 255), (226, 43, 138), (204, 209, 72),
            ]
            for i in range(masks_np.shape[0]):
                mask_2d = masks_np[i, 0]
                mask_binary = (mask_2d > 0).astype(np.uint8)
                if mask_binary.shape != (image_bgr.shape[0], image_bgr.shape[1]):
                    mask_binary = cv2.resize(
                        mask_binary,
                        (image_bgr.shape[1], image_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if not np.any(mask_binary):
                    continue
                color = palette[i % len(palette)]
                color_layer = np.zeros_like(image_bgr, dtype=np.uint8)
                color_layer[mask_binary > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.4, 0)

            boxes_xyxy_np = cxcywh_to_xyxy(
                result.boxes, result.image_width, result.image_height, normalized=True
            )
            if isinstance(boxes_xyxy_np, torch.Tensor):
                boxes_xyxy_np = boxes_xyxy_np.cpu().numpy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.35, min(0.5, 450 / image_bgr.shape[0]))
            thickness = max(1, int(1.5 * font_scale))
            h_img, w_img = image_bgr.shape[:2]
            padding = 4

            for i, box in enumerate(boxes_xyxy_np):
                x1, y1, x2, y2 = map(int, box)
                color = palette[i % len(palette)]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                if i < len(result.labels):
                    label_text = f"{result.labels[i]}: {result.scores[i]:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    if y1 - th - padding - baseline >= 0:
                        label_y = y1 - padding - baseline
                    elif y2 + th + padding + baseline <= h_img:
                        label_y = y2 + th + padding
                    else:
                        label_y = y1 + th + padding
                    label_y = max(th + padding, min(int(label_y), h_img - baseline - padding - 2))
                    label_x = max(0, min(x1, w_img - tw - padding * 2 - 2))
                    r_y1 = max(0, label_y - th - padding)
                    r_y2 = min(h_img, label_y + baseline + padding)
                    r_x1 = max(0, int(label_x))
                    r_x2 = min(w_img, int(label_x) + tw + padding * 2)
                    cv2.rectangle(overlay, (r_x1, r_y1), (r_x2, r_y2), color, -1)
                    cv2.putText(
                        overlay, label_text, (label_x + padding, label_y),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
                    )
            output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_overlay_path), overlay)
            result_dict["overlay_image"] = f"/outputs/overlays/{output_overlay_path.name}"

    return result_dict, result


def get_converters() -> tuple:
    """Export용 변환기 (DINO/SAM 모델 로드 없이 경량 반환)"""
    global _coco_converter, _yolo_converter
    cfg = get_config()
    if _coco_converter is None:
        _coco_converter = COCOConverter(cfg.class_mapping.copy())
    if _yolo_converter is None:
        _yolo_converter = YOLOConverter(cfg.class_mapping.copy())
    return _coco_converter, _yolo_converter


def _coerce_boxes_xyxy_pixel(
    boxes: List[List[float]],
    image_width: int,
    image_height: int,
    boxes_format: Optional[str] = None,
) -> np.ndarray:
    if not boxes:
        return np.array([]).reshape(0, 4)

    boxes_np = np.array(boxes, dtype=float)
    if boxes_np.size == 0:
        return boxes_np.reshape(0, 4)

    if boxes_format == "xyxy_pixel":
        return boxes_np
    if boxes_format == "xyxy_normalized":
        return denormalize_boxes(boxes_np, image_width, image_height)
    if boxes_format == "cxcywh_normalized":
        return cxcywh_to_xyxy(boxes_np, image_width, image_height, normalized=True)

    max_val = float(np.max(boxes_np))
    min_val = float(np.min(boxes_np))
    if 0 <= min_val and max_val <= 1.5:
        if np.any(boxes_np[:, 2] < boxes_np[:, 0]) or np.any(boxes_np[:, 3] < boxes_np[:, 1]):
            return cxcywh_to_xyxy(boxes_np, image_width, image_height, normalized=True)
        return denormalize_boxes(boxes_np, image_width, image_height)

    return boxes_np


def _decode_rle_mask(rle: dict) -> np.ndarray:
    counts = rle.get("counts", [])
    size = rle.get("size", [])
    if len(size) != 2:
        return np.zeros((0, 0), dtype=np.uint8)

    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        if run > 0:
            flat[idx:idx + run] = val
        idx += run
        val = 1 - val

    return flat.reshape((height, width), order='F')


def _rebuild_masks(
    result_data: dict,
    image_width: int,
    image_height: int,
) -> Optional[List[np.ndarray]]:
    masks_rle = result_data.get("masks_rle", [])
    if masks_rle:
        return [_decode_rle_mask(rle).astype(np.uint8) for rle in masks_rle]

    masks_coords = result_data.get("masks_coords", [])
    if not masks_coords:
        return None

    mask_size = result_data.get("mask_size")
    height, width = image_height, image_width
    if mask_size and len(mask_size) == 2:
        height, width = mask_size

    masks = []
    for coords in masks_coords:
        mask = np.zeros((height, width), dtype=np.uint8)
        y_coords = np.array(coords.get("y", []), dtype=int)
        x_coords = np.array(coords.get("x", []), dtype=int)
        if y_coords.size > 0 and x_coords.size > 0:
            y_coords = np.clip(y_coords, 0, height - 1)
            x_coords = np.clip(x_coords, 0, width - 1)
            mask[y_coords, x_coords] = 1
        masks.append(mask)

    return masks


# Request/Response Models
class LabelRequest(BaseModel):
    image_id: str
    prompt: str
    confidence_threshold: Optional[float] = None


class SegmentPointRequest(BaseModel):
    image_id: str
    point_x: int
    point_y: int
    point_label: int = 1  # 1: foreground, 0: background


class ExportRequest(BaseModel):
    image_ids: List[str]
    format: str = "coco"  # "coco" or "yolo"


# 정적 파일 서빙
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 출력 폴더 서빙 (오버레이 이미지 등)
cfg = get_config()
output_dir = Path(cfg.output_dir)
if output_dir.exists():
    app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")


@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
    import torch
    import numpy as np
    import random
    
    # Random seed 고정 (완전한 재현성)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CUDA determinism 설정 (전역)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(f"CUDA deterministic mode enabled (seed={seed})")
    
    cfg = get_config()
    # 파이프라인 사전 로드 (첫 요청 대기 시간 단축)
    get_pipeline()
    logger.info("DINO + SAM 모델 사전 로드 완료")
    
    # 업로드 디렉토리 생성
    Path(cfg.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"서버 시작: http://{cfg.host}:{cfg.port}")
    logger.info(f"업로드 디렉토리: {cfg.upload_dir}")
    logger.info(f"출력 디렉토리: {cfg.output_dir}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>DINO-SAM Labeling Agent</h1><p>Static files not found.</p>")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드"""
    try:
        cfg = get_config()
        
        # 파일 확장자 검증
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(400, f"지원하지 않는 파일 형식: {file_ext}")
        
        # 고유 ID 생성
        image_id = str(uuid.uuid4())
        filename = f"{image_id}{file_ext}"
        file_path = Path(cfg.upload_dir) / filename
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"이미지 업로드: {filename}")
        
        return {
            "success": True,
            "image_id": image_id,
            "filename": filename,
            "path": str(file_path),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"업로드 실패: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/label")
async def label_image(request: LabelRequest):
    """이미지 라벨링 실행 (스레드풀에서 블로킹 연산 실행)"""
    try:
        cfg = get_config()
        upload_dir = Path(cfg.upload_dir)
        image_files = list(upload_dir.glob(f"{request.image_id}.*"))

        if not image_files:
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {request.image_id}")

        image_path = image_files[0]
        output_overlay_path = Path(cfg.output_dir) / "overlays" / f"{request.image_id}_overlay.png"

        # 라벨링 + 오버레이 생성을 스레드풀에서 실행 (이벤트 루프 블로킹 방지)
        result_dict, result = await asyncio.to_thread(
            _run_labeling_and_overlay,
            str(image_path),
            request.prompt,
            request.confidence_threshold,
            request.image_id,
            output_overlay_path,
        )

        labeling_results_cache[request.image_id] = result

        if output_overlay_path.exists():
            logger.info(f"오버레이 이미지 저장: {output_overlay_path}")

        return {
            "success": True,
            "image_id": request.image_id,
            "result": result_dict,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"라벨링 실패: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/segment-point")
async def segment_point(request: SegmentPointRequest):
    """클릭한 점에 대한 세그멘테이션 마스크 생성 (SAM point prompt)"""
    try:
        cfg = get_config()
        pipe = get_pipeline()
        
        # 이미지 경로 찾기
        upload_dir = Path(cfg.upload_dir)
        image_files = list(upload_dir.glob(f"{request.image_id}.*"))
        
        if not image_files:
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {request.image_id}")
        
        image_path = image_files[0]
        
        # SAM용 이미지 로드
        from agent.utils.image_loader import load_image_for_sam
        image_sam = load_image_for_sam(str(image_path))
        image_height, image_width = image_sam.shape[:2]
        
        logger.info(f"Point segmentation: 이미지={image_path}, 클릭=({request.point_x}, {request.point_y})")
        
        # SAM으로 point prompt 세그멘테이션
        from segment_anything import SamPredictor
        
        predictor = SamPredictor(pipe.sam.sam_model)
        predictor.set_image(image_sam)
        
        # Point coordinates (x, y) and labels (1=foreground, 0=background)
        point_coords = np.array([[request.point_x, request.point_y]], dtype=np.float32)
        point_labels = np.array([request.point_label], dtype=np.int32)
        
        # 마스크 예측
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,  # 3개의 마스크 후보 생성
        )
        
        logger.info(f"Point segmentation 완료: masks shape={masks.shape}, scores={scores}")
        
        # 가장 높은 점수의 마스크 선택
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])
        
        # 오버레이 이미지 생성
        image_bgr = cv2.imread(str(image_path))
        overlay = image_bgr.copy()
        
        # 마스크 색상 (초록색)
        color = (60, 179, 113)  # BGR: 초록색
        color_layer = np.zeros_like(image_bgr, dtype=np.uint8)
        color_layer[best_mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.5, 0)
        
        # 클릭한 점 표시
        cv2.circle(overlay, (request.point_x, request.point_y), 5, (0, 0, 255), -1)  # 빨간 점
        
        # 임시 폴더에 저장
        output_dir = Path(cfg.output_dir) / "point_segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        overlay_filename = f"{request.image_id}_point_{request.point_x}_{request.point_y}.png"
        overlay_image_path = output_dir / overlay_filename
        cv2.imwrite(str(overlay_image_path), overlay)
        
        logger.info(f"Point segmentation 오버레이 저장: {overlay_image_path}")
        
        return {
            "success": True,
            "image_id": request.image_id,
            "point": {"x": request.point_x, "y": request.point_y},
            "score": best_score,
            "mask_pixels": int(np.sum(best_mask)),
            "overlay_image": f"/outputs/point_segments/{overlay_filename}",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Point segmentation 실패: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ========================================
# Stub endpoints (프론트엔드 호환용 - 기능 비활성화됨)
# ========================================


@app.get("/api/stats")
async def get_stats():
    """피드백 통계 (HITL 비활성화로 빈 응답)"""
    return {
        "success": True,
        "stats": {
            "total": 0,
            "by_status": {"pending": 0, "approved": 0, "rejected": 0},
        },
    }


@app.get("/api/feedback/pending")
async def get_pending_reviews(limit: int = Query(default=50, le=100)):
    """리뷰 대기 목록 (HITL 비활성화로 빈 응답)"""
    return {"success": True, "count": 0, "items": []}


@app.post("/api/feedback")
async def submit_feedback():
    """피드백 제출 (HITL 비활성화)"""
    return JSONResponse(
        status_code=501,
        content={"detail": "Feedback (HITL) is currently disabled"},
    )


@app.post("/api/export")
async def export_labels(request: ExportRequest):
    """라벨 내보내기 (COCO 또는 YOLO 포맷)"""
    try:
        if not request.image_ids:
            raise HTTPException(400, "image_ids가 비어 있습니다")
        
        format_type = (request.format or "coco").lower()
        if format_type not in ("coco", "yolo"):
            raise HTTPException(400, f"지원하지 않는 포맷: {format_type}. (coco 또는 yolo)")
        
        # 캐시에서 결과 조회
        missing = [iid for iid in request.image_ids if iid not in labeling_results_cache]
        if missing:
            raise HTTPException(
                404,
                f"라벨링 결과를 찾을 수 없습니다. 먼저 Run Labeling을 실행하세요: {missing[:3]}{'...' if len(missing) > 3 else ''}",
            )
        
        cfg = get_config()
        coco_conv, yolo_conv = get_converters()  # 모델 로드 없이 변환기만 사용
        output_dir = Path(cfg.output_dir) / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        export_id = str(uuid.uuid4())[:8]
        
        if format_type == "coco":
            coco_datasets = []
            for i, image_id in enumerate(request.image_ids):
                result = labeling_results_cache[image_id]
                boxes_xyxy = cxcywh_to_xyxy(
                    result.boxes,
                    result.image_width,
                    result.image_height,
                    normalized=True,
                )
                boxes_np = boxes_xyxy.cpu().numpy() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy
                scores_np = result.scores.cpu().numpy() if isinstance(result.scores, torch.Tensor) else result.scores
                masks_list = None
                if result.has_masks:
                    masks_np = result.masks.cpu().numpy()
                    masks_list = [masks_np[j, 0] for j in range(masks_np.shape[0])]
                coco_data = coco_conv.convert(
                    boxes=boxes_np,
                    scores=scores_np,
                    labels=result.labels,
                    masks=masks_list,
                    image_id=i + 1,
                    image_width=result.image_width,
                    image_height=result.image_height,
                    image_filename=Path(result.image_path).name,
                    use_rle=False,
                )
                coco_datasets.append(coco_data)
            merged = coco_conv.merge(coco_datasets)
            output_path = output_dir / f"export_{export_id}.json"
            coco_conv.save(merged, str(output_path))
            logger.info(f"COCO 내보내기 완료: {output_path}")
            return {
                "success": True,
                "format": "coco",
                "output_path": str(output_path),
                "output_url": f"/outputs/exports/export_{export_id}.json",
                "count": len(request.image_ids),
            }
        
        else:  # yolo
            yolo_dir = output_dir / f"yolo_{export_id}"
            yolo_dir.mkdir(parents=True, exist_ok=True)
            for image_id in request.image_ids:
                result = labeling_results_cache[image_id]
                boxes_xyxy = cxcywh_to_xyxy(
                    result.boxes,
                    result.image_width,
                    result.image_height,
                    normalized=True,
                )
                boxes_np = boxes_xyxy.cpu().numpy() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy
                masks_list = None
                if result.has_masks:
                    masks_np = result.masks.cpu().numpy()
                    masks_list = [masks_np[j, 0] for j in range(masks_np.shape[0])]
                yolo_str = yolo_conv.convert(
                    boxes=boxes_np,
                    labels=result.labels,
                    masks=masks_list,
                    image_width=result.image_width,
                    image_height=result.image_height,
                )
                label_path = yolo_dir / f"{image_id}.txt"
                yolo_conv.save(yolo_str, str(label_path))
            classes_path = yolo_dir / "classes.txt"
            yolo_conv.save_classes(str(classes_path))
            logger.info(f"YOLO 내보내기 완료: {yolo_dir}")
            return {
                "success": True,
                "format": "yolo",
                "output_path": str(yolo_dir),
                "output_url": f"/outputs/exports/yolo_{export_id}/",
                "count": len(request.image_ids),
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"내보내기 실패: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """업로드된 이미지 조회"""
    try:
        cfg = get_config()
        upload_dir = Path(cfg.upload_dir)
        image_files = list(upload_dir.glob(f"{image_id}.*"))
        
        if not image_files:
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {image_id}")
        
        return FileResponse(str(image_files[0]))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 조회 실패: {e}")
        raise HTTPException(500, str(e))


def main():
    """서버 실행"""
    cfg = Config.from_env()
    uvicorn.run(
        "agent.app:app",
        host=cfg.host,
        port=cfg.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
