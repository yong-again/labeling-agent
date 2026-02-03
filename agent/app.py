"""
FastAPI 웹 서버
DINO-SAM 라벨링 Web UI
"""

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

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
# Note: pipeline is NOT cached - we create a fresh one per request to avoid state pollution


def get_config() -> Config:
    global config
    if config is None:
        config = Config.from_env()
    return config


def get_pipeline() -> LabelingPipeline:
    # CRITICAL: Create fresh pipeline for each request to avoid state pollution
    # The DINO model maintains internal state that gets corrupted on reuse
    return LabelingPipeline(get_config())


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
    """이미지 라벨링 실행"""
    try:
        cfg = get_config()
        pipe = get_pipeline()
        
        # 이미지 경로 찾기
        upload_dir = Path(cfg.upload_dir)
        image_files = list(upload_dir.glob(f"{request.image_id}.*"))
        
        if not image_files:
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {request.image_id}")
        
        image_path = image_files[0]
        
        # 라벨링 실행
        result = pipe.process_image(
            str(image_path),
            request.prompt,
            request.confidence_threshold,
        )

        # 결과를 딕셔너리로 변환
        result_dict = result.to_dict()
        
        # 시각화용 데이터 추가
        import numpy as np
        
        # 박스를 픽셀 좌표 xyxy 형식으로 변환
        boxes_xyxy = cxcywh_to_xyxy(
            result.boxes,
            result.image_width,
            result.image_height,
            normalized=True
        )
        
        # 박스를 퍼센트로 변환 (웹 표시용)
        result_dict["boxes_percent"] = []
        if len(boxes_xyxy) > 0:
            for box in boxes_xyxy:
                result_dict["boxes_percent"].append({
                    "x": float(box[0] / result.image_width * 100),
                    "y": float(box[1] / result.image_height * 100),
                    "width": float((box[2] - box[0]) / result.image_width * 100),
                    "height": float((box[3] - box[1]) / result.image_height * 100),
                })
        
        # 서버 측에서 마스크 오버레이 이미지 생성
        overlay_image_path = None
        if result.has_masks:
            # 원본 이미지 읽기
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise HTTPException(500, f"이미지를 읽을 수 없습니다: {image_path}")
            
            # 마스크 오버레이
            masks_np = result.masks.cpu().numpy()  # (N, 1, H, W)
            overlay = image_bgr.copy()
            
            # 색상 팔레트
            palette = [
                (0, 99, 255),      # 주황색
                (255, 144, 30),    # 파란색
                (113, 179, 60),    # 녹색
                (0, 215, 255),     # 노란색
                (226, 43, 138),    # 자주색
                (204, 209, 72),    # 청록색
            ]
            
            for i in range(masks_np.shape[0]):
                mask_2d = masks_np[i, 0]  # (H, W)
                mask_binary = (mask_2d > 0).astype(np.uint8)
                
                # 마스크 크기가 이미지 크기와 다르면 리사이즈
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
            
            # 박스 그리기
            boxes_xyxy = cxcywh_to_xyxy(
                result.boxes,
                result.image_width,
                result.image_height,
                normalized=True
            )
            if isinstance(boxes_xyxy, torch.Tensor):
                boxes_xyxy = boxes_xyxy.cpu().numpy()
            
            for i, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = map(int, box)
                color = palette[i % len(palette)]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 표시
                if i < len(result.labels):
                    label = f"{result.labels[i]}: {result.scores[i]:.2f}"
                    cv2.putText(overlay, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 임시 폴더에 저장
            output_dir = Path(cfg.output_dir) / "overlays"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            overlay_filename = f"{request.image_id}_overlay.png"
            overlay_image_path = output_dir / overlay_filename
            cv2.imwrite(str(overlay_image_path), overlay)
            
            logger.info(f"오버레이 이미지 저장: {overlay_image_path}")
            
            # 상대 경로로 변환 (웹에서 접근 가능하도록)
            result_dict["overlay_image"] = f"/outputs/overlays/{overlay_filename}"
        
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


# Export 기능은 추후 구현 예정


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
    main()
