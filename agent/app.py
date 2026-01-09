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

from agent.config import Config
from agent.pipeline import LabelingPipeline, LabelingResult
from agent.feedback import FeedbackManager, FeedbackStatus

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
feedback_manager: Optional[FeedbackManager] = None


def get_config() -> Config:
    global config
    if config is None:
        config = Config.from_env()
    return config


def get_pipeline() -> LabelingPipeline:
    global pipeline
    if pipeline is None:
        pipeline = LabelingPipeline(get_config())
    return pipeline


def get_feedback_manager() -> FeedbackManager:
    global feedback_manager
    if feedback_manager is None:
        feedback_manager = FeedbackManager(get_config().feedback_db_path)
    return feedback_manager


# Request/Response Models
class LabelRequest(BaseModel):
    image_id: str
    prompt: str
    confidence_threshold: Optional[float] = None


class FeedbackRequest(BaseModel):
    image_id: str
    status: str  # "approved", "rejected", "corrected"
    corrections: Optional[dict] = None
    notes: Optional[str] = None


class ExportRequest(BaseModel):
    image_ids: List[str]
    format: str = "coco"  # "coco" or "yolo"


# 정적 파일 서빙
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
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
        fm = get_feedback_manager()
        
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
        
        # 피드백 DB에 저장 (PENDING 상태)
        if cfg.hitl_enabled:
            fm.save_feedback(
                image_path=str(image_path),
                image_id=request.image_id,
                prompt=request.prompt,
                labeling_result=result_dict,
                status=FeedbackStatus.PENDING,
            )
        
        # 시각화용 데이터 추가 (픽셀 좌표를 퍼센트로 변환)
        result_dict["boxes_percent"] = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                result_dict["boxes_percent"].append({
                    "x": float(box[0] / result.image_width * 100),
                    "y": float(box[1] / result.image_height * 100),
                    "width": float((box[2] - box[0]) / result.image_width * 100),
                    "height": float((box[3] - box[1]) / result.image_height * 100),
                })
        
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


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """HITL 피드백 제출"""
    try:
        fm = get_feedback_manager()
        
        # 상태 매핑
        status_map = {
            "approved": FeedbackStatus.APPROVED,
            "rejected": FeedbackStatus.REJECTED,
            "corrected": FeedbackStatus.CORRECTED,
        }
        
        if request.status not in status_map:
            raise HTTPException(400, f"잘못된 상태: {request.status}")
        
        status = status_map[request.status]
        
        # 피드백 업데이트
        success = fm.update_status(
            image_id=request.image_id,
            status=status,
            corrections=request.corrections,
            notes=request.notes,
        )
        
        if not success:
            raise HTTPException(404, f"피드백을 찾을 수 없습니다: {request.image_id}")
        
        return {
            "success": True,
            "image_id": request.image_id,
            "status": request.status,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"피드백 제출 실패: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/feedback/pending")
async def get_pending_reviews(limit: int = Query(default=50, le=100)):
    """리뷰 대기 목록"""
    try:
        fm = get_feedback_manager()
        items = fm.get_pending_reviews(limit=limit)
        
        return {
            "success": True,
            "count": len(items),
            "items": [item.to_dict() for item in items],
        }
        
    except Exception as e:
        logger.error(f"pending 조회 실패: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/stats")
async def get_stats():
    """피드백 통계"""
    try:
        fm = get_feedback_manager()
        stats = fm.get_stats()
        
        return {
            "success": True,
            "stats": stats,
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/export")
async def export_labels(request: ExportRequest):
    """라벨 내보내기"""
    try:
        cfg = get_config()
        pipe = get_pipeline()
        fm = get_feedback_manager()
        
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_id = str(uuid.uuid4())[:8]
        
        if request.format.lower() == "coco":
            # COCO format으로 내보내기
            coco_datasets = []
            
            for i, image_id in enumerate(request.image_ids):
                feedback = fm.get_feedback(image_id)
                if not feedback:
                    continue
                
                # 수정된 데이터가 있으면 사용
                result_data = feedback.corrections or feedback.labeling_result
                
                # result_data에서 boxes, labels, scores 추출
                boxes = result_data.get("boxes", [])
                labels = result_data.get("labels", [])
                scores = result_data.get("scores", [])
                image_width = result_data.get("image_width", 1920)
                image_height = result_data.get("image_height", 1080)
                
                # numpy array로 변환
                import numpy as np
                boxes_np = np.array(boxes) if boxes else np.array([]).reshape(0, 4)
                scores_np = np.array(scores) if scores else np.array([])
                
                # COCO converter를 사용하여 annotation 생성
                coco_data = pipe.coco_converter.convert(
                    boxes=boxes_np,
                    scores=scores_np,
                    labels=labels,
                    masks=None,  # 마스크는 피드백에 저장되지 않음
                    image_id=i + 1,
                    image_width=image_width,
                    image_height=image_height,
                    image_filename=Path(feedback.image_path).name,
                )
                coco_datasets.append(coco_data)
            
            # 파일 저장
            output_path = output_dir / f"export_{export_id}.json"
            merged = pipe.coco_converter.merge(coco_datasets)
            pipe.coco_converter.save(merged, str(output_path))
            
            return {
                "success": True,
                "format": "coco",
                "output_path": str(output_path),
                "count": len(request.image_ids),
            }
            
        elif request.format.lower() == "yolo":
            # YOLO format으로 내보내기
            yolo_dir = output_dir / f"yolo_{export_id}"
            yolo_dir.mkdir(exist_ok=True)
            
            import numpy as np
            
            for image_id in request.image_ids:
                feedback = fm.get_feedback(image_id)
                if not feedback:
                    continue
                
                result_data = feedback.corrections or feedback.labeling_result
                
                # result_data에서 boxes, labels 추출
                boxes = result_data.get("boxes", [])
                labels = result_data.get("labels", [])
                image_width = result_data.get("image_width", 1920)
                image_height = result_data.get("image_height", 1080)
                
                # numpy array로 변환
                boxes_np = np.array(boxes) if boxes else np.array([]).reshape(0, 4)
                
                # YOLO annotation 생성 및 저장
                yolo_data = pipe.yolo_converter.convert(
                    boxes=boxes_np,
                    labels=labels,
                    masks=None,
                    image_width=image_width,
                    image_height=image_height,
                )
                
                label_path = yolo_dir / f"{image_id}.txt"
                with open(label_path, 'w') as f:
                    f.write(yolo_data)
            
            # classes.txt 저장
            pipe.yolo_converter.save_classes(str(yolo_dir / "classes.txt"))
            
            return {
                "success": True,
                "format": "yolo",
                "output_path": str(yolo_dir),
                "count": len(request.image_ids),
            }
        
        else:
            raise HTTPException(400, f"지원하지 않는 포맷: {request.format}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"내보내기 실패: {e}")
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
