# 수정사항 요약

## 1. GroundingDINO CUDA 빌드 오류 수정

**파일:** `GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu`, `ms_deform_attn.h`

**원인:** PyTorch C++ API 변경 (DeprecatedTypeProperties → ScalarType)

**변경:**
- `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` → `AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), ...)`
- `tensor.type().is_cuda()` → `tensor.is_cuda()`

---

## 2. uv로 GroundingDINO 빌드

**원인:** uv 빌드 환경에 pip/torch가 없어 빌드 실패

**해결:**
```bash
uv pip install torch torchvision
uv pip install --no-build-isolation -e .
```

---

## 3. BertModelWarper - transformers 5.0 호환

**파일:** `GroundingDINO/groundingdino/models/GroundingDINO/bertwarper.py`

### 3-1. get_head_mask 제거 대응

**원인:** transformers 5.0에서 `BertModel.get_head_mask` 메서드 제거

**변경:**
- `bert_model.get_head_mask`가 있으면 사용, 없으면 `_get_head_mask`, `_convert_head_mask_to_5d` 자체 구현 사용

### 3-2. get_extended_attention_mask / invert_attention_mask 대체

**원인:** transformers 5.0에서 시그니처 변경 → `TypeError: to() received (dtype=torch.device)` 발생

**변경:**
- `_get_extended_attention_mask`, `_invert_attention_mask` 로컬 구현으로 완전 대체

---

## 4. 웹/로컬 세그멘트 마스크 결과 불일치 해결

**파일:** `agent/config.py`

**원인:** `Config.from_env()`에서 `SAM_CHECKPOINT_PATH` 미설정 시 `sam_checkpoint_path=None` → SAM이 랜덤 초기화 상태로 동작 → 전체 이미지 마스크

**변경:**
```python
sam_checkpoint_path=sam_checkpoint or cls.sam_checkpoint_path
```
- 환경변수가 없으면 기본값 사용

---

## 5. 기존 repo 대비 추가 변경사항

### 5-1. Config

| 항목 | 이전 | 변경 후 |
|------|------|---------|
| SAM 모델 | `vit_l` | `vit_h` |
| SAM 체크포인트 | `sam_vit_l_0b3195.pth` | `sam_vit_h_4b8939.pth` |

### 5-2. Image Loader (`agent/utils/image_loader.py`)

- **DINO/SAM 이미지 로드 분리**
  - `load_image()`: DINO용 → `(image_np, image_transformed)` 반환 (groundingdino transforms 사용)
  - `load_image_for_sam()`: 경로 → RGB numpy 직접 반환 (파일 경로 인자로 변경)
- `get_image_size()` 반환 순서: `(w, h)` → `(h, w)`
- DINO 전처리: groundingdino transforms 사용 (기존 `dino_load_image` 제거)

### 5-3. SAM 래퍼 (`agent/models/sam.py`)

- `predict_from_boxes()` 시그니처: `(image, boxes, image_width, image_height)` → `(image, boxes)` (width/height 제거)
- **상태 오염 방지**: 호출 시마다 SAM 모델·predictor 재생성
- `sam.eval()`, `self.sam_model` 저장 추가
- 디버그 로그 추가 (transformed_boxes, masks shape, first_pixel 등)

### 5-4. Pipeline (`agent/pipeline.py`)

- DINO/SAM 이미지 로드 분리 (`load_image`, `load_image_for_sam`)
- `LabelingResult.to_dict()`: `boxes_format: xyxy_pixel`, `masks_polygon`, `mask_size`, `masks_format: polygon` 추가
- `_mask_to_polygon()` 유틸 추가
- `get_mask_cordinates()` 호출로 마스크 디버그
- 로그 레벨: `debug` → `info`

### 5-5. DINO 래퍼 (`agent/models/dino.py`)

- import: `torchvision.transforms` → `groundingdino.datasets.transforms`
- 로그 레벨: `debug` → `info`
- 불필요한 `print(label)` 추가 (제거 권장)

### 5-6. FastAPI 앱 (`agent/app.py`)

- **Pipeline 캐시 제거**: 요청마다 새 pipeline 생성 (상태 오염 방지)
- **Feedback/HITL 제거**: `FeedbackManager`, `/api/feedback/*`, `/api/stats`, `/api/export` 삭제
- **Point segmentation 추가**: `/api/segment-point` (클릭 좌표로 SAM 마스크)
- **라벨 API 개선**: 서버 측 오버레이 이미지 생성, `boxes_percent`, 마스크 폴리곤 반환
- `_coerce_boxes_xyxy_pixel`, `_decode_rle_mask`, `_rebuild_masks` 유틸 추가
- `/outputs` 정적 마운트 (오버레이 이미지 서빙)
- startup: random seed, CUDA determinism 설정
- **버그**: `main()` 중복 호출 (제거 필요)

### 5-7. Data Collector (`agent/training/data_collector.py`)

- `_coerce_boxes_xyxy_pixel()` 추가: `boxes_format`에 따라 xyxy 픽셀 좌표로 정규화
- COCO/YOLO 저장 시 `boxes_format` 지원 (xyxy_pixel, xyxy_normalized, cxcywh_normalized)
- YOLO 변환: 정규화 좌표 가정 → 픽셀 좌표 기반 정규화로 수정

### 5-8. 프론트엔드 (`agent/static/app.js`)

- **Point segmentation 모드**: `pointSegmentMode`, canvas 클릭 핸들러
- **오버레이 우선**: 서버 생성 오버레이 이미지가 있으면 클라이언트 렌더 스킵
- 마스크 렌더: `masks_base64` → `masks_raw` 또는 서버 오버레이
- `DEBUG_MASK_RENDER`, `ctx.imageSmoothingEnabled = false` 추가

### 5-9. 추가/삭제 파일

| 구분 | 파일 |
|------|------|
| 추가 | `agent/utils/util.py` (get_mask_cordinates) |
| 추가 | `debug/` (check_outputs, test_*.py 등) |
| 추가 | `docs/MODIFICATIONS_SUMMARY.md` |
| 삭제 | `MASK_COORDINATE_ISSUE.md` |
| 삭제 | `temp.py`, `test_mask_fix.py`, `test_transforms.py` |

### 5-10. .gitignore

- `.vscode/`, `*.pyc` 등 Python 산출물
- `/GroundingDINO/` (GroundingDINO 폴더 무시)

### 5-11. 기타

- `feedback.db` 크기 증가 (수정 시 주의)

---

## 요약표

| 구분 | 파일 | 내용 |
|------|------|------|
| CUDA | `ms_deform_attn_cuda.cu`, `ms_deform_attn.h` | PyTorch C++ API 호환 |
| BERT | `bertwarper.py` | transformers 5.0 호환 (get_head_mask, attention mask) |
| 설정 | `agent/config.py` | SAM vit_h, 체크포인트 fallback |
| 이미지 | `image_loader.py` | DINO/SAM 분리, RGB 처리 |
| SAM | `sam.py` | 상태 오염 방지, 시그니처 변경 |
| 파이프라인 | `pipeline.py` | to_dict 확장, 마스크 폴리곤 |
| API | `app.py` | Feedback 제거, Point segmentation, 오버레이 |
| 학습 | `data_collector.py` | boxes_format 지원 |
| 프론트 | `app.js` | 오버레이, point segment 모드 |
