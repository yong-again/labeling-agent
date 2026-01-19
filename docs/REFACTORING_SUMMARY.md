# Model Output 리팩토링 완료

## 개요
DINO 모델의 좌표 변환 로직을 분리하여 파이프라인에서 명시적으로 처리하도록 리팩토링했습니다.

## 변경 사항

### 1. 새로운 파일: `agent/utils/box_transforms.py`
바운딩 박스 좌표 변환을 위한 유틸리티 모듈을 생성했습니다.

**주요 함수:**
- `cxcywh_to_xyxy()`: [cx, cy, w, h] → [x1, y1, x2, y2] 변환
- `xyxy_to_cxcywh()`: [x1, y1, x2, y2] → [cx, cy, w, h] 변환
- `normalize_boxes()`: 픽셀 좌표 → 정규화 좌표 (0-1)
- `denormalize_boxes()`: 정규화 좌표 → 픽셀 좌표

**특징:**
- PyTorch Tensor와 NumPy array 모두 지원
- 이미지 경계를 벗어나는 좌표 자동 클리핑
- 정규화/비정규화 옵션 제공

### 2. `agent/models/dino.py` 수정

**변경 전:**
```python
def predict(...) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # DINO 예측
    boxes, logits, phrases = predict(...)
    
    # 내부에서 좌표 변환 수행
    # [cx, cy, w, h] (정규화) -> [x1, y1, x2, y2] (픽셀)
    boxes_xyxy = convert_coordinates(...)
    
    return boxes_xyxy, scores, labels  # 픽셀 좌표 반환
```

**변경 후:**
```python
def predict(...) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    # DINO 예측
    boxes, logits, phrases = predict(...)
    
    # 변환 없이 원본 출력 그대로 반환
    return boxes, scores, labels  # [cx, cy, w, h] (정규화) 반환
```

**주요 변경:**
- 반환 타입: `np.ndarray` → `torch.Tensor` (boxes)
- 반환 형식: `[x1, y1, x2, y2] (픽셀)` → `[cx, cy, w, h] (정규화 0-1)`
- 좌표 변환 로직 제거 (약 20줄 제거)
- `predict_batch()` 메서드도 동일하게 수정

### 3. `agent/pipeline.py` 수정

**변경 전:**
```python
def process_image(...):
    # DINO 검출
    boxes, scores, labels = self.dino.predict(...)  # [x1, y1, x2, y2] 픽셀 좌표
    
    # SAM에 바로 전달
    masks = self.sam.predict_from_boxes(image_source, boxes)
```

**변경 후:**
```python
def process_image(...):
    # DINO 검출
    boxes_cxcywh, scores, labels = self.dino.predict(...)  # [cx, cy, w, h] 정규화 좌표
    
    # 좌표 변환: [cx, cy, w, h] (정규화) -> [x1, y1, x2, y2] (픽셀)
    boxes_xyxy = cxcywh_to_xyxy(
        boxes_cxcywh,
        image_width=image_width,
        image_height=image_height,
        normalized=True,
    )
    
    # torch.Tensor -> numpy array 변환
    boxes_xyxy_np = boxes_xyxy.cpu().numpy()
    
    # SAM에 전달
    masks = self.sam.predict_from_boxes(image_source, boxes_xyxy_np)
```

**주요 변경:**
- `box_transforms` 모듈 import 추가
- DINO와 SAM 사이에 명시적인 좌표 변환 단계 추가
- 파이프라인 흐름: **input → dino → 좌표변환 → sam → output**

## 테스트

`test_transforms.py` 스크립트로 좌표 변환 검증 완료:

```
입력: [[0.5, 0.5, 0.4, 0.3]]  (정규화된 중앙 박스)
출력: [[300, 280, 700, 520]]  (1000x800 이미지에서 픽셀 좌표)
역변환: [[0.5, 0.5, 0.4, 0.3]]  (완벽하게 복원)
최대 차이: 0.0
```

## 장점

1. **관심사의 분리 (Separation of Concerns)**
   - DINO는 순수하게 객체 검출만 담당
   - 좌표 변환은 별도 유틸리티로 분리
   - 파이프라인에서 명시적으로 변환 수행

2. **재사용성**
   - `box_transforms` 유틸리티는 다른 모델/작업에서도 사용 가능
   - DINO 출력을 다른 용도로 사용할 때도 유연하게 대응

3. **명확성**
   - 파이프라인 코드만 봐도 데이터 흐름 이해 가능
   - 각 단계에서 어떤 좌표 형식을 사용하는지 명확

4. **유지보수성**
   - 좌표 변환 로직이 한 곳에 집중
   - 버그 수정이나 기능 개선 시 한 곳만 수정

## 파일 구조

```
labeling-agent/
├── agent/
│   ├── models/
│   │   ├── dino.py          (수정: 좌표 변환 로직 제거)
│   │   └── sam.py           (변경 없음)
│   ├── utils/
│   │   └── box_transforms.py (신규: 좌표 변환 유틸리티)
│   └── pipeline.py          (수정: 명시적 좌표 변환 추가)
└── test_transforms.py       (신규: 테스트 스크립트)
```

## 추후 개선 가능 사항

1. `box_transforms.py`에 더 많은 좌표 형식 지원 (e.g., 회전된 박스)
2. 배치 처리 최적화
3. GPU 텐서 지원 강화
