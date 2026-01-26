# 마스크 좌표 추출 문제 분석

## 문제 요약

SAM 마스크에서 좌표를 추출할 때, `sam.py`, `segment_anything lib`, `pipeline.py`에서 서로 다른 결과가 나옵니다.

## 결과 비교

### 1. sam.py 결과
```python
sam_masks_array = sam_masks.to('cpu').numpy()  # (5, 1, 479, 854)
sam_indices = sam_masks_array[4].nonzero()  # (1, 479, 854)에서 nonzero() 호출
# 결과: (array([0, 0, 0, ...]), array([0, 0, 0, ...]), array([7, 8, 9, ...]))
# shape: (183966,) - 3개의 배열 반환 (batch=0, height, width)
```

### 2. segment_anything lib 결과
```python
mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)  # (479, 854)
sam_lib_indices = mask_binary.nonzero()  # (479, 854)에서 nonzero() 호출
# 결과: (array([112, 112, ...]), array([302, 303, ...]))
# shape: (37979,) - 2개의 배열 반환 (height, width) ✅ 올바른 결과
```

### 3. pipeline.py 결과
```python
pipeline_masks_array = pipeline_masks.to('cpu').numpy()  # (5, 1, 479, 854)
pipeline_indices = pipeline_masks_array[4].nonzero()  # (1, 479, 854)에서 nonzero() 호출
# 결과: (array([0, 0, 0, ...]), array([0, 0, 0, ...]), array([0, 1, 2, ...]))
# shape: (117612,) - 3개의 배열 반환 (batch=0, height, width)
```

## 문제 원인

1. **Shape 차이**: 
   - `sam.py`와 `pipeline.py`: `(1, 479, 854)` 형태에서 `nonzero()` 호출 → 3개 배열 반환
   - `segment_anything lib`: `(479, 854)` 형태에서 `nonzero()` 호출 → 2개 배열 반환 ✅

2. **올바른 사용법**:
   - `(1, H, W)` 형태의 마스크에서 좌표를 추출하려면 먼저 `squeeze(0)` 또는 `[0]`으로 `(H, W)` 형태로 변환해야 합니다.

## 해결 방법

### 올바른 좌표 추출 방법

```python
# ❌ 잘못된 방법
masks_array = masks.to('cpu').numpy()  # (5, 1, 479, 854)
indices = masks_array[4].nonzero()  # (1, 479, 854) → 3개 배열 반환

# ✅ 올바른 방법 1: squeeze 사용
masks_array = masks.to('cpu').numpy()  # (5, 1, 479, 854)
mask_2d = masks_array[4, 0]  # 또는 masks_array[4].squeeze(0) → (479, 854)
indices = mask_2d.nonzero()  # (479, 854) → 2개 배열 반환 (height, width)

# ✅ 올바른 방법 2: numpy where 사용
masks_array = masks.to('cpu').numpy()  # (5, 1, 479, 854)
mask_2d = masks_array[4, 0]  # (479, 854)
y_coords, x_coords = np.where(mask_2d > 0)  # (height, width) 좌표 반환
```

### segment_anything lib의 올바른 사용법

```python
# segment_anything lib에서 올바르게 사용한 예시
masks, scores, logits = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=True,  # 또는 False
)
# masks shape: (5, 3, 479, 854) 또는 (5, 1, 479, 854)

# 각 마스크에 대해
for mask_batch in masks:
    # 3개의 채널 중 첫 번째 채널만 선택 (3, 479, 854) -> (479, 854)
    mask = mask_batch[0].cpu().numpy()  # ✅ 올바른 방법
    
    # 0보다 큰 값을 1로 이진화
    mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    
    # 좌표 추출
    coords = mask_binary.nonzero()  # ✅ (height, width) 좌표 반환
```

## 수정 사항

1. `sam.py`에 주석 추가: 좌표 추출 시 올바른 차원 사용 방법 명시
2. 사용자가 `masks[i, 0]` 또는 `masks[i].squeeze(0)`을 사용하여 `(H, W)` 형태로 변환한 후 좌표 추출해야 함을 명시

## 참고

- `predict_torch`는 `multimask_output=False`일 때 `(N, 1, H, W)` 형태를 반환합니다.
- `multimask_output=True`일 때 `(N, 3, H, W)` 형태를 반환합니다.
- 좌표 추출 시 항상 `(H, W)` 형태의 2D 배열에서 `nonzero()` 또는 `np.where()`를 호출해야 합니다.
