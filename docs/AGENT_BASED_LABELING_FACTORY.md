# Agent-based Labeling Factory Architecture
## Label Studio + DINO + SAM 기반 최적화 설계

---

## 1부: 문제 정의

### 1.1 핵심 도전과제

**3-Way Trade-off 해결**
```
┌─────────────────────────────────────────────────────────┐
│  Label Quality ↑  /  Annotation Cost ↓  /  TAT ↓       │
│  ─────────────────────────────────────────────────────  │
│  기존: 2개만 최적화 → 3개 동시 최적화 필요                │
└─────────────────────────────────────────────────────────┘
```

**현실적 제약사항**
- **비용**: 라벨러 시간 = $X/시간 × N시간 → 자동화율 ↑ 필요
- **품질**: Inter-annotator agreement < 0.85 → 검증/재작업 비용 ↑
- **시간**: 배치 처리 지연 → 실시간 피드백 루프 단절

### 1.2 기존 파이프라인의 한계

| 문제 영역 | 기존 방식 | 한계점 |
|---------|---------|--------|
| **Pre-labeling** | 단순 SAM 적용 | Confidence threshold 고정 → 오분류 누적 |
| **Quality Control** | 랜덤 샘플링 | High-risk 샘플 놓침 → Rework Rate ↑ |
| **Cost Control** | 수동 할당 | 라벨러 역량 미반영 → 비효율 |
| **Model Update** | 주기적 재학습 | Drift 감지 지연 → 성능 저하 누적 |

### 1.3 목표 지표 (Target Metrics)

```
┌─────────────────────────────────────────────────────────┐
│  Primary KPIs                                            │
│  ─────────────────────────────────────────────────────  │
│  • Label Accuracy Proxy:     ≥ 0.92 (IoU-based)         │
│  • Auto-Label Acceptance:    ≥ 60% (Level 2+)           │
│  • Rework Rate:              ≤ 8%                        │
│  • Mean Time To Label:       ≤ 2.5 min/image            │
│  • Cost per Label:           ↓ 40% vs baseline          │
└─────────────────────────────────────────────────────────┘
```

---

## 2부: 에이전트 구조 설계

### 2.1 4-Agent 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent-based Labeling Factory                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Orchestration │    │ Quality Gate  │    │  Cost Control │
│    Agent      │◄───┤     Agent     │◄───┤     Agent     │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │ Drift Detection│
                    │     Agent     │
                    └───────────────┘
```

### 2.2 Orchestration Agent

**역할**: 전체 파이프라인 워크플로우 제어 및 라우팅

**책임**
- 입력 이미지 → 자동화 레벨 결정 (Level 1/2/3)
- DINO + SAM 파이프라인 실행 순서 최적화
- Label Studio 작업 큐 관리
- 에이전트 간 메시지 브로커 역할

**핵심 로직**
```python
class OrchestrationAgent:
    def route_image(self, image, metadata):
        # 1. Drift Detection Agent에게 체크 요청
        drift_score = self.drift_agent.check(image)
        
        # 2. 자동화 레벨 결정
        if drift_score > 0.3:
            return Level.THREE  # Active Learning
        elif self.quality_agent.predict_confidence(image) > 0.85:
            return Level.TWO   # Auto-accept
        else:
            return Level.ONE   # Pre-label only
        
        # 3. Cost Control Agent에게 리소스 할당 요청
        reviewer = self.cost_agent.assign_reviewer(image)
```

**의사결정 테이블**
| 조건 | 레벨 | 액션 |
|-----|------|------|
| Confidence > 0.85 & Drift < 0.2 | Level 2 | Auto-accept → QA 샘플링만 |
| Confidence 0.7-0.85 | Level 1 | Pre-label → Reviewer 검토 |
| Confidence < 0.7 or Drift > 0.3 | Level 3 | Active Learning → Expert 라벨링 |

### 2.3 Quality Gate Agent

**역할**: 라벨 품질 검증 및 리스크 평가

**책임**
- DINO + SAM 출력 신뢰도 평가
- Box → Mask error propagation 예측
- Ambiguous case 자동 분기
- Inter-annotator agreement 예측

**핵심 메트릭**
```python
class QualityGateAgent:
    def evaluate_label_quality(self, image, dino_output, sam_output):
        metrics = {
            'dino_confidence': self._calc_dino_confidence(dino_output),
            'sam_iou_stability': self._calc_sam_stability(sam_output),
            'box_mask_consistency': self._calc_consistency(dino_output, sam_output),
            'edge_case_score': self._detect_ambiguous(image)
        }
        
        # Composite Quality Score
        quality_score = (
            0.3 * metrics['dino_confidence'] +
            0.3 * metrics['sam_iou_stability'] +
            0.2 * metrics['box_mask_consistency'] +
            0.2 * (1 - metrics['edge_case_score'])
        )
        
        return quality_score, metrics
```

**Threshold 설계**
| Quality Score | 액션 | Reviewer 할당 |
|--------------|------|--------------|
| ≥ 0.90 | Auto-accept | None (QA 샘플링만) |
| 0.75 - 0.90 | Standard Review | Junior Reviewer |
| 0.60 - 0.75 | Expert Review | Senior Reviewer |
| < 0.60 | Double-blind | 2x Senior Reviewer |

**Box → Mask Error Propagation 최소화**
```
DINO Box (x, y, w, h) → SAM Prompt
    │
    ├─ [검증 1] Box 크기 검증
    │   └─ 너무 작은 box (< 32px) → 확장 또는 제외
    │
    ├─ [검증 2] Box-Mask IoU 검증
    │   └─ IoU < 0.7 → SAM 재실행 또는 수동 검토
    │
    └─ [검증 3] Mask 경계 품질 검증
        └─ Edge smoothness < threshold → 후처리 또는 재생성
```

### 2.4 Cost Control Agent

**역할**: 라벨링 비용 최적화 및 리소스 할당

**책임**
- Reviewer 역량 기반 작업 할당
- 우선순위 큐 관리 (High-value 샘플 우선)
- 병렬 처리 최적화
- 비용 예측 및 예산 모니터링

**리소스 할당 전략**
```python
class CostControlAgent:
    def assign_reviewer(self, image, quality_score, priority):
        # Reviewer 역량 매트릭스
        reviewers = self._get_available_reviewers()
        
        # 작업 복잡도 기반 매칭
        complexity = self._estimate_complexity(image, quality_score)
        
        # 최적 매칭: 역량 × 가용성 × 비용
        best_match = max(
            reviewers,
            key=lambda r: (
                r.expertise_score[complexity] *
                r.availability *
                (1 / r.hourly_rate)  # 비용 역가중
            )
        )
        
        return best_match
```

**Load Balancing 전략**
- **Round-robin**: 기본 할당
- **Skill-based**: 클래스별 전문가 매칭
- **Dynamic**: 실시간 작업량 모니터링 → 재분배

**비용 최적화 포인트**
| 전략 | 절감율 | 적용 조건 |
|------|--------|----------|
| Auto-accept (Level 2) | 60% | High confidence |
| Batch Processing | 20% | 동일 클래스 그룹핑 |
| Smart Sampling | 30% | High-risk만 검토 |

### 2.5 Drift Detection Agent

**역할**: 데이터/모델 드리프트 감지 및 대응

**책임**
- 입력 데이터 분포 변화 감지
- 모델 성능 저하 조기 감지
- 신규 클래스/패턴 자동 탐지
- 재학습 트리거 결정

**드리프트 감지 메커니즘**
```python
class DriftDetectionAgent:
    def detect_drift(self, new_batch, reference_distribution):
        # 1. Feature Space Drift (DINO embeddings)
        feature_drift = self._kl_divergence(
            new_batch.dino_features,
            reference_distribution.dino_features
        )
        
        # 2. Label Distribution Drift
        label_drift = self._chi_square_test(
            new_batch.label_distribution,
            reference_distribution.label_distribution
        )
        
        # 3. Performance Drift (Confidence drop)
        perf_drift = self._confidence_trend_analysis(new_batch)
        
        # Composite Drift Score
        drift_score = (
            0.4 * feature_drift +
            0.3 * label_drift +
            0.3 * perf_drift
        )
        
        return drift_score
```

**드리프트 대응 전략**
| Drift Score | 액션 | 자동화 레벨 |
|------------|------|------------|
| < 0.2 | Normal | Level 1/2 유지 |
| 0.2 - 0.4 | Caution | Level 1로 강등 |
| 0.4 - 0.6 | Warning | Level 3 (Active Learning) |
| > 0.6 | Critical | 재학습 트리거 + 수동 라벨링 |

---

## 3부: 파이프라인 흐름

### 3.1 3단계 자동화 레벨

#### Level 1: Pre-label Only
```
Input Image
    │
    ├─ DINO: Object Detection → Boxes
    │   └─ Confidence Filter (threshold: 0.6)
    │
    ├─ SAM: Box → Mask Conversion
    │   └─ Quality Check (IoU stability > 0.8)
    │
    ├─ Quality Gate: Quality Score 계산
    │   └─ Score < 0.75 → Flag for Review
    │
    └─ Label Studio: Pre-label 전송
        └─ Reviewer 검토 → 수정/승인
```

**특징**
- 모든 라벨은 Reviewer 검토 필수
- Auto-accept 없음
- **목표**: Rework Rate < 10%

#### Level 2: Confidence-based Auto-accept
```
Input Image
    │
    ├─ DINO + SAM (동일)
    │
    ├─ Quality Gate: Quality Score ≥ 0.90
    │   └─ Auto-accept 결정
    │
    ├─ QA Sampling: 20% 랜덤 + High-risk 샘플
    │   └─ 통과 시 최종 승인
    │
    └─ Label Studio: 승인된 라벨만 저장
```

**특징**
- High confidence만 자동 승인
- QA 샘플링으로 검증
- **목표**: Auto-accept Rate ≥ 60%, Accuracy ≥ 0.92

#### Level 3: Active Learning + Drift-aware
```
Input Image
    │
    ├─ Drift Detection: Drift Score 계산
    │   └─ Score > 0.3 → Active Learning 모드
    │
    ├─ Uncertainty Sampling: High uncertainty 샘플 선별
    │   └─ Expected Model Change (EMC) 최대화
    │
    ├─ Expert Labeling: Senior Reviewer 할당
    │   └─ Double-blind 검증 (필요 시)
    │
    └─ Model Update: 재학습 트리거
        └─ Fine-tuning 또는 Full Retraining
```

**특징**
- 드리프트/불확실성 샘플 우선 라벨링
- Expert 리소스 집중 투입
- **목표**: Model Performance 유지, Drift Recovery < 1주

### 3.2 DINO + SAM 통합 전략

**Box → Mask Pipeline**
```
┌─────────────────────────────────────────────────────────┐
│  Step 1: DINO Detection                                 │
│  ─────────────────────────────────────────────────────  │
│  Input: Image                                            │
│  Output: Boxes [(x, y, w, h), confidence, class]        │
│  Filter: confidence > 0.6                               │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Box Validation                                  │
│  ─────────────────────────────────────────────────────  │
│  • Size Check: min(w, h) ≥ 32px                         │
│  • Overlap Check: IoU < 0.5 (NMS)                        │
│  • Aspect Ratio: 0.1 < w/h < 10                         │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: SAM Mask Generation                            │
│  ─────────────────────────────────────────────────────  │
│  Input: Image + Box (prompt)                            │
│  Output: Mask [H × W binary]                            │
│  Multi-run: 3회 실행 → IoU stability 계산                │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: Quality Validation                             │
│  ─────────────────────────────────────────────────────  │
│  • Box-Mask IoU: ≥ 0.7                                   │
│  • Mask Area: reasonable (0.1% - 50% of image)          │
│  • Edge Smoothness: contour regularity                  │
│  • Class Consistency: DINO class vs mask region         │
└─────────────────────────────────────────────────────────┘
```

**Ambiguous Case 자동 분기**
```python
def handle_ambiguous_case(image, dino_output, sam_output):
    # Case 1: Overlapping Objects
    if detect_overlap(dino_output.boxes) > 0.3:
        return Action.SPLIT_AND_REVIEW  # 분리 후 검토
    
    # Case 2: Partial Occlusion
    if sam_output.mask_coverage < 0.7:
        return Action.EXPAND_BOX_AND_RERUN  # Box 확장 후 재실행
    
    # Case 3: Edge Cases (boundary)
    if is_near_boundary(sam_output.mask, image.shape):
        return Action.MANUAL_REVIEW  # 수동 검토
    
    # Case 4: Low Confidence Chain
    if dino_output.confidence < 0.7 and sam_output.iou_stability < 0.8:
        return Action.ACTIVE_LEARNING  # Level 3로 전환
    
    return Action.AUTO_ACCEPT
```

### 3.3 Human-in-the-loop 최적화

**QA 샘플링 전략**
```
전체 라벨 풀
    │
    ├─ [20%] 랜덤 샘플링
    │   └─ 통계적 대표성 확보
    │
    ├─ [30%] High-risk 샘플링
    │   └─ Quality Score < 0.80
    │
    ├─ [30%] Edge Case 샘플링
    │   └─ Ambiguous case flag
    │
    └─ [20%] Stratified 샘플링
        └─ 클래스별 균등 분포
```

**Reviewer Load Balancing**
```python
class ReviewerBalancer:
    def balance_workload(self, reviewers, tasks):
        # 1. 현재 작업량 계산
        current_load = {r.id: len(r.active_tasks) for r in reviewers}
        
        # 2. 역량 가중치 적용
        capacity = {
            r.id: r.max_concurrent * r.efficiency_score 
            for r in reviewers
        }
        
        # 3. 할당 최적화 (Hungarian Algorithm)
        assignment = self._hungarian_assign(tasks, reviewers, capacity)
        
        return assignment
```

**Double-blind Annotation 전략**
- **적용 조건**: Quality Score < 0.60 or 신규 클래스
- **프로세스**:
  1. 2명의 Senior Reviewer에게 동시 할당
  2. 서로의 라벨 확인 불가
  3. Inter-annotator Agreement 계산
  4. IoU < 0.85 → 3rd Reviewer (Arbiter) 할당

---

## 4부: 최적화 포인트 요약표

### 4.1 메트릭 설계

| 메트릭 | 정의 | 목표값 | 측정 방법 |
|--------|------|--------|----------|
| **Label Accuracy Proxy** | IoU 기반 정확도 | ≥ 0.92 | (Predicted ∩ Ground Truth) / (Predicted ∪ Ground Truth) |
| **Auto-Label Acceptance Rate** | 자동 승인 비율 | ≥ 60% | Auto-accepted / Total Labels |
| **Rework Rate** | 재작업 비율 | ≤ 8% | Reworked / Total Labels |
| **Mean Time To Label (MTTL)** | 평균 라벨링 시간 | ≤ 2.5 min/image | Total Time / Total Images |
| **Cost per Label** | 라벨당 비용 | ↓ 40% | (Reviewer Time × Rate) / Labels |
| **Drift Detection Latency** | 드리프트 감지 지연 | ≤ 1 day | Time from Drift to Detection |
| **Model Update Frequency** | 모델 업데이트 빈도 | Weekly | 재학습 주기 |

### 4.2 Threshold 최적화 매트릭스

| 컴포넌트 | Threshold | 조정 조건 | 영향도 |
|---------|-----------|----------|--------|
| **DINO Confidence** | 0.6 (기본) | Auto-accept ↑ → 0.7로 상향 | Precision ↑, Recall ↓ |
| **SAM IoU Stability** | 0.8 (기본) | 불안정 시 → 0.75로 하향 | Coverage ↑, Quality ↓ |
| **Box-Mask IoU** | 0.7 (기본) | 엄격 모드 → 0.8로 상향 | Accuracy ↑, Rework ↑ |
| **Quality Score (Auto-accept)** | 0.90 (기본) | 보수 모드 → 0.95로 상향 | Rework ↓, Auto-accept ↓ |
| **Drift Score (Alert)** | 0.3 (기본) | 민감 모드 → 0.2로 하향 | Early Detection ↑, False Positive ↑ |

### 4.3 비용 최적화 전략

| 전략 | 절감 메커니즘 | 예상 절감율 | 리스크 |
|------|--------------|------------|--------|
| **Level 2 Auto-accept** | Reviewer 시간 60% 절감 | 40% | Quality 저하 가능 |
| **Smart QA Sampling** | 검증 샘플 50% 감소 | 15% | 오류 누락 가능 |
| **Batch Processing** | 컨텍스트 스위칭 감소 | 10% | 지연 증가 가능 |
| **Skill-based Assignment** | 역량 매칭으로 효율 ↑ | 20% | 리소스 불균형 |
| **Active Learning** | 불필요한 라벨링 감소 | 25% | 모델 편향 가능 |

---

## 5부: 실무 적용 체크리스트

### 5.1 초기 구축 단계

**Phase 1: 인프라 구축 (Week 1-2)**
- [ ] Label Studio 서버 설정 및 워크스페이스 구성
- [ ] DINO 모델 배포 (ONNX/TensorRT 최적화)
- [ ] SAM 모델 배포 (GPU 메모리 최적화)
- [ ] 에이전트 간 메시지 큐 설정 (Redis/RabbitMQ)
- [ ] 메트릭 수집 인프라 (Prometheus/Grafana)

**Phase 2: 파이프라인 통합 (Week 3-4)**
- [ ] DINO → SAM 파이프라인 연결
- [ ] Quality Gate Agent 로직 구현
- [ ] Label Studio API 연동
- [ ] 기본 워크플로우 테스트 (Level 1)

**Phase 3: 에이전트 고도화 (Week 5-6)**
- [ ] Orchestration Agent 구현
- [ ] Cost Control Agent 구현
- [ ] Drift Detection Agent 구현
- [ ] Level 2/3 자동화 레벨 구현

### 5.2 운영 최적화 단계

**Week 7-8: 캘리브레이션**
- [ ] Threshold 튜닝 (Validation Set 기준)
- [ ] Reviewer 역량 매트릭스 구축
- [ ] Baseline 메트릭 수집
- [ ] A/B 테스트 (Level 1 vs Level 2)

**Week 9-12: 스케일링**
- [ ] 병렬 처리 최적화
- [ ] 큐 관리 및 우선순위 시스템
- [ ] 자동 재학습 파이프라인 구축
- [ ] 대시보드 구축 (실시간 모니터링)

### 5.3 운영 시나리오별 대응

#### 시나리오 1: 데이터 급증 시
```
┌─────────────────────────────────────────────────────────┐
│  대응 전략                                               │
│  ─────────────────────────────────────────────────────  │
│  1. Auto-accept Threshold 하향 (0.90 → 0.85)            │
│  2. QA 샘플링 비율 감소 (20% → 10%)                     │
│  3. Batch Size 증가 (병렬 처리)                         │
│  4. 임시 Reviewer 추가 (On-demand)                      │
│  5. 우선순위 큐: High-value 샘플만 선별                  │
└─────────────────────────────────────────────────────────┘
```

#### 시나리오 2: 신규 클래스 추가 시
```
┌─────────────────────────────────────────────────────────┐
│  대응 전략                                               │
│  ─────────────────────────────────────────────────────  │
│  1. 해당 클래스 → Level 3 (Active Learning) 강제        │
│  2. Expert Reviewer 전담 할당                           │
│  3. Double-blind Annotation 필수                        │
│  4. Few-shot 학습 데이터 우선 수집                       │
│  5. 모델 Fine-tuning 트리거 (샘플 100개 도달 시)        │
└─────────────────────────────────────────────────────────┘
```

#### 시나리오 3: 모델 성능 하락 시
```
┌─────────────────────────────────────────────────────────┐
│  대응 전략                                               │
│  ─────────────────────────────────────────────────────  │
│  1. Drift Detection Agent → Alert 발송                  │
│  2. 모든 샘플 → Level 1로 강등 (Auto-accept 중단)       │
│  3. High-confidence 샘플만 선별하여 재학습 데이터 구축   │
│  4. 모델 재학습 트리거 (즉시 또는 주기적)               │
│  5. 성능 회복 확인 후 Level 2/3 점진적 복구              │
└─────────────────────────────────────────────────────────┘
```

#### 시나리오 4: 라벨러 인력 부족 시
```
┌─────────────────────────────────────────────────────────┐
│  대응 전략                                               │
│  ─────────────────────────────────────────────────────  │
│  1. Auto-accept Threshold 상향 (0.90 → 0.95)            │
│  2. QA 샘플링 비율 감소 (20% → 5%)                      │
│  3. 우선순위 큐: Critical 샘플만 처리                    │
│  4. Batch Processing 최대화 (효율 ↑)                   │
│  5. Crowdsourcing 플랫폼 연동 (필요 시)                 │
└─────────────────────────────────────────────────────────┘
```

### 5.4 모니터링 체크리스트

**일일 모니터링**
- [ ] Auto-accept Rate 추이
- [ ] Rework Rate 추이
- [ ] Reviewer 작업량 분포
- [ ] 큐 대기 시간

**주간 모니터링**
- [ ] Label Accuracy Proxy (Validation Set)
- [ ] Drift Score 추이
- [ ] Cost per Label 추이
- [ ] 모델 성능 메트릭 (mAP, IoU)

**월간 모니터링**
- [ ] 전체 파이프라인 ROI 분석
- [ ] Threshold 재최적화
- [ ] Reviewer 역량 재평가
- [ ] 장기 트렌드 분석

### 5.5 리스크 관리

| 리스크 | 영향도 | 완화 전략 |
|--------|--------|----------|
| **Auto-accept 오류 누적** | High | QA 샘플링 + Confidence Threshold 보수적 설정 |
| **Drift 미감지** | High | 다중 드리프트 감지 메커니즘 + 주기적 재학습 |
| **Reviewer 역량 편차** | Medium | Skill-based Assignment + 정기 교육 |
| **인프라 장애** | Medium | Redundancy + 자동 Failover |
| **비용 초과** | Low | 실시간 비용 모니터링 + 예산 알림 |

---

## 부록: Agent-based Labeling Factory 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Agent-based Labeling Factory                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   Input Image Queue           │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Orchestration Agent         │
                    │   • Route to Level 1/2/3      │
                    │   • Queue Management          │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  Level 1      │          │  Level 2      │          │  Level 3      │
│  Pre-label    │          │  Auto-accept  │          │  Active Learn │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────┬───────────┴───────────┬──────────────┘
                       │                       │
        ┌──────────────▼───────────────────────▼──────────────┐
        │         DINO + SAM Pipeline                         │
        │  ┌──────────┐         ┌──────────┐                 │
        │  │   DINO   │────────▶│   SAM    │                 │
        │  │ Detection│         │  Mask    │                 │
        │  └──────────┘         └──────────┘                 │
        └──────────────┬───────────────────────┬──────────────┘
                       │                       │
        ┌──────────────▼───────────────────────▼──────────────┐
        │         Quality Gate Agent                           │
        │  • Quality Score 계산                                │
        │  • Box-Mask Validation                              │
        │  • Ambiguous Case Detection                          │
        └──────────────┬───────────────────────┬──────────────┘
                       │                       │
        ┌──────────────▼───────────────────────▼──────────────┐
        │         Cost Control Agent                          │
        │  • Reviewer Assignment                              │
        │  • Load Balancing                                   │
        │  • Priority Queue Management                        │
        └──────────────┬───────────────────────┬──────────────┘
                       │                       │
        ┌──────────────▼───────────────────────▼──────────────┐
        │         Label Studio                                │
        │  • Human-in-the-loop                                │
        │  • QA Sampling                                      │
        │  • Double-blind Annotation                          │
        └──────────────┬───────────────────────┬──────────────┘
                       │                       │
        ┌──────────────▼───────────────────────▼──────────────┐
        │         Drift Detection Agent                        │
        │  • Feature Space Drift                              │
        │  • Performance Drift                                 │
        │  • Re-training Trigger                              │
        └──────────────┬───────────────────────┬──────────────┘
                       │                       │
        ┌──────────────▼───────────────────────▼──────────────┐
        │         Metrics & Monitoring                         │
        │  • Label Accuracy Proxy                              │
        │  • Rework Rate                                       │
        │  • MTTL                                              │
        │  • Cost per Label                                    │
        └──────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Model Update Pipeline       │
                    │   • Fine-tuning               │
                    │   • Full Retraining           │
                    └───────────────────────────────┘
```

---

## 결론

이 아키텍처는 **"최소 비용 / 최대 품질 / 지속적 학습"**을 동시에 달성하기 위해:

1. **3단계 자동화 레벨**로 점진적 신뢰 구축
2. **4-Agent 시스템**으로 책임 분리 및 최적화
3. **DINO + SAM 통합**으로 Box → Mask 품질 보장
4. **Human-in-the-loop 최적화**로 리소스 효율 극대화
5. **Drift Detection**으로 지속적 학습 보장

**예상 성과**
- 비용: 40% 절감 (Auto-accept + Smart Sampling)
- 품질: Accuracy ≥ 0.92 (Quality Gate + QA)
- 시간: MTTL ≤ 2.5 min/image (병렬 처리 + 우선순위)

**핵심 성공 요인**
- Threshold 캘리브레이션 (도메인 특화)
- Reviewer 역량 매트릭스 구축
- 실시간 모니터링 및 피드백 루프

