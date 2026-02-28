# 🚶‍♂️ AI 기반 파킨슨병 조기 진단 및 보행 분석 시스템
> **Advanced Computer Vision(MediaPipe) & Hybrid Decision Logic을 활용한 비침습적 스크리닝 파이프라인**

본 프로젝트는 고가의 마커 베이스 장비 없이 RGB 영상만으로 파킨슨병의 운동학적 징후를 포착합니다. 특히, 머신러닝 모델의 신뢰도를 높이기 위해 **영상 단계의 물리적 전처리(안정화/정규화)**와 **데이터 단계의 수치적 정제**를 결합한 고도화된 파이프라인을 구축했습니다.

---

## 🛠 Computer Vision Pipeline & Preprocessing
머신러닝(Random Forest, SVM 등) 모델이 노이즈에 오염되지 않은 '순수 특징점'을 학습할 수 있도록 2단계 전처리 체계를 운영합니다.

### **Phase 1: 영상 레벨 전처리 (Visual Stabilization & Normalization)**
좌표를 추출하기 전, 원본 영상 자체를 최적화하여 MediaPipe 추출 성능을 극대화합니다.
1. **Video Stabilization**: 카메라 흔들림으로 인한 관절 좌표의 가공된 변동(Jitter)을 방지하기 위해 **Optical Flow 기반 아핀 변환(Affine Transform)**을 적용하여 배경을 물리적으로 고정합니다.
2. **Lighting Normalization (CLAHE)**: 조명 환경에 구애받지 않고 일관된 랜드마크 추출을 위해 **대비 제한 적응형 히스토그램 평활화(CLAHE)**를 적용하여 가시성을 확보합니다.
3. **Background Subtraction**: 배경의 불필요한 노이즈를 제거하여 모델이 '사람의 실루엣'과 '보행 패턴'에만 집중하도록 환경을 정제합니다.



### **Phase 2: 데이터 레벨 정제 (Coordinate Noise Reduction)**
추출된 좌표의 신뢰도를 확보하기 위해 3단계 필터링을 거칩니다.
1. **Outlier Processing**: 조명/배경 간섭으로 인한 좌표 튐 현상을 통계적 임계치로 보정합니다.
2. **Jittering Removal**: 이동 평균 필터(Moving Average)를 통해 MediaPipe 좌표의 미세한 떨림을 억제합니다.
3. **Data Smoothing**: **Butterworth Low-pass Filter**를 적용하여 생체역학적으로 타당한 부드러운 보행 곡선을 산출합니다.

---

## 🔍 하이브리드 판별 알고리즘 (Hybrid Decision Logic)
본 시스템은 조기 진단 스크리닝의 핵심인 **'미진단 최소화'**를 위해 **의료 지표(Domain Knowledge)**와 **ML 확률(Pattern Recognition)**을 결합한 이중 안전망 체계를 채택하였습니다.



### **1. 이중 안전망(Double Safety-Net) 판정 원칙**
* **정상(Normal) 판정**: 의료 지표(Condition A)와 ML 확률(Condition B)을 **모두 충족(Pass)**할 때만 최종 '정상' 분류.
* **의심(Suspected) 판정**: 두 조건 중 **하나라도** 임계치를 벗어나면 즉시 '의심'으로 분류하여 정밀 검사 권고.

### **2. 판정 임계값 설정 (High-Recall 전략)**
* **Condition A (Clinical)**: Knee ROM > 25° **AND** Trunk Lean < 10° (파킨슨 전형 징후 반영)
* **Condition B (ML Logic)**: **Parkinson Probability < 40%** (보수적 임계값 설정을 통한 재현율 극대화)

---

## 📂 프로젝트 구조 (Project Structure)

```text
Parkinson_Gait_Analysis/
├── gavd_data_1~5/              # 원본 대용량 영상 저장소 (Raw Data)
├── data/                       # 클래스별(Normal, Ambiguous 등) 분리된 데이터셋
├── stabilized_videos/          # 흔들림 보정 및 전처리가 완료된 영상 출력 폴더
├── video_mapping.xlsx          # 잘린 영상과 원본 영상 간의 매핑 테이블
├── stabilize_videos.py         # 영상 안정화 및 전처리 메인 엔진
├── final_gait_extraction.py    # 안정화된 영상에서 특징점 추출 및 필터링
├── gait_pure_learning.py       # 머신러닝 학습 및 특징 중요도 분석
└── final_demo.py               # 실시간 분석 및 진단 대시보드

---

## 📊 분석 결과 및 시각화 (Analysis Results & Visualization)

본 시스템은 단순한 분류를 넘어, 머신러닝 모델의 판단 근거를 시각화하여 진단의 신뢰성과 투명성(Explainability)을 제공합니다.

---

### 1. 영상 안정화 전후 비교 (Stabilization Impact)
영상 안정화(Stabilization) 처리는 머신러닝 모델의 입력값인 좌표 데이터의 순도를 결정짓는 핵심 단계입니다.

* **Before**: 카메라의 미세한 흔들림이 신체 랜드마크의 좌표에 더해져, 실제 보행과 무관한 '가짜 노이즈(Artifacts)'가 발생합니다. 이는 모델이 환자의 신체 떨림으로 오인할 위험을 초래합니다.
* **After**: Optical Flow 기반 보정을 통해 배경을 고정함으로써, 순수한 **생체역학적 움직임(Biomechanical Movement)**만을 추출합니다. 이를 통해 특징점 변동 폭의 표준편차를 약 15~20% 감소시켜 모델의 안정성을 확보했습니다.



---

### 2. 보행 지표 분석 (Gait Feature Insights)
추출된 4대 핵심 지표를 분석한 결과, 정상군과 의심군 사이에서 통계적으로 유의미한 차이가 발견되었습니다.

| 지표 (Features) | 정상군 (Normal) | 의심군 (Suspected) | 비고 |
| :--- | :--- | :--- | :--- |
| **Knee ROM** | 25° ~ 45° (일정함) | 25° 미만 (협소함) | 파킨슨병의 서동증 반영 |
| **Trunk Lean** | 0° ~ 5° (직립) | 10° 이상 (전굴) | 전형적인 구부정한 자세 |
| **Gait Rhythm** | 낮은 변동성 (주기 일정) | 높은 변동성 (불규칙) | 보행 동결 및 리듬 장애 |
| **Step Length** | 일정하고 넓은 보폭 | 좁고 불규칙한 보폭 | 보행 효율성 저하 관찰 |

---

### 3. 머신러닝 특징 중요도 (Feature Importance)
Random Forest 모델을 통해 분석한 결과, 판별에 가장 큰 기여를 하는 변수는 다음과 같습니다.

1.  **Knee ROM (42%)**: 무릎 가동 범위의 축소는 파킨슨병 판별의 가장 강력한 지표입니다.
2.  **Gait Rhythm Consistency (28%)**: 보행 주기가 얼마나 일정하게 유지되는지가 두 번째로 중요한 요소입니다.
3.  **Trunk Lean (15%)**: 상체의 기울기는 자세 불안정성을 판단하는 주요 근거가 됩니다.



---

### 4. 실시간 진단 대시보드 (Real-time Diagnostic Demo)
`final_demo.py`를 통해 제공되는 대시보드는 사용자에게 즉각적인 피드백을 제공합니다.

* **Skeleton Overlay**: MediaPipe를 통한 실시간 관절 추적 및 시각화.
* **Indicator Gauges**: Knee ROM, Trunk Lean 등 주요 수치를 실시간 게이지 형태로 출력.
* **Probability Score**: 머신러닝 모델이 계산한 위험 확률을 실시간으로 업데이트하여 'All-Pass' 로직에 따른 최종 상태(`Normal` / `Suspected`)를 표시합니다.

> **Note**: 본 시각화 결과는 전문의의 최종 진단을 대체할 수 없으며, 정밀 검사가 필요한 대상자를 선별하는 스크리닝 목적으로만 사용됩니다.
