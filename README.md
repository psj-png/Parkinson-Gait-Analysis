# 🚶‍♂️ AI 기반 파킨슨병 조기 진단 및 보행 분석 시스템
> **Computer Vision(MediaPipe)과 Hybrid Decision Logic을 활용한 비침습적 스크리닝 파이프라인**

본 프로젝트는 고가의 마커 베이스 장비 없이, 일반 영상(RGB Video)에서 파킨슨병 특유의 보행 특징을 추출하고 **의료적 지표와 머신러닝 확률을 결합**하여 위험도를 조기에 선별(Screening)하는 AI 시스템입니다.

---

## 🎯 프로젝트 목적 및 철학
* **조기 진단 최적화**: 상세 병명 진단이 아닌, 미세 증상을 포착하여 **'병원 방문 및 정밀 검사 권고'**를 수행하는 것을 최우선 목표로 합니다.
* **High-Recall 전략**: 실제 환자를 정상으로 오진하는 위험(False Negative)을 최소화하기 위해 **[정상(Normal) vs 의심(Suspected)]** 2단계 분류 체계와 보수적인 판정 임계값을 적용합니다.

---

## 🛠 Computer Vision Pipeline & Noise Reduction
영상의 픽셀 데이터를 직접 학습하는 대신, 신체 특징점(Landmarks)을 수치화하는 **Feature-based Learning** 방식을 채택하여 데이터 효율성과 판단 근거의 명확성을 확보했습니다.



### **3단계 데이터 정제 (Noise Reduction)**
추출된 좌표의 신뢰도를 확보하기 위해 다음과 같은 전처리 과정을 거칩니다:
1. **Outlier Processing**: 조명/배경 간섭으로 인한 좌표 튐 현상을 감지하고 통계적 임계치를 활용해 보정합니다.
2. **Jittering Removal**: 이동 평균 필터(Moving Average)를 통해 MediaPipe 좌표의 미세한 떨림을 억제합니다.
3. **Data Smoothing**: **Butterworth Low-pass Filter**를 적용하여 생체역학적으로 타당한 부드러운 보행 곡선을 산출합니다.



---

## 🔍 하이브리드 판별 알고리즘 (Hybrid Decision Logic)
본 시스템은 조기 진단 스크리닝의 핵심인 **'미진단 최소화'**를 위해 **Condition A(의료 지표)**와 **Condition B(ML 확률)**를 결합한 **'All-Pass 이중 안전망'** 체계를 채택하였습니다.



### **1. 이중 안전망(Double Safety-Net) 판정 원칙**
* **정상(Normal) 판정**: Condition A와 Condition B를 **모두 충족(Pass)**할 때만 최종 '정상'으로 분류합니다.
* **의심(Suspected) 판정**: 두 조건 중 **하나라도** 임계치를 벗어나면 즉시 '의심'으로 분류하여 전문의 진료를 권고합니다.

### **2. 세부 판정 조건 및 설정 근거**

| 구분 | 항목 | 판정 기준 (Normal 조건) | 설정 근거 |
| :--- | :--- | :--- | :--- |
| **Condition A** | **의료적 보행 지표** | Knee ROM > 25° **AND** Trunk Lean < 10° | 파킨슨 전형 징후인 서동증 및 전굴 자세 반영 |
| **Condition B** | **ML 예측 확률** | **파킨슨 판정 확률 < 40%** | **High-Recall(민감도) 전략**을 위한 안전 마진 확보 |

#### **※ ML 임계값(Threshold)을 40%로 설정한 이유**
1. **Safety Margin**: 초기 파킨슨 환자의 미세 징후를 놓치지 않도록 판정 문턱을 낮추어 **재현율(Recall)을 극대화**함.
2. **상호 보완**: 의료 수치(A)가 정상이더라도 AI가 포착한 비정형 패턴(B)이 위험 신호를 보낼 경우 선제적으로 대응함.
3. **데이터 편향 보정**: 정상군 데이터의 높은 밀도로 인한 모델의 '정상 편향성'을 공학적으로 보정함.


---

## 📂 프로젝트 구조 (Project Structure)

```text
Parkinson_Gait_Analysis/
├── data/                       # 데이터셋 (Normal / Parkinson)
├── extracted_data/             # 추출된 .csv 수치 데이터
├── models/                     # 학습된 parkinson_model.pkl 저장소
├── utils.py                    # 3단계 노이즈 필터 및 4대 지표 계산 모듈
├── video_classifier.py         # 촬영 시점(정면/측면) 자동 분류기
├── walk_detector.py            # 영상 내 보행 존재 여부 판별기
├── final_gait_extraction.py    # CV 특징 추출 및 필터링 메인 엔진
├── gait_pure_learning.py       # 데이터 재분류 및 모델 학습 스크립트
└── final_demo.py               # 실시간 분석 대시보드 데모

## 📊 분석 결과 시각화 (Visualization)

본 시스템은 단순한 분류 결과를 넘어, **생체역학적 근거(Biomechanical Evidence)**를 시각화하여 AI 판단의 투명성과 신뢰성을 제공합니다.

### **1. 보행 지표 비교 그래프 (Gait Feature Analysis)**
`gait_graph_differ.py`를 통해 추출된 지표를 분석한 결과, 정상군과 의심군 사이의 유의미한 수치 차이를 확인했습니다.

* **정상군 (Normal)**: 무릎 가동 범위(ROM)가 일정하며, 보행 리듬의 변동성이 낮음 (안정적인 사인파 형태).
* **의심군 (Suspected)**: ROM의 진폭이 불규칙하고, 보행 주기 간격의 표준편차가 정상군 대비 약 **2.5배 이상** 높게 나타남.



> *필터링 전(Raw)과 후(Filtered)의 데이터를 비교하여 3단계 노이즈 제거 엔진의 효율을 시각적으로 증명합니다.*

---

### **2. 실시간 진단 대시보드 (Real-time Diagnostic Demo)**
`final_demo.py` 실행 시, 분석 엔진이 영상 위에 실시간으로 진단 데이터를 오버레이(Overlay)하여 직관적인 피드백을 제공합니다.

* **Skeleton Tracking**: MediaPipe를 활용한 실시간 전신 관절 추적 및 시각화.
* **Clinical Dashboard**: 
    * **Risk Score**: 모델이 판단한 파킨슨 위험도(%) 실시간 출력.
    * **Live Indicators**: 현재 프레임의 Knee ROM, Trunk Lean 수치 실시간 업데이트.
    * **Status Label**: 'All-Pass' 로직에 따른 최종 [Normal] / [Suspected] 상태 표시.



---

### **3. 모델 성능 및 중요도 분석 (Model Evaluation)**
조기 진단 목적에 최적화된 모델의 학습 결과입니다.

* **High-Recall (재현율)**: 가중치 조절(1:15) 및 40% 임계값 설정을 통해 환자를 정상으로 오진할 확률(False Negative)을 극소화함.
* **Feature Importance**: Random Forest 분석 결과, **Knee ROM**과 **Gait Rhythm**이 파킨슨 판별에 가장 결정적인 기여를 하는 변수로 확인됨.
