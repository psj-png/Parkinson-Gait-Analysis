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
1. **Jittering Removal**: 이동 평균 필터(Moving Average)를 통해 MediaPipe 좌표의 미세한 떨림을 억제합니다.
2. **Outlier Processing**: 조명/배경 간섭으로 인한 좌표 튐 현상을 감지하고 선형 보간(Interpolation)을 통해 보정합니다.
3. **Data Smoothing**: **Butterworth Low-pass Filter**를 적용하여 고주파 노이즈를 차단하고, 생체역학적으로 타당한 부드러운 보행 곡선을 산출합니다.



---

## 🔍 하이브리드 판별 알고리즘 (Hybrid Decision Logic)
적은 표본 데이터의 한계를 극복하기 위해 **의료적 지식(Condition A)**과 **데이터 통계(Condition B)**를 결합한 이중 안전망을 구축하였습니다.

### **1. Condition A: 4대 핵심 의료 지표 (Clinical Rules)**
다음 지표 중 **단 하나라도** 임계치를 벗어나면 '의심군'으로 선제 분류합니다.
* **무릎 관절 가동 범위 (Knee ROM)**: 보폭 감소 및 보행 동결(Freezing of Gait) 징후 포착.
* **보행 대칭성 (Gait Symmetry)**: 좌우 하중 불균형 및 편측성 보행 이상 감지.
* **상체 경사도 (Trunk Lean)**: 파킨슨 전형의 전굴 자세(Stooped Posture) 확인.
* **보행 리듬 규칙성 (Gait Rhythm)**: 보행 주기의 불규칙한 변동성(Variability) 측정.



### **2. Condition B: ML 확률 판별 (ML Probability)**
* **Model**: Random Forest Classifier
* **Logic**: 모델이 판단한 파킨슨 확률이 **40% 이상**일 경우, 조기 진단 목적에 맞춰 보수적으로 '의심' 판정을 내립니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
Parkinson_Gait_Analysis/
├── data/                       # 데이터셋 (Normal / Suspected)
├── extracted_data/             # 추출된 .csv 수치 데이터
├── models/                     # 학습된 parkinson_model.pkl 저장소
├── utils.py                    # 3단계 노이즈 필터 및 4대 지표 계산 모듈
├── video_classifier.py         # 촬영 시점(정면/측면) 자동 분류기
├── walk_detector.py            # 영상 내 보행 존재 여부 판별기
├── final_gait_extraction.py    # CV 특징 추출 메인 엔진
├── gait_pure_learning.py       # 하이브리드 판별 및 모델 학습 스크립트
└── final_demo.py               # 실시간 분석 대시보드 데모
---

## 📊 분석 결과 시각화 (Visualization)

본 시스템은 단순한 분류 결과를 넘어, **생체역학적 근거(Biomechanical Evidence)**를 시각화하여 AI 판단의 투명성을 제공합니다.

### 1. 보행 지표 비교 그래프 (Gait Feature Analysis)
`gait_graph_differ.py`를 통해 추출된 지표를 분석한 결과, 정상군과 의심군 사이의 유의미한 수치 차이를 확인했습니다.

* **정상군 (Normal)**: 무릎 가동 범위(ROM)가 일정하며, 보행 리듬의 변동성이 낮음 (안정적인 사인파 형태).
* **의심군 (Suspected)**: ROM의 진폭이 불규칙하고, 보행 주기 간격의 표준편차가 정상군 대비 약 **2.5배 이상** 높게 나타남.

![Gait Analysis Graph](https://via.placeholder.com/800x400.png?text=Gait+Angle+and+Rhythm+Comparison+Graph)
> *필터링 전(Raw)과 후(Filtered)의 데이터를 비교하여 노이즈 제거 효율을 시각적으로 증명합니다.*

### 2. 실시간 진단 대시보드 (Real-time Diagnostic Demo)
`final_demo.py` 실행 시, 분석 엔진이 영상 위에 실시간으로 진단 데이터를 오버레이(Overlay)합니다.

* **Skeleton Tracking**: MediaPipe를 활용한 실시간 하체 및 상체 관절 추적.
* **Clinical Dashboard**: 
    - **Risk Score**: 모델이 판단한 파킨슨 위험도(%) 실시간 출력.
    - **Live Indicators**: 현재 프레임의 Knee ROM, Trunk Lean 수치 표시.
    - **Status Label**: 통합 판별 로직에 따른 [Normal] / [Suspected] 상태 표시.

![Real-time Demo Screenshot](https://via.placeholder.com/800x450.png?text=Real-time+Gait+Analysis+Dashboard+Preview)

### 3. 모델 성능 평가 (Model Evaluation)
조기 진단 목적에 최적화된 모델의 성능 지표입니다.
* **Recall (재현율)**: 가중치 조절(1:15)을 통해 실제 환자를 정상으로 오진할 확률을 극도로 낮춤.
* **Feature Importance**: Random Forest 분석 결과, **Knee ROM**과 **Gait Rhythm**이 판별에 가장 큰 영향을 미치는 것으로 나타남.
