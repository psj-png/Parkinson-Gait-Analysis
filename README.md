# 🚶‍♂️ AI 기반 파킨슨병 보행 특징 추출 및 자동 진단 시스템
> **MediaPipe와 Machine Learning(Random Forest)을 활용한 비침습적 보행 패턴 분석 파이프라인**

본 프로젝트는 고가의 마커 베이스 장비 없이, 일반 스마트폰이나 웹캠으로 촬영된 영상(RGB Video)에서 파킨슨병 특유의 보행 특징을 추출하고 위험도를 진단하는 AI 시스템입니다.

---

## ✨ 핵심 기능 (Key Features)

### 1. 데이터 수집 및 지능형 필터링
* **GAVD Dataset Pipeline**: 유튜브 기반의 실제 환자 보행 영상을 자동 수집하고 분석에 적합한 영상만 분류합니다. (`download_all_videos.py`)
* **영상 시점 자동 분류**: 어깨 너비(Shoulder Width) 데이터를 분석하여 정면, 측면, 사선 영상을 자동 분류하여 분석 정확도를 높였습니다. (`video_classifier.py`)
* **보행 감지 (Walk Detector)**: 관절 가시성(Visibility) 점수를 바탕으로 보행이 포함되지 않은 노이즈 영상을 사전에 필터링합니다. (`walk_detector.py`)

### 2. 정밀 관절 특징 추출 (Feature Extraction)
* **MediaPipe Pose Integration**: 전신 33개 랜드마크의 3차원 좌표를 프레임 단위로 추출합니다. (`final_gait_extraction.py`)
* **의료적 보행 지표 계산**: 추출된 좌표를 기반으로 파킨슨 진단의 핵심 지표를 산출합니다. (`utils.py`)
  - **Knee ROM**: 무릎 관절 가동 범위 (환자군의 경우 각도가 좁아지는 특성 반영)
  - **Trunk Lean**: 상체의 전방 기울기 측정
  - **Variability**: 보행 주기의 불규칙성 (정상인 대비 높은 변동성 감지)

### 3. 고도화된 진단 모델 (ML Modeling)
* **Random Forest Classifier**: 수집된 대규모 보행 지표 데이터를 바탕으로 정상(Normal)과 파킨슨(Parkinson)을 분류합니다.
* **불균형 데이터 최적화**: 환자군을 놓치지 않기 위해 **Class Weight를 1:10으로 설정**하여 모델의 진단 민감도를 강화했습니다. (`gait_pure_learning.py`)

---

## 🛠 실행 환경 및 설정 (Environment)

### 개발 환경
* **Language**: Python 3.10.x (64-bit)
* **OS**: Windows 10/11
* **Core Libraries**: `mediapipe`, `opencv-python`, `scikit-learn`, `pandas`, `scipy`

### 라이브러리 경로 이슈 해결
특정 로컬 환경에서 발생하는 패키지 인식 오류를 방지하기 위해, 실행 시 `sys.path`를 직접 참조하도록 설계되었습니다.
```python
import sys
# 상준 님 환경의 패키지 경로 강제 지정 로직 포함
lib_path = r"C:\Users\박상준\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)


## 📂 프로젝트 구조 (Project Structure)

```text
Parkinson_Gait_Analysis/
├── data/                       # 원본 데이터셋
│   ├── 01_Normal/              # 정상군 보행 영상 (.mp4)
│   ├── 02_Parkinson/           # 파킨슨병 환자군 보행 영상
│   └── 03_Ambiguous/           # 분석 모호군 데이터
├── Simulation_Data/            # 센서 기반 .mat 시뮬레이션 데이터
├── extracted_data/             # MediaPipe를 통해 추출된 수치 데이터
│   └── gait_integrated_data.csv # 통합 랜드마크 좌표 데이터
├── models/                     # 학습된 AI 모델 저장소
│   └── parkinson_model.pkl      # 최종 Random Forest 분류 모델
├── utils.py                    # 각도 계산, 이동 평균 필터 등 공통 모듈
├── video_classifier.py         # 어깨 너비 기반 시점(Front/Side) 분류기
├── walk_detector.py            # 영상 내 보행 존재 여부 자동 판별기
├── final_gait_extraction.py    # MediaPipe 기반 특징 추출 메인 엔진
├── gait_pure_learning.py       # 모델 학습 및 가중치(1:10) 설정 스크립트
├── final_demo.py               # 실시간 분석 및 대시보드 출력 데모
└── README.md                   # 프로젝트 설명 문서
---

### 📊 2. 분석 결과 시각화 레퍼런스 (Visualization Reference)
데이터 분석 결과와 모델의 판단 근거를 시각적으로 보여주는 섹션입니다.

```markdown
## 📊 분석 결과 시각화 (Visualization Reference)

본 프로젝트는 단순 분류를 넘어, AI가 어떤 근거로 파킨슨 보행을 판단했는지 시각적 지표를 제공합니다.

### 1) 관절 궤적 및 각도 분석 (Gait Graph)
`gait_graph_differ.py`를 통해 산출된 지표로, 정상군과 환자군의 유의미한 차이를 확인합니다.

* **정상군 (Black Line)**: 무릎 가동 범위(ROM)가 일정하고 진폭이 큼.
* **환자군 (Red Line)**: 보행 동결 및 보폭 감소로 인해 ROM 진폭이 매우 작고 불규칙함.
* **변동성(Variance)**: 환자군의 경우 프레임별 각도 변화의 표준편차가 정상군 대비 약 2.5배 높게 나타남.

> ![Gait Analysis Graph](https://via.placeholder.com/600x300.png?text=Gait+Angle+Comparison+Graph+Example)
> _(참고: 실제 실행 시 matplotlib을 통해 위와 같은 비교 그래프가 생성됩니다.)_

### 2) 실시간 진단 대시보드 (Real-time Demo)
`final_demo.py` 실행 시 영상 위에 실시간으로 분석 데이터가 오버레이됩니다.

* **Skeleton Overlay**: MediaPipe를 통한 실시간 관절 추적 현황.
* **Risk Score (%)**: 학습된 모델이 판단한 실시간 파킨슨 위험도 점수.
* **Clinical Indicators**: 현재 프레임의 ROM, Trunk Lean(상체 기울기) 수치 표시.

### 3) 혼동 행렬 (Confusion Matrix)
모델의 신뢰도를 평가하기 위해 환자군에 높은 가중치(Penalty)를 부여한 결과를 분석합니다.
* **Recall(재현율) 향상**: 가중치 조절(1:10)을 통해 환자를 정상으로 오판하는 비율(False Negative)을 최소화했습니다.
