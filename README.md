# Gait Analysis — Normal vs Abnormal Classification

Optical Flow + CNN (ResNet-18) + Grad-CAM 기반 보행 이상 이진 분류 파이프라인.

## 파이프라인 구조

```
data/
  01_Normal/          원본 정상 보행 영상
  02_Abnormal/        원본 이상 보행 영상

output/
  optical_flow/       Step 1 결과 (Farneback Optical Flow 이미지)
    Normal/
    Abnormal/
  models/             Step 2 결과 (학습된 가중치 .pth)
  results/            Step 3-4 결과 (Grad-CAM 시각화, 진단 리포트)

src/
  01_extract_optical_flow.py   Optical Flow 추출
  02_train_cnn.py              ResNet-18 학습
  03_gradcam.py                Grad-CAM 시각화
  04_diagnosis.py              단일 영상 진단

archive/              이전 파이프라인 파일 백업
```

## 실행 순서

### 1. Optical Flow 추출
```bash
python src/01_extract_optical_flow.py
```
- `data/01_Normal`, `data/02_Abnormal`의 영상을 읽어 Farneback Optical Flow 이미지 추출
- 영상당 30프레임, HSV 색상 맵으로 저장
- 결과: `output/optical_flow/Normal/`, `output/optical_flow/Abnormal/`

### 2. CNN 학습
```bash
python src/02_train_cnn.py
```
- **영상 단위** 80/20 train/val 분리 (seed=42) — 동일 영상의 프레임이 train/val에 섞이지 않음
- ResNet-18 (ImageNet pretrained) fine-tuning
- 매 에포크 Loss / Accuracy / F1 출력, 완료 시 classification report 자동 출력
- 결과: `output/models/cnn_best.pth`

### 3. Grad-CAM 시각화
```bash
python src/03_gradcam.py
```
- 학습된 모델의 `layer4` 기준 Grad-CAM 히트맵 생성
- 결과: `output/results/gradcam/`

### 4. 단일 영상 진단
```bash
python src/04_diagnosis.py --video <영상 경로>
```
- 입력 영상에서 Optical Flow 추출 → 모델 추론 → JSON 리포트 저장
- 결과: `output/results/diagnosis_<영상명>.json`

## 데이터셋

| 클래스 | 영상 수 | Optical Flow 이미지 수 |
|--------|--------|----------------------|
| Normal | 41 | 1,230 |
| Abnormal | 32 | 960 |
| **합계** | **73** | **2,190** |

## 모델 평가 결과

> ResNet-18, 20 epochs, video-level 80/20 split (seed=42)

| 지표 | 값 |
|------|-----|
| Accuracy | **72.38%** |
| F1 (weighted) | **0.7240** |
| F1 (macro) | **0.7184** |

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Abnormal | 0.68 | 0.68 | 0.68 | 180 frames |
| Normal   | 0.76 | 0.75 | 0.76 | 240 frames |

**Confusion Matrix** (val: 14 videos / 420 frames)

|  | Pred Abnormal | Pred Normal |
|--|--------------|-------------|
| True Abnormal | 123 | 57 |
| True Normal   | 59  | 181 |

> 프레임 단위 split 시 acc 100% (데이터 누수) → 영상 단위 split으로 교정 후 72.4%

## 환경

- Python 3.10
- OpenCV 4.13
- PyTorch 2.12 (CPU)
- torchvision 0.27

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy
```
