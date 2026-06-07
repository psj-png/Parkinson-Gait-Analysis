"""
Flask mobile web demo — Gait Analysis
Routes:
  GET  /           업로드 폼
  POST /diagnose   영상 업로드 → 진단 결과 JSON (GradCAM + 프레임별 신뢰도 포함)
  GET  /health     서버 상태 확인
"""

import uuid
import base64
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ── 경로 ───────────────────────────────────────────────────────────────────────

_BASE        = Path(__file__).resolve().parent.parent
MODEL_PATH   = _BASE / "output" / "models" / "cnn_best.pth"
UPLOAD_DIR   = _BASE / "output" / "uploads"
TEMPLATE_DIR = _BASE / "templates"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}
CLASSES     = ["Abnormal", "Normal"]
IMG_SIZE    = 224
# 보행 1사이클 ≈ 1초 → 30fps 기준 1사이클 완전 커버. 학습·진단 조건 통일.
N_FRAMES    = 30

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── 모델 (앱 시작 시 1회 로드) ─────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device).eval()
print(f"[app] model loaded on {device}")

# ── GradCAM ────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, mdl: nn.Module, target_layer: nn.Module):
        self.mdl         = mdl
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, out):
        self.activations = out.detach()

    def _save_grad(self, _, __, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.mdl.zero_grad()
        out = self.mdl(tensor)
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAM(model, model.layer4[-1])

# ── Optical Flow 추출 ──────────────────────────────────────────────────────────

def extract_flow_frames(video_path: Path, n: int = N_FRAMES):
    """BGR optical flow 프레임 리스트 반환."""
    cap = cv2.VideoCapture(str(video_path))
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot open video: {video_path}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        prev_gray = gray
    cap.release()
    return frames


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return TRANSFORM(img).unsqueeze(0).to(device)


def frame_to_b64(bgr: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ── 핵심 진단 로직 ──────────────────────────────────────────────────────────────

def run_diagnosis(video_path: Path) -> dict:
    flow_frames = extract_flow_frames(video_path)
    if not flow_frames:
        raise RuntimeError("No optical flow frames extracted.")

    # 프레임별 추론
    frame_normal_probs = []
    tensors = []
    for bgr in flow_frames:
        t = bgr_to_tensor(bgr)
        tensors.append(t)
        with torch.no_grad():
            probs = F.softmax(model(t), dim=1).cpu().numpy()[0]
        frame_normal_probs.append(float(f"{probs[CLASSES.index('Normal')]:.4f}"))

    mean_probs = np.mean(
        [F.softmax(model(t), dim=1).cpu().detach().numpy()[0] for t in tensors], axis=0
    )
    pred_idx   = int(np.argmax(mean_probs))
    prediction = CLASSES[pred_idx]

    # GradCAM — 예측 클래스 신뢰도가 가장 높은 프레임 선택
    best_idx   = int(np.argmax([frame_normal_probs[i] if pred_idx == CLASSES.index("Normal")
                                else 1 - frame_normal_probs[i]
                                for i in range(len(flow_frames))]))
    best_bgr   = flow_frames[best_idx]
    cam_tensor = bgr_to_tensor(best_bgr)
    cam_tensor.requires_grad_(True)
    cam        = gradcam.generate(cam_tensor, pred_idx)

    # 히트맵 오버레이
    resized = cv2.resize(best_bgr, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized, 0.55, heatmap, 0.45, 0)

    # Optical Flow 프레임 30장 → 200×200 썸네일로 base64 인코딩
    flow_frames_b64 = [
        frame_to_b64(cv2.resize(f, (200, 200)), quality=65)
        for f in flow_frames
    ]

    return {
        "prediction":         prediction,
        "confidence":         {c: float(f"{mean_probs[i]:.4f}") for i, c in enumerate(CLASSES)},
        "frames_analyzed":    len(flow_frames),
        "frame_normal_probs": frame_normal_probs,
        "gradcam_b64":        frame_to_b64(overlay),
        "flow_frames_b64":    flow_frames_b64,
    }


# ── Flask 앱 ───────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/diagnose", methods=["POST"])
def diagnose_route():
    if "video" not in request.files:
        return jsonify({"error": "video 파일이 없습니다."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "파일명이 비어 있습니다."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"지원하지 않는 형식입니다: {ext}"}), 415

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        file.save(tmp_path)
        result = run_diagnosis(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
