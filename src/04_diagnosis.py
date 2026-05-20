"""
Step 4: Run inference on a new video and produce a diagnosis report.

Usage:
    python 04_diagnosis.py --video <path_to_video.mp4>

Output: output/results/diagnosis_<video_name>.json + Grad-CAM overlay
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("../output/models/cnn_best.pth")
RESULT_DIR = Path("../output/results")
IMG_SIZE   = 224
CLASSES    = ["Abnormal", "Normal"]
N_FRAMES   = 30

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def extract_flow_frames(video_path: Path, n: int = N_FRAMES) -> list[np.ndarray]:
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
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        prev_gray = gray
    cap.release()
    return frames


def diagnose(video_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torch.nn as nn
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    flow_frames = extract_flow_frames(video_path)
    if not flow_frames:
        raise RuntimeError("No optical flow frames extracted.")

    probs_list = []
    for frame_bgr in flow_frames:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        probs_list.append(probs)

    mean_probs = np.mean(probs_list, axis=0)
    pred_idx   = int(np.argmax(mean_probs))
    result = {
        "video":      str(video_path),
        "prediction": CLASSES[pred_idx],
        "confidence": {c: float(f"{mean_probs[i]:.4f}") for i, c in enumerate(CLASSES)},
        "frames_analyzed": len(probs_list),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()

    video_path = Path(args.video)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    result = diagnose(video_path)
    out_json = RESULT_DIR / f"diagnosis_{video_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']}")
    print(f"Report saved: {out_json}")


if __name__ == "__main__":
    main()
