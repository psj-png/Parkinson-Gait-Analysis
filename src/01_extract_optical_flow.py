"""
Step 1: Extract optical flow images from video files.

Input : data/01_Normal/, data/02_Abnormal/
Output: output/optical_flow/{Normal,Abnormal}/
"""

import cv2
import numpy as np
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent

INPUT_DIRS = {
    "Normal":   _BASE / "data" / "01_Normal",
    "Abnormal": _BASE / "data" / "02_Abnormal",
}
OUTPUT_BASE = _BASE / "output" / "optical_flow"


# 보행 1사이클(한 걸음 왕복)은 약 1초 → 30fps 영상에서 30프레임이면 1사이클을 완전히 커버.
# 학습 데이터(3~5초 영상)와 동일한 샘플 수를 유지해 train/inference 조건을 통일.
def extract_optical_flow(video_path: Path, out_dir: Path, n_frames: int = 30):
    """Extract Farneback optical flow frames from a video and save as images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    ret, prev_frame = cap.read()
    if not ret:
        print(f"[WARN] Cannot read: {video_path}")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    saved = 0

    while saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(prev_frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out_path = out_dir / f"{video_path.stem}_frame{saved:04d}.png"
        cv2.imwrite(str(out_path), rgb_flow)

        prev_gray = gray
        saved += 1

    cap.release()


def main():
    for label, input_dir in INPUT_DIRS.items():
        out_label_dir = OUTPUT_BASE / label
        videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
        print(f"[{label}] {len(videos)} videos found")
        for video in videos:
            extract_optical_flow(video, out_label_dir / video.stem)
    print("Done. Optical flow saved to", OUTPUT_BASE)


if __name__ == "__main__":
    main()
