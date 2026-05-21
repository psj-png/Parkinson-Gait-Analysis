"""
Step 1b: Extract person-cropped optical flow from videos.

Uses MediaPipe Pose to detect the person bounding box across all frames,
computes one stable crop window per video, then runs Farneback optical
flow on the cropped region — removing background and normalising scale.

Input : data/01_Normal/, data/02_Abnormal/
Output: output/optical_flow_crop/{Normal,Abnormal}/<video_stem>/
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent

INPUT_DIRS = {
    "Normal":   _BASE / "data" / "01_Normal",
    "Abnormal": _BASE / "data" / "02_Abnormal",
}
OUTPUT_BASE = _BASE / "output" / "optical_flow_crop"

N_FRAMES = 30
PADDING  = 0.15   # bbox 상하좌우 패딩 비율


def get_stable_bbox(frames: list, h: int, w: int) -> tuple:
    """
    모든 프레임에서 MediaPipe Pose landmark를 검출하고
    union bounding box + padding을 반환한다.
    detection 실패 시 전체 프레임을 fallback으로 사용.
    """
    all_x, all_y = [], []

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:
        for frame in frames:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    if lm.visibility > 0.3:
                        all_x.append(lm.x * w)
                        all_y.append(lm.y * h)

    if not all_x:
        return 0, 0, w, h  # fallback

    bw = max(all_x) - min(all_x)
    bh = max(all_y) - min(all_y)

    x1 = max(0, int(min(all_x) - bw * PADDING))
    y1 = max(0, int(min(all_y) - bh * PADDING))
    x2 = min(w, int(max(all_x) + bw * PADDING))
    y2 = min(h, int(max(all_y) + bh * PADDING))

    return x1, y1, x2, y2


def extract_cropped_flow(video_path: Path, out_dir: Path, n: int = N_FRAMES):
    cap = cv2.VideoCapture(str(video_path))

    # n+1 프레임 읽기 (optical flow는 연속 두 프레임 필요)
    raw = []
    while len(raw) <= n:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(frame)
    cap.release()

    if len(raw) < 2:
        print(f"  [SKIP] 프레임 부족: {video_path.name}")
        return None

    h, w = raw[0].shape[:2]
    x1, y1, x2, y2 = get_stable_bbox(raw, h, w)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        print(f"  [WARN] bbox 너무 작음 ({x1},{y1},{x2},{y2}), 전체 프레임 사용")
        x1, y1, x2, y2 = 0, 0, w, h

    out_dir.mkdir(parents=True, exist_ok=True)
    prev_gray = cv2.cvtColor(raw[0][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    saved = 0

    for frame in raw[1:n + 1]:
        cropped = frame[y1:y2, x1:x2]
        gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv        = np.zeros_like(cropped)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow    = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(str(out_dir / f"{video_path.stem}_frame{saved:04d}.png"), rgb_flow)
        prev_gray = gray
        saved    += 1

    return (x1, y1, x2, y2)


def main():
    total_ok, total_fail = 0, 0

    for label, input_dir in INPUT_DIRS.items():
        videos = sorted(list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")))
        print(f"\n[{label}] {len(videos)} videos")

        for i, video in enumerate(videos, 1):
            out_dir = OUTPUT_BASE / label / video.stem
            bbox    = extract_cropped_flow(video, out_dir)

            if bbox:
                x1, y1, x2, y2 = bbox
                is_full = (x1 == 0 and y1 == 0)
                flag    = " (fallback)" if is_full else ""
                print(f"  [{i:02d}/{len(videos)}] {video.name}  "
                      f"crop=({x1},{y1})-({x2},{y2}){flag}")
                total_ok += 1
            else:
                total_fail += 1

    print(f"\n완료: {total_ok} 성공 / {total_fail} 실패")
    print(f"저장 위치: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
