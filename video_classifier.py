import cv2
import os
import shutil
import numpy as np

# [최강 우회 전략]
try:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    print("✅ 방법 1: mp.solutions 로드 성공")
except Exception:
    try:
        from mediapipe.python.solutions import pose as mp_pose

        print("✅ 방법 2: mediapipe.python.solutions 로드 성공")
    except Exception as e:
        print(f"❌ 모든 방법 실패. 에러 내용: {e}")
        exit()

# 1. 설정
input_folder = "gavd_data_1"
output_base = "classified_videos"
categories = ["Pure_Lateral", "Valid_Oblique", "Frontal", "Error_Noise"]

for cat in categories:
    os.makedirs(os.path.join(output_base, cat), exist_ok=True)

# Pose 객체 생성
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    model_complexity=1
)


def get_view_angle(video_path):
    cap = cv2.VideoCapture(video_path)
    widths = []
    frame_count = 0

    while cap.isOpened() and frame_count < 45:
        success, image = cap.read()
        if not success: break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 어깨 너비 계산
            shoulder_width = abs(lm[11].x - lm[12].x)
            widths.append(shoulder_width)

        frame_count += 1

    cap.release()
    return np.mean(widths) if widths else None


# 2. 실행
print("🚀 시점 분류를 시작합니다...")
if not os.path.exists(input_folder):
    print(f"❌ 폴더 없음: {input_folder}")
else:
    files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video in files:
        path = os.path.join(input_folder, video)
        avg_w = get_view_angle(path)

        if avg_w is None:
            target = "Error_Noise"
        elif avg_w < 0.12:
            target = "Pure_Lateral"
        elif 0.12 <= avg_w < 0.30:
            target = "Valid_Oblique"
        elif 0.30 <= avg_w < 0.50:
            target = "Error_Noise"
        else:
            target = "Frontal"

        # [수정된 출력부]
        val_display = f"{avg_w:.3f}" if avg_w is not None else "N/A"
        print(f"🎬 {video} [{val_display}] -> {target}")
        shutil.copy(path, os.path.join(output_base, target, video))

print("\n✨ 모든 작업이 완료되었습니다! 'classified_videos' 폴더를 확인하세요.")