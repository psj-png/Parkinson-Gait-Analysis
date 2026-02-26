import cv2
import os
import shutil
import mediapipe as mp

# 1. MediaPipe 설정 (경량화 옵션 적용)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # 0으로 설정하면 정확도는 살짝 낮아지지만 속도가 비약적으로 빨라집니다.
    min_detection_confidence=0.5
)

# 2. 경로 설정
input_folder = "gavd_data_1"
output_valid = "Valid_Walk"
output_invalid = "No_Walk_Detected"

os.makedirs(output_valid, exist_ok=True)
os.makedirs(output_invalid, exist_ok=True)


def is_walking_fast(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    walking_frames = 0
    # 15프레임(약 0.5초)마다 한 번씩만 체크 (초고속 스캔)
    for i in range(0, total_frames, 15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        if not success: break

        # 이미지 크기를 절반으로 줄여서 분석 속도 향상
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 발목 가시성만 빠르게 체크
            if lm[27].visibility > 0.5 and lm[28].visibility > 0.5:
                walking_frames += 1

    cap.release()
    # 띄엄띄엄 봐서 4번 이상만 걸려도 "사람이 걷고 있다"고 판단
    return walking_frames >= 4


# 3. 메인 실행
print("⚡ 초고속 필터링 시작...")

files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

for video in files:
    path = os.path.join(input_folder, video)
    if is_walking_fast(path):
        print(f"✅ [OK] {video}")
        shutil.copy(path, os.path.join(output_valid, video))
    else:
        print(f"❌ [SKIP] {video}")
        shutil.copy(path, os.path.join(output_invalid, video))

print("\n✨ 작업 완료!")