import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm

# 1. 설정 및 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

base_path = r'C:\Gait_Analysis\data'
output_path = r'C:\Gait_Analysis\extracted_data'
os.makedirs(output_path, exist_ok=True)

all_data = []

# 2. 데이터 폴더 탐색 (01_Normal, 02_Parkinson, 03_Hemiplegic 등 전체)
categories = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

print(f"📂 분석 대상 폴더: {categories}")

for category in categories:
    cat_path = os.path.join(base_path, category)
    # 지원하는 영상 확장자 전체 체크
    video_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.mp4', '.avi', '.gif', '.mov'))]

    print(f"\n🎬 [{category}] 폴더 분석 중... (파일 {len(video_files)}개)")

    for video_name in tqdm(video_files):
        video_path = os.path.join(cat_path, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # MediaPipe 처리
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # 33개 관절 데이터 추출
                landmarks = results.pose_landmarks.landmark
                row = [category, video_name, frame_idx]  # 라벨, 파일명, 프레임번호 저장

                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                all_data.append(row)

            frame_idx += 1
        cap.release()

# 3. CSV 저장
# 컬럼명 생성 (Label, FileName, FrameIdx, x0, y0, z0, v0, x1, y1...)
columns = ['label', 'file_name', 'frame_idx']
for i in range(33):
    columns.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

df = pd.DataFrame(all_data, columns=columns)
output_file = os.path.join(output_path, 'total_gait_data.csv')
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✨ 전수 조사 완료! 파일 저장됨: {output_file}")
print(f"📊 총 추출된 데이터 행 수: {len(df)}")