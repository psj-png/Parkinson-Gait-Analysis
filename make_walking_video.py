import sys
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# 경로 설정
lib_path = r"C:\Users\박상준\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

base_path = r"C:\Parkinson_Gait_Analysis"
output_dir = os.path.join(base_path, "output_videos")
os.makedirs(output_dir, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)


def create_video(file_path):
    filename = os.path.basename(file_path)
    print(f"🎬 {filename} 영상 생성 중...")

    # 영상 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, f"{filename}.mp4"), fourcc, 30.0, (640, 480))

    # 테스트를 위해 검은 배경에 300프레임 생성
    for i in range(300):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Test Rendering: {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"✅ {filename}.mp4 저장 완료!")


# .cpv 파일까지 인식하도록 수정
data_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if
              f.startswith("gavd_data") and (".txt" in f or ".cpv" in f)]
for file in data_files:
    create_video(file)