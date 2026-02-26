import sys
import os

# 1. 경로 강제 지정 (상준 님의 파이썬 3.10 주소)
lib_path = r"C:\Users\박상준\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

try:
    import cv2
    import mediapipe as mp
    import pandas as pd

    # 그리기 도구 로드 (영상 제작의 핵심 부품!)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    print("✅ [성공] MediaPipe 영상 그리기 도구 로드 완료!")

    # 2. 데이터 확인
    data_path = r"C:\Parkinson_Gait_Analysis"
    files = [f for f in os.listdir(data_path) if f.startswith("gavd_data") and f.endswith(".txt")]

    if files:
        print(f"✅ [성공] {len(files)}개의 설계도(메모장)를 찾았습니다.")
        print(f"📄 첫 번째 설계도: {files[0]}")
    else:
        print("❌ [실패] 메모장 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

except Exception as e:
    print(f"❌ [에러 발생] 원인: {e}")