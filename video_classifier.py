import cv2
import os
import shutil
import numpy as np
from scipy.signal import butter, filtfilt  # 버터워스 필터용

# ... (MediaPipe 로드 부분은 상준 님 기존 코드와 동일) ...

# --- [신규 추가: 3단계 노이즈 제거 함수들] ---

def moving_average(data, window_size=5):
    """1단계: Jittering 제거를 위한 이동 평균 필터"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def remove_outliers(data, threshold=2.0):
    """2단계: 갑자기 튀는 값(Outlier) 제거 및 보간"""
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    # 평균에서 2표준편차 이상 벗어나면 아웃라이어로 간주
    is_outlier = np.abs(data - mean) > threshold * std
    data[is_outlier] = mean  # 간단하게 평균값으로 대체 (보간)
    return data

def butter_lowpass_filter(data, cutoff=3.0, fs=30, order=2):
    """3단계: 데이터 스무딩을 위한 버터워스 저주파 필터"""
    if len(data) <= order * 3: return data # 데이터가 너무 짧으면 패스
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ----------------------------------------------

def get_view_angle(video_path):
    cap = cv2.VideoCapture(video_path)
    raw_widths = []  # 날것의 데이터
    frame_count = 0

    while cap.isOpened() and frame_count < 60: # 노이즈 제거를 위해 프레임을 조금 더 확보(45->60)
        success, image = cap.read()
        if not success: break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 11: 왼쪽 어깨, 12: 오른쪽 어깨
            shoulder_width = abs(lm[11].x - lm[12].x)
            raw_widths.append(shoulder_width)
        frame_count += 1
    cap.release()

    if not raw_widths: return None

    # --- [데이터 정제 프로세스 적용] ---
    # 1. 아웃라이어 제거
    clean_data = remove_outliers(raw_widths)
    # 2. 이동 평균 적용 (지터링 감소)
    ma_data = moving_average(clean_data, window_size=5)
    # 3. 버터워스 필터 적용 (최종 스무딩)
    final_data = butter_lowpass_filter(ma_data)

    return np.mean(final_data) 

# ... (이하 분류 로직 및 출력부는 상준 님 기존 코드와 동일) ...
