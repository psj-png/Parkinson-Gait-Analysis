import numpy as np
import mediapipe as mp
import cv2
from scipy.signal import butter, filtfilt # 버터워스 필터용 추가

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- [신규 추가: 3단계 노이즈 제거 모듈] ---

def remove_outliers(data, threshold=2.0):
    """1단계: 갑자기 튀는 값 제거 (Outlier Processing)"""
    data = np.array(data)
    if len(data) < 2: return data
    mean = np.mean(data)
    std = np.std(data)
    # 표준편차 기반으로 튀는 값을 평균값으로 대체
    data[np.abs(data - mean) > threshold * std] = mean
    return data

def apply_moving_average(data, window_size=5):
    """2단계: 지터링 제거 (Jittering Removal)"""
    if len(data) < window_size: return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def apply_butterworth_filter(data, cutoff=3.0, fs=30, order=2):
    """3단계: 데이터 스무딩 (Data Smoothing)"""
    if len(data) <= order * 3: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ------------------------------------------

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def get_normalized_indicators(frame):
    # (이 부분은 상준 님 기존 코드와 동일)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks: return None

    lm = results.pose_landmarks.landmark
    # ... (생략: 좌표 추출 로직) ...
    
    # [지표 계산]
    return {
        'knee_angle': calculate_angle(hip, knee, ankle),
        'trunk_lean': calculate_angle(shoulder, hip, [hip[0], 0]),
        'arm_swing': np.sqrt((shoulder[0]-wrist[0])**2 + (shoulder[1]-wrist[1])**2) / trunk_length,
        'step_height': abs(hip[1] - ankle[1]) / trunk_length,
        'landmarks': results.pose_landmarks
    }
