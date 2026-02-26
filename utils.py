import numpy as np
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def apply_moving_average(data, window_size=5):
    if len(data) < window_size: return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_normalized_indicators(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks: return None

    lm = results.pose_landmarks.landmark
    shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
    knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
    ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]

    trunk_length = np.sqrt((shoulder[0]-hip[0])**2 + (shoulder[1]-hip[1])**2)
    if trunk_length < 0.05: return None

    return {
        'knee_angle': calculate_angle(hip, knee, ankle),
        'trunk_lean': calculate_angle(shoulder, hip, [hip[0], 0]),
        'arm_swing': np.sqrt((shoulder[0]-wrist[0])**2 + (shoulder[1]-wrist[1])**2) / trunk_length,
        'step_height': abs(hip[1] - ankle[1]) / trunk_length,
        'landmarks': results.pose_landmarks
    }