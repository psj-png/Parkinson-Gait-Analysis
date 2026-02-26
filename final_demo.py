import cv2
import mediapipe as mp
import joblib
import pandas as pd
import numpy as np

# 1. 환경 설정 및 모델 로드
model = joblib.load('parkinson_model.pkl')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 분석할 영상 경로와 결과 저장 경로
video_path = 'test_video.mp4'  # 분석하고 싶은 새로운 영상 파일명
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('final_diagnosis_result.mp4', fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))


def calculate_angle(a, b, c):
    a = np.array(a);
    b = np.array(b);
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


print("시각화 분석 시작... 잠시만 기다려 주세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 프레임 처리 (MediaPipe)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # 1. 좌표 추출 (예시로 ROM과 Lean만 실시간 계산 표시)
        lm = results.pose_landmarks.landmark
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        current_rom = calculate_angle(hip, knee, ankle)
        current_lean = abs(shoulder[0] - hip[0]) * 100  # 단순화된 기울기 계산 예시

        # 2. 모델 예측 (실제로는 전체 구간 평균값을 넣어야 정확하지만, 데모를 위해 실시간 값 투입)
        # 실제 발표 시에는 미리 계산된 해당 영상의 평균 features를 넣는 것이 좋습니다.
        dummy_features = [current_rom, current_lean, 0.8, 0.1, 1.2]  # 임시 피처
        prob = model.predict_proba([dummy_features])[0][1] * 100

        # 3. 영상 위에 뼈대 그리기
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 4. 정보 오버레이 (대시보드)
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Real-time ROM: {current_rom:.1f} deg", (20, 40), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Real-time Lean: {current_lean:.1f}", (20, 70), font, 0.7, (255, 255, 255), 2)

        # 위험도에 따른 색상 변경 (초록 -> 빨강)
        color = (0, 0, 255) if prob > 50 else (0, 255, 0)
        cv2.putText(frame, f"Parkinson Risk: {prob:.1f}%", (20, 120), font, 1, color, 3)

    out.write(frame)
    cv2.imshow('Parkinson AI Diagnosis Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("분석 완료! 'final_diagnosis_result.mp4' 파일이 생성되었습니다.")