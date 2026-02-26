import sys
import os
import cv2

# 1. 상준 님의 실제 파이썬 패키지 경로로 강제 고정
LIB_PATH = r"C:\Users\박상준\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

try:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe 라이브러리 로드 성공!")
except Exception as e:
    print(f"❌ 라이브러리 로드 에러: {e}")
    sys.exit()

# 2. 분석 도구 설정 (컴퓨터 사양을 고려해 가볍게 설정)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # 0: 빠름, 1: 보통, 2: 정확함
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. [핵심 수정] 영상 파일 대신 웹캠(0번) 사용
# 만약 외장 카메라를 쓰신다면 1이나 2로 바꿀 수도 있습니다.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라(웹캠)를 찾을 수 없습니다. 카메라가 연결되어 있는지 확인해 주세요.")
    sys.exit()

print("🎬 실시간 분석 시작! 화면 앞에서 걸어보거나 움직여보세요.")
print("👉 종료하려면 영상 창을 클릭하고 키보드의 'q'를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전 (거울처럼 보이게 함)
    frame = cv2.flip(frame, 1)

    # 처리 속도를 위해 해상도 조절
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 관절 인식 수행
    results = pose.process(frame_rgb)

    # 결과 그리기 (가상 관절 시각화)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    # 화면에 텍스트 표시
    cv2.putText(frame, "Real-time Gait Analysis", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Parkinson Gait Analysis - Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()