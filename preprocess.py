import cv2
import os
import glob
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


def preprocess_all_videos(base_dir, output_dir):
    # 1. 모든 하위 폴더에서 mp4 파일 목록 가져오기
    # data/**/*.mp4 는 하위 폴더를 모두 뒤진다는 뜻
    video_files = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)

    if not video_files:
        print(f"❌ 영상을 하나도 찾을 수 없습니다. 경로를 확인하세요: {base_dir}")
        return

    print(f"📂 총 {len(video_files)}개의 영상을 찾았습니다. 분석을 시작합니다.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for v_path in video_files:
            print(f"🎬 분석 중: {os.path.basename(v_path)}")
            cap = cv2.VideoCapture(v_path)

            while cap.isOpened():
                success, image = cap.read()
                if not success: break

                # MediaPipe 처리
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # 화면에 관절 포인트 그리기
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Gait Analysis - Phase 1', image)

                # 'q' 누르면 다음 영상으로 넘어가기, 'ESC'는 전체 종료
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('q'):  # q
                    break

            cap.release()
            print(f"✅ {os.path.basename(v_path)} 처리 완료")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 폴더 구조에 맞춘 설정
    DATA_DIR = "data"
    OUTPUT_DIR = "processed_data"

    preprocess_all_videos(DATA_DIR, OUTPUT_DIR)