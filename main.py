import cv2
import joblib
import numpy as np
import os
import mediapipe as mp
import utils
import pandas as pd
import warnings

# 불필요한 경고 메시지 끄기
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = r'C:\Users\박상준\PycharmProjects\Parkinson_Gait_Analysis\parkinson_model.pkl'
VIDEO_DIR = r'C:\test_video'

try:
    clf = joblib.load(MODEL_PATH)
    # 모델 학습 시 사용된 특징 이름 설정 (경고 방지)
    feature_names = ['ROM', 'Lean', 'Swing', 'Height', 'Var']
    print("✅ 최신 지표 모델 로드 완료")
except:
    clf = None
    print("❌ 모델 로드 실패")


def analyze_all_videos():
    # 대소문자 구분 없이 모든 영상 파일 찾기
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print(f"❌ {VIDEO_DIR} 폴더에 영상 파일이 없습니다.")
        return

    all_results = []
    print(f"🎬 총 {len(video_files)}개의 영상을 정밀 분석합니다...")

    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        raw_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (640, 480))
            info = utils.get_normalized_indicators(frame)

            if info:
                raw_data.append([info['knee_angle'], info['trunk_lean'], info['arm_swing'], info['step_height']])
                mp.solutions.drawing_utils.draw_landmarks(frame, info['landmarks'], mp.solutions.pose.POSE_CONNECTIONS)
                cv2.imshow('Parkinson Gait Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()

        if len(raw_data) > 10:
            data_arr = np.array(raw_data)
            clean_knee = utils.apply_moving_average(data_arr[:, 0])

            rom = np.max(clean_knee) - np.min(clean_knee)
            lean = np.mean(data_arr[:, 1])
            swing = np.mean(data_arr[:, 2])
            height = np.mean(data_arr[:, 3])
            var = np.std(clean_knee)

            prob = 0.0
            if clf:
                # 데이터프레임 형태로 전달하여 경고 제거 및 정확도 향상
                input_df = pd.DataFrame([[rom, lean, swing, height, var]], columns=feature_names)
                prob = clf.predict_proba(input_df)[0][0] * 100

            all_results.append({
                '파일명': video_name,
                'ROM': round(rom, 2),
                '기울기': round(lean, 2),
                '변동성': round(var, 2),
                '정상일치도(%)': round(prob, 2)
            })

    cv2.destroyAllWindows()

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 70)
        print("📊 [ 최종 분석 요약 리포트 ]")
        print(df.to_string(index=False))
        print("=" * 70)

        # CSV 저장
        report_path = os.path.join(VIDEO_DIR, 'final_report.csv')
        df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"✅ 결과가 저장되었습니다: {report_path}")


if __name__ == "__main__":
    analyze_all_videos()