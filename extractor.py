import cv2
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# --- MediaPipe 안전 로드 ---
try:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
except (AttributeError, ModuleNotFoundError):
    from mediapipe.python.solutions import pose as mp_pose


# --- 필터 함수 (데이터 정제) ---
def remove_outliers(data, threshold=3.0):
    mean, std = np.mean(data), np.std(data)
    return np.where(np.abs(data - mean) > threshold * std, mean, data) if std > 0 else data


def apply_moving_average(data, window=5):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window) / window, mode='same')


def apply_butterworth_filter(data, cutoff=3.0, fs=30, order=2):
    if len(data) <= order * 3: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# --- 경로 설정 (상준 님의 폴더 구조에 맞춤) ---
INPUT_DIR = r'C:\Gait_Analysis\data'  # 원본 영상이 있는 곳
OUTPUT_DIR = r'C:\Gait_Analysis\extracted_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. 하위 폴더를 포함한 모든 mp4 파일 찾기 (preprocess와 동일한 로직)
    video_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.mp4"), recursive=True)

    if not video_files:
        print(f"❌ 분석할 영상을 찾을 수 없습니다: {INPUT_DIR}")
    else:
        print(f"📂 총 {len(video_files)}개의 영상을 분석하여 데이터를 추출합니다.")

        all_rows = []
        pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

        for v_path in tqdm(video_files, desc="Processing Videos"):
            v_name = os.path.basename(v_path)
            # 폴더명(01_Normal 등)을 라벨로 활용
            parent_folder = os.path.basename(os.path.dirname(v_path))
            label = parent_folder if parent_folder.startswith('0') else 'Unknown'

            cap = cv2.VideoCapture(v_path)
            temp_data = []
            f_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 이미지 변환 및 처리
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(frame_rgb)

                if res.pose_landmarks:
                    row = {'label': label, 'video': v_name, 'frame': f_idx}
                    for i, lm in enumerate(res.pose_landmarks.landmark):
                        row[f'j{i}_x'], row[f'j{i}_y'], row[f'j{i}_z'] = lm.x, lm.y, lm.z
                    temp_data.append(row)
                f_idx += 1
            cap.release()

            # 데이터가 충분히 쌓였을 때만 필터링 적용 및 병합
            if len(temp_data) > 15:  # 최소 프레임 수 기준
                df_t = pd.DataFrame(temp_data)
                # 좌표 컬럼만 추출 (_x, _y, _z)
                coord_cols = [c for c in df_t.columns if any(ax in c for ax in ['_x', '_y', '_z'])]

                for col in coord_cols:
                    # 3중 필터 적용: 이상치 제거 -> 이동평균 -> 버터워스 필터
                    raw_values = df_t[col].values
                    cleaned = remove_outliers(raw_values)
                    smoothed = apply_moving_average(cleaned)
                    filtered = apply_butterworth_filter(smoothed)
                    df_t[col] = filtered

                all_rows.extend(df_t.to_dict('records'))

        # 2. 최종 결과 저장
        if all_rows:
            final_df = pd.DataFrame(all_rows)
            output_path = os.path.join(OUTPUT_DIR, 'gait_integrated_data.csv')
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ 데이터 추출 완료! 파일 저장 위치: {output_path}")
        else:
            print("\n❌ 추출된 데이터가 없습니다.")