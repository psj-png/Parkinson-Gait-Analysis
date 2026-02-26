import os
import sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
# [중요] utils.py에서 필터 함수들을 가져옵니다.
from utils import remove_outliers, apply_moving_average, apply_butterworth_filter


# ... (MediaPipe 로드 및 경로 설정 부분은 동일) ...

def run_extraction():
    all_rows = []
    target_categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    for category in target_categories:
        cat_path = os.path.join(DATA_DIR, category)
        videos = [f for f in os.listdir(cat_path) if f.lower().endswith(('.mp4', '.avi', '.gif', '.mov'))]

        for v_name in tqdm(videos):
            v_path = os.path.join(cat_path, v_name)
            cap = cv2.VideoCapture(v_path)

            # --- [수정] 영상 하나당 데이터를 임시로 모을 리스트 ---
            temp_video_data = []
            f_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    frame_data = {'video': v_name, 'label': category, 'frame': f_idx}
                    for i, lm in enumerate(res.pose_landmarks.landmark):
                        frame_data[f'j{i}_x'] = lm.x
                        frame_data[f'j{i}_y'] = lm.y
                        frame_data[f'j{i}_z'] = lm.z
                        frame_data[f'j{i}_v'] = lm.visibility
                    temp_video_data.append(frame_data)
                f_idx += 1
            cap.release()

            # --- [핵심: 3단계 노이즈 제거 적용] ---
            if len(temp_video_data) > 10:  # 최소 프레임 이상일 때만 필터링
                df_temp = pd.DataFrame(temp_video_data)

                # 모든 관절 좌표(x, y, z)에 대해 필터 적용
                for i in range(33):
                    for axis in ['x', 'y', 'z']:
                        col = f'j{i}_{axis}'
                        # 1단계: Outlier 제거 -> 2단계: Moving Average -> 3단계: Butterworth
                        data = df_temp[col].values
                        data = remove_outliers(data)
                        data = apply_moving_average(data)
                        # 버터워스는 데이터 길이가 충분할 때만 (순서 주의)
                        data = apply_butterworth_filter(data)

                        # 필터링된 데이터를 다시 프레임 수에 맞게 할당
                        # (필터 특성상 길이가 줄어들 수 있으므로 보간 처리 필요할 수 있음)
                        df_temp[col] = pd.Series(data).reindex(df_temp.index).interpolate().bfill()

                # 정제된 데이터를 전체 리스트에 통합
                all_rows.extend(df_temp.to_dict('records'))

    # 3. 결과 통합 저장
    if all_rows:
        df = pd.DataFrame(all_rows)
        save_file = os.path.join(OUTPUT_DIR, 'gait_integrated_data.csv')
        df.to_csv(save_file, index=False, encoding='utf-8-sig')
        print(f"\n✨ 전수 추출 완료! 파일 경로: {save_file}")
        print(f"📊 최종 데이터 행 수: {len(df)}")
    else:
        print("\n❌ 분석할 영상 파일을 찾지 못했습니다. data 폴더 구성을 확인해 주세요.")


if __name__ == "__main__":
    run_extraction()
