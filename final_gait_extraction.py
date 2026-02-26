import os
import sys
import cv2
import pandas as pd
from tqdm import tqdm

# [보안책] 경로 꼬임 방지
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

# 1. MediaPipe 로드
try:
    import mediapipe as mp

    try:
        mp_pose = mp.solutions.pose
    except AttributeError:
        from mediapipe.python.solutions import pose as mp_pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        model_complexity=1
    )
    print("✅ [성공] MediaPipe 관절 모델이 준비되었습니다.")

except Exception as e:
    print(f"❌ [치명적 오류] 라이브러리 로드 실패: {e}")
    sys.exit()

# 2. 경로 설정
BASE_PATH = r'C:\Gait_Analysis'
DATA_DIR = os.path.join(BASE_PATH, 'data')
OUTPUT_DIR = os.path.join(BASE_PATH, 'extracted_data')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run_extraction():
    all_rows = []

    # [수정] data 폴더 내의 모든 하위 폴더를 자동으로 탐색합니다.
    target_categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    print(f"📂 분석 대상 폴더 발견: {target_categories}")

    for category in target_categories:
        cat_path = os.path.join(DATA_DIR, category)
        # 지원하는 모든 영상 및 GIF 확장자 포함
        videos = [f for f in os.listdir(cat_path) if f.lower().endswith(('.mp4', '.avi', '.gif', '.mov'))]

        if not videos:
            continue

        print(f"\n🎬 [{category}] 작업 시작 (총 {len(videos)}개 파일)")

        for v_name in tqdm(videos):
            v_path = os.path.join(cat_path, v_name)
            cap = cv2.VideoCapture(v_path)
            f_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 이미지 처리
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    # 기본 정보 (영상명, 라벨, 프레임 번호)
                    data = {'video': v_name, 'label': category, 'frame': f_idx}

                    # 33개 관절의 x, y, z, 신뢰도(v) 추출
                    for i, lm in enumerate(res.pose_landmarks.landmark):
                        data[f'j{i}_x'] = lm.x
                        data[f'j{i}_y'] = lm.y
                        data[f'j{i}_z'] = lm.z
                        data[f'j{i}_v'] = lm.visibility  # AI 인식 신뢰도 포함

                    all_rows.append(data)
                f_idx += 1
            cap.release()

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