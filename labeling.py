import pandas as pd
import numpy as np
import os


def calculate_angle(a, b, c):
    """세 점을 이용해 관절 각도를 계산합니다."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


if __name__ == "__main__":
    input_path = r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv'
    output_path = r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv'

    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
    else:
        df = pd.read_csv(input_path)
        df = df[~df['video'].str.contains('멈추는|제미나이')].copy()

        print("🔄 4대 보행 지표(ROM, Lean, Symmetry, Rhythm) 산출 중...")


        # 1. 프레임별 각도 및 대칭성 계산
        def get_gait_features(row):
            # [기존] 왼쪽 무릎 각도
            left_k = calculate_angle([row.j23_x, row.j23_y], [row.j25_x, row.j25_y], [row.j27_x, row.j27_y])
            # [추가] 오른쪽 무릎 각도 (대칭성 판단용)
            right_k = calculate_angle([row.j24_x, row.j24_y], [row.j26_x, row.j26_y], [row.j28_x, row.j28_y])
            # [기존] 상체 기울기
            trunk = calculate_angle([row.j11_x, row.j11_y], [row.j23_x, row.j23_y], [row.j23_x, 0])

            # 보행 대칭성: 좌우 무릎 각도의 차이
            symmetry_gap = abs(left_k - right_k)

            return pd.Series([left_k, right_k, trunk, symmetry_gap])


        df[['left_knee', 'right_knee', 'trunk_lean', 'symmetry_gap']] = df.apply(get_gait_features, axis=1)


        # 2. 보행 리듬 규칙성 (Rhythm Variability) 계산
        # 영상별로 무릎 각도의 변화 주기가 얼마나 일정한지(변동성) 산출
        def calculate_rhythm(group):
            # 무릎 각도가 변하는 속도(차이)의 표준편차를 규칙성 지표로 사용
            rhythm_var = group['left_knee'].diff().std()
            group['rhythm_variability'] = rhythm_var if pd.notnull(rhythm_var) else 0
            return group


        df = df.groupby('video', group_keys=False).apply(calculate_rhythm)

        # 3. 하이브리드 진단을 위한 Condition A (Clinical 기준) 적용
        # 무릎 가동성 부족 OR 상체 숙임 OR 대칭성 붕괴 OR 리듬 불규칙 중 하나라도 해당하면 의심
        df['clinical_condition'] = (
                (df['left_knee'] < 150) |
                (df['trunk_lean'] > 10) |
                (df['symmetry_gap'] > 15) |
                (df['rhythm_variability'] > 5.0)
        ).astype(int)

        df.to_csv(output_path, index=False)
        print(f"✅ 4대 지표 반영 및 라벨링 완료: {output_path}")
