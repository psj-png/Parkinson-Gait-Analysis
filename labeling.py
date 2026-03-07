import pandas as pd
import numpy as np
import os


def calculate_angle(a, b, c):
    """세 점을 이용해 관절 각도를 계산합니다."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    # 벡터 연산을 통한 각도 산출
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


if __name__ == "__main__":
    # 1. 2단계에서 추출한 통합 데이터 불러오기
    input_path = r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv'
    output_path = r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv'

    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
    else:
        df = pd.read_csv(input_path)

        # 2. 불필요한 영상 데이터 필터링 (제미나이, 멈추는 등 분석 방해 요소 제거)
        df = df[~df['video'].str.contains('멈추는|제미나이')].copy()


        # 3. 임상 지표(무릎 각도, 상체 기울기) 계산 함수
        def get_clinical(row):
            # 무릎 각도 (골반-무릎-발목)
            k = calculate_angle([row.j23_x, row.j23_y], [row.j25_x, row.j25_y], [row.j27_x, row.j27_y])
            # 상체 기울기 (어깨-골반-수직선)
            t = calculate_angle([row.j11_x, row.j11_y], [row.j23_x, row.j23_y], [row.j23_x, 0])
            return pd.Series([k, t])


        print("🔄 임상 지표(Knee Angle, Trunk Lean) 계산 중...")
        df[['knee_angle', 'trunk_lean']] = df.apply(get_clinical, axis=1)

        # 4. Condition A 반영: 영상별 자동 진단 및 라벨링
        # Future-proof 문법: include_groups=False를 사용하여 경고 제거
        print("⚖️ 진단 기준(Condition A)에 따른 라벨링 수행 중...")
        v_map = df.groupby('video').apply(
            lambda x: '02_Parkinson' if x.knee_angle.min() < 150 or x.trunk_lean.max() > 10 else '01_Normal',
            include_groups=False
        )

        df['label'] = df['video'].map(v_map)

        # 5. 최종 결과 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 라벨링 및 진단 완료! 파일 저장: {output_path}")
        print(f"📊 정상 영상: {sum(v_map == '01_Normal')}개, 파킨슨 의심 영상: {sum(v_map == '02_Parkinson')}개")