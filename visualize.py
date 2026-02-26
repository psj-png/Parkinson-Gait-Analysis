import sys

# 1. 라이브러리 경로 강제 삽입 (가장 먼저 수행)
# 상준 님의 실제 파이썬 패키지 경로입니다.
LIB_PATH = r"C:\Users\박상준\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

# 그 다음 라이브러리들을 부릅니다.
try:
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print("✅ 모든 라이브러리(os, numpy 등) 로드 성공!")
except ImportError as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    print(f"💡 팁: {LIB_PATH} 경로에 해당 라이브러리가 있는지 확인하세요.")
    sys.exit()

# 2. 데이터 로드 설정
base_path = r'C:\Parkinson_Gait_Analysis'
csv_name = 'report.csv'
csv_path = os.path.join(base_path, csv_name)

try:
    df = pd.read_csv(csv_path)
    # 컬럼명이 'Trunk_Lean', 'Knee_Angle'인지 확인 (없으면 첫 번째, 두 번째 컬럼 사용)
    cols = df.columns
    x_col = 'Knee_Angle' if 'Knee_Angle' in cols else cols[2]
    y_col = 'Trunk_Lean' if 'Trunk_Lean' in cols else cols[1]

    # 라벨이 없으면 기본값 생성
    if 'Label' not in df.columns:
        df['Label'] = 0
except Exception as e:
    print(f"❌ 데이터 로드 오류: {e}")
    sys.exit()

# 3. 시각화 (그래프 그리기)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=x_col, y=y_col, hue='Label', s=100, palette='viridis')

plt.title('보행 데이터 분석 시각화')
plt.xlabel(f'무릎 각도 ({x_col})')
plt.ylabel(f'상체 기울기 ({y_col})')
plt.grid(True)

# 결과 저장
save_path = os.path.join(base_path, 'gait_analysis_graph.png')
plt.savefig(save_path)
print(f"📈 그래프 저장 완료: {save_path}")
plt.show()