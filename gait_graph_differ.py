import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv')

# 2. 분석 타겟: 왼쪽 무릎 y좌표
target_feature = 'j25_y'

# 3. 그룹별 데이터 분리
normal_data = df[df['label'] == '01_Normal']
ambiguous_data = df[df['label'] == '03_Ambiguous']

video_list = ['S15_Norm_S_01.mp4', 'S15_Norm_S_02.mp4', 'A11_Act_F_01.mp4']

# 4. 시각화
plt.figure(figsize=(15, 8))

# --- 정상(Normal) 그룹 시각화 ---
sns.lineplot(data=normal_data, x='frame', y=target_feature,
             color='gray', alpha=0.2, label='Normal Range')
normal_mean = normal_data.groupby('frame')[target_feature].mean()
plt.plot(normal_mean.index, normal_mean.values, color='black', linestyle='--', alpha=0.8, label='Normal Mean')

# --- 모호함(Ambiguous) 그룹 시각화 (새로 추가) ---
sns.lineplot(data=ambiguous_data, x='frame', y=target_feature,
             color='skyblue', alpha=0.3, label='Ambiguous Range')
ambiguous_mean = ambiguous_data.groupby('frame')[target_feature].mean()
plt.plot(ambiguous_mean.index, ambiguous_mean.values, color='blue', linestyle=':', alpha=0.8, label='Ambiguous Mean')

# --- 개별 문제 영상들 ---
colors = ['red', 'orange', 'green'] # A11은 초록색으로 변경해 가독성 높임
for vid, color in zip(video_list, colors):
    vid_data = df[df['video'] == vid]
    if not vid_data.empty:
        plt.plot(vid_data['frame'], vid_data[target_feature],
                 color=color, label=f'Target: {vid}', linewidth=2.5)

plt.title(f'Gait Analysis: Normal vs Ambiguous Boundary ({target_feature})', fontsize=15)
plt.xlabel('Frame Number')
plt.ylabel('Y-Coordinate Value')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # 범례를 밖으로 살짝 뺌
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()