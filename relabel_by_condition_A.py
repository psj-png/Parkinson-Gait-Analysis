import pandas as pd
import numpy as np
import os
import csv
from utils import calculate_angle, remove_outliers, apply_moving_average, apply_butterworth_filter

# 1. 데이터 로드 및 강력 텍스트 기반 클리닝
raw_path = r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv'
cleaned_path = r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv'
temp_path = r'C:\Gait_Analysis\extracted_data\temp_cleaned.csv'

print("🔄 데이터 클리닝 및 3초 구간 추출 시작...")

if os.path.exists(cleaned_path):
    os.remove(cleaned_path)
    print("🗑️ 이전 클리닝 파일을 삭제했습니다.")

# [핵심] 텍스트 기반 필터링
with open(raw_path, 'r', encoding='utf-8-sig') as infile, open(temp_path, 'w', encoding='utf-8-sig',
                                                               newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        if '멈추는' not in row[0] and '제미나이' not in row[0]:
            writer.writerow(row)

df = pd.read_csv(temp_path)
os.remove(temp_path)
print(f"📂 원본 데이터 텍스트 필터링 완료 (총 행 수: {len(df)})")

df['video'] = df['video'].astype(str).str.strip()
df = df[df['video'].str.contains('.mp4', na=False)]
print(f"✨ .mp4 필터링 완료 (최종 남은 행 수: {len(df)})")


# 3초 구간 추출
def extract_3sec_window(group):
    total_frames = len(group)
    if total_frames <= 90:
        return group
    mid = total_frames // 2
    return group.iloc[mid - 45: mid + 45]


df = df.groupby('video', group_keys=True).apply(extract_3sec_window, include_groups=False).reset_index()

if 'level_1' in df.columns:
    df = df.drop(columns=['level_1'])
print(f"✅ 데이터 구조 복구 완료. 현재 데이터 행 수: {len(df)}")

# ==============================================================================
# [🔥핵심] 좌표 데이터 노이즈 제거 파이프라인 (Warning 해결)
# ==============================================================================
print("🧹 좌표 데이터 필터링 시작 (노이즈 제거)...")

coord_columns = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]


def filter_group(group):
    for col in coord_columns:
        if col in group.columns:
            cleaned = remove_outliers(group[col].values)
            smoothed = apply_moving_average(cleaned, window_size=5)
            final_data = apply_butterworth_filter(smoothed, cutoff=3.0, fs=30.0)
            group[col] = final_data
    return group


# [Warning 해결] include_groups=False 사용
df = df.groupby('video').apply(filter_group, include_groups=False).reset_index()
print("✅ 좌표 필터링 완료.")


# ==============================================================================

# 2. 지표 계산 함수 (Condition A용)
def get_indicators(row):
    try:
        shoulder = [row['j11_x'], row['j11_y']]
        hip = [row['j23_x'], row['j23_y']]
        knee = [row['j25_x'], row['j25_y']]
        ankle = [row['j27_x'], row['j27_y']]

        knee_angle = calculate_angle(hip, knee, ankle)
        trunk_lean = calculate_angle(shoulder, hip, [hip[0], 0])
        return pd.Series([knee_angle, trunk_lean])
    except Exception:
        return pd.Series([180.0, 0.0])


# 지표 추가 (PerformanceWarning 방지를 위해 데이터 프레임 재구성)
indicators = df.apply(get_indicators, axis=1)
df['knee_angle'] = indicators[0]
df['trunk_lean'] = indicators[1]

# 3. 영상 단위 라벨 결정 (Condition A)
video_stats = df.groupby('video').agg({
    'knee_angle': 'min',
    'trunk_lean': 'max'
}).reset_index()


def judge_video(row):
    if row['knee_angle'] < 150.0 or row['trunk_lean'] > 10.0:
        return '02_Parkinson'
    else:
        return '01_Normal'


video_stats['new_label'] = video_stats.apply(judge_video, axis=1)

# 4. 최종 라벨 매핑 및 저장 (파편화 완전 해소)
label_map = dict(zip(video_stats['video'], video_stats['new_label']))
df['label'] = df['video'].map(label_map)

# 필요한 컬럼만 선택하여 새로운 데이터프레임으로 재구성 (가장 확실한 방법)
needed_cols = ['video', 'frame', 'label', 'knee_angle', 'trunk_lean']
coord_cols = [col for col in df.columns if col.startswith('j')]
df = df[needed_cols + coord_cols].copy()

df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"✅ 모든 공정 완료!")
print(f"💾 저장 경로: {cleaned_path}")
print(f"📊 최종 영상 분포:\n{video_stats['new_label'].value_counts()}")