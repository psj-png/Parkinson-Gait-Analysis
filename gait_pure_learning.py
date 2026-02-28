import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 정제된 데이터 로드 (Step 1에서 만든 파일)
df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv')

# 신뢰도(_v) 컬럼 및 불필요한 지표 제외 (좌표 데이터로만 학습)
coords_only = [col for col in df.columns if not col.endswith('_v')
               and col not in ['knee_angle', 'trunk_lean']]
df_pure = df[coords_only]

X = df_pure.drop(['label'], axis=1)
y = df_pure['label']

# 2. 데이터 분할 (Condition A로 라벨링된 'Actual'을 학습)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_features = X_train.drop(['video', 'frame'], axis=1)
test_features = X_test.drop(['video', 'frame'], axis=1)

# 3. 모델 학습 (가중치 전략 유지)
custom_weights = {'01_Normal': 1, '02_Parkinson': 15}
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight=custom_weights,
    random_state=42
)
model.fit(train_features, y_train)

# 4. 예측 및 확신도 계산
y_pred = model.predict(test_features)
y_probs = model.predict_proba(test_features)

# 5. 영상별 분석 리포트 생성
full_analysis = X_test[['video']].copy()
full_analysis['Actual'] = y_test.values         # Condition A가 정한 정답
full_analysis['Predicted'] = y_pred             # ML이 예측한 결과
full_analysis['Confidence'] = np.max(y_probs, axis=1)

# 6. 영상 단위 최종 요약 (Mode 판정)
video_summary = full_analysis.groupby('video').agg({
    'Actual': 'first',
    'Predicted': lambda x: x.mode()[0],
    'Confidence': 'mean'
}).reset_index()

# All-Pass 로직 적용: Actual이 02거나 Predicted가 02면 최종 Suspected
video_summary['Final_Status'] = video_summary.apply(
    lambda x: 'Suspected' if x['Actual'] == '02_Parkinson' or x['Predicted'] == '02_Parkinson' else 'Normal',
    axis=1
)

# 7. 결과 출력
print("\n" + "="*80)
print("📊 [Gait Hybrid Screening Report] (Condition A Labeling Applied)")
print("="*80)
print(video_summary.sort_values(by='Confidence', ascending=True).head(20))
print("-" * 80)

# 8. 저장
save_path = r'C:\Gait_Analysis\extracted_data\all_video_confidence.csv'
video_summary.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"💾 최종 리포트 저장 완료: {save_path}")
