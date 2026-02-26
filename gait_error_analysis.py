import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 로드
df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv')

# 2. 전처리 (신뢰도 데이터 제거)
v_cols = [c for c in df.columns if c.endswith('_v')]
df_pure = df.drop(v_cols, axis=1)

# 3. 영상 이름을 보존하기 위해 데이터 분할 시 인덱스를 유지
X = df_pure.drop(['label'], axis=1) # video, frame 포함해서 분할
y = df_pure['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 모델 학습 (분석용으로 video, frame 열은 제외하고 학습)
train_features = X_train.drop(['video', 'frame'], axis=1)
test_features = X_test.drop(['video', 'frame'], axis=1)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(train_features, y_train)

# 5. 예측 및 결과 분석
y_pred = model.predict(test_features)
y_probs = model.predict_proba(test_features) # 예측 확신도(확률)

# 6. 틀린 영상 찾기
analysis_df = X_test[['video', 'frame']].copy()
analysis_df['Actual'] = y_test.values
analysis_df['Predicted'] = y_pred
analysis_df['Confidence'] = np.max(y_probs, axis=1)

# 실제와 예측이 다른 데이터만 필터링
errors = analysis_df[analysis_df['Actual'] != analysis_df['Predicted']]

# 영상별로 어떤 프레임에서 주로 틀렸는지 정리
error_summary = errors.groupby('video').agg({
    'Actual': 'first',
    'Predicted': 'first',
    'frame': 'count',
    'Confidence': 'mean'
}).rename(columns={'frame': 'Error_Frames'}).reset_index()

print("\n" + "="*70)
print("❌ [오답 리포트] AI가 판단을 틀린 영상 리스트")
print("="*70)
if error_summary.empty:
    print("현재 모델이 모든 테스트 데이터를 맞혔습니다. (경계가 너무 뚜렷함)")
else:
    print(error_summary.sort_values(by='Error_Frames', ascending=False).head(20))
print("-" * 70)

print("\n💡 팁: 'Error_Frames'가 많은 영상일수록 AI가 보기에 아주 헷갈리는 영상입니다.")
print("해당 영상을 C:\Gait_Analysis\data 폴더에서 찾아 '카메라 각도'를 확인해 보세요.")