import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 로드 및 전처리
df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv')
coords_only = [col for col in df.columns if not col.endswith('_v')]
df_pure = df[coords_only]

X = df_pure.drop(['label'], axis=1)
y = df_pure['label']

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# [기존 3번 코드 위치]

# 3. 모델 학습 (가중치 설정 추가)
train_features = X_train.drop(['video', 'frame'], axis=1)
test_features = X_test.drop(['video', 'frame'], axis=1)

# --- 상준 님의 가중치 전략 반영 ---
custom_weights = {
    '01_Normal': 1,      # 정상: 기준
    '03_Ambiguous': 1,   # 모호함: 기준
    '02_Abnormal': 10    # 파킨슨: 놓치면 벌점 10배 (민감도 강화)
}

# class_weight 파라미터를 추가합니다.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight=custom_weights,  # 가중치 주입!
    random_state=None             # 테스트를 위해 랜덤 고정 해제
)
model.fit(train_features, y_train)


# 4. 전체 예측 및 확신도 계산
y_pred = model.predict(test_features)
y_probs = model.predict_proba(test_features)

# 5. 전체 영상 리스트 정리
full_analysis = X_test[['video']].copy()
full_analysis['Actual'] = y_test.values
full_analysis['Predicted'] = y_pred
full_analysis['Confidence'] = np.max(y_probs, axis=1)
full_analysis['Is_Correct'] = (full_analysis['Actual'] == full_analysis['Predicted'])

# 6. 영상별로 평균 확신도 요약
video_summary = full_analysis.groupby('video').agg({
    'Actual': 'first',
    'Predicted': 'first',
    'Confidence': 'mean',
    'Is_Correct': 'mean' # 1.0이면 모든 프레임 정답, 0.8이면 80% 정답
}).reset_index()

# 7. 결과 출력 (상위 20개)
print("\n" + "="*80)
print("📊 [전체 영상 확신도 리포트] 정답 여부와 Confidence 확인")
print("="*80)
print(video_summary.sort_values(by='Confidence', ascending=True).head(20)) # 낮은 확신도 순으로 정렬
print("-" * 80)

# 8. 저장
save_path = r'C:\Gait_Analysis\extracted_data\all_video_confidence.csv'
video_summary.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"💾 전체 리포트 저장 완료: {save_path}")