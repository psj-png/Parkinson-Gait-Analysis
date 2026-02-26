import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 데이터 로드
df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv')

# 2. [핵심] 가짜 힌트(_v) 제거
# 컬럼 이름에 '_v'가 포함된 모든 열을 리스트에서 제외합니다.
coords_only = [col for col in df.columns if not col.endswith('_v')]
df_pure = df[pure_coords_cols] if 'pure_coords_cols' in locals() else df[coords_only]

# 3. 분석용 특징(X)과 정답(y) 분리
# 영상이름, 라벨, 프레임 번호를 제외한 순수 x, y, z 좌표만 X에 넣습니다.
X = df_pure.drop(['video', 'label', 'frame'], axis=1)
y = df_pure['label']

print(f"✅ 가짜 힌트(_v) 제거 완료! (총 {len(X.columns)}개의 순수 좌표로 학습)")

# 4. 데이터 분할 (학습용 80%, 확인용 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. 모델 학습 (상준 님의 의도대로 경계 분석을 위해 깊이 제한)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 6. AI가 선정한 진짜 핵심 관절 순위 (Top 15)
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\n" + "="*60)
print("🎯 [진짜 성적표] AI가 순수 좌표만 보고 판단한 중요도")
print("="*60)
print(feat_imp.head(15))
print("-" * 60)

# 7. 모델 저장
joblib.dump(model, r'C:\Gait_Analysis\gait_pure_model.pkl')
print("\n💾 순수 좌표 기반 모델 저장 완료!")