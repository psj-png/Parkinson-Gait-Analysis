import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. 데이터 로드
df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_integrated_data.csv')

# 2. [핵심] 신뢰도(_v) 컬럼 및 불필요한 정보 제거
# x, y, z 좌표만 남기고 _v가 들어간 모든 열을 삭제합니다.
v_cols = [c for c in df.columns if c.endswith('_v')]
X = df.drop(['video', 'label', 'frame'] + v_cols, axis=1)
y = df['label']

print(f"✅ 신뢰도 데이터 제거 완료. 현재 특징 수: {X.shape[1]}개")

# 3. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 모델 학습
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. 경계선 분석 (Confusion Matrix)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='YlGnBu', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Pure Coordinate Gait Boundary Analysis')
plt.show()

# 6. 진짜 핵심 지표 확인 (Feature Importance)
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n" + "="*50)
print("🎯 AI가 좌표만 보고 선정한 핵심 판별 지표 (Top 15)")
print("="*50)
print(feat_imp.head(15))
print("-" * 50)

# 7. 모델 저장
joblib.dump(model, r'C:\Gait_Analysis\gait_pure_model.pkl')
print("\n💾 순수 좌표 모델 저장 완료!")