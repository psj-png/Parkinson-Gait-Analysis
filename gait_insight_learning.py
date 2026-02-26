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

# 2. 특징(X)과 라벨(y) 분리
X = df.drop(['video', 'label', 'frame'], axis=1)
y = df['label']

# 3. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"🚀 분석 시작... (데이터 수: {len(X_train)}개)")

# 4. 모델 학습 (이해도를 위해 깊이를 제한)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. 경계선 분석 (Confusion Matrix)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='coolwarm', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Gait Class Boundary Analysis')
plt.show()

# 6. 핵심 지표 매핑 (Feature Importance)
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n" + "="*50)
print("🔍 AI가 선정한 핵심 판별 관절 (4대 지표 매칭용)")
print("="*50)
print(feat_imp.head(15))
print("-" * 50)

# 7. 모델 저장
joblib.dump(model, r'C:\Gait_Analysis\gait_insight_model.pkl')
print("\n💾 모델 저장 완료: gait_insight_model.pkl")