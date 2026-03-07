import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv')
    X = df.drop(['label', 'video', 'frame', 'knee_angle', 'trunk_lean'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, class_weight={'01_Normal': 1, '02_Parkinson': 15}, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'parkinson_gait_model.pkl')
    print(f"모델 저장 완료 (정확도: {model.score(X_test, y_test)*100:.1f}%)")