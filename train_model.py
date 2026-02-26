import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def generate_professional_data(samples=200):
    """상준 님의 분석 결과 수치를 반영한 학습 데이터 생성"""
    # 1. 정상군 (Normal): ROM이 크고(40~70), 상체가 곧으며(0~10), 변동성이 적당함
    normal_data = pd.DataFrame({
        'ROM': np.random.normal(55, 10, samples),
        'Lean': np.random.normal(5, 3, samples),
        'Swing': np.random.normal(0.85, 0.1, samples),
        'Height': np.random.normal(1.1, 0.2, samples),
        'Var': np.random.normal(15.0, 4.0, samples),
        'Label': 0
    })

    # 2. 파킨슨 위험군 (Patient): ROM이 매우 작거나(10~20), 상체가 쏠리고(20~35), 변동성이 낮음(서행)
    patient_data = pd.DataFrame({
        'ROM': np.random.normal(18, 5, samples),
        'Lean': np.random.normal(25, 5, samples),
        'Swing': np.random.normal(0.85, 0.1, samples),  # 팔흔들림은 이번 데이터에서 유사했으므로 유지
        'Height': np.random.normal(1.3, 0.2, samples),
        'Var': np.random.normal(5.0, 2.0, samples),
        'Label': 1
    })

    return pd.concat([normal_data, patient_data]).sample(frac=1).reset_index(drop=True)


def train_new_model():
    df = generate_professional_data()
    X = df.drop('Label', axis=1)
    y = df['Label']

    # 모델 학습 (데이터 특성을 더 잘 파악하도록 깊이 조정)
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X, y)

    save_path = r'C:\Users\박상준\PycharmProjects\Parkinson_Gait_Analysis\parkinson_model.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    joblib.dump(model, save_path)
    print(f"✅ 실제 데이터 경향이 반영된 모델 저장 완료: {save_path}")


if __name__ == "__main__":
    train_new_model()