import joblib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    model = joblib.load('parkinson_gait_model.pkl')
    df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv')
    features = [f'j{i}_{ax}' for i in range(33) for ax in ['x', 'y', 'z']]
    
    for v_name, group in df.groupby('video'):
        prob = np.mean(model.predict_proba(group[features])[:, 1]) * 100
        print(f"영상: {v_name} | 위험도: {prob:.1f}% | 판정: {'🚨위험' if prob > 50 else '✅정상'}")