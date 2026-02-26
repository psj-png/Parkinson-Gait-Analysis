import joblib
import pandas as pd

# 1. 저장된 모델 불러오기
# (학습 시 scaler를 사용했다면 scaler.pkl도 함께 로드해야 합니다)
model = joblib.load('parkinson_model.pkl')


def run_diagnosis(video_name, features):
    # features: [ROM, Lean, Swing, Height, Var] 형태의 리스트
    cols = ['ROM', 'Lean', 'Swing', 'Height', 'Var']
    df_input = pd.DataFrame([features], columns=cols)

    # 2. 파킨슨 확률 예측
    # [정상 확률, 환자 확률]
    probabilities = model.predict_proba(df_input)[0]
    parkinson_risk = probabilities[1] * 100

    print(f"\n" + "=" * 40)
    print(f"       [ 파킨슨 보행 분석 결과 보고서 ]")
    print(f" 대상 영상: {video_name}")
    print("=" * 40)
    print(f" ■ 파킨슨병 위험도: {parkinson_risk:.1f}%")
    print("-" * 40)
    print(f" [ 주요 지표 데이터 ]")
    print(f" 1. 무릎 가동 범위(ROM): {features[0]:.2f}°")
    print(f" 2. 상체 기울기(Lean): {features[1]:.2f}°")
    print(f" 3. 보행 변동성(Var): {features[4]:.2f}")
    print("-" * 40)

    # 3. 위험도에 따른 코멘트
    if parkinson_risk >= 70:
        print(" [판정] 고위험 (High Risk)")
        print(" -> 전형적인 파킨슨 보행 패턴이 강하게 감지됩니다.")
    elif parkinson_risk >= 40:
        print(" [판정] 주의 (Caution)")
        print(" -> 보행 지표 중 일부가 정상 범위를 벗어나 있습니다.")
    else:
        print(" [판정] 정상 (Normal)")
        print(" -> 안정적인 보행 패턴을 유지하고 있습니다.")
    print("=" * 40)


# 실행 예시 (나중에 main.py의 결과를 여기로 전달하게 만듭니다)
# 예: 환자 데이터 (ROM 낮음, Lean 높음)
test_features = [15.5, 25.3, 0.8, 0.1, 1.5]
run_diagnosis("test_sample_01.mp4", test_features)