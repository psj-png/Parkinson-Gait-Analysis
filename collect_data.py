import os
import sys
import glob
import pandas as pd

# 1. MediaPipe 강제 경로 인식
lib_path = r"C:\Users\박상준\AppData\Local\Programs\Python\Python310\Lib\site-packages"
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# 2. 환경 설정 및 데이터 로드
base_path = r"C:\Parkinson_Gait_Analysis"
print(f"현재 작업 폴더: {base_path}")

# --- [수정 포인트 1] 파일 확장자 패턴 변경 ---
# .cpv.txt로 끝나는 모든 파일을 찾도록 수정했습니다.
data_files = glob.glob(os.path.join(base_path, "gavd_data_*.cpv.txt"))
print(f"발견된 데이터 파일: {[os.path.basename(f) for f in data_files]}")

full_data = []
for file in data_files:
    try:
        # --- [수정 포인트 2] 구분자 자동 감지 ---
        # sep=None, engine='python'을 쓰면 쉼표든 탭이든 알아서 맞춰서 읽어옵니다.
        df = pd.read_csv(file, sep=None, engine='python')
        full_data.append(df)
        print(f"로드 성공: {os.path.basename(file)} (데이터 모양: {df.shape})")
    except Exception as e:
        print(f"로드 실패: {os.path.basename(file)} (사유: {e})")

# 3. 데이터 통합 결과 확인
if full_data:
    combined_df = pd.concat(full_data, ignore_index=True)
    print("-" * 30)
    print(f"✅ 총 {len(combined_df)}행의 데이터를 하나로 합쳤습니다.")
    print(f"📊 컬럼 목록: {combined_df.columns.tolist()}")
    print("🚀 데이터 분석 준비 완료!")
else:
    print("-" * 30)
    print("❌ 로드할 데이터가 없습니다. 파일 이름을 다시 확인해주세요.")
    print(f"💡 팁: 폴더에 'gavd_data_1.cpv.txt' 파일이 있는지 확인하세요.")