import scipy.io
import os

# 1. 상준 님의 파일 리스트 중 확인하고 싶은 파일명을 입력하세요.
# 사진을 보니 확장자가 .mat일 것으로 추정됩니다.
file_name = 'feat1.mat'

# 2. 경로 설정
feat_path = os.path.join(r'C:\Gait_Analysis\Simulation_Data', file_name)

try:
    # 파일 불러오기
    feat_data = scipy.io.loadmat(feat_path)

    # 시스템 헤더를 제외한 실제 데이터 키(Key) 추출
    keys = [k for k in feat_data.keys() if not k.startswith('__')]

    print("=" * 50)
    print(f"✅ [{file_name}] 분석 성공!")
    print(f"📋 포함된 데이터 항목(Keys): {keys}")
    print("=" * 50)

    for key in keys:
        content = feat_data[key]
        print(f"🔹 항목명: {key} | 데이터 크기(Shape): {getattr(content, 'shape', 'N/A')}")

        # 데이터가 비어있지 않다면 첫 번째 샘플 출력
        if hasattr(content, '__len__') and len(content) > 0:
            print(f"   - 데이터 샘플: {content[0]}")
        print("-" * 30)

except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다. 경로와 확장자를 확인해주세요.")
    print(f"현재 시도 경로: {feat_path}")
except Exception as e:
    print(f"❌ 에러 발생: {e}")