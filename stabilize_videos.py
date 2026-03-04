import cv2
import numpy as np
import pandas as pd
import os
import glob

# --- 설정 ---
mapping_file_path = r'C:\Gait_Analysis\video_mapping.xlsx'
raw_data_folders = [fr'C:\Gait_Analysis\gavd_data_{i}' for i in range(1, 6)]
output_folder = r'C:\Gait_Analysis\stabilized_videos'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. CLAHE 객체 생성 (전역 설정)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(frame):
    """프레임의 밝기 채널(Y)만 정규화하는 함수"""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def find_video(folder_list, target_name):
    for root_folder in folder_list:
        if not os.path.exists(root_folder): continue
        for ext in ['*.mp4', '*.MP4', '*.mkv', '*.avi']:
            found = glob.glob(os.path.join(root_folder, "**", f"{target_name}{ext}"), recursive=True)
            if found: return found[0]
    return None


# --- 안정화 + CLAHE 통합 함수 ---
def run_stabilization(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    ret, prev_frame = cap.read()
    if not ret: return

    # 첫 프레임부터 CLAHE 적용 후 저장
    prev_frame_balanced = apply_clahe(prev_frame)
    out.write(prev_frame_balanced)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. 안정화 로직 (움직임 계산)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_pts is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            idx = np.where(status == 1)[0]
            if len(idx) > 0:
                m, _ = cv2.estimateAffinePartial2D(prev_pts[idx], curr_pts[idx])
                if m is not None:
                    # 흔들림 보정
                    curr_frame = cv2.warpAffine(curr_frame, m, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # 2. 밝기/대비 정규화 적용 (저장 직전)
        final_frame = apply_clahe(curr_frame)

        out.write(final_frame)
        prev_gray = curr_gray

    cap.release()
    out.release()


# --- 실행부 (기존과 동일) ---
df = pd.read_excel(mapping_file_path)
print("🚀 [안정화 + 밝기 보정 시작] 처리를 시작합니다...\n")

for _, row in df.iterrows():
    target_csv = str(row['CSV파일명']).strip()
    raw_id = str(row['원본영상명']).strip()
    if not raw_id or "기타" in raw_id: continue

    input_path = find_video(raw_data_folders, raw_id)
    output_path = os.path.join(output_folder, f"{target_csv}_stabilized.mp4")

    if input_path:
        print(f"🎬 [처리중] {target_csv} <--- {os.path.basename(input_path)}")
        run_stabilization(input_path, output_path)
        print(f"    ✅ 저장 완료: {os.path.basename(output_path)}")
    else:
        print(f"❌ [미연결] {target_csv} (원본 {raw_id}를 찾을 수 없음)")

print("\n🎉 모든 영상의 전처리가 완료되었습니다!")
