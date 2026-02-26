import os
import glob
import yt_dlp
import random

# 1. 설정
cpv_files = glob.glob("gavd_data_*.cpv.txt")
ADDITIONAL_LIMIT = 10
# [추가] 이미 실패한 ID를 저장할 세트 (메모리 절약)
failed_ids = set(["jzkn287X-84"])

if not cpv_files:
    print("❌ 리스트(.txt) 파일을 찾을 수 없습니다.")
else:
    for cpv in sorted(cpv_files):
        folder_name = cpv.replace('.cpv.txt', '')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        print(f"\n📂 [{folder_name}] 분석 및 랜덤 탐색 시작...")
        existing_files = [f.split('.')[0] for f in os.listdir(folder_name) if f.endswith('.mp4')]

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(folder_name, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            with open(cpv, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[1:]

            # [핵심] 리스트를 랜덤하게 섞어서 삭제된 구간을 탈출합니다.
            random.shuffle(lines)

            new_download_count = 0
            for line in lines:
                if new_download_count >= ADDITIONAL_LIMIT:
                    break

                parts = line.strip().split(',')
                if not parts: continue
                youtube_url = parts[-1].strip()

                if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                    video_id = youtube_url.split('v=')[-1].split('&')[0]

                    # 이미 있거나 이미 실패한 적이 있다면 스킵
                    if video_id in existing_files or video_id in failed_ids:
                        continue

                    try:
                        print(f"🎬 탐색 중 (성공:{new_download_count}/{ADDITIONAL_LIMIT}): {video_id}")
                        ydl.download([youtube_url])

                        if os.path.exists(os.path.join(folder_name, f"{video_id}.mp4")):
                            print(f"✅ [대박] 살아있는 영상 발견! 다운로드 완료.")
                            new_download_count += 1
                            existing_files.append(video_id)
                    except Exception:
                        print(f"⏭️ {video_id}는 없는 영상입니다. 블랙리스트 추가.")
                        failed_ids.add(video_id)  # 다시는 시도 안 함
                        continue

print(f"\n✨ 모든 작업 완료!")