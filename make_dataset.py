import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# 1. 경로 설정 (상준 님이 요청하신 경로로 변경)
source_folder = r'C:\Gait_Analysis\Simulation_Data'
save_folder = r'C:\Gait_Analysis\data\02_Parkinson'

# 폴더가 없으면 자동으로 생성
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"📂 폴더를 새로 생성했습니다: {save_folder}")

# 2. 파일 리스트 가져오기
all_files = [f for f in os.listdir(source_folder) if f.endswith('.mat') and 'dataset' in f]
all_files.sort()
target_files = all_files[:10]

skip = 5
duration_frames = 60


def create_jointed_video(file_name, view_type):
    full_path = os.path.join(source_folder, file_name)
    key = file_name.replace('.mat', '')

    try:
        data = scipy.io.loadmat(full_path)[key]
    except:
        return

    start_pt = len(data) // 2
    angles = data[start_pt: start_pt + (duration_frames * skip): skip, 3]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1.2, 1.2);
    ax.set_ylim(-1.5, 1.2)
    ax.axis('off')

    # 신체 비율 설정
    UPPER_ARM = 0.25;
    LOWER_ARM = 0.25
    THIGH = 0.5;
    CALF = 0.45

    head = plt.Circle((0, 0), 0.08, color='black')
    ax.add_patch(head)
    body, = ax.plot([], [], 'k-', lw=8, solid_capstyle='round')
    arm_l, = ax.plot([], [], 'b-o', lw=3, ms=5)
    arm_r, = ax.plot([], [], 'r-o', lw=3, ms=5)
    leg_l, = ax.plot([], [], 'b-o', lw=6, ms=6)
    leg_r, = ax.plot([], [], 'r-o', lw=6, ms=6)

    def update(i):
        # 각도 데이터 정규화 및 파킨슨 특유의 각도 범위 적용
        t = np.radians(np.interp(angles[i], (angles.min(), angles.max()), (-22, 22)))
        view_mod = {'front': 0.1, 'back': 0.1, 'right': 1.0, 'left': -1.0}[view_type]

        # 상체 경사
        stoop = np.radians(18) + (t * 0.05)
        nx, ny = 0.45 * np.sin(stoop) * view_mod, 0.45 * np.cos(stoop)
        head.center = (nx + (0.05 * view_mod), ny + 0.1)
        body.set_data([0, nx], [0, ny])

        # 팔 (관절 분리)
        swing = -t * 0.5
        el_lx, el_ly = nx + UPPER_ARM * np.sin(swing), ny - UPPER_ARM * np.cos(swing)
        wr_lx, wr_ly = el_lx + LOWER_ARM * np.sin(swing * 1.2), el_ly - LOWER_ARM
        arm_l.set_data([nx, el_lx, wr_lx], [ny - 0.05, el_ly, wr_ly])

        el_rx, el_ry = nx + UPPER_ARM * np.sin(-swing), ny - UPPER_ARM * np.cos(-swing)
        wr_rx, wr_ry = el_rx + LOWER_ARM * np.sin(-swing * 1.2), el_ry - LOWER_ARM
        arm_r.set_data([nx, el_rx, wr_rx], [ny - 0.05, el_ry, wr_ry])

        # 다리 (관절 분리)
        width = 0.5 if 'right' in view_type or 'left' in view_type else 0.2
        kn_lx, kn_ly = width * np.sin(t) * view_mod, -THIGH * np.cos(t)
        an_lx, an_ly = kn_lx + (width * 0.5 * np.sin(t * 1.5) * view_mod), kn_ly - CALF
        leg_l.set_data([0, kn_lx, an_lx], [0, kn_ly, an_ly])

        kn_rx, kn_ry = width * np.sin(-t) * view_mod, -THIGH * np.cos(-t)
        an_rx, an_ry = kn_rx + (width * 0.5 * np.sin(-t * 1.5) * view_mod), kn_ry - CALF
        leg_r.set_data([0, kn_rx, an_rx], [0, kn_ry, an_ry])

        return head, body, arm_l, arm_r, leg_l, leg_r

    ani = animation.FuncAnimation(fig, update, frames=len(angles), blit=True)

    # 파일명 설정 (3초 분량임을 명시)
    save_path = os.path.join(save_folder, f"{key}_{view_type}_3s.gif")
    ani.save(save_path, writer='pillow', fps=20)
    plt.close(fig)


# 실행
print(f"🚀 '{save_folder}' 위치로 영상 생성을 시작합니다...")
for f_name in target_files:
    for vp in ['front', 'back', 'right', 'left']:
        create_jointed_video(f_name, vp)
        print(f"✅ 저장 완료: {f_name}_{vp}")

print(f"\n🎉 모든 작업이 끝났습니다! 지정하신 경로를 확인해 보세요.")