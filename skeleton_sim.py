import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re

# 1. 설정 및 파일 자동 검색
folder_path = r'C:\Gait_Analysis\Simulation_Data'

# 폴더 내 모든 .mat 파일을 가져와서 숫자 순서대로 정렬
all_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and 'dataset' in f]
# 파일 이름에서 숫자를 추출해 정렬 (1, 2, 10 순서 맞추기)
all_files.sort(key=lambda f: int(re.findall(r'\d+', f)[0]))

# 상위 10개만 선택
target_files = all_files[:10]
all_data = []
actual_names = []

print(f"📂 총 {len(target_files)}개의 파일을 찾았습니다.")

for f_name in target_files:
    full_path = os.path.join(folder_path, f_name)
    key_name = f_name.replace('.mat', '')

    try:
        data = scipy.io.loadmat(full_path)[key_name]
        # 데이터가 너무 길면 메모리를 많이 먹으니 10배속 + 처음 5000프레임만
        all_data.append(data[:50000:10, 3])
        actual_names.append(f_name.split('.')[0])
        print(f"✅ 로드 성공: {f_name}")
    except Exception as e:
        print(f"❌ 로드 실패 ({f_name}): {e}")

if not all_data:
    print("‼️ 데이터를 하나도 불러오지 못했습니다. 경로와 파일명을 확인해주세요.")
    exit()

min_len = min(len(d) for d in all_data)
all_data = [d[:min_len] for d in all_data]

# 2. 시뮬레이션 설정 (2행 5열)
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.3, hspace=0.4)

heads, bodies, arms_l, arms_r, legs_l, legs_r = [], [], [], [], [], []

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xlim(-1, 1);
    ax.set_ylim(-1.3, 1.2)
    ax.axis('off')

    if i < len(actual_names):
        ax.set_title(actual_names[i], fontsize=9)
        h = plt.Circle((0, 0), 0.07, color='black', fill=True)
        ax.add_patch(h)
        heads.append(h)

        b, = ax.plot([], [], 'k-', lw=8)
        al, = ax.plot([], [], 'b-o', lw=3, ms=4, alpha=0.6)
        ar, = ax.plot([], [], 'r-o', lw=3, ms=4, alpha=0.6)
        ll, = ax.plot([], [], 'b-o', lw=5, ms=5)
        lr, = ax.plot([], [], 'r-o', lw=5, ms=5)

        bodies.append(b);
        arms_l.append(al);
        arms_r.append(ar)
        legs_l.append(ll);
        legs_r.append(lr)
    else:
        ax.set_visible(False)  # 파일이 10개 미만이면 빈칸 숨기기


# 3. 업데이트 함수
def update(frame):
    updated_items = []
    for i in range(len(all_data)):
        raw_val = all_data[i][frame]
        t = np.radians(np.interp(raw_val, (all_data[i].min(), all_data[i].max()), (-30, 30)))

        tilt = t * 0.15 + np.radians(5)
        nx, ny = 0.4 * np.sin(tilt), 0.4 * np.cos(tilt)

        heads[i].center = (nx + 0.02, ny + 0.1)
        bodies[i].set_data([0, nx], [0, ny])

        at_l, at_r = -t * 0.7, t * 0.7
        arms_l[i].set_data([nx, nx + 0.3 * np.sin(at_l)], [ny - 0.05, ny - 0.35 * np.cos(at_l)])
        arms_r[i].set_data([nx, nx + 0.3 * np.sin(at_r)], [ny - 0.05, ny - 0.35 * np.cos(at_r)])

        klx, kly = 0.45 * np.sin(t), -0.45 * np.cos(t)
        alx, aly = klx + 0.4 * np.sin(t + 0.15), kly - 0.4 * np.cos(t + 0.15)
        legs_l[i].set_data([0, klx, alx], [0, kly, aly])

        tr = -t
        krx, kry = 0.45 * np.sin(tr), -0.45 * np.cos(tr)
        arx, ary = krx + 0.4 * np.sin(tr + 0.15), kry - 0.4 * np.cos(tr + 0.15)
        legs_r[i].set_data([0, krx, arx], [0, kry, ary])

        updated_items.extend([heads[i], bodies[i], arms_l[i], arms_r[i], legs_l[i], legs_r[i]])

    return updated_items


ani = animation.FuncAnimation(fig, update, frames=min_len, interval=30, blit=True)
plt.show()