import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. 파일 불러오기
mat_data = scipy.io.loadmat(r'C:\Gait_Analysis\Simulation_Data\dataset_1Giorgia.mat')
raw_data = mat_data['dataset_1Giorgia']

# 2. 영상처럼 보여줄 애니메이션 설정
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], 'r-', lw=2)
ax.set_xlim(0, 100) # 가로축: 시간 프레임
ax.set_ylim(-15, 15) # 세로축: 움직임 강도
ax.set_title("Parkinson Gait Simulation Video (Data Stream)")

# 3. 프레임 업데이트 함수
def update(i):
    # 실시간으로 데이터가 흘러가는 모습을 시뮬레이션
    x = range(i)
    y = raw_data[:i, 3] # 4번째 열(회전 데이터) 시각화
    line.set_data(x, y)
    if i > 100:
        ax.set_xlim(i-100, i)
    return line,

# 4. 애니메이션 실행 (약 500프레임만 보기)
ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=True)
print("🎥 시뮬레이션 영상 재생 중...")
plt.show()