import scipy.io
import matplotlib.pyplot as plt
import numpy as np

file_path = r'C:\Gait_Analysis\Simulation_Data\dataset_1Giorgia.mat'

try:
    mat_data = scipy.io.loadmat(file_path)
    raw_data = mat_data['dataset_1Giorgia']

    # 데이터가 너무 길어서 앞부분 1000프레임만 분석합니다 (약 10~20초 분량)
    subset_data = raw_data[:1000, :]

    plt.figure(figsize=(12, 6))

    # 4, 5, 6번 열이 보통 회전(각도 관련) 데이터입니다.
    plt.subplot(2, 1, 1)
    plt.plot(subset_data[:, 3], label='X-axis (Pitch)', color='r')
    plt.plot(subset_data[:, 4], label='Y-axis (Roll)', color='g')
    plt.plot(subset_data[:, 5], label='Z-axis (Yaw)', color='b')
    plt.title("Gait Sensor Data (Rotation/Angle)")
    plt.legend()
    plt.grid(True)

    # 1, 2, 3번 열은 가속도(움직임의 세기)입니다.
    plt.subplot(2, 1, 2)
    plt.plot(subset_data[:, 0], label='Acc X', color='orange')
    plt.plot(subset_data[:, 1], label='Acc Y', color='purple')
    plt.title("Gait Acceleration (Movement)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    print("📈 그래프 창이 떴습니다! 확인해 보세요.")
    plt.show()

except Exception as e:
    print(f"❌ 에러 발생: {e}")