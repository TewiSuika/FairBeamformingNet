import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
Nt = 16  # 发射天线数
Nr = 16  # 接收天线数
K = 4  # 通信用户数
theta_users = np.array([-60, -30, 30, 60]) * np.pi / 180  # 通信用户角度(弧度)
theta_target = 15 * np.pi / 180  # 感知目标角度(弧度)
wavelength = 1  # 波长
d = wavelength / 2  # 天线间距


# 生成阵列响应向量
def array_response(theta, N, d):
    n = np.arange(N)
    a = np.exp(1j * 2 * np.pi * d * n * np.sin(theta) / wavelength)
    return a / np.sqrt(N)


# 生成混合波束形成矩阵
def hybrid_beamforming(theta_users, theta_target, Nt, K):
    # 数字波束形成矩阵
    F_BB = np.zeros((K + 1, K + 1), dtype=complex)

    # 模拟波束形成矩阵
    F_RF = np.zeros((Nt, K + 1), dtype=complex)

    # 为通信用户设计波束
    for k in range(K):
        a_user = array_response(theta_users[k], Nt, d)
        F_RF[:, k] = a_user
        F_BB[k, k] = 1  # 数字部分为单位矩阵

    # 为感知目标设计波束
    a_target = array_response(theta_target, Nt, d)
    F_RF[:, K] = a_target
    F_BB[K, K] = 1

    # 归一化功率
    F = F_RF @ F_BB
    F = F / np.linalg.norm(F, 'fro') * np.sqrt(Nt)

    return F_RF, F_BB, F


# 生成波束形成矩阵
F_RF, F_BB, F = hybrid_beamforming(theta_users, theta_target, Nt, K)
print('F_RF:', F_RF)
print('F_BB:', F_BB)


# 计算波束方向图
def beam_pattern(F, theta_range):
    N_theta = len(theta_range)
    pattern = np.zeros(N_theta, dtype=complex)

    for i, theta in enumerate(theta_range):
        a = array_response(theta, Nt, d)
        pattern[i] = np.abs(a.conj().T @ F @ F.conj().T @ a)

    return pattern


# 角度范围
theta_range = np.linspace(-90, 90, 181) * np.pi / 180
pattern = beam_pattern(F, theta_range)

# 绘制波束方向图
plt.figure(figsize=(10, 6))
plt.plot(theta_range * 180 / np.pi, 10 * np.log10(np.abs(pattern)), linewidth=2)
plt.xlabel('Angle (degrees)', fontsize=12)
plt.ylabel('Beam Pattern (dB)', fontsize=12)
plt.title('Hybrid Beamforming Pattern', fontsize=14)
plt.grid(True)

# 标记用户位置
for theta in theta_users:
    plt.axvline(x=theta * 180 / np.pi, color='r', linestyle='--', linewidth=1)
plt.axvline(x=theta_target * 180 / np.pi, color='g', linestyle='--', linewidth=1)

plt.legend(['Beam Pattern', 'Communication Users', 'Sensing Target'], loc='upper right')
plt.xlim([-90, 90])
plt.ylim([-20, 30])
plt.show()

