"""
性能评估模块
"""
import numpy as np
import torch
from config import *
from normfun.modulation import qam16_modulate, qam16_demodulate
import time

# ====================== 评估函数 ======================
def evaluate_beamforming(weights, label=None, ax=None, plot_annotations=True, is_combined=False, color=None,
                         linestyle='-'):
    w_cplx = weights[:num_antennas] + 1j * weights[num_antennas:]
    pattern = []
    for theta in theta_range:
        sv = np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(theta)) / wavelength)
        pattern.append(np.abs(w_cplx @ sv.conj()))
    pattern = 20 * np.log10(np.array(pattern) / np.max(pattern) + 1e-8)

    if ax is not None:
        # Plot the beam pattern
        ax.plot(theta_range, pattern, linewidth=2, label=label, color=color, linestyle=linestyle)

        if plot_annotations:
            # User angle colors (more visible)
            user_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Target angle colors (more visible)
            target_colors = ['#ff9896', '#98df8a', '#ffbb78', '#c5b0d5', '#dbdb8d']

            # Annotate user angles
            for idx, angle in enumerate(user_angles):
                ax.axvline(angle, color=user_colors[idx], linestyle='--',
                           label=f'User {idx + 1}' if not is_combined else None, alpha=0.7)

            # Annotate target angles
            for idx, angle in enumerate(target_angles):
                ax.axvline(angle, color=target_colors[idx], linestyle='-.',
                           label=f'Target {idx + 1}' if not is_combined else None, alpha=0.7)

            # Set plot properties
            ax.set_ylim(-50, 5)
            ax.grid(alpha=0.3)
    return pattern

# ====================== 改进的速率计算函数 ======================
def calculate_user_rates(w, user_steering_vectors, snr_db, num_symbols=10000):
    # real_part = w[:num_antennas]
    # imag_part = w[num_antennas:]
    # w_cplx = real_part + 1j * imag_part
    # w_cplx /= np.linalg.norm(w_cplx)
    #
    # user_rates = []
    # snr_linear = 10 ** (snr_db / 10)
    #
    # for k in range(len(user_steering_vectors)):
    #     h_k = user_steering_vectors[k]
    #     effective_channel = np.dot(w_cplx.conj(), h_k)
    #
    #     bits = np.random.randint(0, 2, 4 * num_symbols)
    #     tx_symbols = qam16_modulate(bits)
    #     rx_symbols = effective_channel * tx_symbols
    #     noise_power = 1.0 / snr_linear
    #     noise = np.sqrt(noise_power / 2) * (np.random.randn(len(rx_symbols)) + 1j * np.random.randn(len(rx_symbols)))
    #     rx_symbols += noise
    #     rx_bits = qam16_demodulate(rx_symbols / effective_channel)
    #     error_bits = np.sum(np.abs(np.array(rx_bits) - bits))
    #     ber = error_bits / len(bits)
    #     user_rates.append((1 - ber) * np.log2(16))
    #
    # return user_rates


    """基于香农公式的速率计算"""
    # 转换为复数权重并归一化
    real = w[:num_antennas]
    imag = w[num_antennas:]
    w_cplx = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)  # 重要：功率归一化

    snr_linear = 10 ** (snr_db / 10)
    user_rates = []

    for k in range(len(user_steering_vectors)):
        h_k = user_steering_vectors[k]

        # 计算有效信道增益
        effective_gain = np.abs(np.dot(w_cplx.conj(), h_k))

        # 计算信噪比
        signal_power = (effective_gain ** 2) * snr_linear  # 考虑发射功率
        noise_power = 1.0  # 归一化噪声功率

        # 香农容量公式
        rate = np.log2(1 + signal_power / noise_power)
        user_rates.append(rate)

    return user_rates

# ====================== Steering Vector Generation =======================
def generate_a_theta(Nt, theta):
    theta_rad = torch.deg2rad(torch.tensor(theta, device=device))
    n = torch.arange(Nt, device=device) - (Nt - 1) / 2
    a = torch.exp(1j * torch.pi * n * torch.sin(theta_rad))
    return a.reshape(-1, 1)

def generate_da_theta(Nt, theta, a):
    theta_rad = torch.deg2rad(torch.tensor(theta, device=device))
    n = torch.arange(Nt, device=device) - (Nt - 1) / 2
    derivative_factor = 1j * torch.pi * n * torch.cos(theta_rad)
    da = derivative_factor * a.squeeze()
    return da.reshape(-1, 1)

# ====================== CRLB Calculation =======================
def calculate_CRLB(Nt, Nr, target_angle, precoder, snr_db):
    # 类型和设备处理

    if isinstance(precoder, np.ndarray):
        precoder = torch.tensor(precoder, dtype=torch.complex64, device=device)
    precoder = precoder.to(torch.complex64)

    SNR_radar = 10 ** (snr_db / 10)
    Pt = 1

    # 生成导向向量（确保complex64类型）
    a = generate_a_theta(Nt, target_angle).to(torch.complex64)
    da = generate_da_theta(Nt, target_angle, a).to(torch.complex64)
    b = generate_a_theta(Nr, target_angle).to(torch.complex64)
    db = generate_da_theta(Nr, target_angle, b).to(torch.complex64)

    # 归一化预编码器
    precoder = precoder / torch.norm(precoder)

    # 计算相关矩阵
    A = a @ b.T.conj()
    dot_A = da @ b.T.conj() + a @ db.T.conj()
    Rx = precoder.reshape(-1, 1) @ precoder.reshape(-1, 1).T.conj()

    # 计算各项（确保使用.real取实部）
    term1 = torch.trace(A @ Rx @ A.T.conj()).real
    term2 = torch.trace(dot_A @ Rx @ dot_A.T.conj()).real
    term3 = torch.trace(A @ Rx @ dot_A.T.conj()).real

    # # 处理可能的数值问题
    # denominator = term1 * term2 - term3 ** 2
    # if denominator <= 1e-6:
    #     return torch.inf

    alpha = Pt / (2 * SNR_radar)

    pi = torch.pi

    CRLB = torch.sqrt(alpha * (term1.real / (term2.real * term1.real - (abs(term3.real)) ** 2))) * 180 / pi
    return torch.sqrt(CRLB)

# ====================== 新增性能评估函数 ======================
def calculate_ber(w, user_steering_vectors, snr_db, num_symbols=10000):
    """计算误码率(BER)"""
    w_cplx = w[:num_antennas] + 1j * w[num_antennas:]
    w_cplx /= np.linalg.norm(w_cplx)
    bers = []

    for h_k in user_steering_vectors:
        effective_gain = np.abs(np.dot(w_cplx.conj(), h_k))
        snr_linear = 10 ** (snr_db / 10)

        bits = np.random.randint(0, 2, 4 * num_symbols)
        tx_symbols = qam16_modulate(bits)
        rx_symbols = effective_gain * tx_symbols
        noise = np.sqrt(0.5 / snr_linear) * (np.random.randn(len(rx_symbols)) + 1j * np.random.randn(len(rx_symbols)))
        rx_symbols += noise
        rx_bits = qam16_demodulate(rx_symbols / effective_gain)
        ber = np.mean(np.abs(np.array(rx_bits) - bits))
        bers.append(ber)

    return np.mean(bers)  # 返回平均误码率

# ================== 计时函数 ==================
def time_method(method_func, *args, num_runs=10, **kwargs):
    """运行方法并返回平均时间（秒）"""
    times = []
    for _ in range(num_runs):
        start = time.time()
        method_func(*args, **kwargs)
        times.append(time.time() - start)
    return np.mean(times)

