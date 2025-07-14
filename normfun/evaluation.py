"""
性能评估模块
"""
import numpy as np
import torch
from config import *
from normfun.modulation import qam16_modulate, qam16_demodulate
import time
import matplotlib.pyplot as plt

# ====================== Evaluation functions ======================
def evaluate_beamforming(weights, label=None, ax=None, plot_annotations=True, is_combined=False, color=None,
                         linestyle='-', verbose=True):
    w_cplx = weights[:num_antennas] + 1j * weights[num_antennas:]
    pattern = []
    for theta in theta_range:
        sv = np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(theta)) / wavelength)
        pattern.append(np.abs(w_cplx @ sv.conj()))
    pattern = 20 * np.log10(np.array(pattern) / np.max(pattern) + 1e-8)

    # ====================== Added analysis features ======================
    if verbose:
        print("\n" + "=" * 40)
        print("Multi-beam Performance Analysis Report")
        print("=" * 40)

        # 1. Key Angle Gain Report
        print("\n[1] Key Angle Gains (dB):")
        print(f"{'Type':<8} | {'Angle(°)':<8} | {'Gain(dB)':<10} | {'Normalized':<12}")
        print("-" * 45)

        # Target angle
        for i, angle in enumerate(target_angles):
            gain = pattern[np.abs(theta_range - angle).argmin()]
            print(f"Target{i + 1} | {angle:<8.1f} | {gain:<10.2f} | {10 ** (gain / 20):<12.4f}")

        # User perspective
        for i, angle in enumerate(user_angles):
            gain = pattern[np.abs(theta_range - angle).argmin()]
            print(f"UE{i + 1}    | {angle:<8.1f} | {gain:<10.2f} | {10 ** (gain / 20):<12.4f}")

        # 2. Beamwidth analysis
        def get_beamwidth(angle, threshold=3):
            idx = np.abs(theta_range - angle).argmin()
            hp_level = pattern[idx] - threshold
            left = np.where(pattern[:idx] < hp_level)[0]
            right = np.where(pattern[idx:] < hp_level)[0]
            bw = theta_range[idx + right[0]] - theta_range[left[-1]] if len(left) > 0 and len(right) > 0 else 0
            return bw

        print("\n[2] Beamwidth Analysis:")
        for angle in target_angles + user_angles:
            bw = get_beamwidth(angle)
            print(f"- {angle}° direction 3dB BW: {bw:.1f}°")

        # 3. Interference analysis
        print("\n[3] Interference Analysis:")
        all_nodes = target_angles + user_angles
        for i, angle in enumerate(all_nodes):
            other_angles = all_nodes[:i] + all_nodes[i + 1:]
            interference = max(pattern[np.abs(theta_range - a).argmin()] for a in other_angles)
            print(f"- {angle}°The maximum interference from neighboring nodes: {interference:.2f}dB")

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

# ====================== Rate calculation function ======================
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

    real = w[:num_antennas]
    imag = w[num_antennas:]
    w_cplx = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)  # 重要：功率归一化

    snr_linear = 10 ** (snr_db / 10)
    user_rates = []

    for k in range(len(user_steering_vectors)):
        h_k = user_steering_vectors[k]

        # Calculate the effective channel gain
        effective_gain = np.abs(np.dot(w_cplx.conj(), h_k))

        # Calculate the signal-to-noise ratio
        signal_power = (effective_gain ** 2) * snr_linear  # 考虑发射功率
        noise_power = 1.0  # 归一化噪声功率

        # Shannon Capacity Formula
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

    if isinstance(precoder, np.ndarray):
        precoder = torch.tensor(precoder, dtype=torch.complex64, device=device)
    precoder = precoder.to(torch.complex64)

    SNR_radar = 10 ** (snr_db / 10)
    Pt = 1

    a = generate_a_theta(Nt, target_angle).to(torch.complex64)
    da = generate_da_theta(Nt, target_angle, a).to(torch.complex64)
    b = generate_a_theta(Nr, target_angle).to(torch.complex64)
    db = generate_da_theta(Nr, target_angle, b).to(torch.complex64)

    # Normalize the preencoder
    precoder = precoder / torch.norm(precoder)

    # Calculate the correlation matrix
    A = a @ b.T.conj()
    dot_A = da @ b.T.conj() + a @ db.T.conj()
    Rx = precoder.reshape(-1, 1) @ precoder.reshape(-1, 1).T.conj()

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

# ====================== Performance evaluation function ======================
def calculate_ber(w, user_steering_vectors, snr_db, num_symbols=10000):
    """Calculating the Bit Error Rate (BER)"""
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

    return np.mean(bers)

# def calculate_ber(w, user_steering_vectors, snr_db, num_symbols=10000):
#     """计算误码率并绘制星象图"""
#     # 更新样式设置 - 使用新版样式名称
#     plt.style.use('seaborn-v0_8')  # 替代原来的'seaborn'
#     plt.rcParams['font.sans-serif'] = ['Arial']
#     snr_db=30
#
#     # 复数权重处理
#     w_cplx = w[:num_antennas] + 1j * w[num_antennas:]
#     w_cplx /= np.linalg.norm(w_cplx)
#
#     # 创建图形
#     fig = plt.figure(figsize=(12, 8), dpi=100)
#     plt.suptitle(f"QAM16 Constellation and BER Evaluation (SNR={snr_db}dB)", y=1.02)
#
#     bers = []
#     snr_linear = 10 ** (snr_db / 10)
#
#     for i, h_k in enumerate(user_steering_vectors):
#         # 计算有效增益
#         effective_gain = np.abs(np.dot(w_cplx.conj(), h_k))
#
#         # 生成测试数据
#         bits = np.random.randint(0, 2, 4 * num_symbols)
#         tx_symbols = qam16_modulate(bits)
#
#         # 模拟接收信号
#         rx_symbols = effective_gain * tx_symbols
#         noise = np.sqrt(0.5 / snr_linear) * (np.random.randn(len(rx_symbols)) + 1j * np.random.randn(len(rx_symbols)))
#         rx_symbols += noise
#
#         # 计算BER
#         rx_bits = qam16_demodulate(rx_symbols / effective_gain)
#         ber = np.mean(np.abs(np.array(rx_bits) - bits))
#         bers.append(ber)
#
#         # --- 绘制星象图 ---
#         ax = plt.subplot(2, 2, i + 1)
#
#         # 绘制接收信号（采样前500个点避免过密）
#         sample_idx = np.random.choice(len(rx_symbols), size=min(500, len(rx_symbols)), replace=False)
#         ax.scatter(np.real(rx_symbols[sample_idx]), np.imag(rx_symbols[sample_idx]),
#                    s=8, alpha=0.6, color='blue',
#                    edgecolor='white', linewidth=0.2)
#
#         # # 绘制理想QAM16星座点
#         # ideal_symbols = qam16_modulate_all_possible()
#         # ax.scatter(np.real(ideal_symbols), np.imag(ideal_symbols),
#         #            s=40, color='red', marker='o',
#         #            edgecolor='black', linewidth=0.5,
#         #            label='Ideal')
#
#         # 图形标注
#         ax.set_xlim(-1.8, 1.8)
#         ax.set_ylim(-1.8, 1.8)
#         ax.grid(alpha=0.3)
#         ax.set_xlabel("In-phase")
#         ax.set_ylabel("Quadrature")
#         ax.set_title(f"User {i + 1} (BER={ber:.2e})", pad=10)
#         ax.legend()
#
#     plt.tight_layout()
#     plt.savefig(f'QAM16_Constellation_SNR{snr_db}dB.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#     return np.mean(bers)

# ================== Timing function ==================
def time_method(method_func, *args, num_runs=10, **kwargs):
    """Run the method and return the average time (seconds)"""
    times = []
    for _ in range(num_runs):
        start = time.time()
        method_func(*args, **kwargs)
        times.append(time.time() - start)
    return np.mean(times)


# def calculate_efficiency(self, weights, Hc_r, Hc_i):
#     """
#     计算能效(EE): 总速率/总功耗
#     """
#     # 转换为复数权重
#     weights = weights[:, :self.num_antennas] + 1j * weights[:, self.num_antennas:]
#
#     # 计算发射功率 (W)
#     transmit_power = torch.sum(torch.abs(weights) ** 2, dim=1) / power_params['power_scaling']
#
#     # 计算总功耗
#     total_power = transmit_power / power_params['PA_efficiency'] + power_params['circuit_power']
#
#     # 计算信道容量 (bps/Hz)
#     Hc = Hc_r + 1j * Hc_i
#     user_rates = torch.log2(1 + torch.abs(Hc @ weights.unsqueeze(-1)) ** 2)
#     sum_rate = torch.sum(user_rates)
#
#     # 能效 = 总速率/总功耗 (bps/Hz/W)
#     ee = sum_rate / total_power
#
#     return ee

