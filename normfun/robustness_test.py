"""
robustness_evaluation.py
信道误差鲁棒性评估（实部虚部分开处理再合并）
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
from models.FBN_model import FairBeamformingNet, get_model_name
from models.channel_data_generate import generate_communication_channel, generate_sensing_channel
from algorithms.optimization import traditional_optimizer
from normfun.evaluation import calculate_user_rates, calculate_ber, calculate_CRLB

def channel_estimate(error_rate: float, real_channel: torch.Tensor) -> torch.Tensor:
    """
    信道误差模拟函数
    Args:
        error_rate: 误差率 (0-1)
        real_channel: 真实信道 (实数张量)
    Returns:
        带误差的信道估计
    """
    # 计算信道功率
    channel_power = torch.norm(real_channel, 2)

    # 生成随机误差基
    error_base = torch.rand(real_channel.shape)

    # 计算误差功率
    error_power = torch.norm(error_base, 2)

    # 计算误差校正因子
    error_correction = torch.sqrt(torch.tensor(error_rate)) / (error_power / channel_power)

    # 应用误差
    error_channel = error_correction * error_base
    channel_with_error = error_channel + real_channel

    # 功率归一化
    power_est = torch.norm(channel_with_error, 2)
    return channel_with_error * (channel_power / power_est)

def add_channel_errors(H, error_percentage):
    """
    向信道添加误差（实部虚部分开处理再合并）
    Args:
        H: 原始信道矩阵 (复数numpy数组)
        error_percentage: 误差百分比 (0-1)
    Returns:
        带误差的信道矩阵
    """
    if H is None:
        return None

    # 分离实部和虚部
    H_real = H.real
    H_imag = H.imag

    # 转换为PyTorch张量
    H_real_tensor = torch.tensor(H_real, dtype=torch.float32)
    H_imag_tensor = torch.tensor(H_imag, dtype=torch.float32)

    # 合并实部和虚部
    H_combined = torch.cat((H_real_tensor, H_imag_tensor), dim=0)

    # 应用信道误差模型
    noisy_combined = channel_estimate(error_percentage, H_combined)

    # 分离回实部和虚部
    num_antennas = H_real.shape[0]
    noisy_real = noisy_combined[:num_antennas].numpy()
    noisy_imag = noisy_combined[num_antennas:].numpy()

    # 重新组合为复数
    noisy_H = noisy_real + 1j * noisy_imag

    return noisy_H

def evaluate_robustness(model_path=None, error_levels=[0.01, 0.05, 0.1, 0.2], snr_dBs=[0, 5, 10]):
    """
    主评估函数
    Args:
        model_path: 训练模型路径
        error_levels: 误差百分比列表
        snr_dBs: 信噪比列表
    """
    # 初始化方法
    methods = ["DE", "PSO", "GWO", "WOA", "ZF", "MMSE"]
    if model_path and os.path.exists(model_path):
        methods.insert(0, "Deep Learning")

    # 初始化结果存储
    metrics = {
        'sum_rate': {method: np.zeros((len(error_levels), len(snr_dBs))) for method in methods},
        'ber': {method: np.zeros((len(error_levels), len(snr_dBs))) for method in methods},
        'crlb': {method: np.zeros((len(error_levels), len(snr_dBs))) for method in methods}
    }

    # 加载模型（如果可用）
    model = None
    if model_path and os.path.exists(model_path):
        model = FairBeamformingNet(input_size, hidden_size=512, max_users=num_users).to(device)
        model.load_state_dict(torch.load(model_path))
        print(f"已加载模型: {model_path}")

    # 生成标称信道
    print("\n生成标称信道...")
    nominal_Hc = generate_communication_channel(num_antennas, user_angles)
    nominal_Hs = generate_sensing_channel(num_antennas, target_angles) if len(target_angles) > 0 else None

    # 生成导向向量
    steering_vectors = [
        np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
        for angle in user_angles
    ]

    # 主评估循环
    print("\n评估信道误差影响:")

    for err_idx, error_pct in enumerate(tqdm(error_levels, desc="误差水平")):
        # 添加信道误差（实部虚部分开处理再合并）
        noisy_Hc = add_channel_errors(nominal_Hc, error_pct)
        noisy_Hs = add_channel_errors(nominal_Hs, error_pct) if nominal_Hs is not None else None

        # 准备张量
        Hc_r = torch.FloatTensor(noisy_Hc.real).unsqueeze(0).to(device)
        Hc_i = torch.FloatTensor(noisy_Hc.imag).unsqueeze(0).to(device)
        Hs_r = torch.FloatTensor(noisy_Hs.real).unsqueeze(0).to(device) if noisy_Hs is not None else None
        Hs_i = torch.FloatTensor(noisy_Hs.imag).unsqueeze(0).to(device) if noisy_Hs is not None else None
        rho_tensor = torch.FloatTensor([rho]).to(device)

        # 获取所有方法的权重
        weights_dict = {}
        if model:
            with torch.no_grad():
                weights_dict["Deep Learning"] = model(
                    Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor
                ).cpu().numpy()[0]

        for method in [m for m in methods if m != "Deep Learning"]:
            weights_dict[method] = traditional_optimizer(
                method, Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor
            )

        # 在每个SNR下评估
        for snr_idx, snr_db in enumerate(snr_dBs):
            for method, weights in weights_dict.items():
                # 计算和速率
                rates = calculate_user_rates(weights, steering_vectors, snr_db)
                metrics['sum_rate'][method][err_idx, snr_idx] = np.sum(rates)

                # 计算误码率
                metrics['ber'][method][err_idx, snr_idx] = calculate_ber(
                    weights, steering_vectors, snr_db
                )

                # 如果存在感知目标，则计算CRLB
                if len(target_angles) > 0:
                    precoder = torch.tensor(weights[:num_antennas]) + 1j * torch.tensor(weights[num_antennas:])
                    precoder = precoder / torch.norm(precoder)
                    metrics['crlb'][method][err_idx, snr_idx] = calculate_CRLB(
                        num_antennas, num_receiver_antennas, target_angles[0], precoder, snr_db
                    )

    # 绘制并保存结果
    plot_results(metrics, methods, error_levels, snr_dBs)
    save_results(metrics, methods, error_levels, snr_dBs)

def plot_results(metrics, methods, error_levels, snr_dBs):
    """生成性能图"""
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    error_labels = [f"{x*100:.0f}%" for x in error_levels]

    plt.figure(figsize=(15, 10))
    plt.suptitle("信道估计误差下的性能表现（实部虚部分开处理再合并）", fontsize=16, fontweight='bold')

    # 子图1: 不同误差水平下的和速率 (SNR=10dB)
    ax1 = plt.subplot(2, 2, 1)
    snr_idx = snr_dBs.index(10) if 10 in snr_dBs else -1
    for i, method in enumerate(methods):
        ax1.plot(error_labels,
                metrics['sum_rate'][method][:, snr_idx],
                color=colors[i], marker=markers[i], linewidth=2, label=method)
    ax1.set_xlabel("误差水平", fontsize=12)
    ax1.set_ylabel("和速率 (bps/Hz)", fontsize=12)
    ax1.set_title("不同误差水平下的和速率 (SNR=10dB)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 子图2: 不同SNR下的性能下降 (误差20%)
    ax2 = plt.subplot(2, 2, 2)
    err_idx = error_levels.index(0.2) if 0.2 in error_levels else -1
    for i, method in enumerate(methods):
        # 计算性能下降百分比
        baseline = metrics['sum_rate'][method][0, :]  # 0%误差
        current = metrics['sum_rate'][method][err_idx, :]
        degradation = (baseline - current) / baseline * 100

        ax2.plot(snr_dBs, degradation,
                color=colors[i], marker=markers[i], linewidth=2, label=method)
    ax2.set_xlabel("SNR (dB)", fontsize=12)
    ax2.set_ylabel("性能下降 (%)", fontsize=12)
    ax2.set_title("20%误差下的性能下降", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 子图3: BER曲线 (误差10%, SNR=10dB)
    ax3 = plt.subplot(2, 2, 3)
    err_idx = error_levels.index(0.1) if 0.1 in error_levels else -1
    for i, method in enumerate(methods):
        ax3.semilogy(error_labels,
                    metrics['ber'][method][:, snr_idx],
                    color=colors[i], marker=markers[i], linewidth=2, label=method)
    ax3.set_xlabel("误差水平", fontsize=12)
    ax3.set_ylabel("误码率 (BER)", fontsize=12)
    ax3.set_title("不同误差水平下的误码率 (SNR=10dB)", fontsize=14)
    ax3.grid(True, which="both", alpha=0.3)

    # 子图4: CRLB曲线 (如果有感知目标)
    if len(target_angles) > 0:
        ax4 = plt.subplot(2, 2, 4)
        for i, method in enumerate(methods):
            ax4.plot(error_labels,
                    metrics['crlb'][method][:, snr_idx],
                    color=colors[i], marker=markers[i], linewidth=2, label=method)
        ax4.set_xlabel("误差水平", fontsize=12)
        ax4.set_ylabel("CRLB (度)", fontsize=12)
        ax4.set_title("不同误差水平下的感知精度 (CRLB)", fontsize=14)
        ax4.grid(True, alpha=0.3)

    # 添加图例
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=min(4, len(methods)),
                  bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("信道误差鲁棒性评估_实部虚部分开再合并.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(metrics, methods, error_levels, snr_dBs):
    """保存数值结果到文件"""
    os.makedirs("results", exist_ok=True)
    filename = "results/信道误差鲁棒性评估_实部虚部分开再合并.csv"

    with open(filename, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("方法,误差水平,SNR(dB),和速率(bps/Hz),误码率(BER),CRLB(度)\n")

        # 写入数据
        for method in methods:
            for err_idx, err_pct in enumerate(error_levels):
                for snr_idx, snr_db in enumerate(snr_dBs):
                    crlb = metrics['crlb'][method][err_idx, snr_idx] if len(target_angles) > 0 else 0
                    line = f"{method},{err_pct:.2f},{snr_db}," + \
                           f"{metrics['sum_rate'][method][err_idx, snr_idx]:.4f}," + \
                           f"{metrics['ber'][method][err_idx, snr_idx]:.4e},{crlb:.4f}\n"
                    f.write(line)

    print(f"结果已保存至: {filename}")

def print_summary(metrics, methods, error_levels, snr_dBs):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("信道误差鲁棒性评估摘要（实部虚部分开处理再合并）")
    print("="*80)

    # 打印和速率结果 (SNR=10dB)
    snr_idx = snr_dBs.index(10) if 10 in snr_dBs else -1
    print("\n和速率 (SNR=10dB):")
    header = "方法\t\t" + "\t".join([f"{pct*100:.0f}%" for pct in error_levels])
    print(header)
    print("-"*len(header))

    for method in methods:
        rates = metrics['sum_rate'][method][:, snr_idx]
        rate_str = "\t".join([f"{rate:.2f}" for rate in rates])
        print(f"{method[:12]:<12}\t{rate_str}")

    # 打印误码率结果 (SNR=10dB)
    print("\n误码率 (SNR=10dB):")
    header = "方法\t\t" + "\t".join([f"{pct*100:.0f}%" for pct in error_levels])
    print(header)
    print("-"*len(header))

    for method in methods:
        bers = metrics['ber'][method][:, snr_idx]
        ber_str = "\t".join([f"{ber:.2e}" for ber in bers])
        print(f"{method[:12]:<12}\t{ber_str}")

    # 打印CRLB结果 (如果有感知目标)
    if len(target_angles) > 0:
        print("\n感知精度CRLB (度):")
        header = "方法\t\t" + "\t".join([f"{pct*100:.0f}%" for pct in error_levels])
        print(header)
        print("-"*len(header))

        for method in methods:
            crlbs = metrics['crlb'][method][:, snr_idx]
            crlb_str = "\t".join([f"{crlb:.4f}" for crlb in crlbs])
            print(f"{method[:12]:<12}\t{crlb_str}")

if __name__ == "__main__":
    print("="*80)
    print("开始信道误差鲁棒性评估（实部虚部分开处理再合并）")
    print("="*80)

    # 获取模型路径
    model_path = get_model_name(user_angles, target_angles, num_antennas, rho)
    print(f"模型路径: {model_path}")

    # 设置测试参数
    error_levels = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, 20% 误差
    snr_dBs = [0, 5, 10]  # 测试的信噪比

    print(f"测试参数:")
    print(f"- 误差水平: {[f'{pct*100:.0f}%' for pct in error_levels]}")
    print(f"- 信噪比 (dB): {snr_dBs}")

    # 运行鲁棒性评估
    evaluate_robustness(
        model_path=model_path,
        error_levels=error_levels,
        snr_dBs=snr_dBs
    )