import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from itertools import cycle
from tqdm import tqdm
import matplotlib
import torch.nn as nn
from models.channel_data_generate import generate_sensing_channel,generate_communication_channel

# # 设置字体为黑体，避免中文乱码
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

# Configure system parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_antennas = 16
wavelength = 1
d = wavelength / 2
theta_range = np.linspace(-90, 90, 361)
snr_db = 20



# 模型加载函数
def load_model(user_angles, target_angles, num_antennas, rho):
    class FairBeamformingNet(nn.Module):
        def __init__(self, input_size, hidden_size=512, max_users=4, num_rf_chains=8):
            super().__init__()
            self.max_users = max_users  # 最大用户数（动态用户数需≤此值）
            self.num_rf_chains = num_rf_chains
            self.num_antennas = 16  # 输出 32 = 16 * 2

            # 检查RF链数是否足够支持最大用户数
            assert max_users <= num_rf_chains, "max_users cannot exceed num_rf_chains"

            # Shared network
            self.shared_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )

            # Digital branches (每个用户输出2维复数)
            self.digital_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 2)
                ) for _ in range(max_users)  # 初始化最大可能的分支
            ])

            # Analog beamformer (输出 num_rf_chains * num_antennas * 2)
            self.analog_beamformer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_rf_chains * num_antennas * 2)
            )

            self.norm = nn.LayerNorm(num_antennas * 2)

        def forward(self, Hc_real, Hc_imag, Hs_real, Hs_imag, rho, num_users=None):
            """
            Args:
                num_users: 当前batch的动态用户数（需≤max_users）
                          若为None，默认使用max_users
            """
            if num_users is None:
                num_users = self.max_users
            assert num_users <= self.max_users, f"num_users ({num_users}) > max_users ({self.max_users})"

            # 输入处理
            Hc = torch.cat([Hc_real.flatten(1), Hc_imag.flatten(1)], dim=1)
            Hs = torch.cat([Hs_real.flatten(1), Hs_imag.flatten(1)], dim=1)
            x = torch.cat([Hc, Hs, rho.view(-1, 1)], dim=1)
            x = self.shared_net(x)

            # Digital: 仅激活前num_users个分支 [B, num_users, 2]
            digital_outputs = [self.digital_branches[i](x).view(-1, 1, 2)
                               for i in range(num_users)]
            digital_combined = torch.cat(digital_outputs, dim=1)

            # 补零到num_rf_chains维度 [B, num_rf_chains, 2]
            if num_users < self.num_rf_chains:
                padding = torch.zeros(x.size(0),
                                      self.num_rf_chains - num_users,
                                      2, device=x.device)
                digital_combined = torch.cat([digital_combined, padding], dim=1)

            # Analog: [B, num_rf_chains, num_antennas, 2]
            analog_output = self.analog_beamformer(x).view(-1, self.num_rf_chains, self.num_antennas, 2)

            # Hybrid: 数字权重 * 模拟矩阵 [B, num_antennas, 2]
            hybrid_output = torch.einsum('brf,brnf->bnf', digital_combined, analog_output)
            hybrid_output = hybrid_output.flatten(1)  # [B, 32]

            return self.norm(hybrid_output)

    num_users = len(user_angles)
    num_targets = len(target_angles)
    input_size = 2 * (num_users * num_antennas + num_targets * num_antennas) + 1
    model = FairBeamformingNet(input_size, hidden_size=512, max_users=num_users).to(device)
    model_name = f"model_U{len(user_angles)}_S{len(target_angles)}_A{num_antennas}_rho{rho:.2f}.pth".replace(".",
                                                                                                                 "_")

    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
        print(f"The model was loaded successfully: {model_name}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_name}")

    return model


# 速率计算函数
def calculate_rates(weights, user_angles, snr_db):
    """Calculate both sum rate and average rate"""
    w_cplx = weights[:num_antennas] + 1j * weights[num_antennas:]
    w_cplx = w_cplx / np.linalg.norm(w_cplx)

    steering_vectors = np.array([
        np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
        for angle in user_angles
    ])

    snr_linear = 10 ** (snr_db / 10)
    gains = np.abs(np.dot(w_cplx.conj(), steering_vectors.T))
    rates = np.log2(1 + (gains ** 2) * snr_linear)

    return np.sum(rates), np.mean(rates)  # (sum_rate, avg_rate)


def plot_rates_vs_users():
    user_configs = {
        3: [-10, 20, 50],
        4: [-15, 15, 30, 45],
        5: [-10, 0, 10, 20, 30],
        6: [-10, 0, 10, 20, 30, 40],
        7: [-10, 0, 10, 20, 30, 40, 50],
        8: [-10, 0, 10, 20, 30, 40, 50, 60]
    }
    target_angles = [-45]
    rhos = [0.3, 0.5, 0.7, 0.9]

    # Store all rate data
    all_sum_rates = {rho: [] for rho in rhos}
    all_avg_rates = {rho: [] for rho in rhos}

    # Create the first graph: sum rate
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(rhos)))
    line_styles = ['-', '--', '-.', ':']
    style_cycle = cycle(zip(colors, line_styles))

    for rho in rhos:
        color, linestyle = next(style_cycle)
        user_counts = sorted(user_configs.keys())
        sum_rates = []

        for n_users in tqdm(user_counts, desc=f"Rho={rho}"):
            user_angles = user_configs[n_users]
            try:
                model = load_model(user_angles, target_angles, num_antennas, rho)
                with torch.no_grad():
                    # test_input = torch.FloatTensor([user_angles + target_angles])
                    # weights = model(test_input).numpy()[0]
                    test_Hc = generate_communication_channel(num_antennas, user_angles)
                    test_Hs = generate_sensing_channel(num_antennas, target_angles)

                    # 准备输入数据
                    Hc_r = torch.FloatTensor(test_Hc.real).unsqueeze(0).to(device)
                    Hc_i = torch.FloatTensor(test_Hc.imag).unsqueeze(0).to(device)
                    Hs_r = torch.FloatTensor(test_Hs.real).unsqueeze(0).to(device)
                    Hs_i = torch.FloatTensor(test_Hs.imag).unsqueeze(0).to(device)
                    rho_tensor = torch.FloatTensor([rho]).to(device)

                    weights = model(Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor).numpy()[0]

                sum_rate, avg_rate = calculate_rates(weights, user_angles, snr_db)
                sum_rates.append(sum_rate)
                all_avg_rates[rho].append(avg_rate)
                print(f"ρ={rho}, Users={n_users}: Sum Rate={sum_rate:.4f}, Avg Rate={avg_rate:.4f}")
            except Exception as e:
                print(f"Error for rho={rho}, users={n_users}: {str(e)}")
                sum_rates.append(np.nan)
                all_avg_rates[rho].append(np.nan)

        all_sum_rates[rho] = sum_rates
        plt.plot(user_counts, sum_rates,
                 color=color, linestyle=linestyle,
                 marker='o', markersize=8,
                 label=f'ρ={rho}')

    plt.xlabel("Number of users", fontsize=20)
    plt.ylabel("Sum Rate (bps/Hz)", fontsize=20)
    plt.xlim(3, 8)
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(user_configs.keys()))
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('sum_rate_vs_users.png', dpi=300)
    plt.show()

    # Create a second graph: average rate
    plt.figure(figsize=(10, 6))
    style_cycle = cycle(zip(colors, line_styles))

    for rho in rhos:
        color, linestyle = next(style_cycle)
        user_counts = sorted(user_configs.keys())
        avg_rates = all_avg_rates[rho]

        plt.plot(user_counts, avg_rates,
                 color=color, linestyle=linestyle,
                 marker='o', markersize=8,
                 label=f'ρ={rho}')

    plt.xlabel("Number of users", fontsize=20)
    plt.ylabel("Rate (bps/Hz)", fontsize=20)
    plt.xlim(3,8)
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(user_configs.keys()))
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('avg_rate_vs_users.png', dpi=300)
    plt.show()

    # Print all rate data
    print("\nAll rate data is aggregated:")
    for rho in rhos:
        print(f"\nρ = {rho}:")
        for i, n_users in enumerate(sorted(user_configs.keys())):
            print(f"  Users={n_users}: Sum Rate={all_sum_rates[rho][i]:.4f}, Avg Rate={all_avg_rates[rho][i]:.4f}")


if __name__ == "__main__":
    plot_rates_vs_users()