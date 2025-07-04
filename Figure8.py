import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
from models.channel_data_generate import generate_sensing_channel,generate_communication_channel
from algorithms.optimization import traditional_optimizer
# ====================== System parameter configuration =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_angles = [-20, 15, 25, 55]

target_angles = [-45]
num_antennas = 16
num_receiver_antennas = 16
num_users = len(user_angles)
num_targets = len(target_angles)
snr_db = 10
wavelength = 1
d = wavelength / 2
rho_values = np.arange(0, 1.1, 0.1)
CRLB_SCALE_FACTOR = 100
SEED = 19


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


# ====================== load_model ========================
def load_model(rho):
    model_name = (
        f"model_U{num_users}_S{len(target_angles)}_A{num_antennas}_rho{rho:.2f}.pth"
    ).replace(".", "_")
    # model = FairBeamformingNet(input_size=num_users + len(target_angles))

    input_size = 2 * (num_users * num_antennas + num_targets * num_antennas) + 1
    model = FairBeamformingNet(input_size, hidden_size=512, max_users=num_users).to(device)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded model: {model_name}")
    else:
        raise FileNotFoundError(f"Model {model_name} not found")
    return model.to(device)


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
def calculate_CRLB(Nt, Nr, target_angle, precoder, snr_db, Pt=1):
    snr_linear = 10 ** (snr_db / 10)

    if isinstance(precoder, np.ndarray):
        precoder = torch.tensor(precoder, dtype=torch.complex64, device=device)
    precoder = precoder.to(device=device, dtype=torch.complex64)

    a = generate_a_theta(Nt, target_angle).to(torch.complex64)
    da = generate_da_theta(Nt, target_angle, a).to(torch.complex64)
    b = generate_a_theta(Nr, target_angle).to(torch.complex64)
    db = generate_da_theta(Nr, target_angle, b).to(torch.complex64)

    precoder = precoder / torch.norm(precoder)

    A = a @ b.T.conj()
    dot_A = da @ b.T.conj() + a @ db.T.conj()
    Rx = precoder.reshape(-1, 1) @ precoder.reshape(-1, 1).T.conj()

    term1 = (A @ Rx @ A.T.conj()).diagonal().sum().real
    term2 = (dot_A @ Rx @ dot_A.T.conj()).diagonal().sum().real
    term3 = (A @ Rx @ dot_A.T.conj()).diagonal().sum().real

    numerator = term1
    denominator = term1 * term2 - (term3) ** 2

    if denominator <= 1e-10:
        return torch.tensor(float('inf'), device=device)

    crlb_radians_squared = 1 / (2 * snr_linear * denominator / numerator)
    crlb_degrees = torch.sqrt(crlb_radians_squared) * (180 / torch.pi)

    return crlb_degrees


# ====================== Sum Rate Calculation =======================
def calculate_sum_rate(weights, user_angles, snr_db):
    """Calculate sum rate for communication users"""
    # Convert weights to complex
    w_cplx = weights[:num_antennas] + 1j * weights[num_antennas:]
    w_cplx = w_cplx / torch.norm(w_cplx)

    # Generate steering vectors for all users
    steering_vectors = []
    for angle in user_angles:
        theta_rad = torch.deg2rad(torch.tensor(angle, device=device))
        n = torch.arange(num_antennas, device=device) - (num_antennas - 1) / 2
        sv = torch.exp(1j * torch.pi * n * torch.sin(theta_rad))
        steering_vectors.append(sv)
    steering_vectors = torch.stack(steering_vectors)

    snr_linear = 10 ** (snr_db / 10)
    gains = torch.abs(torch.matmul(w_cplx.conj(), steering_vectors.T))
    rates = torch.log2(1 + (gains ** 2) * snr_linear)

    return torch.sum(rates).item()


# ====================== The main execution process ========================
def main():
    # methods = ['DNN', 'PSO', 'DE', 'CS', 'GWO', 'WOA', 'ZF', 'MMSE']
    methods = ['DNN', 'PSO', 'DE', 'GWO', 'WOA', 'ZF', 'MMSE']
    results = {m: {'crlb': [], 'sum_rate': []} for m in methods}

    for rho in rho_values:
        print(f"\nProcessing ρ={rho:.1f}")

        test_Hc = generate_communication_channel(num_antennas, user_angles, random_seed=SEED)
        test_Hs = generate_sensing_channel(num_antennas, target_angles, random_seed=SEED)
        # test_Hc = generate_communication_channel(num_antennas, user_angles)
        # test_Hs = generate_sensing_channel(num_antennas, target_angles)
        # print(test_Hc)

        # 准备输入数据
        Hc_r = torch.FloatTensor(test_Hc.real).unsqueeze(0).to(device)
        Hc_i = torch.FloatTensor(test_Hc.imag).unsqueeze(0).to(device)
        Hs_r = torch.FloatTensor(test_Hs.real).unsqueeze(0).to(device)
        Hs_i = torch.FloatTensor(test_Hs.imag).unsqueeze(0).to(device)
        rho_tensor = torch.FloatTensor([rho]).to(device)

        # DNN方法
        try:
            model = load_model(rho)

            with torch.no_grad():
                # weights = model(input_data).squeeze()
                weights = model(Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor).squeeze()

            # Calculate CRLB
            precoder = weights[:num_antennas] + 1j * weights[num_antennas:]
            precoder = precoder / torch.norm(precoder)
            crlb = calculate_CRLB(num_antennas, num_receiver_antennas,
                                  target_angles[0], precoder, snr_db, 1)
            sum_rate = calculate_sum_rate(weights, user_angles, snr_db)
            results['DNN']['crlb'].append(crlb.item())
            results['DNN']['sum_rate'].append(sum_rate)
            print(f"ρ={rho:.1f}: CRLB = {crlb:.4f}°, Sum Rate = {sum_rate:.4f} bps/Hz")
        except Exception as e:
            print(f"DNN Error: {str(e)}")
            results['DNN']['crlb'].append(np.nan)
            results['DNN']['sum_rate'].append(np.nan)

        for method in methods[1:]:
            try:

                weights = traditional_optimizer(method, Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)  # 不再需要.squeeze()
                # 在计算前转换为 PyTorch 张量
                weights = torch.from_numpy(weights)  # 转换为 PyTorch 张量
                weights = weights.to(torch.complex64)
                # Calculate CRLB
                precoder = weights[:num_antennas] + 1j * weights[num_antennas:]
                precoder = precoder / torch.norm(precoder)
                crlb = calculate_CRLB(num_antennas, num_receiver_antennas,
                                      target_angles[0], precoder, snr_db, 1)
                sum_rate = calculate_sum_rate(weights, user_angles, snr_db)
                results[method]['crlb'].append(crlb.item())
                results[method]['sum_rate'].append(sum_rate)
                print(
                    f"{method}: CRLB={results[method]['crlb'][-1]:.2f}°, Rate={results[method]['sum_rate'][-1]:.2f}bps/Hz")
            except Exception as e:
                print(f"{method} Error: {str(e)}")
                results[method]['crlb'].append(np.nan)
                results[method]['sum_rate'].append(np.nan)

    # == == == == == == == == == == == Visualize the results == == == == == == == == == == == ==
    # plt.figure(figsize=(12, 7))
    plt.rcParams.update({'font.size': 20})

    # Create the graph and the first Y-axis
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Communication-Sensing Weight (ρ)", fontsize=20)
    ax1.set_ylabel("CRLB (°)", fontsize=20)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Draw Sum Rate (Right Y Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Sum Rate (bps/Hz)", fontsize=20)

    # Color palette for methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Line styles - CRLB solid, sum rate dashed
    crlb_style = '-'
    rate_style = '--'

    # Markers for each method
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']

    for i, method in enumerate(methods):
        if method in results:
            # Set line width - thicker for DNN
            line_width = 5 if method == 'DNN' else 2

            # Only plot CRLB for non-ZF/MMSE methods
            if method not in ['ZF', 'MMSE']:
                # CRLB Curve (Primary Axis - solid)
                ax1.plot(rho_values, results[method]['crlb'],
                         marker=markers[i], linestyle=crlb_style,
                         color=colors[i], markersize=8, linewidth=line_width,
                         label=f'{method} CRLB')

            # Sum Rate Curve (Secondary Axis - dashed)
            ax2.plot(rho_values, results[method]['sum_rate'],
                     marker=markers[i], linestyle=rate_style,
                     color=colors[i], markersize=8, linewidth=line_width,
                     label=f'{method} Sum Rate')

    # Set axis limits
    crlb_max = max(max(res['crlb']) for res in results.values())
    rate_max = max(max(res['sum_rate']) for res in results.values())
    ax1.set_ylim(0, crlb_max * 1.4)
    ax2.set_ylim(0, rate_max * 1.4)
    ax1.set_xlim(0, 1.0)  # Strictly [0,1] range

    # Merge and organize legend into two columns
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine all lines and labels
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    # Create legend with two columns
    ax1.legend(all_lines, all_labels,
               loc='upper left',
               ncol=2,
               fontsize=12,
               framealpha=1.0)

    # Adjust layout
    fig.tight_layout()

    # Save & Display
    plt.savefig('crlb_sumrate_comparison_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
