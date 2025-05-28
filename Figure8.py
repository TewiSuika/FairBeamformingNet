import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
from scipy.optimize import differential_evolution
from pyswarm import pso
from models.channel_data_generate import generate_sensing_channel,generate_communication_channel

# ====================== System parameter configuration =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_angles = [-15, 15, 30, 45]
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


# ====================== Model Loading Function ========================
class FairBeamformingNet(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_users=4):
        super().__init__()
        self.num_users = num_users

        self.shared_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.user_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_antennas * 2)
            ) for _ in range(num_users)
        ])


        self.norm = nn.LayerNorm(num_antennas * 2)

    def forward(self, Hc_real, Hc_imag, Hs_real, Hs_imag, rho):

        Hc = torch.cat([Hc_real.flatten(1), Hc_imag.flatten(1)], dim=1)
        Hs = torch.cat([Hs_real.flatten(1), Hs_imag.flatten(1)], dim=1)
        x = torch.cat([Hc, Hs, rho.view(-1, 1)], dim=1)

        x = self.shared_net(x)
        branch_outputs = [branch(x) for branch in self.user_branches]
        combined = torch.mean(torch.stack(branch_outputs), dim=0)

        return self.norm(combined)


# ====================== load_model ========================
def load_model(rho):
    model_name = (
        f"model_U{num_users}_S{len(target_angles)}_A{num_antennas}_rho{rho:.2f}.pth"
    ).replace(".", "_")
    # model = FairBeamformingNet(input_size=num_users + len(target_angles))

    input_size = 2 * (num_users * num_antennas + num_targets * num_antennas) + 1
    model = FairBeamformingNet(input_size, hidden_size=512, num_users=num_users).to(device)
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

# ====================== 优化算法实现 ======================
# def cuckoo_search(objective, bounds, num_cuckoos=15, max_iter=100, pa=0.25):
#     dim = len(bounds)
#     nests = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(num_cuckoos)])
#     fitness = np.array([objective(nest) for nest in nests])
#     best_nest = nests[np.argmin(fitness)]
#     best_fitness = np.min(fitness)
#
#     for _ in range(max_iter):
#         for i in range(num_cuckoos):
#             step_size = 0.01 * (nests[i] - best_nest) * np.random.standard_cauchy(dim)
#             new_nest = nests[i] + step_size
#             new_nest = np.clip(new_nest, [b[0] for b in bounds], [b[1] for b in bounds])
#             new_fitness = objective(new_nest)
#             if new_fitness < fitness[i]:
#                 nests[i] = new_nest
#                 fitness[i] = new_fitness
#
#         for i in range(num_cuckoos):
#             if np.random.rand() < pa:
#                 nests[i] = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
#                 fitness[i] = objective(nests[i])
#
#         if np.min(fitness) < best_fitness:
#             best_nest = nests[np.argmin(fitness)]
#             best_fitness = np.min(fitness)
#     return best_nest


def grey_wolf_optimizer(objective, bounds, num_wolves=30, max_iter=100):
    dim = len(bounds)
    wolves = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(num_wolves)])
    fitness = np.array([objective(wolf) for wolf in wolves])
    alpha, beta, delta = wolves[np.argsort(fitness)[:3]]
    alpha_fitness, beta_fitness, delta_fitness = np.sort(fitness)[:3]

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        for i in range(num_wolves):
            A1 = a * (2 * np.random.rand(dim) - 1)
            C1 = 2 * np.random.rand(dim)
            X1 = alpha - A1 * np.abs(C1 * alpha - wolves[i])
            A2 = a * (2 * np.random.rand(dim) - 1)
            C2 = 2 * np.random.rand(dim)
            X2 = beta - A2 * np.abs(C2 * beta - wolves[i])
            A3 = a * (2 * np.random.rand(dim) - 1)
            C3 = 2 * np.random.rand(dim)
            X3 = delta - A3 * np.abs(C3 * delta - wolves[i])
            new_wolf = (X1 + X2 + X3) / 3
            new_wolf = np.clip(new_wolf, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fitness = objective(new_wolf)
            if new_fitness < fitness[i]:
                wolves[i] = new_wolf
                fitness[i] = new_fitness

        sorted_indices = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_indices[:3]]
        alpha_fitness, beta_fitness, delta_fitness = fitness[sorted_indices[:3]]
    return alpha


def whale_optimization_algorithm(objective, bounds, num_whales=30, max_iter=100):
    dim = len(bounds)
    whales = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(num_whales)])
    fitness = np.array([objective(whale) for whale in whales])
    best_whale = whales[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)
        a2 = -1 + t * (-1 / max_iter)
        for i in range(num_whales):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = (a2 - 1) * np.random.rand() + 1
            p = np.random.rand()

            if p < 0.5:
                if np.abs(A) < 1:
                    D = np.abs(C * best_whale - whales[i])
                    new_whale = best_whale - A * D
                else:
                    rand_index = np.random.randint(0, num_whales)
                    rand_whale = whales[rand_index]
                    D = np.abs(C * rand_whale - whales[i])
                    new_whale = rand_whale - A * D
            else:
                D = np.abs(best_whale - whales[i])
                new_whale = D * np.exp(l) * np.cos(2 * np.pi * l) + best_whale

            new_whale = np.clip(new_whale, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fitness = objective(new_whale)
            if new_fitness < fitness[i]:
                whales[i] = new_whale
                fitness[i] = new_fitness

        if np.min(fitness) < best_fitness:
            best_whale = whales[np.argmin(fitness)]
            best_fitness = np.min(fitness)
    return best_whale

# ====================== MMSE ======================
class MMSEBeamformer:

    def __init__(self, user_angles, snr_db=10):
        self.user_angles = user_angles
        self.snr_db = snr_db
        self.H = np.array([
            [np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(-angle)) / wavelength)
             for n in range(num_antennas)]
            for angle in user_angles
        ])

    def get_weights(self):
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = 1.0 / snr_linear

        H = self.H
        I = np.eye(num_antennas)
        mmse_weights = np.linalg.inv(H.conj().T @ H + noise_power * I) @ H.conj().T

        W_mmse = mmse_weights.mean(axis=1)

        real_part = np.real(W_mmse).flatten()
        imag_part = np.imag(W_mmse).flatten()

        return np.concatenate([real_part, imag_part])


# ====================== ZF ======================
class ZFBeamformer:

    def __init__(self, user_angles):
        self.user_angles = user_angles
        self.H = np.array([
            [np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(-angle)) / wavelength)
             for n in range(num_antennas)]
            for angle in user_angles
        ])
        self.H_pinv = np.linalg.pinv(self.H)

    def get_weights_for_jcas(self, target_angles, rho=0.8):
        W_comm = self.H_pinv.mean(axis=1)  # 形状(num_antennas,)

        # a_target = np.array([
        #     [np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
        #      for n in range(num_antennas)]
        #     for angle in target_angles
        # ]).mean(axis=0)  # 形状(num_antennas,)

        # W_joint = alpha * W_comm + (1 - alpha) * a_target  # 形状(num_antennas,)
        W_joint = W_comm

        real_part = np.real(W_joint).flatten()  # 形状(num_antennas,)
        imag_part = np.imag(W_joint).flatten()  # 形状(num_antennas,)

        return np.concatenate([real_part, imag_part])  # 形状(2*num_antennas,)

    def apply(self, weights):
        w_cplx = weights[:, :num_antennas] + 1j * weights[:, num_antennas:]
        w_zf = w_cplx @ self.H_pinv
        return np.concatenate([w_zf.real, w_zf.imag], axis=1)

# def objective(w):
#     w_cplx = w[:num_antennas] + 1j * w[num_antennas:]
#     gains = []
#     for angle in user_angles + target_angles:
#         sv = np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
#         gains.append(np.abs(w_cplx @ sv.conj()))
#     min_gain = np.min(gains[:len(user_angles)])
#     return -min_gain + 0.1 * np.sum(gains)


# ====================== Objective function ======================
def objective(w, rho_set=0.1):
    """Objective function with communication-sensing weights"""
    w_cplx = w[:num_antennas] + 1j * w[num_antennas:]

    # Communication Performance Calculation (User Direction)
    user_gains = []
    for angle in user_angles:
        sv = np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
        user_gains.append(np.abs(w_cplx @ sv.conj()))
        min_user_gain = np.min(user_gains)
        sum_user_gain = np.sum(user_gains)

    # Perceptual Performance Computing (Target Direction)
    target_gains = []
    for angle in target_angles:
        sv = np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
    target_gains.append(np.abs(w_cplx @ sv.conj()))
    avg_target_gain = np.mean(target_gains)

    # Combine multiple targets
    comm_perf = min_user_gain + 0.1 * sum_user_gain
    sens_perf = avg_target_gain
    return -(rho_set * 2.5 * comm_perf + (1 - rho_set) * sens_perf)
#
#
def traditional_optimizer(method='ZF', rho=0.5):
    bounds = [(-1, 1)] * (2 * num_antennas)

    def torch_objective(w):
        w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        return objective(w_tensor, rho).item()

    if method == 'DE':
        result = differential_evolution(torch_objective, bounds, maxiter=100, popsize=15)
        weights = torch.tensor(result.x, dtype=torch.float32, device=device)
    elif method == 'PSO':
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        xopt, _ = pso(torch_objective, lb, ub, swarmsize=30, maxiter=100)
        weights = torch.tensor(xopt, dtype=torch.float32, device=device)
    # elif method == 'CS':
    #     xopt = cuckoo_search(torch_objective, bounds)
    #     weights = torch.tensor(xopt, dtype=torch.float32, device=device)
    elif method == 'GWO':
        xopt = grey_wolf_optimizer(torch_objective, bounds)
        weights = torch.tensor(xopt, dtype=torch.float32, device=device)
    elif method == 'WOA':
        xopt = whale_optimization_algorithm(torch_objective, bounds)
        weights = torch.tensor(xopt, dtype=torch.float32, device=device)
    elif method == 'ZF':
        zf_bf = ZFBeamformer(user_angles)
        weights = zf_bf.get_weights_for_jcas(target_angles, rho=rho)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    elif method == 'MMSE':
        mmse_bf = MMSEBeamformer(user_angles)
        weights = mmse_bf.get_weights()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


# ====================== The main execution process ========================
def main():
    # methods = ['DNN', 'PSO', 'DE', 'CS', 'GWO', 'WOA', 'ZF', 'MMSE']
    methods = ['DNN', 'PSO', 'DE', 'GWO', 'WOA', 'ZF', 'MMSE']
    results = {m: {'crlb': [], 'sum_rate': []} for m in methods}

    for rho in rho_values:
        print(f"\nProcessing ρ={rho:.1f}")

        # DNN方法
        try:
            model = load_model(rho)
            test_Hc = generate_communication_channel(num_antennas, user_angles)
            test_Hs = generate_sensing_channel(num_antennas, target_angles)

            # 准备输入数据
            Hc_r = torch.FloatTensor(test_Hc.real).unsqueeze(0).to(device)
            Hc_i = torch.FloatTensor(test_Hc.imag).unsqueeze(0).to(device)
            Hs_r = torch.FloatTensor(test_Hs.real).unsqueeze(0).to(device)
            Hs_i = torch.FloatTensor(test_Hs.imag).unsqueeze(0).to(device)
            rho_tensor = torch.FloatTensor([rho]).to(device)

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

                weights = traditional_optimizer(method, rho)  # 不再需要.squeeze()

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

    # Create the graph and the first Y-axis
    fig, ax1 = plt.subplots(figsize=(12, 7))


    plt.rcParams.update({'font.size': 20})

    # Draw CRLB (Left Y-axis - Blue)
    color_crlb = '#2E86C1'
    ax1.set_xlabel("Communication-Sensing Weight (ρ)", fontsize=20)
    ax1.set_ylabel("CRLB (°)", fontsize=20, color=color_crlb)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.grid(True, linestyle='--', alpha=0.7)


    # Draw Sum Rate (Right Y Axis - Red)
    color_rate = '#E74C3C'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Sum Rate (bps/Hz)", fontsize=20, color=color_rate)

    # Draw a double indicator curve for each method
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']  # 8 different markup styles
    line_styles = [
        '-',
        '--',
        '-.',
        ':',
        (0, (3, 1, 1, 1)),
        (0, (5, 5)),
        (0, (1, 1)),
        (0, (5, 1, 1, 1, 1, 1))
    ]

    for i, method in enumerate(methods):
        if method in results:
            # CRLB Curve (Principal Axis)
            ax1.plot(rho_values, results[method]['crlb'],
                     marker=markers[i], linestyle=line_styles[i],
                     color=color_crlb, markersize=8, linewidth=2,
                     label=f'{method} CRLB')

            # Sum Rate Curve (Minor Axis)
            ax2.plot(rho_values, results[method]['sum_rate'],
                     marker=markers[i], linestyle=line_styles[i],
                     color=color_rate, markersize=8, linewidth=2,
                     label=f'{method} Sum Rate')

    # Merge legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper right',
        ncol=1,
        fontsize=10,
        framealpha=0.9
    )

    # Set the axis range
    crlb_max = max(max(res['crlb']) for res in results.values())
    rate_max = max(max(res['sum_rate']) for res in results.values())
    ax1.set_ylim(0, crlb_max * 1.1)
    ax2.set_ylim(0, rate_max * 1.1)


    # Title & Layout
    # plt.title('Joint Communication and Sensing Performance', fontsize=22, pad=20)
    fig.tight_layout()

    # Save & Display
    plt.savefig('crlb_sumrate_comparison_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
