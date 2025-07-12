# ====================== main.py ======================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.optim as optim
from config import *
from models.FBN_model import FairBeamformingNet, check_model_exists, MultiTaskLoss, get_model_name
from models.training import generate_training_data
from algorithms.traditional import ZFBeamformer, MMSEBeamformer
from algorithms.optimization import traditional_optimizer
from normfun.evaluation import calculate_user_rates, evaluate_beamforming, calculate_CRLB, calculate_ber, time_method
# from normfun.visualization import plot_beam_pattern, plot_metric_comparison
from models.channel_data_generate import generate_sensing_channel,generate_communication_channel
from models.training import train_model
from normfun.print_data import *


# # ====================== Model training ======================
# def train_model(rho = 0.8):
#     model = FairBeamformingNet(input_size=len(user_angles + target_angles))
#     criterion = MultiTaskLoss(user_angles, target_angles, rho = rho)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
#     # Generate standardized model names
#     model_name = get_model_name(user_angles, target_angles, num_antennas, rho)
#
#     # Add a learning rate scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#
#     # Generate training data
#     X_train, Y_train = generate_training_data()
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(X_train, Y_train),
#         batch_size=32, shuffle=True
#     )
#
#     best_loss = float('inf')
#     for epoch in range(100):
#         model.train()
#         total_loss = 0
#         for x_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(x_batch)
#             loss = criterion(outputs)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(train_loader)
#         scheduler.step(avg_loss)
#
#         # Early stop mechanism
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), model_name)
#             patience = 0
#         else:
#             patience += 1
#             if patience >= 10:
#                 print(f"Early stopping at epoch {epoch}")
#                 break
#
#         print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
#
#     return model, model_name

def plot_dnn_semicircular_beam_pattern(weights):
    plt.figure(figsize=(10, 10))

    # Parameter settings
    radius = 10  # Half-circle radius
    center = (0, 10)  # Base station coordinates

    # Get beam pattern data
    theta_range = np.linspace(-90, 90, 361)
    pattern = evaluate_beamforming(weights, plot_annotations=False)


    # Convert the dB mode to a linear scale and normalize it
    pattern_linear = 10 ** (pattern / 20)
    pattern_linear = (pattern_linear - np.min(pattern_linear)) / (
            np.max(pattern_linear) - np.min(pattern_linear)) * radius

    # Filter the half-circle angle range and rotate it by 90 degrees
    semicircle_mask = (theta_range >= -90) & (theta_range <= 90)
    semicircle_theta = theta_range[semicircle_mask]
    semicircle_pattern = pattern_linear[semicircle_mask]

    print_beam_pattern_data(theta_range[semicircle_mask], pattern[semicircle_mask], "DNN")

    # Rotate 90 degrees (0° pointing to the right) and convert to radians
    plot_angles = np.deg2rad(-semicircle_theta)

    # Calculate coordinates
    x = center[0] + semicircle_pattern * np.cos(plot_angles)
    y = center[1] + semicircle_pattern * np.sin(plot_angles)

    # Draw a semicircular beammap
    plt.plot(x, y, color='#1f77b4', linewidth=3, label='Beampattern')
    plt.fill_betweenx(y, center[0], x, color='#1f77b4', alpha=0.15)

    # 绘制基站（长方形，宽//x轴，长//y轴）
    bs_width = 1.5  # 长方形的宽（x轴方向）
    bs_height = 5 # 长方形的长（y轴方向）
    bs_center = (-1.55, 10)  # 中心点

    # 绘制基站长方形
    bs_rect = plt.Rectangle(
        (bs_center[0] - bs_width / 2, bs_center[1] - bs_height / 2),  # 左下角坐标
        bs_width, bs_height,  # 宽和高
        edgecolor='black', linewidth=2, label='Base Station'
    )
    plt.gca().add_patch(bs_rect)

    # 在右侧长边上绘制 Y 形天线
    num_antennas = 6  # 天线数量
    antenna_length = 0.8  # 天线长度
    antenna_spacing = bs_height / (num_antennas + 1)  # 天线间距

    for i in range(num_antennas):
        # 天线根部位置（右侧边的中点）
        ant_base_x = bs_center[0] + bs_width / 2
        ant_base_y = bs_center[1] - bs_height / 2 + (i + 1) * antenna_spacing

        # Y 形天线的三个分支（斜向上、斜向下、主干）
        plt.plot(
            [ant_base_x, ant_base_x + antenna_length*0.5],  # 主干（水平向右）
            [ant_base_y, ant_base_y],
            color='black', linewidth=1.5
        )
        plt.plot(
            [ant_base_x + antenna_length * 0.5, ant_base_x + antenna_length],  # 斜向上
            [ant_base_y, ant_base_y + antenna_length * 0.3],
            color='black', linewidth=1.2
        )
        plt.plot(
            [ant_base_x + antenna_length * 0.5, ant_base_x + antenna_length],  # 斜向下
            [ant_base_y, ant_base_y - antenna_length * 0.3],
            color='black', linewidth=1.2
        )

    distance = [12, 17, 15, 11]
    user_labels = ['UE1', 'UE2', 'UE3', 'UE4']

    # Draw the communication user
    for i, (angle, label) in enumerate(zip(user_angles, user_labels)):
        plot_angle = np.deg2rad(-angle)
        patient_x = center[0] + distance[i] * np.cos(plot_angle)
        patient_y = center[1] + distance[i] * np.sin(plot_angle)
        plt.scatter(patient_x, patient_y, color='#2ca02c', s=150,
                    edgecolor='black', linewidth=1, label=f'Communicating User' if i == 0 else "")
        plt.text(patient_x + 0.5, patient_y + 0.5, label, fontsize=24, fontweight='bold')

    # Draw the sensing target
    for i, angle in enumerate(target_angles):
        plot_angle = np.deg2rad(-angle)
        target_x = center[0] + 11 * np.cos(plot_angle)
        target_y = center[1] + 11 * np.sin(plot_angle)
        plt.scatter(target_x, target_y, color='#ff7f0e', s=200, marker='X',
                    edgecolor='black', linewidth=1.5, label=f'Sensing Target' if i == 0 else "")
        plt.text(target_x + 0.5, target_y + 0.5, 'Target1', fontsize=24, fontweight='bold')

    # Set the graphics properties
    plt.xlim(-3, 20)
    plt.ylim(0, 20)
    plt.xlabel('Distance (meters)', fontsize=24, fontweight='bold')
    plt.ylabel('Distance (meters)', fontsize=24, fontweight='bold')
    # plt.title('DNN-based beamforming',
    #           fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create a legend in upper right corner
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    plt.legend(unique_handles, unique_labels, loc='upper right',
               framealpha=1, edgecolor='black', fontsize=15)

    plt.tight_layout()
    plt.savefig('DNN-based beamforming.png', dpi=300, bbox_inches='tight')
    plt.show()



# ====================== Main function ======================
def main():
    # Prioritize checking existing models
    model_name = check_model_exists(user_angles, target_angles, num_antennas, rho)

    if model_name:
        # Load an existing model
        model = FairBeamformingNet(input_size, hidden_size=512, max_users=num_users).to(device)
        try:
            model.load_state_dict(torch.load(model_name))
            print(f"The pretrained model was successfully loaded: {model_name}")
        except Exception as e:
            print(f"The model fails to load and will be retrained. error message: {str(e)}")
            model, model_name = train_model(rho_set=rho)
    else:
        # 训练新模型
        print("No matching pre-trained model found, start training a new model...")
        model, model_name = train_model(rho_set=rho)

    # test_input = torch.tensor([user_angles + target_angles], dtype=torch.float32)

    # test_Hc = generate_communication_channel(num_antennas, user_angles, random_seed=SEED)
    # test_Hs = generate_sensing_channel(num_antennas, target_angles, random_seed=SEED)
    test_Hc = generate_communication_channel(num_antennas, user_angles)
    test_Hs = generate_sensing_channel(num_antennas, target_angles)

    # Prepare the input data
    Hc_r = torch.FloatTensor(test_Hc.real).unsqueeze(0).to(device)
    Hc_i = torch.FloatTensor(test_Hc.imag).unsqueeze(0).to(device)
    Hs_r = torch.FloatTensor(test_Hs.real).unsqueeze(0).to(device)
    Hs_i = torch.FloatTensor(test_Hs.imag).unsqueeze(0).to(device)
    rho_tensor = torch.FloatTensor([rho]).to(device)

    # Get the weights of all methods
    with torch.no_grad():
        # dl_weights = model(test_input).numpy()[0]
        dl_weights = model(Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor).numpy()[0]
        de_weights = traditional_optimizer('DE', Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)
        pso_weights = traditional_optimizer('PSO', Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)
        gwo_weights = traditional_optimizer('GWO', Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)
        woa_weights = traditional_optimizer('WOA', Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)
        zf_weights = traditional_optimizer('ZF', Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)
        mmse_weights = traditional_optimizer('MMSE', Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor)

    plot_dnn_semicircular_beam_pattern(dl_weights)

    # plot_constellation_dnn(dl_weights, user_angles, snr_db=10)

    # ========== Plot Combined Beam Patterns ==========
    plt.figure(figsize=(12, 6))
    ax_combined = plt.gca()


    # Define methods with their corresponding colors and line styles
    methods = [
        (dl_weights, "Deep Learning", '#FF6B6B', '-'),  # Red, Solid
        (de_weights, "Differential Evolution", '#4D96FF', '--'),  # Blue, Dashed
        (pso_weights, "PSO", '#6BCB77', '-.'),  # Green, Dash-dot
        # (cs_weights, "Cuckoo Search", '#FFD700', ':'),  # Gold, Dotted
        (gwo_weights, "Grey Wolf Optimizer", '#8A2BE2', (0, (3, 1, 1, 1))),  # Purple, Custom dash
        (woa_weights, "Whale Optimization Algorithm", '#FF1493', (0, (5, 5))),  # Pink, Loosely dashed
        (zf_weights, "ZF Beamforming", '#00CED1', (0, (1, 1))),
        (mmse_weights, "MMSE Beamforming", '#FF8C00', (0, (3, 5, 1, 5)))
    ]
    # print(zf_weights.shape)



    # Plot beam patterns for all methods
    for weights, label, color, linestyle in methods:
        print(label)
        # print(weights.shape)
        evaluate_beamforming(weights, label=label, ax=ax_combined, is_combined=True, color=color, linestyle=linestyle)

    # Add user and target annotations
    # User angle colors (more visible)
    user_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Target angle colors (more visible)
    target_colors = ['#ff9896', '#98df8a', '#ffbb78', '#c5b0d5', '#dbdb8d']
    from matplotlib.lines import Line2D
    legend_elements = [
                          Line2D([0], [0], color=user_colors[idx], linestyle='--', label=f'User {idx + 1}')
                          for idx in range(len(user_angles))
                      ] + [
                          Line2D([0], [0], color=target_colors[idx], linestyle='-.', label=f'Target {idx + 1}')
                          for idx in range(len(target_angles))
                      ]

    # Add legend for methods
    method_legend = ax_combined.legend(loc='lower right', title="Methods")
    ax_combined.add_artist(method_legend)  # Add method legend first
    ax_combined.legend(handles=legend_elements, loc='lower left', title="Annotations")  # Add annotations legend

    # ax_combined.set_title("Beam Pattern Comparison")
    plt.xlabel("Angle (degree)")
    plt.ylabel("Beampattern (dB)")
    plt.xlim(-90, 90)
    plt.tight_layout()
    plt.show()

    # ========== Draw individual beammaps ==========
    num_methods = len(methods)
    rows = int(np.ceil(np.sqrt(num_methods)))
    cols = int(np.ceil(num_methods / rows))

    plt.figure(figsize=(5 * cols, 5 * rows))
    for idx, (weights, method, color, linestyle) in enumerate(methods, 1):
        ax = plt.subplot(rows, cols, idx)
        evaluate_beamforming(weights, label=method, ax=ax, color=color, linestyle=linestyle)
        ax.set_title(f"{method} Beam Pattern")
    plt.tight_layout()


    plt.tight_layout()
    plt.show()



    # Rate comparison
    user_steering_vectors = [
        np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
        for angle in user_angles
    ]

    # methods = ["Deep Learning", "DE", "PSO", "CS", "GWO", "WOA", "ZF", "MMSE"]
    methods = ["Deep Learning", "DE", "PSO", "GWO", "WOA", "ZF", "MMSE"]
    weights_dict = {
        "Deep Learning": dl_weights,
        "DE": de_weights,
        "PSO": pso_weights,
        # "CS": cs_weights,
        "GWO": gwo_weights,
        "WOA": woa_weights,
        "ZF": zf_weights,
        "MMSE": mmse_weights
    }
    sum_rates = {method: [] for method in methods}
    user_rates = {method: {f"User {i + 1}": [] for i in range(len(user_angles))} for method in methods}

    for method in methods:
        print(f"\nCalculating rates for {method}...")
        weights = weights_dict[method]
        for snr_db in snr_dBs:
            rates = calculate_user_rates(weights, user_steering_vectors, snr_db)
            sum_rates[method].append(np.sum(rates))
            for i, rate in enumerate(rates):
                user_key = f"User {i + 1}"
                user_rates[method][user_key].append(rate)
                print(f"[Debug] Stored {user_key} rate: {rate} at SNR {snr_db}dB")

    # Data validation
    for method in methods:
        print(f"\nData verification for {method}:")
        for user in user_rates[method]:
            data_points = len(user_rates[method][user])
            print(f"  {user}: {data_points} data points")
            if data_points != len(snr_dBs):
                print(f"  !!! Data length mismatch for {user}")

    # Drawing corrections
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4D96FF', '#6BCB77', '#FFD700', '#8A2BE2', '#FF1493', '#00CED1', '#FF8C00']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    for idx, (method, rates) in enumerate(sum_rates.items()):
        plt.plot(snr_dBs, rates, color=colors[idx], marker=markers[idx], label=method)
    plt.title("Sum Rate Comparison")
    plt.xlabel("SNR (dB)", fontsize=20)
    plt.ylabel("Sum Rate (bps/Hz)", fontsize=20)
    plt.xlim(0, 10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # 在性能分析部分添加能效可视化
    plot_efficiency_comparison(methods, weights_dict, snr_dBs, colors, markers, user_steering_vectors)
    plot_power_consumption(methods, weights_dict, colors)

    # 每个用户的速率
    # plt.figure(figsize=(15, 10))
    # users = sorted(user_rates["Deep Learning"].keys(), key=lambda x: int(x.split()[1]))
    # for idx, user in enumerate(users, 1):
    #     plt.subplot(2, 2, idx)
    #     for method_idx, (method, rates_dict) in enumerate(user_rates.items()):
    #         plt.plot(snr_dBs, rates_dict[user],
    #                  color=colors[method_idx],
    #                  marker=markers[method_idx],
    #                  linewidth=2,
    #                  label=method)
    #     plt.title(f"{user} Rate")
    #     plt.xlabel("SNR (dB)")
    #     plt.ylabel("Rate (bps/Hz)")
    #     plt.grid(alpha=0.3)
    #     plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ========== 新增性能分析图 ==========

    # Calculate the bit error rate and CRLB
    ber_results = {method: [] for method in methods}
    mse_results = {method: [] for method in methods}
    crlb_results ={method: [] for method in methods}

    user_steering_vectors = [
        np.exp(1j * 2 * np.pi * d * np.arange(num_antennas) * np.sin(np.deg2rad(angle)) / wavelength)
        for angle in user_angles
    ]

    for method in methods:
        print(f"\nCalculating BER and CRLB for {method}...")
        weights = weights_dict[method]
        for snr_db in snr_dBs:
            # Calculate the bit error rate
            ber = calculate_ber(weights, user_steering_vectors, snr_db)
            ber_results[method].append(ber)

            # # 计算波束MSE
            # mse = calculate_beam_mse(weights, user_angles, target_angles)
            # mse_results[method].append(mse)

            # Calculate CRLB (only when there is a target)
            if len(target_angles) > 0:
                # Convert to complex precoder
                precoder = torch.tensor(weights[:num_antennas]) + 1j * torch.tensor(weights[num_antennas:])
                precoder = precoder / torch.norm(precoder)

                crlb = calculate_CRLB(num_antennas, num_receiver_antennas,
                                      target_angles[0], precoder,snr_db)
                crlb_results[method].append(crlb)
                print(f"snr ={snr_db:.1f}: CRLB = {crlb:.4f}°")
            else:
                crlb_results[method].append(0)  # Set to 0 when there is no target

    # Plot the bit error rate curve
    plt.figure(figsize=(10, 6))

    for idx, method in enumerate(methods):
        plt.semilogy(snr_dBs, ber_results[method],
                     color=colors[idx],
                     marker=markers[idx],
                     label=method,
                     linewidth=2)

    # plt.title("误码率(BER)随SNR变化曲线", fontsize=12)
    plt.xlabel("SNR (dB)", fontsize=20)
    plt.ylabel("BER", fontsize=20)
    plt.xlim(0, 10)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=9)
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.show()

    # # 绘制波束MSE曲线
    # plt.figure(figsize=(10, 6))
    # for idx, method in enumerate(methods):
    #     plt.plot(snr_dBs, mse_results[method],
    #              color=colors[idx],
    #              marker=markers[idx],
    #              label=method,
    #              linewidth=2)
    #
    # plt.title("波束成形MSE随SNR变化曲线", fontsize=12)
    # plt.xlabel("SNR (dB)", fontsize=10)
    # plt.ylabel("波束MSE", fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend(fontsize=9)
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.show()

    # Added CRLB plots
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", n_colors=len(methods))

    for idx, method in enumerate(methods):
        if len(target_angles) == 0:  # Skip aimless situations
            continue

        plt.plot(snr_dBs,crlb_results[method],
                 color=colors[idx], marker=markers[idx], linestyle='-',
                 linewidth=2, markersize=8, label=method)

    # plt.title("角度估计性能比较 (CRLB)")
    # plt.ylim(1e-3, 1e5)  # 添加固定范围
    plt.xlabel("SNR (dB)", fontsize=20)
    plt.ylabel("CRLB", fontsize=20)
    plt.xlim(0, 10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_efficiency_comparison(methods, weights_dict, snr_dBs, colors, markers, user_steering_vectors):
    """绘制能效比较图"""
    plt.figure(figsize=(10, 6))

    # 计算各方法在不同SNR下的能效
    ee_results = {method: [] for method in methods}

    for method in methods:
        weights = weights_dict[method]
        for snr_db in snr_dBs:
            # 计算能效 (简化的能效计算)
            transmit_power = np.sum(np.abs(weights) ** 2) / power_params['power_scaling']
            total_power = transmit_power / power_params['PA_efficiency'] + power_params['circuit_power']

            # 计算速率
            rates = calculate_user_rates(weights, user_steering_vectors, snr_db)
            sum_rate = np.sum(rates)

            # 能效 = 总速率/总功耗 (bps/Hz/W)
            ee = sum_rate / total_power
            ee_results[method].append(ee)

    # 绘制能效曲线
    for idx, method in enumerate(methods):
        plt.plot(snr_dBs, ee_results[method],
                 color=colors[idx],
                 marker=markers[idx],
                 linewidth=2,
                 label=method)

    plt.xlabel("SNR (dB)", fontsize=20)
    plt.ylabel("Energy Efficiency (bps/Hz/W)", fontsize=20)
    plt.xlim(0, 10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_power_consumption(methods, weights_dict, colors):
    """绘制功耗比较图"""
    plt.figure(figsize=(10, 6))

    power_consumption = []
    labels = []

    for method in methods:
        weights = weights_dict[method]
        # 计算发射功率 (W)
        transmit_power = np.sum(np.abs(weights) ** 2) / power_params['power_scaling']
        # 计算总功耗
        total_power = transmit_power / power_params['PA_efficiency'] + power_params['circuit_power']

        power_consumption.append(total_power)
        labels.append(method)

    # 绘制柱状图
    bars = plt.bar(labels, power_consumption, color=colors[:len(methods)])

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f} W', ha='center', va='bottom')

    plt.ylabel("Total Power Consumption (W)", fontsize=20)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=20)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    # # 优先检查现有模型
    # model_name = check_model_exists(user_angles, target_angles, num_antennas, rho)
    #
    # if model_name:
    #     # 加载现有模型
    #     model = FairBeamformingNet(input_size=len(user_angles + target_angles))
    #     try:
    #         model.load_state_dict(torch.load(model_name))
    #         print(f"成功加载预训练模型: {model_name}")
    #     except Exception as e:
    #         print(f"模型加载失败，将重新训练。错误信息: {str(e)}")
    #         model, model_name = train_model(rho = rho)
    # else:
    #     # 训练新模型
    #     print("未找到匹配的预训练模型，开始训练新模型...")
    #     model, model_name = train_model(rho = rho)
    #
    # test_input = torch.tensor([user_angles + target_angles], dtype=torch.float32)
    #
    # # 获取所有方法的权重
    # with torch.no_grad():
    #     dl_weights = model(test_input).numpy()[0]
    # de_weights = traditional_optimizer('DE')
    # pso_weights = traditional_optimizer('PSO')
    # # cs_weights = traditional_optimizer('CS')
    # gwo_weights = traditional_optimizer('GWO')
    # woa_weights = traditional_optimizer('WOA')
    # zf_weights = traditional_optimizer('ZF')
    # mmse_weights = traditional_optimizer('MMSE')
    #
    #
    # # ================== run time ==================
    # methods = {
    #     "Deep Learning": lambda: model(test_input),  # 假设model已定义
    #     "Differential Evolution": lambda: traditional_optimizer('DE'),
    #     "PSO": lambda: traditional_optimizer('PSO'),
    #     # "Cuckoo Search": lambda: traditional_optimizer('CS'),
    #     "Grey Wolf": lambda: traditional_optimizer('GWO'),
    #     "Whale Optimization": lambda: traditional_optimizer('WOA'),
    #     "ZF": lambda: traditional_optimizer('ZF'),
    #     "MMSE": lambda: traditional_optimizer('MMSE')
    # }
    #
    # # 测试不同天线数量下的运行时间
    # # antenna_configs = [8, 16, 32, 64]  # 测试不同天线规模
    # antenna_configs = [16]  # 测试不同天线规模
    # results = {name: [] for name in methods}
    #
    # # ================== 运行测试 ==================
    # print("开始运行时间测试...")
    # for num_antennas in tqdm(antenna_configs, desc="天线配置"):
    #     # 更新全局参数（实际代码需调整您的全局变量）
    #     globals()['num_antennas'] = num_antennas
    #
    #     for name, func in tqdm(methods.items(), desc="方法", leave=False):
    #         # 深度学习的测试需要特殊处理
    #         if name == "Deep Learning":
    #             if num_antennas != 16:  # 假设模型是16天线训练的
    #                 results[name].append(np.nan)
    #                 continue
    #
    #         avg_time = time_method(func)
    #         results[name].append(avg_time)
    #
    # # ================== 可视化结果 ==================
    # plt.figure(figsize=(12, 6))
    #
    # # 柱状图（比较不同方法在16天线时的绝对时间）
    # # plt.subplot(1, 2, 1)
    # plt.subplot(1, 1, 1)
    # bars = plt.bar(methods.keys(), [results[name][antenna_configs.index(16)] for name in methods])
    # plt.title("Comparison of runtime at 32 antennas")
    # plt.ylabel("time(s)")
    # plt.xticks(rotation=45)
    # plt.yscale('log')  # 对数坐标显示数量级差异
    #
    # # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height,
    #              f'{height:.3f}s', ha='center', va='bottom')
    #
    # plt.show()