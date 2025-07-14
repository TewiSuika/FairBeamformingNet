"""
Deep learning model definition module
"""
import numpy as np
import torch
import torch.nn as nn
import os
from config import *


class FairBeamformingNet(nn.Module):
    def __init__(self, input_size, hidden_size=512, max_users=num_users, num_rf_chains=4):
        super().__init__()
        self.max_users = max_users
        self.num_rf_chains = num_rf_chains
        self.num_antennas = 16

        # assert max_users <= num_rf_chains, "max_users cannot exceed num_rf_chains"

        # Enter the feature dimension calculation:
        # Hc_real + Hc_imag = 2 * (max_users * num_antennas)
        # rho = 1
        # target_angles = num_targets
        # input = 2 * max_users * num_antennas + 1 + num_targets
        self.input_size = input_size

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Digital branches
        self.digital_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
            ) for _ in range(max_users)
        ])

        # Analog beamformer
        self.analog_beamformer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_antennas*num_rf_chains * 2)
        )

        self.norm = nn.LayerNorm(num_antennas * 2)

    def forward(self, Hc_real, Hc_imag, target_angles, rho,  num_users=None):
        if num_users is None:
            num_users = self.max_users
        else:
            # Make sure num_users is a Python scalar and not a tensor
            if torch.is_tensor(num_users):
                num_users = num_users.item()

        # assert num_users <= self.max_users, f"num_users ({num_users}) > max_users ({self.max_users})"

        # Handle the communication channel
        Hc = torch.cat([Hc_real.flatten(1), Hc_imag.flatten(1)], dim=1)

        # Working with Target Angles (Use All Angle Values)
        # target_angles : [batch_size, num_targets]

        # Stitch together all input features
        x = torch.cat([Hc, target_angles.view(-1, 1), rho.view(-1, 1) ], dim=1)
        x = self.shared_net(x)

        # Generate digital beamforming
        digital_outputs = [self.digital_branches[i](x).view(-1, 1, 2)
                           for i in range(num_users)]
        digital_combined = torch.cat(digital_outputs, dim=1)

        if num_users < self.num_rf_chains:
            padding = torch.zeros(x.size(0),
                                  self.num_rf_chains - num_users,
                                  2, device=x.device)
            digital_combined = torch.cat([digital_combined, padding], dim=1)

        # Generate analog beamforming
        analog_output = self.analog_beamformer(x).view(-1, self.num_rf_chains, self.num_antennas, 2)

        # Hybrid beamforming
        hybrid_output = torch.einsum('brf,brnf->bnf', digital_combined, analog_output)
        hybrid_output = hybrid_output.flatten(1)  # [B, 32]

        return self.norm(hybrid_output)


class MultiTaskLoss(nn.Module):
    def __init__(self, rho=0.7, lambda_reg=0.1, min_power_weight=5.0,
                 d=0.5, wavelength=1.0, num_antennas=16, P_max=1.0):
        super().__init__()
        # Configure parameters
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.min_power_weight = min_power_weight
        self.d = d
        self.wavelength = wavelength
        self.num_antennas = num_antennas
        self.P_max = P_max
        self.adjacent_weight = 0.01  # Neighbor user interference penalty weight

        # Array parameters
        self.n = torch.arange(num_antennas).float()

    def generate_steering_vectors(self, angles_deg):
        """Vectorization to generate guided vectors (batch processing is supported)"""
        device = angles_deg.device
        theta_rad = torch.deg2rad(angles_deg)  # [B, num_angles]
        n = self.n.to(device).view(1, 1, -1)  # [1, 1, A]

        # Phase delay calculations
        phase_delay = 2 * np.pi * self.d * torch.sin(theta_rad).unsqueeze(-1) * n / self.wavelength

        # Generate complex guided vectors
        steering_vec = torch.complex(
            torch.cos(phase_delay),
            torch.sin(phase_delay)
        )

        # Normalization
        norm_factor = torch.sqrt(torch.tensor(self.num_antennas, dtype=torch.float32, device=device))
        return steering_vec / norm_factor

    def forward(self, W, Hc, target_angles):
        # print(W.shape)
        """
        Calculate multitasking losses

        Args:
            W: Beamforming vectors [batch_size, num_antennas*2]
            Hc: Communication Channels (Plural) [batch_size, num_users, num_antennas]
            target_angles: Perceive the target angle [batch_size, num_targets]
        """
        batch_size, num_users, num_antennas = Hc.shape

        # ===== Beamforming vector processing =====
        real_part = W[:, :num_antennas]
        imag_part = W[:, num_antennas:]
        w_cplx = torch.complex(real_part, imag_part)  # [B, A]
        # print(w_cplx.shape)
        # Hc = Hc.squeeze(0).T

        # ===== Communication Performance Calculation (Direct Use of Channel Information) =====
        # Calculate the channel gain for each user: |w^H * h_i|
        # Communication Performance Calculation (using Hc)
        # print(Hc.squeeze(0).T).shape)
        # print(w_cplx.shape)
        # print((Hc[0].T).shape)
        user_gains = torch.abs(w_cplx @ Hc[0].T)
        min_user_gain = torch.min(user_gains)
        sum_user_gain = torch.sum(user_gains)
        # user_gains = torch.zeros(batch_size, num_users, device=W.device)
        # for i in range(num_users):
        #     h_i = Hc[:, i, :]  # [B, A]
        #     #  w^H * h_i
        #     gain = torch.abs(torch.sum(w_cplx.conj() * h_i, dim=1))  # [B]
        #     user_gains[:, i] = gain
        #
        # # Minimum user gain and total user gain
        # min_user_gain = torch.min(user_gains, dim=1)[0]  # [B]
        # sum_user_gain = torch.sum(user_gains, dim=1)  # [B]

        # Neighbor User Interference Calculations (Directly using channel information)
        interference = torch.zeros(batch_size, device=W.device)
        if num_users > 1:
            for i in range(num_users - 1):
                # Calculate the interference between adjacent user pairs
                h_i = Hc[:, i, :]  # [B, A]
                h_j = Hc[:, i + 1, :]  # [B, A]

                # |w^H * h_i| 和 |w^H * h_j|
                gain_i = torch.abs(torch.sum(w_cplx.conj() * h_i, dim=1))
                gain_j = torch.abs(torch.sum(w_cplx.conj() * h_j, dim=1))
                cross_gain = gain_i * gain_j

                interference += cross_gain

        # ===== Generate goal-oriented vectors =====
        # Goal-Oriented Vector (Angle Directly Using Input)
        target_steering = self.generate_steering_vectors(target_angles)  # [B, T, A]

        # ===== Perceptual performance computing =====
        # Calculate the average gain for all targets |w^H * a_t|
        target_gains = torch.zeros(batch_size, target_steering.shape[1], device=W.device)
        for t in range(target_steering.shape[1]):
            a_t = target_steering[:, t, :]  # [B, A]
            gain = torch.abs(torch.sum(w_cplx.conj() * a_t, dim=1))  # [B]
            target_gains[:, t] = gain

        avg_target_gain = torch.mean(target_gains, dim=1)  # [B]

        # ===== Multi-tasking loss combinations =====
        comm_loss = -self.rho * 0.02 * (min_user_gain + 1.5 * sum_user_gain - self.adjacent_weight * interference)
        sens_loss = -(1 - self.rho) * 3 * avg_target_gain

        return torch.mean(comm_loss + sens_loss)

# class FairBeamformingNet(nn.Module):
#     def __init__(self, input_size, hidden_size=512, max_users=4, num_rf_chains=4):
#         super().__init__()
#         self.max_users = max_users  # Maximum number of users (dynamic number of users needs to ≤ this value)
#         self.num_rf_chains = num_rf_chains
#         self.num_antennas = 16
#
#         assert max_users <= num_rf_chains, "max_users cannot exceed num_rf_chains"
#
#         # Shared network
#         self.shared_net = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )
#
#         # Digital branches
#         self.digital_branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size, 2)
#             ) for _ in range(max_users)
#         ])
#
#         # Analog beamformer (num_rf_chains * num_antennas * 2)
#         self.analog_beamformer = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_rf_chains * num_antennas * 2)
#         )
#
#         self.norm = nn.LayerNorm(num_antennas * 2)
#
#     # def forward(self, Hc_real, Hc_imag, Hs_real, Hs_imag, rho, num_users=None):
#     def forward(self, Hc_real, Hc_imag, Hs_real, Hs_imag, rho, num_users=None):
#
#         if num_users is None:
#             num_users = self.max_users
#         assert num_users <= self.max_users, f"num_users ({num_users}) > max_users ({self.max_users})"
#
#         Hc = torch.cat([Hc_real.flatten(1), Hc_imag.flatten(1)], dim=1)
#         Hs = torch.cat([Hs_real.flatten(1), Hs_imag.flatten(1)], dim=1)
#         x = torch.cat([Hc, Hs, rho.view(-1, 1)], dim=1)
#         x = self.shared_net(x)
#
#         digital_outputs = [self.digital_branches[i](x).view(-1, 1, 2)
#                           for i in range(num_users)]
#         digital_combined = torch.cat(digital_outputs, dim=1)
#
#         if num_users < self.num_rf_chains:
#             padding = torch.zeros(x.size(0),
#                                  self.num_rf_chains - num_users,
#                                  2, device=x.device)
#             digital_combined = torch.cat([digital_combined, padding], dim=1)
#
#         # Analog: [B, num_rf_chains, num_antennas, 2]
#         analog_output = self.analog_beamformer(x).view(-1, self.num_rf_chains, self.num_antennas, 2)
#
#         # Hybrid: [B, num_antennas, 2]
#         hybrid_output = torch.einsum('brf,brnf->bnf', digital_combined, analog_output)
#         hybrid_output = hybrid_output.flatten(1)  # [B, 32]
#
#         return self.norm(hybrid_output)
#
# # ====================== Loss function ======================
# class MultiTaskLoss(nn.Module):
#     def __init__(self, rho=0.8, lambda_reg=0.1, min_power_weight=5.0,
#                  d=0.5, wavelength=1.0, num_antennas=16, P_max=1.0):
#         super().__init__()
#         # Configure basic parameters
#         self.rho = rho
#         self.lambda_reg = lambda_reg
#         self.min_power_weight = min_power_weight
#         self.d = d
#         self.wavelength = wavelength
#         self.num_antennas = num_antennas
#         self.P_max = P_max
#
#         # 新增参数
#         self.adjacent_weight = 0.01
#         # Adjacent user interference penalty weight
#
#         # Array parameter calculation
#         self.n = torch.arange(num_antennas).float()
#
#     def estimate_angles(self, H):
#         """Angle Estimation Method"""
#         device = H.device
#         batch_size, num_targets, _ = H.shape
#
#         phases = torch.angle(H)  # [B, T, A]
#         phases_unwrapped = torch.zeros_like(phases)
#
#         for b in range(batch_size):
#             for t in range(num_targets):
#                 phase_diff = torch.diff(phases[b, t], dim=0)
#                 phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
#                 phases_unwrapped[b, t, 1:] = torch.cumsum(phase_diff, dim=0)
#
#         # Linear regression solves for phase slope
#         X = torch.stack([self.n.to(device), torch.ones_like(self.n.to(device))], dim=1)
#         X = X.unsqueeze(0).unsqueeze(0)  # [1, 1, A, 2]
#         X = X.expand(batch_size, num_targets, -1, -1)
#
#         y = phases_unwrapped.unsqueeze(-1)  # [B, T, A, 1]
#
#         Xt = X.transpose(-1, -2)  # [B, T, 2, A]
#         XtX = torch.matmul(Xt, X)
#         XtY = torch.matmul(Xt, y)
#         beta = torch.linalg.solve(XtX, XtY)  # [B, T, 2, 1]
#
#         slopes = beta[..., 0, :].squeeze(-1)  # [B, T]
#         sin_theta = (slopes * self.wavelength) / (2 * np.pi * self.d)
#         sin_theta = torch.clamp(sin_theta, -1.0, 1.0)
#         return -torch.rad2deg(torch.arcsin(sin_theta))  # [B, T]
#
#     # def generate_steering_vectors(self, angles_deg):
#     #     """动态生成导向向量"""
#     #     theta_rad = torch.deg2rad(angles_deg)  # [B, T]
#     #     n = self.n.to(angles_deg.device)  # [A]
#     #
#     #     # 计算相位延迟
#     #     phase_delay = 2 * np.pi * self.d * torch.einsum('bt,a->bta',
#     #                                                     torch.sin(theta_rad),
#     #                                                     n) / self.wavelength
#     #
#     #     # 生成导向向量并归一化
#     #     steering_vec = torch.exp(1j * phase_delay)  # [B, T, A]
#     #     return steering_vec / torch.norm(steering_vec, dim=2, keepdim=True)
#
#     def forward(self, W, Hc, Hs):
#         # W_reduced = torch.mean(W, dim=2)  # PyTorch uses `dim` instead of `axis`
#         # ===== 角度估计 =====
#         # user_angles = self.estimate_angles(Hc)  # [B, U]
#         # target_angles = self.estimate_angles(Hs)  # [B, T]
#         user_angles = self.estimate_angles(Hc)[0]  # shape [U]
#         target_angles = self.estimate_angles(Hs)[0]  # shape [T]
#         # print('user_angles:', user_angles)
#         # print('target_angles', target_angles)
#         # User-Oriented Vectors
#         user_steering_vectors = torch.stack([
#             torch.tensor([
#                 np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
#                 for n in range(num_antennas)
#             ], dtype=torch.complex64)
#             for angle in user_angles
#         ])
#         # Perceiving Goal-Oriented Vectors
#         target_steering_vectors = torch.stack([
#             torch.tensor([
#                 np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
#                 for n in range(num_antennas)
#             ], dtype=torch.complex64)
#             for angle in target_angles
#         ])
#         # print(W.shape)
#
#         # print(W.shape)
#         real = W[:, :num_antennas]
#         imag = W[:, num_antennas:]
#         w_cplx = torch.complex(real, imag)
#         # w_cplx = W
#
#         # Communication performance calculation
#         user_gains = torch.abs(w_cplx @ user_steering_vectors.T.conj())
#         min_user_gain = torch.min(user_gains, dim=1)[0]
#         sum_user_gain = torch.sum(user_gains, dim=1)
#
#         # Neighboring users interfere with calculations
#         interference = torch.zeros_like(min_user_gain)
#         num_users = user_steering_vectors.shape[0]
#
#         if num_users > 1:
#             for i in range(num_users - 1):
#                 cross_gain = torch.abs(
#                     w_cplx @ user_steering_vectors[i].conj() *
#                     (w_cplx @ user_steering_vectors[i + 1].conj())
#                 )
#                 interference += cross_gain
#
#         # Perceptual performance computing
#         target_gains = torch.abs(w_cplx @ target_steering_vectors.T.conj())
#         avg_target_gain = torch.mean(target_gains, dim=1)
#
#         # Multitasking loss combinations
#         comm_loss = -self.rho * (min_user_gain + 1.5 * sum_user_gain - self.adjacent_weight * interference)
#         sens_loss = -(1 - self.rho) * 2.5 * avg_target_gain  # Loss of sensing
#
#         return torch.mean(comm_loss + sens_loss)


def get_model_name(user_angles, target_angles, num_antennas, rho):
    """Generate Standardized Model Name"""
    num_users = len(user_angles)
    num_targets = len(target_angles)
    return f"model_U{num_users}_S{num_targets}_A{num_antennas}_rho{rho:.2f}.pth".replace(".", "_")


def check_model_exists(user_angles, target_angles, num_antennas, rho):
    """Check whether the model for the corresponding parameter already exists"""
    model_name = get_model_name(user_angles, target_angles, num_antennas, rho)
    if os.path.exists(model_name):
        print(f"Locate the existing model: {model_name}")
        return model_name
    return None
