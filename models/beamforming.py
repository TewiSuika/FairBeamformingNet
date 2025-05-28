"""
Deep learning model definition module
"""
import numpy as np
import torch
import torch.nn as nn
import os
from config import *


# ====================== DNN Model ======================
# class FairBeamformingNet(nn.Module):
#     def __init__(self, input_size=input_size, hidden_size=512):
#         super().__init__()
#         self.shared_net = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )
#         self.user_branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size, num_antennas * 2)
#             ) for _ in range(len(user_angles))]
#         )
#         self.norm = nn.LayerNorm(num_antennas * 2)
#
#     def forward(self, x):
#         x = self.shared_net(x)
#         branch_outputs = [branch(x) for branch in self.user_branches]
#         combined = torch.mean(torch.stack(branch_outputs), dim=0)
#         return self.norm(combined)

# ====================== DNN Model  ======================
class FairBeamformingNet(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_users=4):
        super().__init__()
        self.num_users = num_users

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # User branching
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


# ====================== Loss function ======================
class MultiTaskLoss(nn.Module):
    def __init__(self, rho=0.8, lambda_reg=0.1, min_power_weight=5.0,
                 d=0.5, wavelength=1.0, num_antennas=16, P_max=1.0):
        super().__init__()
        # Configure basic parameters
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.min_power_weight = min_power_weight
        self.d = d
        self.wavelength = wavelength
        self.num_antennas = num_antennas
        self.P_max = P_max

        # Array parameter calculation
        self.n = torch.arange(num_antennas).float()

    def estimate_angles(self, H):
        """Angle Estimation Method"""
        device = H.device
        batch_size, num_targets, _ = H.shape

        phases = torch.angle(H)  # [B, T, A]
        phases_unwrapped = torch.zeros_like(phases)

        for b in range(batch_size):
            for t in range(num_targets):
                phase_diff = torch.diff(phases[b, t], dim=0)
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                phases_unwrapped[b, t, 1:] = torch.cumsum(phase_diff, dim=0)

        # 线性回归求解相位斜率
        X = torch.stack([self.n.to(device), torch.ones_like(self.n.to(device))], dim=1)
        X = X.unsqueeze(0).unsqueeze(0)  # [1, 1, A, 2]
        X = X.expand(batch_size, num_targets, -1, -1)

        y = phases_unwrapped.unsqueeze(-1)  # [B, T, A, 1]

        Xt = X.transpose(-1, -2)  # [B, T, 2, A]
        XtX = torch.matmul(Xt, X)
        XtY = torch.matmul(Xt, y)
        beta = torch.linalg.solve(XtX, XtY)  # [B, T, 2, 1]

        slopes = beta[..., 0, :].squeeze(-1)  # [B, T]
        sin_theta = (slopes * self.wavelength) / (2 * np.pi * self.d)
        sin_theta = torch.clamp(sin_theta, -1.0, 1.0)
        return -torch.rad2deg(torch.arcsin(sin_theta))  # [B, T]

    # def generate_steering_vectors(self, angles_deg):
    #     """动态生成导向向量"""
    #     theta_rad = torch.deg2rad(angles_deg)  # [B, T]
    #     n = self.n.to(angles_deg.device)  # [A]
    #
    #     # 计算相位延迟
    #     phase_delay = 2 * np.pi * self.d * torch.einsum('bt,a->bta',
    #                                                     torch.sin(theta_rad),
    #                                                     n) / self.wavelength
    #
    #     # 生成导向向量并归一化
    #     steering_vec = torch.exp(1j * phase_delay)  # [B, T, A]
    #     return steering_vec / torch.norm(steering_vec, dim=2, keepdim=True)

    def forward(self, W, Hc, Hs):
        # W_reduced = torch.mean(W, dim=2)  # PyTorch uses `dim` instead of `axis`
        # ===== 角度估计 =====
        # user_angles = self.estimate_angles(Hc)  # [B, U]
        # target_angles = self.estimate_angles(Hs)  # [B, T]
        user_angles = self.estimate_angles(Hc)[0]  # shape [U]
        target_angles = self.estimate_angles(Hs)[0]  # shape [T]
        # print('user_angles:', user_angles)
        # print('target_angles', target_angles)
        # User-Oriented Vectors
        user_steering_vectors = torch.stack([
            torch.tensor([
                np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
                for n in range(num_antennas)
            ], dtype=torch.complex64)
            for angle in user_angles
        ])
        # Perceiving Goal-Oriented Vectors
        target_steering_vectors = torch.stack([
            torch.tensor([
                np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
                for n in range(num_antennas)
            ], dtype=torch.complex64)
            for angle in target_angles
        ])

        # print(W.shape)
        real = W[:, :num_antennas]
        imag = W[:, num_antennas:]
        w_cplx = torch.complex(real, imag)
        # w_cplx = W

        # Communication performance calculation
        user_gains = torch.abs(w_cplx @ user_steering_vectors.T.conj())
        min_user_gain = torch.min(user_gains, dim=1)[0]
        sum_user_gain = torch.sum(user_gains, dim=1)

        # Perceptual performance computing
        target_gains = torch.abs(w_cplx @ target_steering_vectors.T.conj())
        avg_target_gain = torch.mean(target_gains, dim=1)

        # Multitasking loss combinations
        comm_loss = -self.rho * 2.5 * (min_user_gain + 0.1 * sum_user_gain)  # Loss of communications
        sens_loss = -(1 - self.rho) * avg_target_gain  # Loss of sensing

        return torch.mean(comm_loss + sens_loss)
# class MultiTaskLoss(nn.Module):
#     def __init__(self, user_angles, target_angles, rho=0.8):
#         super().__init__()
#         self.rho = rho  # 通信权重(0-1)，感知权重为1-alpha
#         # 用户导向向量
#         self.user_steering_vectors = torch.stack([
#             torch.tensor([
#                 np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
#                        for n in range(num_antennas)
#             ], dtype=torch.complex64)
#             for angle in user_angles
#         ])
#         # 感知目标导向向量
#         self.target_steering_vectors = torch.stack([
#             torch.tensor([
#                 np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
#                        for n in range(num_antennas)
#             ], dtype=torch.complex64)
#             for angle in target_angles
#         ])
#
#     def forward(self, pred_weights):
#         # 复数权重转换
#         real = pred_weights[:, :num_antennas]
#         imag = pred_weights[:, num_antennas:]
#         w_cplx = torch.complex(real, imag)
#         # print(w_cplx.shape)
#         # print(self.user_steering_vectors.shape)
#         # print(self.target_steering_vectors.shape)
#
#         # 通信性能计算
#         user_gains = torch.abs(w_cplx @ self.user_steering_vectors.T.conj())
#         min_user_gain = torch.min(user_gains, dim=1)[0]
#         sum_user_gain = torch.sum(user_gains, dim=1)
#
#         # 感知性能计算
#         target_gains = torch.abs(w_cplx @ self.target_steering_vectors.T.conj())
#         avg_target_gain = torch.mean(target_gains, dim=1)
#
#         # 多任务损失组合
#         comm_loss = -self.rho * (min_user_gain + 0.1 * sum_user_gain)  # 通信损失
#         sens_loss = -(1 - self.rho) * avg_target_gain  # 感知损失
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
