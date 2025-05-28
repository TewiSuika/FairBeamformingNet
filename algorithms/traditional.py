"""
Conventional beamforming algorithm module
"""
import numpy as np
from config import num_antennas, d, wavelength


# ====================== ZF beamforming method ======================
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
        # Communication Beam (Average Beam for All Users)
        W_comm = self.H_pinv.mean(axis=1)

        # # 感知波束 (平均所有目标的波束)
        # a_target = np.array([
        #     [np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(angle)) / wavelength)
        #      for n in range(num_antennas)]
        #     for angle in target_angles
        # ]).mean(axis=0)  # 形状(num_antennas,)

        # 联合波束
        # W_joint = alpha * W_comm + (1 - alpha) * a_target  # 形状(num_antennas,)
        W_joint = W_comm

        # Convert to real form and make sure the shape is correct
        real_part = np.real(W_joint).flatten()
        imag_part = np.imag(W_joint).flatten()

        return np.concatenate([real_part, imag_part])  # 形状(2*num_antennas,)

    def apply(self, weights):
        w_cplx = weights[:, :num_antennas] + 1j * weights[:, num_antennas:]
        w_zf = w_cplx @ self.H_pinv
        return np.concatenate([w_zf.real, w_zf.imag], axis=1)


# ====================== MMSE beamforming method ======================
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
        """Generate MMSE weights"""
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = 1.0 / snr_linear

        H = self.H
        I = np.eye(num_antennas)
        mmse_weights = np.linalg.inv(H.conj().T @ H + noise_power * I) @ H.conj().T

        # Average the weight of all users
        W_mmse = mmse_weights.mean(axis=1)

        # Convert to real form
        real_part = np.real(W_mmse).flatten()
        imag_part = np.imag(W_mmse).flatten()

        return np.concatenate([real_part, imag_part])


