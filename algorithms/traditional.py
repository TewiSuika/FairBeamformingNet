
"""
Conventional beamforming algorithm module
"""
import numpy as np
from config import num_antennas


# ====================== ZF beamforming method ======================
class ZFBeamformer:

    def __init__(self, Hc_r, Hc_i):
        """
        Initialize with real and imaginary parts of channel matrix
        Args:
            Hc_r: Real part of channel matrix (num_users, num_antennas)
            Hc_i: Imaginary part of channel matrix (num_users, num_antennas)
        """
        self.H = (Hc_r + 1j * Hc_i).T
        self.H_pinv = np.linalg.pinv(self.H)

    def get_weights_for_jcas(self, Hs_r, Hs_i, rho=0.8):
        """
        Get joint communication and sensing weights
        Args:
            Hs_r: Real part of sensing channel matrix (num_targets, num_antennas)
            Hs_i: Imaginary part of sensing channel matrix (num_targets, num_antennas)
            rho: Weight parameter between communication and sensing
        Returns:
            Combined weights in real form [real_part, imag_part]
        """
        # Communication Beam (Average Beam for All Users)
        W_comm = self.H_pinv.mean(axis=1)

        # # Sensing Beam (Average Beam for All Targets)
        # Hs = Hs_r + 1j * Hs_i
        # W_sens = Hs.mean(axis=0)  # Average across targets
        #
        # # Joint beam with weighted combination
        # W_joint = rho * W_comm + (1 - rho) * W_sens

        # Convert to real form and make sure the shape is correct
        real_part = np.real(W_comm).flatten()
        imag_part = np.imag(W_comm).flatten()

        return np.concatenate([real_part, imag_part])  # Shape (2*num_antennas,)

    def apply(self, weights):
        w_cplx = weights[:, :num_antennas] + 1j * weights[:, num_antennas:]
        w_zf = w_cplx @ self.H_pinv
        return np.concatenate([w_zf.real, w_zf.imag], axis=1)


# ====================== MMSE beamforming method ======================
class MMSEBeamformer:

    def __init__(self, Hc_r, Hc_i, snr_db=20):
        """
        Initialize with real and imaginary parts of channel matrix
        Args:
            Hc_r: Real part of channel matrix (num_users, num_antennas)
            Hc_i: Imaginary part of channel matrix (num_users, num_antennas)
            snr_db: Signal-to-noise ratio in dB
        """
        self.H = Hc_r + 1j * Hc_i
        self.snr_db = snr_db

    def get_weights(self, Hs_r=None, Hs_i=None, rho=1.0):
        """
        Generate MMSE weights (optionally for joint communication and sensing)
        Args:
            Hs_r: Real part of sensing channel matrix (num_targets, num_antennas)
            Hs_i: Imaginary part of sensing channel matrix (num_targets, num_antennas)
            rho: Weight parameter between communication and sensing (1.0 for pure communication)
        Returns:
            Combined weights in real form [real_part, imag_part]
        """
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = 1.0 / snr_linear

        H = self.H.T
        # print(H.shape)
        I = np.eye(num_antennas)
        mmse_weights = np.linalg.inv(H.conj().T @ H + noise_power * I) @ H.conj().T

        # Communication weights (average across users)
        W_comm = mmse_weights.mean(axis=1)

        # if Hs_r is not None and Hs_i is not None and rho < 1.0:
        #     # Include sensing component if provided and rho < 1
        #     Hs = Hs_r + 1j * Hs_i
        #     W_sens = Hs.mean(axis=0)  # Average across targets
        #     W_joint = rho * W_comm + (1 - rho) * W_sens
        # else:
        #     W_joint = W_comm

        # Convert to real form
        real_part = np.real(W_comm).flatten()
        imag_part = np.imag(W_comm).flatten()

        return np.concatenate([real_part, imag_part])
