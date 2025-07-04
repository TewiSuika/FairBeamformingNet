import numpy as np
from config import num_antennas, num_rf_chains, d, wavelength

class HybridBeamformerBase:
    def __init__(self, user_angles, target_angles):
        self.user_angles = user_angles
        self.target_angles = target_angles
        self.num_users = len(user_angles)
        self.num_targets = len(target_angles)
        
    def _get_array_response(self, angles):
        """Generate an array response matrix (the basis of the simulation part)"""
        return np.array([
            [np.exp(1j * 2 * np.pi * d * n * np.sin(np.deg2rad(-angle)) / wavelength)
             for n in range(num_antennas)]
            for angle in angles
        ])

    def _project_analog_weights(self, W_analog):
        """Projecting to Unit Modulus Constraints (Force Simulation Weights to Adjust Phase Only)"""
        return np.exp(1j * np.angle(W_analog)) / np.sqrt(num_antennas)