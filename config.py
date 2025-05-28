# ====================== config.py ======================
"""
System global parameter configuration module
"""
import numpy as np
import torch

# ======================= System parameter configuration =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_antennas = 16  # Number of transmitting antennas
num_receiver_antennas = 16   # Number of receiving antennas
# user_angles = [-10, 20,  50]  # User angle can be modified (number is the number of users)
# user_angles = [-15, 15, 30, 45]
# user_angles = [-10, 0, 10, 20, 30]
# user_angles = [-10, 0, 10, 20, 30, 40]
# user_angles = [-10, 0, 10, 20, 30, 40, 50]
user_angles = [-10, 0, 10, 20, 30, 40, 50, 60]
target_angles = [-45]  # You can modify the angle of the perceived target (the number is the number of targets)
# target_angles = [-45, -15]
# user_angles = [-10, 0, 10, 20, 30, 40, 50]

# target_angles = [-40, -20]
snr_dBs = np.arange(0, 12, 2)  # SNR range
theta_range = np.linspace(-90, 90, 361)  # Beam scan range
rho = 0.3  # communication-sensing weights

# The relevant parameters are automatically calculated
wavelength = 1
d = wavelength / 2
num_users = len(user_angles)
num_targets = len(target_angles)
input_size = 2 * (num_users * num_antennas + num_targets * num_receiver_antennas) + 1  # Input dimensions are calculated dynamically
