
from config import *
from tqdm import tqdm

# ====================== Channel generation function ======================
def generate_array_response(num_antennas, angle_deg):
    theta = np.deg2rad(angle_deg)
    n = np.arange(num_antennas)
    return np.exp(-1j * np.pi * n * np.sin(theta)) / np.sqrt(num_antennas)


def generate_sensing_channel(num_antennas, target_angles, num_paths=3):
    H_s = np.zeros((len(target_angles), num_antennas), dtype=np.complex64)
    for t, target_angle in enumerate(target_angles):
        # LOS
        a_t = generate_array_response(num_antennas, target_angle)
        g_los = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

        # NLOS
        # g_nlos = 0
        # for _ in range(num_paths - 1):
        #     spread_angle = target_angle + np.random.uniform(-5, 5)
        #     a_spread = generate_array_response(num_antennas, spread_angle)
        #     g_nlos += (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) * a_spread

        # H_s[t] = (g_los * a_t + g_nlos / np.sqrt(num_paths - 1)) / np.sqrt(2)
        H_s[t] = (g_los * a_t) / np.sqrt(2)

    return H_s


def generate_communication_channel(num_antennas, user_angles, num_paths=3):
    H_c = np.zeros((len(user_angles), num_antennas), dtype=np.complex64)
    for u, user_angle in enumerate(user_angles):
        a_u = generate_array_response(num_antennas, user_angle)
        g_scatter = 0
        for _ in range(num_paths - 1):
            spread_angle = user_angle + np.random.uniform(-15, 15)
            a_spread = generate_array_response(num_antennas, spread_angle)
            g_scatter += (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) * a_spread

        H_c[u] = (a_u + g_scatter / np.sqrt(num_paths - 1)) / np.sqrt(2)

    return H_c


# # ====================== Data generation ======================
# def generate_dataset(num_samples=10000):
#     Hc_real, Hc_imag = [], []
#     Hs_real, Hs_imag = [], []
#     rho_values = []
#
#     for _ in tqdm(range(num_samples)):
#         H_c = generate_communication_channel(num_antennas, user_angles)
#         H_s = generate_sensing_channel(num_antennas, target_angles)
#
#         Hc_real.append(torch.FloatTensor(H_c.real))
#         Hc_imag.append(torch.FloatTensor(H_c.imag))
#         Hs_real.append(torch.FloatTensor(H_s.real))
#         Hs_imag.append(torch.FloatTensor(H_s.imag))
#         rho_values.append(torch.FloatTensor([np.random.beta(2, 2)]))
#
#     return (torch.stack(Hc_real), torch.stack(Hc_imag),
#             torch.stack(Hs_real), torch.stack(Hs_imag),
#             torch.stack(rho_values))