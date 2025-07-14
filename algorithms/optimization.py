
"""
Optimization algorithm module for hybrid beamforming
"""
import numpy as np
from scipy.optimize import differential_evolution
from pyswarm import pso
from config import *
from algorithms.traditional import ZFBeamformer, MMSEBeamformer


# # ====================== Objective function ======================
# def hybrid_objective(w, Hc, Hs, rho_set, num_antennas, num_rf_chains):
#     """Objective function for hybrid beamforming with communication-sensing weights"""
#     # Split variables into analog and digital parts
#     analog_real = w[:num_antennas * num_rf_chains]
#     analog_imag = w[num_antennas * num_rf_chains:2 * num_antennas * num_rf_chains]
#     digital_real = w[2 * num_antennas * num_rf_chains:2 * num_antennas * num_rf_chains + num_rf_chains]
#     digital_imag = w[2 * num_antennas * num_rf_chains + num_rf_chains:]
#
#     # Construct analog beamforming matrix (Nt x Nrf)
#     F_RF = (analog_real + 1j * analog_imag).reshape(num_antennas, num_rf_chains)
#     # Construct digital beamforming vector (Nrf x 1)
#     F_BB = (digital_real + 1j * digital_imag).reshape(num_rf_chains, 1)
#
#     # Normalize analog beamforming (constant modulus constraint)
#     F_RF = F_RF / np.abs(F_RF)
#     # Normalize digital beamforming (power constraint)
#     F_BB = F_BB / np.linalg.norm(F_RF @ F_BB, 'fro')
#
#     # Combined beamforming vector (16x1)
#     w_cplx = (F_RF @ F_BB).flatten()
#
#     # Communication Performance Calculation (using Hc)
#     user_gains = np.abs(w_cplx @ Hc)
#     min_user_gain = np.min(user_gains)
#     sum_user_gain = np.sum(user_gains)
#
#     # Sensing Performance Computing (using Hs)
#     target_gains = np.abs(w_cplx @ Hs)
#     avg_target_gain = np.mean(target_gains)
#
#     # Combine multiple targets
#     comm_perf = min_user_gain + 1.5 * sum_user_gain
#     sens_perf = avg_target_gain
#     # return -(rho_set * 2.5 * comm_perf + (1 - rho_set) * sens_perf)
#     return -(rho_set * comm_perf + 2.5*(1 - rho_set) * sens_perf)


# ====================== Objective function ======================
def hybrid_objective(w, Hc, target_angles_set, rho_set, num_antennas, num_rf_chains):
    # print(w.shape)
    # print(Hc.shape)
    # 确保target_angles是可迭代对象(将单个角度转换为数组)
    if isinstance(target_angles_set, (float, np.float64)):
        target_angles_set = np.array([target_angles_set])

    """Objective function using direct angle-based sensing calculation"""
    # Split variables into analog and digital parts
    analog_real = w[:num_antennas * num_rf_chains]
    analog_imag = w[num_antennas * num_rf_chains:2 * num_antennas * num_rf_chains]
    digital_real = w[2 * num_antennas * num_rf_chains:2 * num_antennas * num_rf_chains + num_rf_chains]
    digital_imag = w[2 * num_antennas * num_rf_chains + num_rf_chains:]

    # Construct analog beamforming matrix (Nt x Nrf)
    F_RF = (analog_real + 1j * analog_imag).reshape(num_antennas, num_rf_chains)
    # Construct digital beamforming vector (Nrf x 1)
    F_BB = (digital_real + 1j * digital_imag).reshape(num_rf_chains, 1)

    # Normalize analog beamforming (constant modulus constraint)
    F_RF = F_RF / np.abs(F_RF)
    # Normalize digital beamforming (power constraint)
    F_BB = F_BB / np.linalg.norm(F_RF @ F_BB, 'fro')

    # Combined beamforming vector (Nt x 1)
    w_cplx = (F_RF @ F_BB).flatten()


    # Communication Performance Calculation (using Hc)
    user_gains = np.abs(w_cplx @ Hc)  # Hc shape: [num_antennas, num_users]
    min_user_gain = np.min(user_gains)
    sum_user_gain = np.sum(user_gains)

    # Sensing Performance Calculation (using target angles)
    theta_rad = np.deg2rad(target_angles_set)
    n = np.arange(num_antennas)
    d = 0.5  # antenna spacing (wavelength)
    # print((w_cplx.conj()).shape)

    # Calculate steering vectors for all targets
    target_gains = []
    for theta in theta_rad:
        a_t = np.exp(1j * 2 * np.pi * d * n * np.sin(theta))
        gain = np.abs(np.dot(w_cplx.conj(), a_t))
        target_gains.append(gain)

    avg_target_gain = np.mean(target_gains)

    # Combined objective
    comm_perf = min_user_gain + 1.5 * sum_user_gain
    sens_perf = avg_target_gain
    return -(rho_set * comm_perf + 0.5*(1 - rho_set) * sens_perf)


def _convert_hybrid_to_final(w_hybrid, num_antennas, num_rf_chains):
    """Convert 272-dim hybrid weights to 32-dim final beamforming vector"""
    # Extract analog and digital parts
    analog_real = w_hybrid[:num_antennas * num_rf_chains]
    analog_imag = w_hybrid[num_antennas * num_rf_chains:2 * num_antennas * num_rf_chains]
    digital_real = w_hybrid[2 * num_antennas * num_rf_chains:2 * num_antennas * num_rf_chains + num_rf_chains]
    digital_imag = w_hybrid[2 * num_antennas * num_rf_chains + num_rf_chains:]

    # Construct beamforming matrices
    F_RF = (analog_real + 1j * analog_imag).reshape(num_antennas, num_rf_chains)
    F_BB = (digital_real + 1j * digital_imag).reshape(num_rf_chains, 1)

    # Get final beamforming vector and convert to real/imag parts
    w_cplx = (F_RF @ F_BB).flatten()
    return np.concatenate([w_cplx.real, w_cplx.imag])  # (32,)

# ====================== Optimization Algorithms (Modified) ======================
def differential_evolution_optimizer(objective, bounds, args):
    result = differential_evolution(objective, bounds, args=args, maxiter=100, popsize=15)
    w_optim = result.x
    # Convert hybrid weights to final beamforming vector (32,)
    return _convert_hybrid_to_final(w_optim, args[3], args[4])  # num_antennas, num_rf_chains


def particle_swarm_optimization(objective, lb, ub, args):
    xopt, _ = pso(objective, lb, ub, args=args, swarmsize=30, maxiter=100)
    # Convert hybrid weights to final beamforming vector (32,)
    return _convert_hybrid_to_final(xopt, args[3], args[4])  # num_antennas, num_rf_chains


def grey_wolf_optimizer(objective, bounds, args, num_wolves=30, max_iter=100):
    alpha = grey_wolf_optimizer_original(objective, bounds, args, num_wolves, max_iter)
    # Convert hybrid weights to final beamforming vector (32,)
    return _convert_hybrid_to_final(alpha, args[3], args[4])  # num_antennas, num_rf_chains


def whale_optimization_algorithm(objective, bounds, args, num_whales=30, max_iter=100):
    best_whale = whale_optimization_algorithm_original(objective, bounds, args, num_whales, max_iter)
    # Convert hybrid weights to final beamforming vector (32,)
    return _convert_hybrid_to_final(best_whale, args[3], args[4])  # num_antennas, num_rf_chains


# ====================== GWO/WOA implementations ======================
def grey_wolf_optimizer_original(objective, bounds, args, num_wolves=30, max_iter=100):
    dim = len(bounds)
    wolves = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(num_wolves)])
    fitness = np.array([objective(wolf, *args) for wolf in wolves])
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
            new_fitness = objective(new_wolf, *args)
            if new_fitness < fitness[i]:
                wolves[i] = new_wolf
                fitness[i] = new_fitness

        sorted_indices = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_indices[:3]]
        alpha_fitness, beta_fitness, delta_fitness = fitness[sorted_indices[:3]]
    return alpha


def whale_optimization_algorithm_original(objective, bounds, args, num_whales=30, max_iter=100):
    dim = len(bounds)
    whales = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(num_whales)])
    fitness = np.array([objective(whale, *args) for whale in whales])
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
            new_fitness = objective(new_whale, *args)
            if new_fitness < fitness[i]:
                whales[i] = new_whale
                fitness[i] = new_fitness

        if np.min(fitness) < best_fitness:
            best_whale = whales[np.argmin(fitness)]
            best_fitness = np.min(fitness)
    return best_whale


# ====================== Unified Optimizer Interface ======================
# def traditional_optimizer(method, Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor, num_rf_chains=8):
def traditional_optimizer(method, Hc_r, Hc_i, target_angles_tensor, rho_tensor, num_rf_chains=8):
    """Unified optimizer interface for hybrid beamforming with the same inputs as DL model"""
    # Convert tensors to numpy arrays
    Hc = Hc_r.numpy() + 1j * Hc_i.numpy()
    # Hs = Hs_r.numpy() + 1j * Hs_i.numpy()
    target_angles_set = target_angles_tensor.item()
    rho_set = rho_tensor.item()

    # Make sure the target_angles is a numpy array
    if isinstance(target_angles_set, torch.Tensor):
        target_angles_set = target_angles_set.numpy()
    elif isinstance(target_angles_set, (float, int)):
        target_angles_set = np.array([target_angles_set])

    # Flatten Hc and Hs to (num_antennas, num_users/targets)
    # Hc = Hc.squeeze(0).T  # Remove batch dim and transpose
    Hc = Hc[0].T
    # Hs = Hs.squeeze(0).T  # Remove batch dim and transpose
    num_antennas = Hc.shape[0]  # Get actual number of antennas from input

    # Prepare arguments for objective function
    # args = (Hc, Hs, rho_set, num_antennas, num_rf_chains)
    args = (Hc, target_angles_set, rho_set, num_antennas, num_rf_chains)

    # Define bounds for optimization variables
    bounds = [(-1, 1)] * (2 * num_antennas * num_rf_chains + 2 * num_rf_chains)

    if method == 'DE':
        return differential_evolution_optimizer(hybrid_objective, bounds, args)
    elif method == 'PSO':
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        return particle_swarm_optimization(hybrid_objective, lb, ub, args)
    elif method == 'GWO':
        return grey_wolf_optimizer(hybrid_objective, bounds, args)
    elif method == 'WOA':
        return whale_optimization_algorithm(hybrid_objective, bounds, args)
    elif method == 'ZF':
        # For ZF, we still need angles - assuming they're in config
        zf_bf = ZFBeamformer(Hc.real, Hc.imag)
        return zf_bf.get_weights_for_jcas(Hc.real, Hc.imag, rho=rho_set)
    elif method == 'MMSE':
        # For MMSE, we still need angles - assuming they're in config
        mmse_bf = MMSEBeamformer(Hc.real, Hc.imag)
        return mmse_bf.get_weights(Hc.real, Hc.imag, rho=rho_set)
    else:
        raise ValueError(f"Unknown method: {method}")
