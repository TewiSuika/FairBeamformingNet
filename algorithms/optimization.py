"""
Optimization algorithm module
"""
import numpy as np
from scipy.optimize import differential_evolution
from pyswarm import pso
from config import *
from algorithms.traditional import ZFBeamformer, MMSEBeamformer


# ====================== Objective function ======================
def objective(w, Hc, Hs, rho_set):
    """Objective function with communication-sensing weights"""
    num_antennas = len(w) // 2
    w_cplx = w[:num_antennas] + 1j * w[num_antennas:]

    # Communication Performance Calculation (using Hc)
    user_gains = np.abs(w_cplx @ Hc)
    min_user_gain = np.min(user_gains)
    sum_user_gain = np.sum(user_gains)

    # Perceptual Performance Computing (using Hs)
    target_gains = np.abs(w_cplx @ Hs)
    avg_target_gain = np.mean(target_gains)

    # Combine multiple targets
    comm_perf = min_user_gain + 0.1 * sum_user_gain
    sens_perf = avg_target_gain
    return -(rho_set * 2.5 * comm_perf + (1 - rho_set) * sens_perf)

# ====================== Optimization Algorithms ======================
def differential_evolution_optimizer(objective, bounds, args):
    return differential_evolution(objective, bounds, args=args, maxiter=100, popsize=15)

def particle_swarm_optimization(objective, lb, ub, args):
    return pso(objective, lb, ub, args=args, swarmsize=30, maxiter=100)

def grey_wolf_optimizer(objective, bounds, args, num_wolves=30, max_iter=100):
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

def whale_optimization_algorithm(objective, bounds, args, num_whales=30, max_iter=100):
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
def traditional_optimizer(method, Hc_r, Hc_i, Hs_r, Hs_i, rho_tensor):
    """Unified optimizer interface with the same inputs as DL model"""
    # Convert tensors to numpy arrays
    Hc = Hc_r.numpy() + 1j * Hc_i.numpy()
    Hs = Hs_r.numpy() + 1j * Hs_i.numpy()
    rho_set = rho_tensor.item()

    # Flatten Hc and Hs to (num_antennas, num_users/targets)
    Hc = Hc.squeeze(0).T  # Remove batch dim and transpose
    Hs = Hs.squeeze(0).T  # Remove batch dim and transpose

    # Prepare arguments for objective function
    args = (Hc, Hs, rho_set)

    # Define bounds for optimization variables
    bounds = [(-1, 1)] * (2 * num_antennas)

    if method == 'DE':
        result = differential_evolution_optimizer(objective, bounds, args)
        return result.x
    elif method == 'PSO':
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        xopt, _ = particle_swarm_optimization(objective, lb, ub, args)
        return xopt
    elif method == 'GWO':
        return grey_wolf_optimizer(objective, bounds, args)
    elif method == 'WOA':
        return whale_optimization_algorithm(objective, bounds, args)
    elif method == 'ZF':
        # For ZF, we still need angles - assuming they're in config
        zf_bf = ZFBeamformer(user_angles)
        return zf_bf.get_weights_for_jcas(target_angles, rho=rho_set)
    elif method == 'MMSE':
        # For MMSE, we still need angles - assuming they're in config
        mmse_bf = MMSEBeamformer(user_angles)
        return mmse_bf.get_weights()
    else:
        raise ValueError(f"Unknown method: {method}")