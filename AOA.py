import time
import numpy as np

# Artificial Orcas Algorithm (AOA)
def AOA(X, objective_function, VRmin, VRmax, max_iter):
    num_individuals, num_dimensions = X.shape
    lower_bound = VRmin[0, :]
    upper_bound = VRmax[0, :]

    # Parameters
    gamma = 0.5
    delta = 0.25
    alpha = 0.75
    w0 = 0.25
    fmin, fmax = 0, 2
    max_echo = 30
    max_hunt = 30

    # Initialize
    vel = np.zeros_like(X)
    fitness = np.array([objective_function(ind) for ind in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    Leader_score = fitness[best_idx]
    Convergence_curve = np.zeros((max_iter, 1))

    start_time = time.time()

    # Main loop
    for t in range(max_iter):
        # Velocity & position update
        for i in range(num_individuals):
            w_i = w0
            vel[i] = (w_i * vel[i] +
                      np.random.rand(num_dimensions) * (X_best - X[i]) +
                      np.random.rand(num_dimensions) * (X_best - X[i]))
            X[i] += vel[i]
            L = 1.0
            tau = L / (np.linalg.norm(vel[i]) + 1e-9)
            X[i] -= gamma * np.sin((2 * np.pi * tau) / L)

        # Boundary control
        X = np.clip(X, lower_bound, upper_bound)

        # Echolocation phase
        A = np.ones(num_individuals)
        for _ in range(max_echo):
            beta = np.random.rand(num_individuals)
            f = fmin + (fmax - fmin) * beta
            A *= delta
            vel = (f[:, None] * A[:, None]) * np.ones((num_individuals, num_dimensions))
            X += vel
        X = np.clip(X, lower_bound, upper_bound)

        # Hunting phase
        for _ in range(max_hunt):
            for i in range(num_individuals):
                if np.random.rand() < alpha:
                    r = np.random.rand()
                    X[i] += r * (X_best - X[i])
                else:
                    l = np.random.rand()
                    X[i] += l * 2 * np.pi
        X = np.clip(X, lower_bound, upper_bound)

        # Fitness update
        for i in range(num_individuals):
            fit = objective_function(X[i])
            if fit < fitness[i]:
                fitness[i] = fit
                if fit < Leader_score:
                    Leader_score = fit
                    X_best = X[i].copy()

        Convergence_curve[t] = Leader_score

    exec_time = time.time() - start_time
    return Leader_score, Convergence_curve, X_best, exec_time
