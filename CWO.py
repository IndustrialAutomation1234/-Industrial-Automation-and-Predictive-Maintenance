# Carpet Weaver Optimization (CWO)

import time
import numpy as np


def CWO(X, objective_function, VRmin, VRmax, max_iter):

    num_individuals, num_dimensions = X.shape
    lower_bound = VRmin[0, :]
    upper_bound = VRmax[0, :]

    # Parameters
    warp_factor = 0.7        # exploration control
    weft_factor = 0.5        # exploitation control
    tighten_factor = 0.3     # intensification
    pattern_decay = 0.99     # adaptive reduction

    # Initialization
    fitness = np.array([objective_function(ind) for ind in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    Leader_score = fitness[best_idx]

    Convergence_curve = np.zeros((max_iter, 1))
    start_time = time.time()

    # Main loop
    for t in range(max_iter):

        # Adaptive tightening
        adaptive_tight = tighten_factor * (1 - t / max_iter)

        for i in range(num_individuals):

            # ---------------- Warp Weaving (Exploration) ----------------
            r1 = np.random.rand(num_dimensions)
            warp_move = warp_factor * r1 * (X_best - X[i])

            # ---------------- Weft Weaving (Exploitation) ----------------
            j = np.random.randint(num_individuals)
            r2 = np.random.rand(num_dimensions)
            weft_move = weft_factor * r2 * (X[j] - X[i])

            # ---------------- Pattern Tightening ----------------
            tighten_move = adaptive_tight * np.random.randn(num_dimensions)

            # Position update
            X_new = X[i] + warp_move + weft_move + tighten_move

            # Boundary control
            X_new = np.clip(X_new, lower_bound, upper_bound)

            # Fitness evaluation
            new_fitness = objective_function(X_new)

            if new_fitness < fitness[i]:
                X[i] = X_new
                fitness[i] = new_fitness

                if new_fitness < Leader_score:
                    Leader_score = new_fitness
                    X_best = X_new.copy()

        # Gradual decay of weaving strength
        warp_factor *= pattern_decay
        weft_factor *= pattern_decay

        Convergence_curve[t] = Leader_score

    exec_time = time.time() - start_time

    return Leader_score, Convergence_curve, X_best, exec_time

