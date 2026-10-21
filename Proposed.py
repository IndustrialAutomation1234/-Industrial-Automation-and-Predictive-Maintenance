# Improved Hiking Optimization Algorithm (IHOA) - Update starts from line no 34


import numpy as np
import time


def Proposed(population, objective_function, lb, ub, max_iter):

    pop_size, dim = population.shape
    lower_bound = lb[0, :]
    upper_bound = ub[0, :]

    # Parameters
    exploration_rate = 0.8
    exploitation_rate = 0.4
    ridge_factor = 0.6

    # Initialize fitness
    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = np.zeros((max_iter, 1))
    start_time = time.time()

    for t in range(max_iter):

        adaptive_factor = 1 - (t / max_iter)

        for i in range(pop_size):

            # -------- Trail Exploration (Global Search) --------
            r1 = fitness[i] / (fitness[i] + max(fitness) + np.mean(fitness))
            trail_move = exploration_rate * r1 * (best_solution - population[i])

            # -------- Ridge Traversal (Peer Learning) --------
            j = np.random.randint(pop_size)
            r2 = np.random.rand(dim)
            ridge_move = ridge_factor * r2 * (population[j] - population[i])

            # -------- Camp Adjustment (Local Exploitation) --------
            camp_move = exploitation_rate * adaptive_factor * np.random.randn(dim)

            # Position update
            new_position = population[i] + trail_move + ridge_move + camp_move

            # Boundary control
            new_position = np.clip(new_position, lower_bound, upper_bound)

            # Fitness evaluation
            new_fitness = objective_function(new_position)

            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_position.copy()

        convergence_curve[t] = best_fitness

    exec_time = time.time() - start_time

    return best_fitness, convergence_curve, best_solution, exec_time
