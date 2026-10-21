import time

import numpy as np


def _clamp(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


# American Zebra Optimization Algorithm (AZO)
def AZO(pop, objective, lb, ub, max_iter):
    pop_size, dim = pop.shape[0], pop.shape[1]
    rng = np.random.default_rng(5)
    PC = 0.1  # crossover probability (paper default 0.1). :contentReference[oaicite:5]{index=5}
    SP = 0.5
    WR_center = None  # water reserves position (optional). if None, use global best approximation
    # Evaluate fitness
    fitness = np.array([objective(ind) for ind in pop])
    n_stallions = max(1, int(np.round(SP * pop_size)))
    # Track best
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_f = fitness[best_idx]

    # helper: index grouping
    # We'll partition population into groups: assign each stallion and its foals
    # Simpler: assign stallions as first n_stallions after sorting by fitness each iter (or random initially)
    history = [best_f]

    # ---- Utility functions used in equations ----
    def R2_value(t):
        # Eq.(3): R2 = 1 - t*(1/T)
        return 1.0 - (t * (1.0 / max_iter))

    def random_R(val_min, val_max, size=None):
        return rng.uniform(val_min, val_max, size=size)

    def crossover(a, b):
        # paper: "Crossover(Za_i, Zb_j)" — not precisely described.
        # We'll implement simple arithmetic crossover (blend) with random mixing coefficient.
        alpha = rng.random(size=a.shape)
        return alpha * a + (1 - alpha) * b

    ct = time.time()
    for t in range(max_iter):
        # Optionally sort population (not required by paper, but helps reproducibility)
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        # reassign best
        if fitness[0] < best_f:
            best_f = fitness[0]
            best = pop[0].copy()

        # Choose water reserves center (WR). If not given, use current global best (paper describes WR)
        if WR_center is None:
            WR = best.copy()
        else:
            WR = np.array(WR_center)

        # Determine groups: assign first n_stallions as stallions, next foals distributed among groups
        # We'll partition foals evenly across stallions.
        stallions_idx = list(range(n_stallions))
        foals_idx = list(range(n_stallions, pop_size))

        # Create groups: for fairness, create a list of foal indices per stallion
        groups = [[] for _ in range(n_stallions)]
        for i, fidx in enumerate(foals_idx):
            groups[i % n_stallions].append(fidx)

        # PHASE 1 (Feeding activity) & PHASE 2 (Breeding activity) for each foal in each group
        for s_idx, group_foals in enumerate(groups):
            # stallion position
            Zs = pop[s_idx].copy()
            for fi in group_foals:
                Zi = pop[fi].copy()
                # Eq.(1): feeding update around the stallion with R1 in [-2,2], R2 adaptive, R3 random
                R1 = random_R(-2.0, 2.0)
                R2 = R2_value(t)
                R3 = rng.random()
                diff = Zs - Zi
                if R3 < 0.5:
                    Zi_new = 2.0 * R1 * np.sin(2.0 * np.pi * R2) * np.abs(diff) + Zs
                else:
                    Zi_new = 2.0 * R1 * np.cos(2.0 * np.pi * R2) * np.abs(diff) + Zs
                # clamp
                Zi_new = _clamp(Zi_new, lb, ub)

                # Eq.(2): only accept if new fitness is better (paper uses comparison with Fji)
                f_old = fitness[fi]
                f_new = objective(Zi_new)
                if f_new[i] < f_old:
                    pop[fi] = Zi_new[i]
                    fitness[fi] = f_new[i]
                # else remain

                # PHASE 3: Breeding activity (crossover among groups)
                # Paper: baby zebra chooses another group j or k and does crossover
                # Eq.(4)-(5) define two crossover possibilities with probability PC
                r = rng.random()
                # choose two random different groups (j and k) distinct from i's group if possible
                other_group_indices = [idx for idx in range(n_stallions) if idx != s_idx]
                if len(other_group_indices) >= 1:
                    j = rng.choice(other_group_indices)
                    # pick random individual from group j (if any foals), otherwise pick its stallion
                    candidates_j = groups[j] if len(groups[j]) > 0 else [j]
                    pick_j = rng.choice(candidates_j + [j]) if isinstance(candidates_j, list) else j
                    Zbj = pop[pick_j] if pick_j < pop.shape[0] else pop[j]

                    # pick k as another group's random individual (if available)
                    # fallback: pick random index in pop
                    other_group_indices2 = [idx for idx in other_group_indices if idx != j]
                    if len(other_group_indices2) >= 1:
                        k = rng.choice(other_group_indices2)
                        candidates_k = groups[k] if len(groups[k]) > 0 else [k]
                        pick_k = rng.choice(candidates_k + [k])
                        Zck = pop[pick_k] if pick_k < pop.shape[0] else pop[k]
                    else:
                        # fallback: random individual not equal to fi
                        rand_idx = rng.choice([ii for ii in range(pop_size) if ii != fi])
                        Zck = pop[rand_idx]
                else:
                    # no other groups — pick random agents
                    idxA, idxB = rng.choice([ii for ii in range(pop_size) if ii != fi], size=2, replace=False)
                    Zbj = pop[idxA]
                    Zck = pop[idxB]

                if r < PC:
                    # Eq.(4): crossover Za_i with Zb_j
                    child = crossover(pop[fi], Zbj)
                else:
                    # Eq.(5): crossover Za_i with Zc_k
                    child = crossover(pop[fi], Zck)

                child = _clamp(child, lb, ub)
                f_child = objective(child)
                if f_child[i] < fitness[fi]:
                    pop[fi] = child[i]
                    fitness[fi] = f_child[i]

        # PHASE 4: Group leadership (stallion moves toward water reserves) — Eq.(6)-(8)
        # For each stallion (index s_idx), update its position relative to WR with R4 in [-2,2], R5 adaptive, R6 random
        for s_idx in stallions_idx:
            Zs = pop[s_idx].copy()
            R4 = random_R(-2.0, 2.0)
            R5 = R2_value(t)  # same form as Eq.(8) (R5 = 1 - t*(1/T))
            R6 = rng.random()
            diffWR = WR - Zs
            if R6 < 0.5:
                Zs_new = 2.0 * R4 * np.sin(2.0 * np.pi * R5) * np.abs(diffWR) + WR
            else:
                Zs_new = 2.0 * R4 * np.cos(2.0 * np.pi * R5) * np.abs(diffWR) + WR
            Zs_new = _clamp(Zs_new, lb, ub)
            f_new = objective(Zs_new)
            # Eq.(7): accept update if fitness improved
            if f_new[i] < fitness[s_idx]:
                pop[s_idx] = Zs_new[i]
                fitness[s_idx] = f_new[i]

        # PHASE 5: Leadership transition (select a new leader if some foal is better than stallion) — Eq.(9)
        # For each group, check if any member has better fitness than current stallion; if yes, replace stallion
        for s_idx, group_foals in enumerate(groups):
            # check all members in group (stallion + foals)
            members = [s_idx] + group_foals
            # find best among them
            best_local_idx = min(members, key=lambda idx: fitness[idx])
            if fitness[best_local_idx] < fitness[s_idx]:
                # new stallion becomes leader (swap positions in pop arrays to keep stallion ordering)
                # We'll swap the positions (so stallions remain first n_stallions indices)
                if best_local_idx != s_idx:
                    # swap pop rows and fitness values
                    pop[[s_idx, best_local_idx]] = pop[[best_local_idx, s_idx]]
                    fitness[[s_idx, best_local_idx]] = fitness[[best_local_idx, s_idx]]
                    # also update group indexes: find which group contained that foal and update lists
                    # (simple approach: reassign groups next iteration because we reorder by stallion / foals each iter)
                    # so no further action needed here.

        # end of iteration bookkeeping
        # recompute global best
        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < best_f:
            best_f = fitness[it_best_idx]
            best = pop[it_best_idx].copy()

        history.append(best_f)
    ct = time.time() - ct
    return best_f, history, best, ct
