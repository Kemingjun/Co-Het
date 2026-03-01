import os
import sys
import time
import math
import random
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

# ==============================================================================
# Project Environment Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from Baseline.Util.load_data import read_excel
from Baseline.Util.operators import (
    destroy_random,
    destroy_worst_cost,
    destroy_worst_distance,
    destroy_worst_tardiness,
    repair_greedy,
    repair_greedy_urgency,
    repair_greedy_cost
)
from Baseline.Util.generate_init_solution import generate_solution_nearest

# ==============================================================================
# ALNS Configuration & Operators
# ==============================================================================
DESTRUCT_OPERATORS = [
    destroy_random,
    destroy_worst_cost,
    destroy_worst_distance,
    destroy_worst_tardiness
]

CONSTRUCT_OPERATORS = [
    repair_greedy,
    repair_greedy_urgency,
    repair_greedy_cost
]

# Scoring Parameters for Weight Adaptation
SIGMA_1 = 33  # New global best found
SIGMA_2 = 13  # Better than current solution or a new accepted solution
SIGMA_3 = 9  # Worse than current but accepted (Simulated Annealing)
RHO = 0.1  # Reaction factor (decay rate for weights)
L_S = 10  # Segment length (iterations between weight updates)


def select_operators(w_destruct: List[float], w_construct: List[float]) -> Tuple[int, int]:
    """Selects destruction and construction operators using Roulette Wheel Selection."""
    p_destruct = np.array(w_destruct) / sum(w_destruct)
    p_construct = np.array(w_construct) / sum(w_construct)

    d_idx = np.random.choice(np.arange(len(w_destruct)), p=p_destruct)
    c_idx = np.random.choice(np.arange(len(w_construct)), p=p_construct)

    return int(d_idx), int(c_idx)


def run_alns(file_path: str, max_iter: int, time_limit: int) -> Tuple[Any, float, float]:
    """
    Executes the Adaptive Large Neighborhood Search (ALNS) algorithm.
    """
    start_t = time.time()
    print(f"[*] Loading instance: {file_path}")
    instance = read_excel(file_path)

    # 1. Initialization
    d_op_num = len(DESTRUCT_OPERATORS)
    c_op_num = len(CONSTRUCT_OPERATORS)

    w_destruct = [1.0] * d_op_num
    w_construct = [1.0] * c_op_num
    score_destruct = [0.0] * d_op_num
    score_construct = [0.0] * c_op_num
    count_destruct = [0] * d_op_num
    count_construct = [0] * c_op_num

    solution_table = {}  # History of explored solution hash keys
    task_num = len(instance)
    d_num = math.ceil(task_num * 0.2)  # Destruction rate: 20% of tasks

    # Initial Temperature for Simulated Annealing (Metropolis Criterion)
    temperature = 0.1

    # Generate Initial Solution
    current_sol = generate_solution_nearest(instance)
    solution_table[current_sol.hash_key] = current_sol

    current_fitness = current_sol.get_fitness()
    best_sol = current_sol
    best_fitness = current_fitness

    print(f"[*] Initial Fitness: {current_fitness:.4f}")
    print(f"[*] Starting ALNS Optimization (Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Main Optimization Loop
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # A. Operator Selection
        d_idx, c_idx = select_operators(w_destruct, w_construct)

        # B. Destroy and Repair
        # destroy operators return: (destroyed_sequence, path_map, task_list)
        destroyed_data = DESTRUCT_OPERATORS[d_idx](current_sol, d_num)
        new_sol = CONSTRUCT_OPERATORS[c_idx](*destroyed_data, current_sol)

        # Check for novelty
        is_new = new_sol.hash_key not in solution_table
        if is_new:
            solution_table[new_sol.hash_key] = new_sol

        new_fitness = new_sol.get_fitness()
        current_score = 0
        accepted = False

        # C. Acceptance Criterion (Simulated Annealing)
        if new_fitness < current_fitness:
            accepted = True
            if new_fitness < best_fitness:
                best_sol = new_sol
                best_fitness = new_fitness
                current_score = SIGMA_1
                print(f"[Best] Iter {iteration}: Fitness improved to {best_fitness:.4f}")
            else:
                current_score = SIGMA_2 if is_new else 0
        elif new_fitness == current_fitness:
            accepted = True if is_new else False
            current_score = SIGMA_2 if is_new else 0
        else:
            # Metropolis acceptance probability
            prob = math.exp((current_fitness - new_fitness) / temperature)
            if random.random() < prob:
                accepted = True
                current_score = SIGMA_3 if is_new else 0

        # Update Current State
        if accepted:
            current_sol = new_sol
            current_fitness = new_fitness

        # D. Update Operator Statistics
        count_destruct[d_idx] += 1
        count_construct[c_idx] += 1
        score_destruct[d_idx] += current_score
        score_construct[c_idx] += current_score

        # E. Adaptive Weight Update (Segment Basis)
        if iteration % L_S == 0:
            for i in range(d_op_num):
                usage = max(1, count_destruct[i])
                w_destruct[i] = w_destruct[i] * (1 - RHO) + RHO * (score_destruct[i] / usage)
                score_destruct[i] = 0
                count_destruct[i] = 0
            for i in range(c_op_num):
                usage = max(1, count_construct[i])
                w_construct[i] = w_construct[i] * (1 - RHO) + RHO * (score_construct[i] / usage)
                score_construct[i] = 0
                count_construct[i] = 0

        if iteration % 20 == 0:
            elapsed = time.time() - start_t
            print(
                f"Iter {iteration:4d} | Best: {best_fitness:8.2f} | Curr: {current_fitness:8.2f} | T: {elapsed:6.2f}s")

    return best_sol, best_fitness, time.time() - start_t


# ==============================================================================
# Execution Entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALNS Solver for HRSP")

    parser.add_argument(
        '-i', '--instance',
        type=str,
        required=True,
        help="Name or path of the instance Excel file (without .xlsx)."
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=500,
        help="Maximum number of ALNS iterations (default: 500)."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=300,
        help="Time limit in seconds (default: 300)."
    )

    args = parser.parse_args()

    # Ensure file extension is handled
    file_path = args.instance if args.instance.endswith(".xlsx") else f"{args.instance}.xlsx"

    # Run Algorithm
    best_solution, best_fit, total_time = run_alns(file_path, args.iter, args.time)

    # Output Results
    print("\n" + "=" * 60)
    print(" ALNS OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Final Fitness    : {best_fit:.4f}")
    print(f"Total Distance   : {best_solution.distance:.2f}")
    print(f"Total Tardiness  : {best_solution.tardiness:.2f}")
    print(f"Total Time       : {total_time:.4f}s")
    print(f"Solution Code    : {best_solution.code}")
    print("=" * 60)