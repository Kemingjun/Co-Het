import os
import sys
import time
import math
import random
import argparse
from typing import List, Tuple, Any, Optional

# ==============================================================================
# Project Environment Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from Baseline.Util.load_data import read_excel
from Baseline.Util.generate_init_solution import generate_solution_nearest
from Baseline.Util.operators import destroy_random, repair_greedy
from Baseline.Util.Solution import Solution
from Baseline.Util.util import *
from Baseline.Util.Config import Config

# ==============================================================================
# IGA Hyperparameters
# ==============================================================================
D_NUM_COEFFICIENT = 0.2  # Proportion of tasks to destroy
T_COEFFICIENT = 0.1  # Temperature coefficient for acceptance criterion


# ==============================================================================
# Helper Functions: Search & Operators
# ==============================================================================

def destruct_construct(current_solution: Solution, d_num: int) -> Solution:
    """
    Applies the destruction and construction phases of the IG algorithm.
    """
    destroyed_info = destroy_random(current_solution, d_num)
    new_solution = repair_greedy(*destroyed_info, current_solution)
    return new_solution


def local_search_type2(solution: Solution, start_t: float, duration: int) -> Solution:
    """
    Performs a local search by attempting to improve the position of each task.
    """
    task_list = list(range(1, solution.task_num + 1))
    current_fitness = solution.get_fitness()
    current_solution = solution
    random.shuffle(task_list)

    for task in task_list:
        if time.time() - start_t > duration:
            break

        sequence_map = current_solution.get_sequence_map()
        path_init_task_map = current_solution.get_path_init_task_map()

        # Remove task and greedily re-insert to find potential improvement
        remove_(sequence_map, path_init_task_map, task)
        new_solution = repair_greedy(sequence_map, path_init_task_map, [task], solution)

        if new_solution.get_fitness() < current_fitness:
            current_solution = new_solution
            current_fitness = new_solution.get_fitness()
            break  # First improvement strategy

    return current_solution


# ==============================================================================
# Core IGA Algorithm
# ==============================================================================

def run_iga(file_path: str, max_iter: int, time_limit: int) -> Tuple[Solution, float, float]:
    """
    Executes the Iterative Greedy Algorithm (IGA) for the HRSP.
    """
    start_t = time.time()
    print(f"[*] Loading instance: {file_path}")
    instance = read_excel(file_path)

    # 1. Initialization
    task_num = len(instance)
    d_num = math.ceil(task_num * D_NUM_COEFFICIENT)

    # Generate initial solution using Nearest Neighbor heuristic
    solution = generate_solution_nearest(instance)
    current_fitness = solution.get_fitness()

    best_solution = solution
    best_fitness = current_fitness

    print(f"[*] Initial Fitness: {current_fitness:.4f}")
    print(f"[*] Starting IGA Optimization (Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Optimization Loop
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # A. Local Search Phase
        neighbor_solution = local_search_type2(solution, start_t, time_limit)

        # B. Destruction and Construction Phase
        new_solution = destruct_construct(neighbor_solution, d_num)
        new_fitness = new_solution.get_fitness()

        # C. Acceptance Criterion (Simulated Annealing/Metropolis)
        if new_fitness < current_fitness:
            # Improvement found
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                print(f"[Best] Iter {iteration}: New best fitness: {best_fitness:.4f}")
        elif new_fitness == current_fitness:
            # Neutral move
            solution = new_solution
        else:
            # Worse solution: Accept with a certain probability
            prob = math.exp((current_fitness - new_fitness) / T_COEFFICIENT)
            if random.random() < prob:
                solution = new_solution
                current_fitness = new_fitness

        # Periodic logging
        if iteration % 20 == 0:
            elapsed = time.time() - start_t
            print(
                f"Iter {iteration:4d} | Best: {best_fitness:8.2f} | Curr: {current_fitness:8.2f} | Time: {elapsed:6.2f}s")

    total_time = time.time() - start_t
    return best_solution, best_fitness, total_time


# ==============================================================================
# Execution Entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative Greedy Algorithm (IGA) Solver")

    parser.add_argument(
        '-i', '--instance',
        type=str,
        required=True,
        help="Name or path of the instance Excel file (without .xlsx)."
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=100,
        help="Maximum number of iterations (default: 100)."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=1800,
        help="Time limit in seconds (default: 1800)."
    )

    args = parser.parse_args()

    # Normalize file path
    file_path = args.instance if args.instance.endswith(".xlsx") else f"{args.instance}.xlsx"

    # Execute algorithm
    best_sol, best_fit, elapsed = run_iga(file_path, args.iter, args.time)

    # Output Summary
    print("\n" + "=" * 60)
    print(" IGA OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Final Best Fitness : {best_fit:.4f}")
    print(f"Total Distance    : {best_sol.distance:.2f}")
    print(f"Total Tardiness   : {best_sol.tardiness:.2f}")
    print(f"Execution Time    : {elapsed:.4f}s")
    print(f"Solution Code     : {best_sol.code}")
    print("=" * 60)