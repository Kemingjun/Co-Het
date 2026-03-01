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
from Baseline.Util.util import *
from Baseline.Util.Solution import Solution
from Baseline.Util.generate_init_solution import generate_solution_nearest
from Baseline.Util.Config import Config

# ==============================================================================
# DIWO Algorithm Parameters
# ==============================================================================
POP_INITIAL_SIZE = 150  # Initial population of weeds
POP_MAX_SIZE = 200  # Maximum population size (Competitive Exclusion limit)
S_MAX = 40  # Maximum number of seeds per weed
S_MIN = 1  # Minimum number of seeds per weed


# ==============================================================================
# Neighbor Operators (Local Search)
# ==============================================================================

def neighbor_insertion(solution: Solution) -> Solution:
    """Performs a random task re-insertion neighbor search."""
    task = random.randint(1, solution.task_num)
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    remove_(sequence_map, path_init_task_map, task)

    for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
        feasible_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, robot_type)
        position = random.sample(list(feasible_positions), 1)[0]
        insert_(sequence_map, path_init_task_map, task, position, robot_type)

    return Solution(solution.instance, sequence_map, path_init_task_map)


def neighbor_swap(solution: Solution) -> Solution:
    """Performs a random task swap between two distinct tasks."""
    task_list = list(range(1, solution.task_num + 1))
    task_1, task_2 = random.sample(task_list, 2)

    path_map = solution.get_path_map()

    def get_task_pos(p_map, t_id):
        pos = []
        for p_idx, path in p_map.items():
            for i, task in enumerate(path):
                if task == t_id:
                    pos.append((p_idx, i))
        return pos

    t1_pos = get_task_pos(path_map, task_1)
    t2_pos = get_task_pos(path_map, task_2)

    # Swap task IDs at identified positions
    for pos in t1_pos:
        path_map[pos[0]][pos[1]] = task_2
    for pos in t2_pos:
        path_map[pos[0]][pos[1]] = task_1

    seq_map, p_init_map = path_map2sequence_map(path_map)
    return Solution(solution.instance, seq_map, p_init_map)


def get_neighbor_solution(solution: Solution) -> Solution:
    """Randomly selects and applies a neighborhood move to generate a 'seed'."""
    operators = [neighbor_insertion, neighbor_swap]
    return random.choice(operators)(solution)


# ==============================================================================
# Core DIWO Algorithm
# ==============================================================================

def run_diwo(file_path: str, max_iter: int, time_limit: int) -> Tuple[Solution, float, float]:
    """
    Executes the Discrete Invasive Weed Optimization (DIWO) algorithm.
    """
    start_t = time.time()
    print(f"[*] Loading instance: {file_path}")
    instance = read_excel(file_path)

    # 1. Initialization (Dispersal of initial weeds)
    weed_list: List[Solution] = []

    # Generate initial best and diverse population
    best_solution = generate_solution_nearest(instance)
    best_fitness = best_solution.get_fitness()
    weed_list.append(best_solution)

    for _ in range(POP_INITIAL_SIZE - 1):
        init_sol = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
        weed_list.append(init_sol)

    print(f"[*] Initial Best Fitness: {best_fitness:.4f}")
    print(f"[*] Starting DIWO Optimization (Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Main Optimization Loop
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # A. Reproduction & Seeding
        current_fitness_list = [sol.get_fitness() for sol in weed_list]
        min_fit = min(current_fitness_list)
        max_fit = max(current_fitness_list)

        seed_list: List[Solution] = []
        for weed in weed_list:
            weed_fit = weed.get_fitness()

            # Linear distribution of seeds based on fitness ranking
            try:
                if abs(max_fit - min_fit) < 1e-5:
                    seed_num = random.randint(S_MIN, S_MAX)
                else:
                    # Higher fitness (lower value) produces more seeds (closer to S_MAX)
                    seed_num = math.floor(S_MAX - (weed_fit - min_fit) / (max_fit - min_fit) * (S_MAX - S_MIN))
            except ZeroDivisionError:
                seed_num = random.randint(S_MIN, S_MAX)

            # Generate seeds (neighbor solutions)
            for _ in range(seed_num):
                seed_list.append(get_neighbor_solution(weed))

        # B. Competitive Exclusion
        # Combine parents and offspring, then select the best survivors
        weed_seed_list = weed_list + seed_list
        weed_seed_list.sort(key=lambda x: x.get_fitness())

        # Retain population within capacity
        weed_list = weed_seed_list[:POP_MAX_SIZE]

        # Update Global Best
        current_best = weed_list[0]
        if current_best.get_fitness() < best_fitness:
            best_solution = current_best
            best_fitness = best_solution.get_fitness()
            print(f"[Best] Iter {iteration}: New best fitness found: {best_fitness:.4f}")

        # Logging iteration progress
        if iteration % 20 == 0:
            elapsed = time.time() - start_t
            print(f"Iter {iteration:4d} | Best: {best_fitness:8.2f} | Time: {elapsed:6.2f}s")

    return best_solution, best_fitness, time.time() - start_t


# ==============================================================================
# Execution Entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Invasive Weed Optimization (DIWO) Solver for HRSP")

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
        help="Maximum number of DIWO generations (default: 100)."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=300,
        help="Time limit in seconds (default: 300)."
    )

    args = parser.parse_args()

    # File path normalization
    file_path = args.instance if args.instance.endswith(".xlsx") else f"{args.instance}.xlsx"

    # Execution
    best_sol, best_fit, total_time = run_diwo(file_path, args.iter, args.time)

    # Result Summary Output
    print("\n" + "=" * 60)
    print(" DIWO OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Final Fitness    : {best_fit:.4f}")
    print(f"Total Distance   : {best_sol.distance:.2f}")
    print(f"Total Tardiness  : {best_sol.tardiness:.2f}")
    print(f"Execution Time   : {total_time:.4f}s")
    print(f"Solution Code    : {best_sol.code}")
    print("=" * 60)