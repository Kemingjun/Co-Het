import random
import logging
from typing import List, Dict, Tuple, Any, Optional

from Baseline.Util.util import (
    remove_,
    insert_,
    cal_fitness,
    get_feasible_insert_position,
    copy_dict_int_dict,
    copy_dict_int_int
)
from Baseline.Util.Solution import Solution


# ==============================================================================
# Destruction Operators
# ==============================================================================

def destroy_random(solution: Solution, d_num: int) -> Tuple[Dict, Dict, List[int]]:
    """
    Randomly removes a specified number of tasks from the current solution.
    """
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_sequence_map = solution.get_sequence_map()

    # Sample tasks randomly from the total task set
    all_tasks = list(range(1, solution.task_num + 1))
    destroyed_task_list = random.sample(all_tasks, d_num)

    for task in destroyed_task_list:
        remove_(destroyed_sequence_map, path_init_task_map, task)

    return destroyed_sequence_map, path_init_task_map, destroyed_task_list


def destroy_worst_cost(solution: Solution, d_num: int) -> Tuple[Dict, Dict, List[int]]:
    """
    Removes tasks that have the highest contribution to the total travel cost.
    """
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    task_info_map = solution.info_map

    task_cost_list = []
    for task in list(sequence_map.keys()):
        # Retrieve 'distance' as the primary cost metric from the solution info
        cost = task_info_map[task]['distance']
        task_cost_list.append([task, cost])

    # Sort tasks by cost in descending order to identify the 'worst' tasks
    task_cost_list.sort(key=lambda x: x[1], reverse=True)
    destroyed_task_list = [task_cost_list[i][0] for i in range(min(d_num, len(task_cost_list)))]

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)

    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_worst_distance(solution: Solution, d_num: int) -> Tuple[Dict, Dict, List[int]]:
    """
    Removes tasks associated with the longest travel distances.
    """
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    # Ensure solution metrics are evaluated to populate info_map
    if solution.info_map is None:
        solution.get_fitness()

    task_info_map = solution.info_map
    task_distance_list = []

    for task in list(sequence_map.keys()):
        distance = task_info_map[task]['distance']
        task_distance_list.append([task, distance])

    task_distance_list.sort(key=lambda x: x[1], reverse=True)
    destroyed_task_list = [task_distance_list[i][0] for i in range(min(d_num, len(task_distance_list)))]

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)

    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_worst_tardiness(solution: Solution, d_num: int) -> Tuple[Dict, Dict, List[int]]:
    """
    Removes tasks with the highest tardiness (violation of deadlines).
    """
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    task_info_map = solution.info_map

    task_tardiness_list = []
    for task in list(sequence_map.keys()):
        tardiness = task_info_map[task]['tardiness']
        task_tardiness_list.append([task, tardiness])

    task_tardiness_list.sort(key=lambda x: x[1], reverse=True)
    destroyed_task_list = [task_tardiness_list[i][0] for i in range(min(d_num, len(task_tardiness_list)))]

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)

    return sequence_map, path_init_task_map, destroyed_task_list


# ==============================================================================
# Repair (Construction) Operators
# ==============================================================================

def repair_greedy(destroyed_sequence_map: Dict,
                  path_init_task_map: Dict,
                  destroyed_task_list: List[int],
                  current_solution: Solution) -> Solution:
    """
    Standard greedy repair: Re-inserts tasks in a randomized order into the best feasible positions.
    """
    instance = current_solution.instance
    random.shuffle(destroyed_task_list)

    for task in destroyed_task_list:
        greedy_seq_list = [None]
        greedy_path_list = [None]
        fitness_min = [1e9]
        required_robot = instance[task - 1][5]

        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=task,
            fitness_min_list=fitness_min,
            greedy_destroyed_sequence_map_list=greedy_seq_list,
            greedy_path_init_task_map_list=greedy_path_list
        )

        if greedy_seq_list[0] is None:
            logging.error(f"Feasibility error: Task {task} has no valid insertion point.")

        destroyed_sequence_map = greedy_seq_list[0]
        path_init_task_map = greedy_path_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


def repair_greedy_urgency(destroyed_sequence_map: Dict,
                          path_init_task_map: Dict,
                          destroyed_task_list: List[int],
                          current_solution: Solution) -> Solution:
    """
    Urgency-based repair: Re-inserts tasks sorted by their deadlines (Earliest Deadline First).
    """
    instance = current_solution.instance

    # Sort by deadline (index 3 in instance data)
    urgency_list = sorted([[t, instance[t - 1][3]] for t in destroyed_task_list], key=lambda x: x[1])

    for task_info in urgency_list:
        task = task_info[0]
        greedy_seq_list = [None]
        greedy_path_list = [None]
        fitness_min = [1e9]
        required_robot = instance[task - 1][5]

        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=task,
            fitness_min_list=fitness_min,
            greedy_destroyed_sequence_map_list=greedy_seq_list,
            greedy_path_init_task_map_list=greedy_path_list
        )

        if greedy_seq_list[0] is None:
            logging.error(f"Feasibility error: Task {task} has no valid insertion point.")

        destroyed_sequence_map = greedy_seq_list[0]
        path_init_task_map = greedy_path_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


def repair_greedy_cost(destroyed_sequence_map: Dict,
                       path_init_task_map: Dict,
                       destroyed_task_list: List[int],
                       current_solution: Solution) -> Solution:
    """
    Cost-based repair: Re-inserts tasks based on their original contribution to the objective cost.
    """
    info_map = current_solution.info_map
    instance = current_solution.instance

    # Sort by previous cost in descending order
    cost_list = sorted([[t, info_map[t]['cost']] for t in destroyed_task_list], key=lambda x: x[1], reverse=True)

    for task_info in cost_list:
        task = task_info[0]
        greedy_seq_list = [None]
        greedy_path_list = [None]
        fitness_min = [1e9]
        required_robot = instance[task - 1][5]

        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=task,
            fitness_min_list=fitness_min,
            greedy_destroyed_sequence_map_list=greedy_seq_list,
            greedy_path_init_task_map_list=greedy_path_list
        )

        if greedy_seq_list[0] is None:
            logging.error(f"Feasibility error: Task {task} has no valid insertion point.")

        destroyed_sequence_map = greedy_seq_list[0]
        path_init_task_map = greedy_path_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


# ==============================================================================
# Helper Search Function
# ==============================================================================

def search_insertions(instance: List[List],
                      required_robot: List[int],
                      current_robot_type: int,
                      destroyed_sequence_map: Dict,
                      path_init_task_map: Dict,
                      destroyed_task: int,
                      fitness_min_list: List[float],
                      greedy_destroyed_sequence_map_list: List[Optional[Dict]],
                      greedy_path_init_task_map_list: List[Optional[Dict]]) -> None:
    """
    Recursively searches for optimal insertion positions for multi-robot tasks
    considering heterogeneous requirements.
    """
    # Base Case: All required robot types have been processed
    if current_robot_type == len(required_robot) + 1:
        fitness, _ = cal_fitness(instance, destroyed_sequence_map, path_init_task_map)
        if fitness < fitness_min_list[0]:
            fitness_min_list[0] = fitness
            greedy_destroyed_sequence_map_list[0] = copy_dict_int_dict(destroyed_sequence_map)
            greedy_path_init_task_map_list[0] = copy_dict_int_int(path_init_task_map)
        return

    # Skip current robot type if it is not required for the specific task
    if required_robot[current_robot_type - 1] == 0:
        search_insertions(instance, required_robot, current_robot_type + 1,
                          destroyed_sequence_map, path_init_task_map,
                          destroyed_task, fitness_min_list,
                          greedy_destroyed_sequence_map_list, greedy_path_init_task_map_list)
    else:
        # Explore all feasible positions for the current robot type
        feasible_positions = get_feasible_insert_position(destroyed_sequence_map,
                                                          path_init_task_map,
                                                          destroyed_task,
                                                          current_robot_type)
        for pos in feasible_positions:
            # Create local copies for this recursion branch to maintain state
            seq_tmp = copy_dict_int_dict(destroyed_sequence_map)
            path_tmp = copy_dict_int_int(path_init_task_map)

            insert_(seq_tmp, path_tmp, destroyed_task, pos, current_robot_type)

            search_insertions(instance, required_robot, current_robot_type + 1,
                              seq_tmp, path_tmp, destroyed_task,
                              fitness_min_list, greedy_destroyed_sequence_map_list,
                              greedy_path_init_task_map_list)