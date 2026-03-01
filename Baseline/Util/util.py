import math
import random
import bisect
import logging
from typing import List, Dict, Set, Tuple, Any, Union, Optional
from Baseline.Util.Config import Config



# ==============================================================================
# Deep Copy Utility Functions
# ==============================================================================

def copy_set_int(original_set: Set[int]) -> Set[int]:
    """Performs a deep copy of a set containing integers."""
    return {item for item in original_set}


def copy_list_int(original_list: List[int]) -> List[int]:
    """Performs a deep copy of a list containing integers."""
    return [item for item in original_list]


def copy_dict_int_int(original_dict: Dict[int, int]) -> Dict[int, int]:
    """Performs a deep copy of a dictionary mapping integers to integers."""
    return {key: value for key, value in original_dict.items()}


def copy_dict_int_list(original_dict: Dict[int, List[Any]]) -> Dict[int, List[Any]]:
    """Performs a deep copy of a dictionary mapping integers to lists."""
    return {key: [v for v in value] for key, value in original_dict.items()}


def copy_dict_int_dict(original_dict: Dict[int, Dict]) -> Dict[int, Dict]:
    """Performs a deep copy of a nested dictionary."""
    return {key: {ik: iv for ik, iv in inner_dict.items()} for key, inner_dict in original_dict.items()}


# ==============================================================================
# Mapping & Transformation Functions
# ==============================================================================

def code2path_map(code: List[int]) -> Dict[int, List[int]]:
    """
    Decodes a flat list representation into a path map for each robot.
    Tasks are separated by '0' (depot).
    """
    path_map = {}
    path = []
    path_index = 1
    for task_index, task in enumerate(code):
        if task == 0:
            if task_index != 0:
                path_map[path_index] = path
                path = []
                path_index += 1
            else:
                continue
        else:
            path.append(task)
    path_map[path_index] = path
    return path_map


def path_map2sequence_map(path_map: Dict[int, List[int]]) -> Tuple[Dict, Dict]:
    """
    Converts path maps into sequence maps containing predecessor/successor info 
    per robot type for synchronization analysis.
    """
    sequence_map = {}
    path_init_task_map = {}

    for path_index in range(1, Config.ROBOT_NUM + 1):
        # Determine robot type based on index thresholds
        path_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
        path = path_map.get(path_index, [])

        if not path:
            path_init_task_map[path_index] = 0
            continue

        for task_index, task in enumerate(path):
            task_info = sequence_map.get(task, {})
            task_info[f'robot_{path_type}'] = path_index

            # Linking logic
            if task_index == 0:
                task_info[f'robot_{path_type}_pre_task'] = 0
                path_init_task_map[path_index] = task
            else:
                task_info[f'robot_{path_type}_pre_task'] = path[task_index - 1]

            if task_index == len(path) - 1:
                task_info[f'robot_{path_type}_next_task'] = 0
            else:
                task_info[f'robot_{path_type}_next_task'] = path[task_index + 1]

            sequence_map[task] = task_info

    return sequence_map, path_init_task_map


# ==============================================================================
# Mathematical & Heuristic Operators
# ==============================================================================

def get_distance(source_position: List[float], destination_position: List[float]) -> float:
    """Computes the Euclidean distance between two 2D coordinates."""
    return math.hypot(source_position[0] - destination_position[0],
                      source_position[1] - destination_position[1])


def remove_(sequence_map: Dict, path_init_task_map: Dict, task: int) -> Tuple[Dict, Dict]:
    """Removes a task from the current solution and repairs the sequence links."""
    for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
        if f'robot_{robot_type}' not in sequence_map[task]:
            continue

        pre_task = sequence_map[task][f'robot_{robot_type}_pre_task']
        next_task = sequence_map[task][f'robot_{robot_type}_next_task']

        if pre_task == 0:
            robot_id = sequence_map[task][f'robot_{robot_type}']
            path_init_task_map[robot_id] = next_task
        else:
            sequence_map[pre_task][f'robot_{robot_type}_next_task'] = next_task

        if next_task != 0:
            sequence_map[next_task][f'robot_{robot_type}_pre_task'] = pre_task

    del sequence_map[task]
    return sequence_map, path_init_task_map


def insert_(sequence_map: Dict, path_init_task_map: Dict, task: int,
            position: Tuple[int, int], robot_type: int) -> Tuple[Dict, Dict]:
    """
    Inserts a task into the specified position for a given robot type.
    Position format: (neighbor_task, direction_flag) or (0, robot_id).
    """
    if position[0] != 0:
        neighbor_task = position[0]
        # direction: 0 for 'pre', 1 for 'next'
        dir_name = 'pre' if position[1] == 0 else 'next'
        opp_name = 'next' if position[1] == 0 else 'pre'

        target_robot_id = sequence_map[neighbor_task][f'robot_{robot_type}']
        original_link_task = sequence_map[neighbor_task][f'robot_{robot_type}_{dir_name}_task']

        # Update neighbor's link
        sequence_map[neighbor_task][f'robot_{robot_type}_{dir_name}_task'] = task

        if original_link_task != 0:
            sequence_map[original_link_task][f'robot_{robot_type}_{opp_name}_task'] = task
        elif dir_name == 'pre':
            path_init_task_map[target_robot_id] = task

        # Initialize or update task entry
        task_info = sequence_map.get(task, {})
        task_info[f'robot_{robot_type}'] = target_robot_id
        task_info[f'robot_{robot_type}_{dir_name}_task'] = original_link_task
        task_info[f'robot_{robot_type}_{opp_name}_task'] = neighbor_task
        sequence_map[task] = task_info
    else:
        # Inserting into an empty path or as the first task
        robot_id = position[1]
        path_init_task_map[robot_id] = task
        task_info = sequence_map.get(task, {})
        task_info[f'robot_{robot_type}'] = robot_id
        task_info[f'robot_{robot_type}_pre_task'] = 0
        task_info[f'robot_{robot_type}_next_task'] = 0
        sequence_map[task] = task_info

    return sequence_map, path_init_task_map


# ==============================================================================
# Feasibility & Search Functions
# ==============================================================================

def get_all_position(sequence_map: Dict, path_init_task_map: Dict, robot_type: int) -> Set[Tuple]:
    """Finds all potential insertion positions for a specific robot type."""
    tasks_assigned = [
        tid for tid, info in sequence_map.items() if f"robot_{robot_type}" in info
    ]

    # Position flags: 0 for 'Before', 1 for 'After'
    position_set = {(task, 0) for task in tasks_assigned}
    position_set.update({(task, 1) for task in tasks_assigned})

    # Deduplication: Position after task A is equivalent to position before task B if A->B
    to_delete = []
    for pos in position_set:
        task, direction = pos
        if direction == 0:
            pre_task = sequence_map[task][f'robot_{robot_type}_pre_task']
            if pre_task and (pre_task, 1) in position_set:
                to_delete.append((pre_task, 1))
        else:
            next_task = sequence_map[task][f'robot_{robot_type}_next_task']
            if next_task and (next_task, 0) in position_set:
                to_delete.append((task, 1))

    for pos in to_delete:
        position_set.discard(pos)

    # Add empty path start positions
    for robot, init_task in path_init_task_map.items():
        type_of_robot = bisect.bisect_left(Config.INDEX_LIST, robot) + 1
        if type_of_robot == robot_type and init_task == 0:
            position_set.add((0, robot))

    return position_set


def cal_fitness(instance: List[List], sequence_map: Dict, path_init_task_map: Dict) -> Tuple[float, bool]:
    """
    Evaluates the current solution. Returns (fitness_value, feasibility_flag).
    Used primarily during heuristic searches.
    """
    task_num = len(sequence_map)
    info_map = {task: {} for task in sequence_map.keys()}
    total_dist, total_tard = 0.0, 0.0

    enabled_tasks = []
    task_readiness = {task: [0] * Config.ROBOT_TYPE_NUM for task in sequence_map.keys()}

    # Initialize task queue
    for r_idx, init_task in path_init_task_map.items():
        if init_task != 0:
            r_type = bisect.bisect_left(Config.INDEX_LIST, r_idx) + 1
            task_readiness[init_task][r_type - 1] = 1
            info_map[init_task][f'robot_{r_type}_pre_complete_time'] = 0.0
            if instance[init_task - 1][5] == task_readiness[init_task]:
                enabled_tasks.append(init_task)

    completed_tasks = []
    while len(completed_tasks) < task_num:
        if not enabled_tasks:
            logging.error("Infeasible solution: No enabled transitions found.")
            return 1e6, False

        current_task = enabled_tasks.pop(random.randrange(len(enabled_tasks)))
        src_pos = [instance[current_task - 1][1], instance[current_task - 1][2]]

        arrival_times = []
        task_dist_acc = 0.0

        for r_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[current_task - 1][5][r_type - 1] == 0:
                continue

            pre_time = info_map[current_task][f'robot_{r_type}_pre_complete_time']
            pre_tid = sequence_map[current_task][f'robot_{r_type}_pre_task']
            pre_pos = Config.DEPOT if pre_tid == 0 else [instance[pre_tid - 1][1], instance[pre_tid - 1][2]]

            dist = get_distance(pre_pos, src_pos)
            arrival_times.append(pre_time + (dist / Config.VELOCITY))
            task_dist_acc += dist

            # Update readiness for the next task on this robot's path
            next_tid = sequence_map[current_task][f'robot_{r_type}_next_task']
            if next_tid != 0:
                task_readiness[next_tid][r_type - 1] = 1
                if task_readiness[next_tid] == instance[next_tid - 1][5]:
                    enabled_tasks.append(next_tid)

        start_time = max(arrival_times)
        finish_times = [start_time + instance[current_task - 1][4][rt - 1]
                        for rt in range(1, Config.ROBOT_TYPE_NUM + 1) if instance[current_task - 1][5][rt - 1] != 0]

        final_finish = max(finish_times)

        # Pass timing info forward
        for r_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            next_tid = sequence_map[current_task][f'robot_{r_type}_next_task']
            if next_tid != 0:
                info_map[next_tid][f'robot_{r_type}_pre_complete_time'] = start_time + instance[current_task - 1][4][
                    r_type - 1]

        total_dist += task_dist_acc
        total_tard += max(final_finish - instance[current_task - 1][3], 0)
        completed_tasks.append(current_task)

    fitness = total_dist * Config.WEIGHT + total_tard * (1 - Config.WEIGHT)
    return fitness, True


def get_feasible_insert_position(sequence_map: Dict,
                                 path_init_task_map: Dict,
                                 destroyed_task: int,
                                 robot_type: int) -> Set[Tuple]:
    """
    Prunes the set of all potential insertion positions to find those that are
    feasible according to temporal and synchronization constraints.

    This function performs a graph traversal to ensure that a task is not inserted
    at a position that violates the predecessor/successor relationships across
    different robot types.
    """
    # Initialize with all theoretically possible positions for the robot type
    feasible_position_set = get_all_position(sequence_map, path_init_task_map, robot_type)

    # If the task is not yet in the sequence map, all positions are considered feasible
    if destroyed_task not in sequence_map.keys():
        return feasible_position_set

    # --------------------------------------------------------------------------
    # Predecessor Pruning: Discard positions that violate "Before" constraints
    # --------------------------------------------------------------------------
    pre_to_explore_set = {destroyed_task}
    pre_explored_list = []

    while pre_to_explore_set:
        temp_set = copy_set_int(pre_to_explore_set)
        for task in temp_set:
            for to_explore_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                # Check if this robot type is assigned to the current task
                if f"robot_{to_explore_type}" in sequence_map[task].keys():
                    pre_task = sequence_map[task][f"robot_{to_explore_type}_pre_task"]

                    if pre_task != 0:
                        # Discard 'Before' position of the predecessor
                        feasible_position_set.discard((pre_task, 0))

                        if pre_task not in pre_explored_list:
                            pre_to_explore_set.add(pre_task)

                        # Pruning on the specific chain of the target robot_type
                        if f"robot_{robot_type}" in sequence_map[pre_task].keys():
                            pre_pre_task = sequence_map[pre_task][f"robot_{robot_type}_pre_task"]
                            if pre_pre_task != 0:
                                # Discard 'After' position of the grandparent task on this chain
                                feasible_position_set.discard((pre_pre_task, 1))

            pre_to_explore_set.remove(task)
            pre_explored_list.append(task)

    # --------------------------------------------------------------------------
    # Successor Pruning: Discard positions that violate "After" constraints
    # --------------------------------------------------------------------------
    next_to_explore_set = {destroyed_task}
    next_explored_list = []

    while next_to_explore_set:
        temp_set = copy_set_int(next_to_explore_set)
        for task in temp_set:
            for to_explore_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if f"robot_{to_explore_type}" in sequence_map[task].keys():
                    next_task = sequence_map[task][f"robot_{to_explore_type}_next_task"]

                    if next_task != 0:
                        # Discard 'After' position of the successor
                        feasible_position_set.discard((next_task, 1))

                        if next_task not in next_explored_list:
                            next_to_explore_set.add(next_task)

                        # Pruning on the specific chain of the target robot_type
                        if f"robot_{robot_type}" in sequence_map[next_task].keys():
                            next_next_task = sequence_map[next_task][f"robot_{robot_type}_next_task"]
                            if next_next_task != 0:
                                # Discard 'Before' position of the grandchild task on this chain
                                feasible_position_set.discard((next_next_task, 0))

            next_to_explore_set.remove(task)
            next_explored_list.append(task)

    return feasible_position_set