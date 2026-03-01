import torch
from typing import NamedTuple, Dict, Optional
from utils.boolmask import mask_long2bool, mask_long_scatter
from problems.hrsp.paramet_hrsp import paramet_hrsp


class StateHRSP(NamedTuple):
    """
    State representation for the Heterogeneous Robot Scheduling Problem (HRSP).
    """
    # Fixed input data
    coords: torch.Tensor  # Coordinates of all tasks: [batch_size, n_loc, 2]
    deadline: torch.Tensor  # Deadlines for all tasks: [batch_size, n_loc]
    operation_time: torch.Tensor  # Execution times for tasks: [batch_size, n_loc]

    # Instance tracking
    ids: torch.Tensor  # Original indices of rows for memory efficiency in beam search

    # Dynamic state variables
    cur_time: torch.Tensor  # Current time for each robot: [batch_size, robot_num]
    cur_coord: torch.Tensor  # Current coordinates of each robot: [batch_size, robot_num, 2]
    visited_: torch.Tensor  # Visited tasks mask (long or uint8)
    length: torch.Tensor  # Total distance traveled by each robot: [batch_size, robot_num]
    tardiness: torch.Tensor  # Accumulated tardiness per batch: [batch_size, 1]
    i: torch.Tensor  # Current step counter (number of tasks visited)

    @property
    def visited(self) -> torch.Tensor:
        """Returns the visited mask as a boolean tensor."""
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(1))

    @property
    def dist(self) -> torch.Tensor:
        """Computes the L1 distance matrix between all task coordinates."""
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=1, dim=-1)

    @staticmethod
    def initialize(input_data: Dict[str, torch.Tensor], visited_dtype=torch.uint8) -> 'StateHRSP':
        """
        Initializes the state from input data.

        Args:
            input_data: Dictionary containing 'source' (coords), 'deadline', and 'operation_time'.
            visited_dtype: Data type for the visited mask.
        """
        coords = input_data['source']  # Note: Does not include depot
        batch_size, n_loc, _ = coords.size()

        device = coords.device
        robot_num = paramet_hrsp.ROBOT_NUM

        cur_time = torch.zeros(batch_size, robot_num, dtype=torch.float, device=device)
        cur_coord = torch.zeros(batch_size, robot_num, 2, dtype=torch.float, device=device)
        length = torch.zeros(batch_size, robot_num, dtype=torch.float, device=device)
        tardiness = torch.zeros(batch_size, 1, dtype=torch.float, device=device)

        # Initialize visited mask (excluding depot)
        if visited_dtype == torch.uint8:
            visited_shape = (batch_size, 1, n_loc)
            visited_mask = torch.zeros(visited_shape, dtype=torch.uint8, device=device)
        else:
            visited_mask = torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=device)

        return StateHRSP(
            coords=coords,
            deadline=input_data['deadline'],
            operation_time=input_data['operation_time'],
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],
            cur_time=cur_time,
            cur_coord=cur_coord,
            visited_=visited_mask,
            length=length,
            tardiness=tardiness,
            i=torch.zeros(1, dtype=torch.int64, device=device)
        )

    def update(self, selected_task: torch.Tensor, selected_robot_one_hot: torch.Tensor) -> 'StateHRSP':
        """
        Updates the state based on the selected task and robot.

        Args:
            selected_task: Index of the chosen task [batch_size].
            selected_robot_one_hot: Boolean mask for the selected robot [batch_size, robot_num].
        """
        assert self.i.size(0) == 1, "State update only supports single step increments."

        selected_robot_one_hot = selected_robot_one_hot.bool()
        batch_indices = self.ids.squeeze()

        # Update current coordinates for selected robots
        cur_coord = self.cur_coord.clone()
        selected_task_coord = self.coords[batch_indices, selected_task]

        # Expand coordinates to match robot dimensions for masked assignment
        expanded_task_coord = selected_task_coord.unsqueeze(1).expand(-1, selected_robot_one_hot.size(1), -1)
        cur_coord[selected_robot_one_hot] = expanded_task_coord[selected_robot_one_hot]

        # Calculate travel distance and update total length
        diff = cur_coord - self.cur_coord
        distance = diff.norm(p=1, dim=-1)
        length = self.length + distance

        # Update current time based on travel and synchronization logic
        travel_time = distance / paramet_hrsp.ROBOT_VELOCITY
        updated_time = self.cur_time.clone()
        updated_time[selected_robot_one_hot] += travel_time[selected_robot_one_hot]

        # Synchronize: selected robots wait for the last one to arrive at the collaborative task
        masked_time = updated_time.masked_fill(~selected_robot_one_hot, float('-inf'))
        max_time_per_batch, _ = masked_time.max(dim=1, keepdim=True)
        updated_time[selected_robot_one_hot] = max_time_per_batch.expand_as(updated_time)[selected_robot_one_hot]

        # Add heterogeneous operation time based on robot types
        op_time_for_task = self.operation_time[batch_indices, selected_task]
        robot_type_indices = paramet_hrsp.robot_type_indices.expand(updated_time.size(0), -1)
        robot_op_time = op_time_for_task.gather(1, robot_type_indices)
        updated_time[selected_robot_one_hot] += robot_op_time[selected_robot_one_hot]

        # Calculate task completion time and tardiness
        masked_final_times = updated_time.masked_fill(~selected_robot_one_hot, float('-inf'))
        complete_time, _ = masked_final_times.max(dim=1)

        selected_task_deadline = self.deadline[batch_indices, selected_task]
        selected_task_tardiness = torch.clamp_min(complete_time - selected_task_deadline, 0)

        # Update visited mask
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, selected_task[:, None, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, selected_task[:, None])

        return self._replace(
            cur_time=updated_time,
            cur_coord=cur_coord,
            visited_=visited_,
            length=length,
            i=self.i + 1,
            tardiness=self.tardiness + selected_task_tardiness.unsqueeze(-1),
        )

    def all_finished(self) -> bool:
        """Checks if all tasks are finished and the step counter is complete."""
        return self.i.item() >= self.coords.size(1) and self.visited.all()

    def get_finished(self) -> torch.Tensor:
        """Returns a boolean mask indicating finished batches."""
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_mask(self) -> torch.Tensor:
        """Returns the current visited mask."""
        return self.visited_.to(torch.bool)