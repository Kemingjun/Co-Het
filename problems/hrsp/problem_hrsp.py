import time
import torch
import os
import pickle
import ast
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from problems.hrsp.state_hrsp import StateHRSP
from problems.hrsp.paramet_hrsp import paramet_hrsp


class HRSP(object):
    """
    Heterogeneous Robot Scheduling Problem (HRSP) definition.
    """
    NAME = 'hrsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Placeholder for cost calculation logic
        pass

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HRSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHRSP.initialize(*args, **kwargs)


class HRSPDataset(Dataset):
    """
    Dataset loader for HRSP, supporting .pkl, .xlsx, and random generation.
    """

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(HRSPDataset, self).__init__()

        self.data = []
        self.data_set = []

        if filename is not None:
            file_ext = os.path.splitext(filename)[1]

            if file_ext == '.pkl':
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

            elif file_ext == '.xlsx':
                # Evaluate a single instance from Excel
                file_path = str(Path(__file__).resolve().parent.parent.parent) + "/Instance/" + filename
                df = pd.read_excel(file_path)
                instance = [list(row) for index, row in df.iterrows()]

                for task_info in instance:
                    operation_time_str = task_info[-2]
                    task_info[-2] = ast.literal_eval(operation_time_str)

                source = [[row[1], row[2]] for row in instance]
                deadline = [row[3] for row in instance]
                operation_time = [row[4] for row in instance]

                self.data = [{
                    'source': torch.tensor(source, dtype=torch.float),
                    'deadline': torch.tensor(deadline, dtype=torch.float),
                    'operation_time': torch.tensor(operation_time, dtype=torch.float),
                }]

            else:
                # Load multiple instances from a directory
                base_dir = Path(__file__).resolve().parent.parent.parent / f"Instance/{filename}"
                all_files = list(base_dir.glob("*.xlsx"))
                for file_path in all_files:
                    df = pd.read_excel(file_path)
                    instance = [list(row) for index, row in df.iterrows()]

                    for task_info in instance:
                        operation_time_str = task_info[-2]
                        task_info[-2] = ast.literal_eval(operation_time_str)

                    source = [[row[1], row[2]] for row in instance]
                    deadline = [row[3] for row in instance]
                    operation_time = [row[4] for row in instance]

                    self.data.append({
                        'source': torch.tensor(source, dtype=torch.float),
                        'deadline': torch.tensor(deadline, dtype=torch.float),
                        'operation_time': torch.tensor(operation_time, dtype=torch.float),
                    })

        else:
            # Generate random synthetic data
            operation_range_list = [[0.3, 0.7], [0.8, 1.2], [0.8, 1.2]]
            self.data = []

            # Generate deadlines with noise and shuffle
            base = torch.arange(size, dtype=torch.float32) * 0.5 + 0.5
            base_batch = base.unsqueeze(0).repeat(num_samples, 1)
            noise_batch = (torch.rand(num_samples, size) / 2.5 - 0.2)
            deadline_unpermuted = base_batch + noise_batch
            permutations = torch.argsort(torch.rand(num_samples, size), dim=1)
            deadline = torch.gather(deadline_unpermuted, 1, permutations)

            source = torch.rand(num_samples, size, 2)
            operation_time = torch.empty(num_samples, size, paramet_hrsp.ROBOT_TYPE_NUM)

            for i in range(paramet_hrsp.ROBOT_TYPE_NUM):
                range_left = operation_range_list[i][0]
                range_right = operation_range_list[i][1]
                # Randomize operation time based on specified ranges per robot type
                operation_time[..., i] = torch.rand(num_samples, size) * (range_right - range_left) + range_left

            for i in range(num_samples):
                self.data.append({
                    'source': source[i],
                    'deadline': deadline[i],
                    'operation_time': operation_time[i],
                })

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def make_instance(args):
    """
    Helper to format raw instance arguments into torch tensors.
    """
    source, destination, deadline, required_robot, operation_time, *extra_args = args
    grid_size = 1
    if len(extra_args) > 0:
        depot_types, customer_types, grid_size = extra_args

    return {
        'source': torch.tensor(source, dtype=torch.float) / grid_size,
        'destination': torch.tensor(destination, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float),
        'required_robot': torch.tensor(required_robot, dtype=torch.float),
        'operation_time': torch.tensor(operation_time, dtype=torch.float)
    }