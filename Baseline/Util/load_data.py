import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Any


def read_excel(file_name: str) -> List[List[Any]]:
    """
    Loads HRSP instance data from an Excel file and parses structured string fields.

    Args:
        file_name (str): Name of the Excel file located in the '/Instance/' directory.

    Returns:
        List[List[Any]]: A list of tasks, where each task is represented as a list of
                         attributes, including parsed robot requirements and operation times.
    """
    # Resolve the absolute path to the Instance directory
    # Note: Path logic remains strictly as per original implementation
    base_path = Path(__file__).resolve().parent.parent.parent
    full_path = str(base_path) + "/Instance/" + file_name

    # Load raw data using pandas
    df = pd.read_excel(full_path)

    # Convert dataframe rows to a list of lists
    instance = [list(row) for index, row in df.iterrows()]

    # Post-processing: Parse string representations of lists into Python objects
    for task_info in instance:
        # Parse Required Robot Types (Expected at index -1)
        required_robot_str = task_info[-1]
        task_info[-1] = ast.literal_eval(required_robot_str)

        # Parse Operation Times (Expected at index -2)
        operation_time_str = task_info[-2]
        task_info[-2] = ast.literal_eval(operation_time_str)

    return instance