import os
from os import makedirs
from os.path import dirname, join

from core import folder


def find_csv_files(script_dir_path):
    csv_files = []
    skip_dirs = ["svm", "pls", "xgboost"]  # Directories to skip

    for root, dirs, files in os.walk(script_dir_path, topdown=True):
        # Modify dirs in-place to skip certain directories
        dirs[:] = [
            d for d in dirs if not any(skip_dir in d.lower() for skip_dir in skip_dirs)
        ]

        for file in files:
            # Check if the file ends with '.csv' and does not contain 'report' in its name
            if file.endswith(".csv") and "report" not in file.lower():
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_file_name(file_path):
    return file_path.split("/")[-1].split(".")[0]


def create_folder_get_output_path(model_name, csv_file_path, suffix="report", ext="csv"):
    """
    Creates a structured folder for storing model outputs in a separate 'outputs' directory.

    Parameters:
        model_name (str): The name of the model (e.g., "PLS_DA", "SVM").
        csv_file_path (str): The path to the input CSV file.
        suffix (str): The suffix to append to the output file name (default: "report").
        ext (str): The file extension for the output file (default: "csv").

    Returns:
        str: The full path to the output file.
    """

    # Define the base output directory
    base_output_dir = "outputs"

    # Extract the input file name without extension
    file_name = folder.get_file_name(csv_file_path)

    # Create a structured output directory
    output_dir = os.path.join(base_output_dir, model_name, file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full output file path
    output_file_name = f"{file_name}_{suffix}.{ext}"
    output_file_path = os.path.join(output_dir, output_file_name)

    return output_file_path