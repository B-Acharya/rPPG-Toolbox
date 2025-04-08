import re
import pathlib
import csv

import csv

def read_csv_file(file_path):
    """
    Reads a CSV file and returns its contents as a dictionary.

    :param file_path: Path to the CSV file.
    :return: Dictionary containing the CSV contents.
    """
    data = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row
        headers = next(reader)

        for row in reader:
            if len(row) == 2:  # Ensure each row has a key and value
                key, value = row
                try:
                    # Convert value to a float if possible
                    data[key] = float(value)
                except ValueError:
                    data[key] = value  # Leave as string if conversion fails
    return data

def save_dict_to_csv(data, file_path):
    """
    Saves a dictionary to a CSV file.

    :param data: Dictionary to save.
    :param file_path: Path to the output CSV file.
    """
    if not isinstance(data, dict):
        raise ValueError("The data must be a dictionary.")

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers (keys)
        writer.writerow(["Key", "Value"])

        # Write each key-value pair
        for key, value in data.items():
            writer.writerow([key, value])


def extract_model_info(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Match the Result block
    # match = re.search(
    #     r"Result\(\n\s+metrics=\{(.*?)\},\n\s+path='(.*?)',",
    #     content,
        # re.DOTALL,
    # )

    match = re.search(
        r"Result\(\n\s+error='RayTaskError\(ValueError\)',\n\s+metrics=\{(.*?)\},\n\s+path='(.*?)',",
        content,
        re.DOTALL,
    )

    if match:
        metrics, path = match.groups()
        # Parse the metrics dictionary
        metrics_dict = eval(f"{{{metrics}}}")
        return metrics_dict, path
    else:
        return None

if __name__ == "__main__":
    log_paths = {
            "tscan": "/homes/bacharya/rPPG-Toolbox/ray/ts-can/",
            "physnet": "/homes/bacharya/rPPG-Toolbox/ray/vmc-mbp/"
    }

    log_patterns = {
        ("tscan", "pseudo-ppg"): "early-stop_tscan_dropout-pseudo-*.log",
        ("tscan", "ppg"): "early-stop_tscan_dropouto_*.log",
        ("physnet", "pseudo-ppg"): "ea5ly-pseudo-physnet_*.log",
        ("physnet", "ppg"): "early-physnet_*.log"
    }

    model = "physnet"
    # model = "physnet"
    experimetn_loss = "ppg"
    # experimetn_loss = "ppg"

    basepath = pathlib.Path(log_paths[model])
    log_files = sorted(basepath.rglob(log_patterns[(model, experimetn_loss)]))

    result_dict = {}

    for log_file in log_files:
        print(log_file)
        model_info = extract_model_info(log_file)
        #get the last integer
        i = log_file.stem[-1]
        if model_info:
            metrics, path = model_info
            print("Model Information:")
            print(f"Metrics: {metrics}")
            print(f"Path: {path}")
            print(pathlib.Path(path).stem)
            print("-"*100)
            result_dict[str(i)] = (path)
        else:
            print("-"*100)
            print("No model information found.")
            print("-"*100)

    save_dict_to_csv(result_dict, str(model) + "_" + str(experimetn_loss)+".csv")
    print(f"saved file to {model}_{experimetn_loss}.csv")
