import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class mlflow_data:
    mlruns_path: str
    experiment_id: str
    run_id: str
    name: str

    def _get_avearged_mae(self) -> dict:
        artifacts_dir = os.path.join(self.mlruns_path, self.experiment_id, self.run_id, "artifacts")

        # Dictionary to store errors per participant and scenario
        errors = defaultdict(list)

        # Iterate over all files in the artifacts directory
        for filename in tqdm(os.listdir(artifacts_dir)):

            # Extract participant ID and scenario ID
            parts = filename.split("_")

            index = parts[0]
            participant_id = parts[1][:-2]
            scenario_id = parts[1][-2:]

            # Read the JSON file
            file_path = os.path.join(artifacts_dir, filename)

            with open(file_path, "r") as f:
                data = json.load(f)

            # Compute absolute error
            mae = abs(data["GT_HR"] - data["Pred_HR"])

            # Store error under participant and scenario
            key = f"{participant_id}_{scenario_id}"
            errors[key].append(mae)

            # Compute mean absolute error per participant and scenario
        mae_values = {k: np.mean(v) for k, v in errors.items()}

        return mae_values

    def get_maes(self) -> list:
        mae_values = self._get_avearged_mae()
        return list(mae_values.values())


if __name__ == "__main__":

    print("started")

    mlflow_face_face = mlflow_data(
        mlruns_path="./mlruns",
        experiment_id="114686732981263784",
        run_id="6be601449d1a42acb8692cc440ae949f",
        name="Face-Face-PPG",
    )

    mlflow_finger_finger = mlflow_data(
        mlruns_path="./mlruns",
        experiment_id="863072867488777331",
        run_id="d52572d5f6bd4a58b8febc4bc09322e8",
        name="Finger-Finger-PPG",
    )

    mlflow_face_finger = mlflow_data(
        mlruns_path="./mlruns",
        experiment_id="713960939873865865",
        run_id="a7f0957eee7e414d8e8d7c8abdbb0ccc",
        name="Face-Finger-PPG",
    )

    mlflow_finger_face = mlflow_data(
        mlruns_path="./mlruns",
        experiment_id="380191366684957392",
        run_id="000b24557909468e8c58e3841031e79d",
        name="Finger-Face-PPG",
    )

    data = [
        mlflow_finger_finger.get_maes(),
        mlflow_face_face.get_maes(),
        mlflow_face_finger.get_maes(),
        mlflow_finger_face.get_maes(),
    ]

    labels = [
        mlflow_finger_finger.name,
        mlflow_face_face.name,
        mlflow_face_finger.name,
        mlflow_finger_face.name,
    ]

    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.violinplot(
        dataset=data,
        showmeans=True
    )
    plt.xticks(np.arange(1, len(data) + 1), labels, rotation=20)
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Box Plot of Errors per Participant and Scenario")

    # Show the plot
    plt.show()
