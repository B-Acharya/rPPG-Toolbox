import pandas as pd

pattern = r"\{('[^']+': \{[^}]+\}(, )?)+\}"
import ast
from collections import defaultdict
import pandas as pd
import requests
import subprocess
import re
import pathlib
import math
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import ClassVar, Literal, Tuple, List, Dict, Optional
import pandas as pd
import scipy.stats as stats


@dataclass
class metrics:
    model: str
    test_Used: str
    t_statistic: float
    p_value: float
    effect_size_r: float
    median_samples1: float
    median_samples2: float
    delta_median: float


@dataclass
class statistical_tests:
    dataframe: pd.DataFrame
    model_name: str
    dataset: Literal["COHFACE", "CHILL", "PURE"]
    HR_threshold: float = 80.0

    def __post_init__(self):

        if self.dataset == "COHFACE":

            self.DARK_SCENARIOS = [2, 3]
            self.BRIGHT_SCENARIOS = [0, 1]

        elif self.dataset == "CHILL":
            print("Using CHILL dataset")
            print("Defining scenarios")

            self.HIGH_HR_SCENARIOS = [2, 3]
            self.LOW_HR_SCENARIOS = [0, 1]

            self.DARK_SCENARIOS = [1, 2]
            self.BRIGHT_SCENARIOS = [0, 3]

        elif self.dataset == "PURE":
            raise NotImplementedError

    def _split_based_on_hr(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.dataset == "CHILL":
            low_HR_setting_mask = self.dataframe["scenario"].isin(self.LOW_HR_SCENARIOS)
            high_HR_setting_mask = self.dataframe["scenario"].isin(self.HIGH_HR_SCENARIOS)

            df_lowHR = self.dataframe[low_HR_setting_mask]
            df_highHR = self.dataframe[high_HR_setting_mask]

            df_lowHR = df_lowHR[df_lowHR["GT_HR"] < self.HR_threshold]
            df_highHR = df_highHR[df_highHR["GT_HR"] >= self.HR_threshold]

        elif self.dataset == "COHFACE" or self.dataset == "PURE":

            df_lowHR = self.dataframe[self.dataframe["GT_HR"] < self.HR_threshold]
            df_highHR = self.dataframe[self.dataframe["GT_HR"] >= self.HR_threshold]

        else:
            raise NotImplementedError

        return df_lowHR, df_highHR

    def _split_based_on_illumination(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df_bright_mask = self.dataframe["scenario"].isin(self.BRIGHT_SCENARIOS)
        df_dark_mask = self.dataframe["scenario"].isin(self.DARK_SCENARIOS)

        df_bright = self.dataframe[df_bright_mask]
        df_dark = self.dataframe[df_dark_mask]

        return df_bright, df_dark

    @staticmethod
    def plot_gt_distribution(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        """
        Plots the ground truth HR distribution of the two different groups
        :param df1:
        :param df2:

        """
        df1["source"] = "source 1"
        df2["source"] = "source 2"

        df_cat = pd.concat([df1, df2], ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_cat, x="GT_HR", hue="source", bins=40, kde=True, multiple="stack")
        plt.show()

    @staticmethod
    def plot_error_distribution(df1: pd.DataFrame, df2: pd.DataFrame) -> None:

        df1["source"] = "source 1"
        df2["source"] = "source 2"

        df_cat = pd.concat([df1, df2], ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_cat, x="Error", hue="source", bins=40, kde=True, multiple="stack")
        plt.show()

    @staticmethod
    def rain_plot(df1: pd.Series, df2: pd.Series, model_name=None) -> None:

        # import PtitPrince as pt
        import ptitprince as pt
        import matplotlib.pyplot as plt

        # Sample data structure (adapt to your actual DataFrames)
        data = {
            'Groups': ['Group1']*len(df1)+ ['Group2']*len(df2),
            'HR': list(df1) + list(df2) ,  # Your HR values
        }

        #adding color
        f, ax = plt.subplots(figsize=(7, 5))

        pt.RainCloud(x="Groups", y="HR", data=data, ax=ax)
        # ax.axhline(80, color='red', linestyle='--', alpha=0.7, label='High HR Threshold')
        # ax.legend(loc='upper right', frameon=False)  # Moved inside
        # ax.axis('off')  # Hide axes
        if model_name is not None:
            plt.title('Error Distribution between groups: {}'.format(model_name))
        else:
            plt.title('Error Distribution between groups')

        plt.ylabel('Error (MAE)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_box(series1: pd.Series, series2: pd.Series) -> None:

        df_cat = pd.DataFrame(
            {
                "source 1": series1,
                "source 2": series2,
            }
        )

        # Create the box plots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_cat)

        # Customize the plot (optional)
        plt.title("Box Plot of Errors by Source")
        plt.xlabel("Source")
        plt.ylabel("Error")

        # Show the plot
        plt.show()

    @staticmethod
    def plot_error_scatterplot(df1: pd.DataFrame, df2: pd.DataFrame) -> None:

        df1["source"] = "source 1"
        df2["source"] = "source 2"

        df_cat = pd.concat([df1, df2], ignore_index=True)

        plt.figure(figsize=(20, 6))
        sns.scatterplot(
            data=df_cat,
            x="GT_HR",
            y="Error",
            hue="source",  # Different marker style for dataset source
            s=100,
            legend=True,
        )

        plt.xlabel("Heart Rate (GT_HR)")
        plt.ylabel("Error")
        plt.title("Scatter Plot of Heart Rate vs. Error by Source")
        plt.grid(True)
        plt.show()

    def _fold_level_mae_per_participant(
        self, df1: pd.DataFrame, df2: pd.DataFrame, debug=False
    ) -> Tuple[pd.Series, pd.Series]:

        participants = set(df1["Participant"]).intersection(set(df2["Participant"]))

        print("No. of Participants in dataset: ", len(participants))

        # Compute fold-level MAE for each scenario
        fold_mae_1 = []
        fold_mae_2 = []

        for participant in participants:

            if debug:
                print("---." * 10)
                print(participant)
                print(len(df1[df1["Participant"] == participant]))

                print("--")
                print(participant)
                print(len(df2[df2["Participant"] == participant]))

            # Get absolute errors for the current fold
            mae_1 = df1[df1["Participant"] == participant]["Error"].abs().mean()
            mae_2 = df2[df2["Participant"] == participant]["Error"].abs().mean()

            if debug:
                print("-" * 50)
                print(mae_1, mae_2)

            fold_mae_1.append(mae_1)
            fold_mae_2.append(mae_2)

        # Convert to pandas Series for easier manipulation
        fold_mae_1 = pd.Series(fold_mae_1)
        fold_mae_2 = pd.Series(fold_mae_2)

        return fold_mae_1, fold_mae_2

    def paired_wilcoxen_test_on_illumination_groups(
        self, debug: Optional[bool] = False, plot: Optional[bool] = False
    ) -> metrics:
        """This can be only used for the illumination setting as there are clear splits for each participant"""

        print("This is performed on illumination splits")

        df_s1, df_s2 = self._split_based_on_illumination()

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            self.plot_error_scatterplot(df_s1.copy(), df_s2.copy())

        fold_mae_1, fold_mae_2 = self._fold_level_mae_per_participant(df_s1, df_s2, debug)

        if plot:
            self.plot_box(fold_mae_1, fold_mae_2)
            self.rain_plot(fold_mae_1, fold_mae_2, self.model_name)

        metrics = self.wilcoxon_test(fold_mae_1, fold_mae_2, self.model_name, debug=debug)

        return metrics

    def paired_wilcoxen_test_on_hr_groups(self, debug: Optional[bool]= False, plot: Optional[bool]=False) -> metrics:

        print("This is performed on hr splits")

        df_s1, df_s2 = self._split_based_on_hr()

        # Step 1: Find common participants
        participants = set(df_s1["Participant"]) & set(df_s2["Participant"])
        print("No. of Participants in dataset: ", len(participants))

        # Step 2: Filter relevant data
        df_s1_filtered = df_s1[df_s1["Participant"].isin(participants)]
        df_s2_filtered = df_s2[df_s2["Participant"].isin(participants)]

        # Step 3: Group by 'Participant' and compute mean of 'Error' and 'GT_HR'
        df_s1_grouped = (
            df_s1_filtered.groupby("Participant")
            .agg({"Error": "mean", "GT_HR": "mean"})
            .rename(columns={"Error": "Error_1", "GT_HR": "GT_HR_1"})
        )
        df_s2_grouped = (
            df_s2_filtered.groupby("Participant")
            .agg({"Error": "mean", "GT_HR": "mean"})
            .rename(columns={"Error": "Error_2", "GT_HR": "GT_HR_2"})
        )

        # Step 4: Merge both grouped dataframes on 'Participant'
        grouped_df = df_s1_grouped.merge(df_s2_grouped, on="Participant", how="inner").reset_index()

        df_s1 = grouped_df[["Error_1", "GT_HR_1"]].rename(columns={"Error_1": "Error", 'GT_HR_1': 'GT_HR'})
        df_s2 = grouped_df[["Error_2", "GT_HR_2"]].rename(columns={"Error_2": "Error", 'GT_HR_2': 'GT_HR'})

        if debug:
            print(df_s1)
            print(df_s2)

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            self.plot_error_scatterplot(df_s1.copy(), df_s2.copy())
            self.plot_box(df_s1["Error"], df_s2["Error"])
            # self.plot_box(fold_mae_1, fold_mae_2)
            self.rain_plot(df_s1['Error'], df_s2["Error"], self.model_name)

        return_metrics = self.wilcoxon_test(
            df_s1["Error"],
            df_s2["Error"],
            self.model_name,
            debug=debug,
        )

        return return_metrics

    @staticmethod
    def wilcoxon_test(
        samples1: pd.Series, samples2: pd.Series, model_name: str, debug: Optional[bool]= False
    )-> metrics:

        return_metrics = metrics(
            model=model_name,
            test_Used="Wilcoxon Signed-Rank Test",
            t_statistic=0.0,
            p_value=0.0,
            effect_size_r=0.0,
            median_samples1=0.0,
            median_samples2=0.0,
            delta_median=0.0
        )

        differences = samples1 - samples2

        if debug:
            print("-" * 50)
            print(differences)

        # Step 1: Check for normality
        shapiro_test = stats.shapiro(differences)
        p_normality = shapiro_test.pvalue

        if p_normality > 0.05:
            print("Can use Paired t-test")

        #Compute the Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(samples1, samples2)

        # Compute the Z-score for W
        n = len(samples1)  # Ensure this excludes zero-difference pairs if needed
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z_score = (w_stat - mu) / sigma

        # Correct effect size calculation
        effect_size_r = z_score / np.sqrt(n)

        return_metrics.t_statistic = w_stat
        return_metrics.p_value = p_value
        return_metrics.effect_size_r = effect_size_r
        return_metrics.median_samples1 = np.median(samples1)
        return_metrics.median_samples2 = np.median(samples2)
        return_metrics.delta_median = return_metrics.median_samples1 - return_metrics.median_samples2

        if p_value < 0.05:
            print("Significant difference detected between scenarios!")
        else:
            print("No significant difference detected between scenarios.")

        return return_metrics

    def perform_Mann_Whitney_test(self, debug=False) -> metrics:

        print("This is performed on HR ")

        df_s1, df_s2 = self._split_based_on_hr()

        if debug:
            self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            self.plot_error_scatterplot(df_s1.copy(), df_s2.copy())

        fold_mae_1, fold_mae_2 = df_s1["Error"], df_s2["Error"]

        metrics = self.Mann_Whitney_test(fold_mae_1, fold_mae_2, self.model_name, debug=debug)

        return metrics

    @staticmethod
    def Mann_Whitney_test(samples1, samples2, model_name, debug=False):

        # TODO: this is not implemented yet

        return_metrics = metrics(
            model=model_name,
            test_Used="Mann-Whitney Test",
            t_statistic=0.0,
            p_value=0.0,
            effect_size_r=0.0,
            median_samples1=0.0,
            median_samples2=0.0,
            delta_median=0.0
        )

        differences = samples1 - samples2

        if debug:
            print("-" * 50)
            print(differences)

        # Step 1: Check for normality
        shapiro_test = stats.shapiro(differences)
        p_normality = shapiro_test.pvalue

        if p_normality > 0.05:
            print("Can use Paired t-test")

        # Step 2: Choose appropriate test
        t_stat, p_value = stats.wilcoxon(samples1, samples2)

        cohen_d = differences.mean() / differences.std(ddof=1)

        n = len(differences)
        # The expected mean (mu) and standard deviation (sigma) of the Wilcoxon statistic (W) are:
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        # Here, t_stat is the Wilcoxon test statistic W.
        z_score = (t_stat - mu) / sigma
        effect_size_r = z_score / np.sqrt(n)

        return_metrics.t_statistic = t_stat
        return_metrics.p_value = p_value
        return_metrics.cohen_d = cohen_d
        return_metrics.effect_size_r = effect_size_r
        return_metrics.median_samples1 = np.median(samples1)
        return_metrics.median_samples2 = np.median(samples2)
        return_metrics.delta_median = return_metrics.median_samples1 - return_metrics.median_samples2

        if p_value < 0.05:
            print("Significant difference detected between scenarios!")
        else:
            print("No significant difference detected between scenarios.")

        return return_metrics

    def perform_spearman_correlation(self, debug=False)-> Tuple[metrics, metrics]:

        df_s1, df_s2 = self._split_based_on_hr()

        if debug:
            self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            self.plot_error_scatterplot(df_s1.copy(), df_s2.copy())


        return_metrics_1 = self.plot_spearman_correlations(df_s1['GT_HR'], df_s1['Error'], name="low-HR", debug=debug)
        return_metrics_2 = self.plot_spearman_correlations(df_s2['GT_HR'], df_s2['Error'], name="high-HR", debug=debug)
        return return_metrics_1, return_metrics_2

    @staticmethod
    def plot_spearman_correlations(series1: pd.Series, series2: pd.Series, name: str, debug: Optional[bool]=False) -> metrics:

        return_metrics =  metrics(
            model="spearman",
            test_Used="spearman",
            t_statistic=0.0,
            p_value=0.0,
            effect_size_r=0.0,
            median_samples1=0.0,
            median_samples2=0.0,
            delta_median=0.0
        )

        _, p1 = stats.shapiro(series1)
        _, p2 = stats.shapiro(series2)

        # Decide which correlation test to use
        if p1 < 0.05 or p2 < 0.05:
            corr_method = "Spearman"
            corr_coef, p_value = stats.spearmanr(series1.tolist(), series2.tolist())
        else:
            corr_method = "Pearson"
            corr_coef, p_value = stats.pearson(series1, series2)

        # Plot the scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=series1, y=series2)

        # Display correlation coefficient on the plot
        plt.title(f'Setting-{name}:{corr_method} Correlation: r = {corr_coef:.3f}, p = {p_value:.3f}')
        plt.xlabel('Series 1')
        plt.ylabel('Series 2')

        # Show the plot
        plt.show()

        # Print correlation details
        print(f"Using {corr_method} correlation:")
        print(f"Correlation coefficient (r): {corr_coef:.3f}")
        print(f"P-value: {p_value:.3f}")

        # Interpretation
        if p_value < 0.05:
            print("Significant correlation detected!")
        else:
            print("No significant correlation.")

        return_metrics.t_statistic = corr_coef
        return_metrics.p_value = p_value

        return return_metrics

# Function to calculate MAE
def calculate_mae(gt_list, pred_list):
    return sum(abs(gt - pred) for gt, pred in zip(gt_list, pred_list)) / len(gt_list)


def plot_mae_per_fold(mae_dict):
    # Convert to numpy array
    sorted_mae_values = [mae_dict[key] for key in sorted(mae_dict.keys())]

    mae_array = np.array(sorted_mae_values)

    # Calculate mean and standard error

    mean_mae = np.mean(mae_array)
    median_mae = np.median(mae_array)

    se_mae = np.std(mae_array, ddof=1) / np.sqrt(len(mae_array))

    # Plot the MAE values

    plt.figure(figsize=(8, 5))

    plt.bar(
        sorted(mae_dict.keys()), mae_array, color="skyblue", edgecolor="black", label="Fold MAE"
    )

    plt.axhline(mean_mae, color="red", linestyle="--", label=f"Mean MAE = {mean_mae:.2f}")
    plt.axhline(median_mae, color="blue", linestyle="--", label=f"Median MAE = {median_mae:.2f}")

    plt.title("MAE per Fold with Mean Line")

    plt.xlabel("Fold")

    plt.ylabel("MAE")

    plt.legend()

    plt.tight_layout()

    plt.show()


def plot_mae_per_scenario(mae_dict):
    # Convert to numpy array
    sorted_mae_values = [mae_dict[key] for key in sorted(mae_dict.keys())]

    mae_array = np.array(sorted_mae_values)

    # Calculate mean and standard error

    mean_mae = np.mean(mae_array)
    median_mae = np.median(mae_array)

    se_mae = np.std(mae_array, ddof=1) / np.sqrt(len(mae_array))

    # Plot the MAE values

    plt.figure(figsize=(8, 5))

    plt.bar(
        sorted(mae_dict.keys()), mae_array, color="skyblue", edgecolor="black", label="Scenario MAE"
    )

    plt.axhline(mean_mae, color="red", linestyle="--", label=f"Mean MAE = {mean_mae:.2f}")
    plt.axhline(median_mae, color="blue", linestyle="--", label=f"Median MAE = {median_mae:.2f}")

    plt.title("MAE per Scenario with Mean/Median Line")

    plt.xlabel("Scenario")

    plt.ylabel("MAE")

    plt.legend()

    plt.tight_layout()

    plt.show()


def get_result_dict(experiment_dict, debug):
    result_dicts = {}
    for exp in experiment_dict:

        match = re.search(pattern, exp.get_output())
        name = exp.name
        if debug:
            print(name)

        if match:
            temp_dict = ast.literal_eval(match.group())
            if debug:
                print("Matched line:", match.group())
                print("-" * 40)
            result_dicts[name[-1]] = temp_dict
        else:
            raise NotImplementedError
    return result_dicts


def calculate_metrics(experiment_dict, debug=False):
    result_dicts = get_result_dict(experiment_dict, debug)

    # Initialize dictionaries to store MAE values
    mae_per_fold = {}

    overall_mae = []
    mae_per_scenario = defaultdict(list)

    errors_per_fold_per_scenario = defaultdict(lambda: defaultdict(list))
    errors_per_fold_per_HR_SPLIT = defaultdict(lambda: defaultdict(list))

    # Iterate through each fold
    for fold, scenarios in result_dicts.items():
        print(mae_per_scenario)

        gt_list = []
        pred_list = []

        # Iterate through each scenario in the fold
        for scenario, values in scenarios.items():
            gt_list.append(values["GT_HR"])
            pred_list.append(values["Pred_HR"])

            # Calculate MAE for each scenario
            scenario_key = scenario.split("_")[-1][-1]  # Extract the last part (e.g., '901')

            mae_per_scenario[scenario_key].append(abs(values["GT_HR"] - values["Pred_HR"]))
            overall_mae.append(abs(values["GT_HR"] - values["Pred_HR"]))

            errors_per_fold_per_scenario[fold][scenario_key].append(
                abs(values["GT_HR"] - values["Pred_HR"])
            )

            if values["GT_HR"] >= 80.0:
                errors_per_fold_per_HR_SPLIT[fold]["Tachycardia"].append(
                    abs(values["GT_HR"] - values["Pred_HR"])
                )
            else:
                errors_per_fold_per_HR_SPLIT[fold]["Bradycardia"].append(
                    abs(values["GT_HR"] - values["Pred_HR"])
                )

        # Calculate MAE for the fold
        mae_per_fold[fold] = calculate_mae(gt_list, pred_list)

    # Calculate MAE for each scenario
    for scenario_key, errors in mae_per_scenario.items():
        mae_per_scenario[scenario_key] = sum(errors) / len(errors)
        # mae_per_scenario[scenario_key] = np.median(errors)

    # Print results
    if debug:
        print("MAE per fold:")
        for fold, mae in mae_per_fold.items():
            print(f"Fold {fold}: {mae}")

    plot_mae_per_fold(mae_per_fold)

    if debug:
        print("\nMAE per scenario:")
        for scenario, mae in mae_per_scenario.items():
            print(f"Scenario {scenario}: {mae}")

    plot_mae_per_scenario(mae_per_scenario)

    # Compute mean and variance of MAE across folds
    mae_values = list(mae_per_fold.values())
    mean_mae = np.mean(mae_values)
    median_mae = np.median(mae_values)

    std_mae = np.std(mae_values)

    print("\nOverall MAE: ", sum(list(mae_per_fold.values())) / len(mae_per_fold.values()))
    print(
        "\nOverall MAE: ",
        sum(list(mae_per_scenario.values())) / len(list(mae_per_scenario.values())),
    )

    return (
        {"MAE": mean_mae, "VAR": std_mae, "Median": median_mae},
        mae_per_scenario,
        errors_per_fold_per_scenario,
        errors_per_fold_per_HR_SPLIT,
    )


def get_dataframe(experiment_dict, debug=False):

    result_dict = get_result_dict(experiment_dict, debug)
    df = pd.DataFrame.from_dict(
        {
            (outerKey, innerKey): values
            for outerKey, innerDict in result_dict.items()
            for innerKey, values in innerDict.items()
        },
        orient="index",
    )

    df.index.names = ["FOLD", "Participant-ID"]
    df["Error"] = abs(df["GT_HR"] - df["Pred_HR"])
    return df




def perform_ttest(samples1, samples2, model_name, debug=False):

    differences = samples1 - samples2

    if debug:
        print("-" * 50)
        print(differences)

    # Step 1: Check for normality
    shapiro_test = stats.shapiro(differences)
    p_normality = shapiro_test.pvalue

    # Step 2: Choose appropriate test
    if p_normality > 0.05:  # Data is normally distributed
        t_stat, p_value = stats.ttest_rel(samples1, samples2)
        test_used = "Paired t-test"
    else:  # Non-normal distribution
        t_stat, p_value = stats.wilcoxon(samples1, samples2)
        test_used = "Wilcoxon signed-rank test"

    cohen_d = differences.mean() / differences.std(ddof=1)

    if test_used == "Wilcoxon signed-rank test":
        n = len(differences)
        # The expected mean (mu) and standard deviation (sigma) of the Wilcoxon statistic (W) are:
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        # Here, t_stat is the Wilcoxon test statistic W.
        z_score = (t_stat - mu) / sigma
        effect_size_r = z_score / np.sqrt(n)
    else:
        effect_size_r = np.nan  # Not computed for the t-test.

    # Print results
    print(f"Test Used: {test_used}")
    print(f"Test Used: {t_stat}")
    print(f"p-value: {p_value:.5f}")
    print(f"cohen_d: {cohen_d:.5f}")
    print(f"effect_size_r: {effect_size_r:.5f}")

    if p_value < 0.05:
        print("Significant difference detected between scenarios!")
    else:
        print("No significant difference detected between scenarios.")

    return {
        "Model": model_name,
        "Test Used": test_used,
        "t-statistic": t_stat,
        "p-value": round(p_value, 3),
        "cohen-d": round(cohen_d, 3),
        "effect-size-r": round(effect_size_r, 3),
    }


def perform_ttest_per_fold(dataframe, model_name, dataset="COHFACE", type="HR", debug=False):

    df = dataframe.copy()

    # Filter data for each scenario for COHFACE
    if dataset == "COHFACE":
        # dark scnearios for cohface
        df_s1 = df[(df["scenario"] == 2) | (df["scenario"] == 3)]
        # clean sceanrios for cohface
        df_s2 = df[(df["scenario"] == 0) | (df["scenario"] == 1)]

    if dataset == "CMBP":
        if type == "HR":
            # high HR scenarios
            df_s1 = df[(df["scenario"] == 2) | (df["scenario"] == 3)]

            # low HR scenarios
            df_s2 = df[(df["scenario"] == 0) | (df["scenario"] == 1)]

        elif type == "Illumination":
            # dark scenarios for cmbp
            df_s1 = df[(df["scenario"] == 1) | (df["scenario"] == 2)]
            # clean scenarios for cmbp
            df_s2 = df[(df["scenario"] == 0) | (df["scenario"] == 3)]

    if debug:
        print("Scenarios 0 and 1")
        print(df_s1)
        print("Scenarios 2 and 3")
        print(df_s2)

    # Ensure both scenarios have the same folds
    common_folds = set(df_s1["FOLD"]).intersection(set(df_s2["FOLD"]))

    print(f"Common Folds: {common_folds}")

    # Compute fold-level MAE for each scenario
    fold_mae_s1 = []
    fold_mae_s2 = []

    for fold in common_folds:
        # Get absolute errors for the current fold
        mae_s1 = df_s1[df_s1["FOLD"] == fold]["Error"].abs().mean()
        mae_s2 = df_s2[df_s2["FOLD"] == fold]["Error"].abs().mean()

        if debug:
            print("-" * 50)
            print(mae_s1, mae_s2)

        fold_mae_s1.append(mae_s1)
        fold_mae_s2.append(mae_s2)

    # Convert to pandas Series for easier manipulation
    fold_mae_s1 = pd.Series(fold_mae_s1)
    fold_mae_s2 = pd.Series(fold_mae_s2)

    return perform_ttest(fold_mae_s1, fold_mae_s2, model_name, debug=debug)


def perform_ttest_participant(dataframe, model_name, dataset="COHFACE", type="HR", debug=False):

    df = dataframe.copy()

    if dataset == "COHFACE":
        # dark scenarios for cohface
        df_s1 = df[(df["scenario"] == 2) | (df["scenario"] == 3)]
        df_s1.hist()
        plt.show()
        # clean scenarios for cohface

        df_s2 = df[(df["scenario"] == 0) | (df["scenario"] == 1)]
        df_s2.hist()
        plt.show()

    if dataset == "CMBP":
        if type == "HR":
            # high HR scenarios
            df_s1 = df[(df["scenario"] == 2) | (df["scenario"] == 3)]
            df_s1.hist()
            plt.show()

            # low HR scenarios
            df_s2 = df[(df["scenario"] == 0) | (df["scenario"] == 1)]
            df_s2.hist()
            plt.show()

        elif type == "Illumination":
            # dark scenarios for cmbp
            df_s1 = df[(df["scenario"] == 1) | (df["scenario"] == 2)]
            # clean scenarios for cmbp
            df_s2 = df[(df["scenario"] == 0) | (df["scenario"] == 3)]

    if debug:
        print("Scenarios 0 and 1")
        print(df_s1)
        print("Scenarios 2 and 3")
        print(df_s2)

    participants = set(df_s1["Participant"]).intersection(set(df_s2["Participant"]))

    print(f"Participant : {participants}")

    # Compute fold-level MAE for each scenario
    fold_mae_s1 = []
    fold_mae_s2 = []

    for participant in participants:

        # Get absolute errors for the current fold
        mae_s1 = df_s1[df_s1["Participant"] == participant]["Error"].abs().mean()
        mae_s2 = df_s2[df_s2["Participant"] == participant]["Error"].abs().mean()

        if debug:
            print("-" * 50)
            print(mae_s1, mae_s2)

        fold_mae_s1.append(mae_s1)
        fold_mae_s2.append(mae_s2)

    # Convert to pandas Series for easier manipulation
    fold_mae_s1 = pd.Series(fold_mae_s1)
    fold_mae_s2 = pd.Series(fold_mae_s2)

    return perform_ttest(fold_mae_s1, fold_mae_s2, model_name, debug=debug)
