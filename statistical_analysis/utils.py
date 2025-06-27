from metrics import HR_metrics, Illumination_metrics, metrics
from plots import plotter

import ast
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

pd.DataFrame.iteritems = pd.DataFrame.items

pattern = r"\{('[^']+': \{[^}]+\}(, )?)+\}"


@dataclass
class statistical_test:

    @staticmethod
    def effect_size(w_stat: float, n: int) -> float:

        # Compute the Z-score for W
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z_score = (w_stat - mu) / sigma

        # Correct effect size calculation
        effect_size_r = z_score / np.sqrt(n)

        return effect_size_r

    @staticmethod
    def wilcoxon_test(
        samples1: pd.Series,
        samples2: pd.Series,
        model_name: str,
        hypothesis: Literal["two-sided", "less", "great"],
        debug: Literal[False] | None,
    ) -> metrics:

        samples1 = samples1.copy()
        samples2 = samples2.copy()

        return_metrics = metrics(
            model=model_name,
            test_Used="Wilcoxon Signed-Rank Test",
            t_statistic=0.0,
            p_value=0.0,
            effect_size_r=0.0,
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

        # Compute the Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(samples1, samples2, alternative=hypothesis)

        n = len(samples1)

        effect_size_r = statistical_test.effect_size(w_stat, n)

        return_metrics.t_statistic = w_stat
        return_metrics.p_value = p_value
        return_metrics.effect_size_r = effect_size_r

        if p_value < 0.05:
            print("Significant difference detected between scenarios!")
        else:
            print("No significant difference detected between scenarios.")

        return return_metrics


@dataclass
class mbp_datasets_statistical_tests:
    dataframe: pd.DataFrame
    model_name: str
    dataset: Literal["COHFACE", "CHILL", "PURE"]
    save_dir: Optional[str] = None
    hr_threshold: float = 80.0
    save_plots: bool = False

    def __post_init__(self):

        if self.dataset == "COHFACE":

            self.dark_scenarios = [2, 3]
            self.bright_scenarios = [0, 1]
            self.has_hr_groups = False

        elif self.dataset == "CHILL":
            print("using chill dataset")
            print("defining scenarios")

            self.high_hr_scenarios = [2, 3]
            self.low_hr_scenarios = [0, 1]
            self.dark_scenarios = [1, 2]
            self.bright_scenarios = [0, 3]
            self.has_hr_groups = True

        elif self.dataset == "PURE":
            raise NotImplementedError

        if self.save_dir is not None:
            pathlib.Path(self.save_dir).mkdir(exist_ok=True, parents=True)

    def _split_based_on_hr(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if self.dataset == "CHILL":

            low_hr_setting_mask = self.dataframe["scenario"].isin(self.low_hr_scenarios)
            high_hr_setting_mask = self.dataframe["scenario"].isin(
                self.high_hr_scenarios
            )

            df_lowhr = self.dataframe[low_hr_setting_mask]
            df_highhr = self.dataframe[high_hr_setting_mask]

            df_lowhr = df_lowhr[df_lowhr["GT_HR"] < self.hr_threshold]
            df_highhr = df_highhr[df_highhr["GT_HR"] >= self.hr_threshold]

        elif self.dataset == "COHFACE" or self.dataset == "PURE":

            df_lowhr = self.dataframe[self.dataframe["GT_HR"] < self.hr_threshold]
            df_highhr = self.dataframe[self.dataframe["GT_HR"] >= self.hr_threshold]

        else:
            raise NotImplementedError

        return df_lowhr, df_highhr

    def _split_based_on_illumination(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        df_bright_mask = self.dataframe["scenario"].isin(self.bright_scenarios)
        df_dark_mask = self.dataframe["scenario"].isin(self.dark_scenarios)

        df_bright = self.dataframe[df_bright_mask]
        df_dark = self.dataframe[df_dark_mask]

        return df_bright, df_dark

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

    def _fold_level_gt_per_participant(
        self, df1: pd.DataFrame, df2: pd.DataFrame, debug: Optional[bool] = False
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
            mae_1 = df1[df1["Participant"] == participant]["GT_HR"].abs().mean()
            mae_2 = df2[df2["Participant"] == participant]["GT_HR"].abs().mean()

            if debug:
                print("-" * 50)
                print(mae_1, mae_2)

            fold_mae_1.append(mae_1)
            fold_mae_2.append(mae_2)

        # Convert to pandas Series for easier manipulation
        fold_mae_1 = pd.Series(fold_mae_1)
        fold_mae_2 = pd.Series(fold_mae_2)

        return fold_mae_1, fold_mae_2

    def paired_less_wilcoxen_test_on_illumination_groups(
        self, debug: Optional[bool] = False, plot: Optional[bool] = False
    ) -> metrics:
        """This can be only used for the illumination setting as there are clear splits for each participant"""

        print("This is performed on illumination splits")

        df_s1, df_s2 = self._split_based_on_illumination()

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_scatterplot(df_s1.copy(), df_s2.copy())

        fold_mae_1, fold_mae_2 = self._fold_level_mae_per_participant(
            df_s1.copy(), df_s2.copy(), debug
        )

        if plot:
            plotter.plot_box(fold_mae_1, fold_mae_2)
            if self.save_dir:
                plotter.rain_plot(
                    fold_mae_1,
                    fold_mae_2,
                    model_name=self.model_name,
                    save_dir=self.save_dir,
                )
            else:
                plotter.rain_plot(fold_mae_1, fold_mae_2, model_name=self.model_name)

        metrics = statistical_test.wilcoxon_test(
            fold_mae_1,
            fold_mae_2,
            self.model_name,
            hypothesis="less",
            debug=debug,
        )

        bright_median = float(np.median(fold_mae_1))
        dark_median = float(np.median(fold_mae_2))

        median_differences = np.median(fold_mae_1 - fold_mae_2)

        # delta_median = bright_median - dark_median

        illu = Illumination_metrics(
            model=metrics.model,
            effect_size_r=metrics.effect_size_r,
            p_value=metrics.p_value,
            t_statistic=metrics.t_statistic,
            test_Used=metrics.test_Used,
            bright_median=bright_median,
            dark_median=dark_median,
            delta_median=median_differences,
        )

        return illu

    def paired_wilcoxem_for_gt_hr_on_illumination_groups(
        self, debug: Optional[bool] = False, plot: Optional[bool] = False
    ):
        print("This is performed on illumination splits")

        df_s1, df_s2 = self._split_based_on_illumination()

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_scatterplot(df_s1.copy(), df_s2.copy())

        fold_mae_1, fold_mae_2 = self._fold_level_gt_per_participant(
            df_s1, df_s2, debug
        )

        if plot:
            plotter.plot_box(fold_mae_1, fold_mae_2)
            if self.save_dir:
                plotter.rain_plot(
                    fold_mae_1,
                    fold_mae_2,
                    model_name=self.model_name,
                    save_dir=self.save_dir,
                )
            else:
                plotter.rain_plot(fold_mae_1, fold_mae_2, model_name=self.model_name)

        metrics_wilk = statistical_test.wilcoxon_test(
            fold_mae_1,
            fold_mae_2,
            self.model_name,
            hypothesis="two-sided",
            debug=debug,
        )

        bright_median = float(np.median(fold_mae_1))
        dark_median = float(np.median(fold_mae_2))

        # delta_median = bright_median - dark_median
        median_differences = np.median(fold_mae_1 - fold_mae_2)

        illu = Illumination_metrics(
            model=metrics_wilk.model,
            effect_size_r=metrics_wilk.effect_size_r,
            p_value=metrics_wilk.p_value,
            t_statistic=metrics_wilk.t_statistic,
            test_Used=metrics_wilk.test_Used,
            bright_median=bright_median,
            dark_median=dark_median,
            delta_median=median_differences,
        )

        return illu

    def paired_wilcoxen_test_on_illumination_groups(
        self, debug: Optional[bool] = False, plot: Optional[bool] = False
    ) -> Tuple[metrics, pd.Series, pd.Series]:
        """This can be only used for the illumination setting as there are clear splits for each participant"""

        print("This is performed on illumination splits")

        df_s1, df_s2 = self._split_based_on_illumination()

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_scatterplot(df_s1.copy(), df_s2.copy())

        fold_mae_1, fold_mae_2 = self._fold_level_mae_per_participant(
            df_s1, df_s2, debug
        )

        if plot:
            plotter.plot_box(fold_mae_1, fold_mae_2)
            if self.save_dir:
                plotter.rain_plot(
                    fold_mae_1,
                    fold_mae_2,
                    model_name=self.model_name,
                    save_dir=self.save_dir,
                )
            else:
                plotter.rain_plot(fold_mae_1, fold_mae_2, model_name=self.model_name)

        metrics_wilk = statistical_test.wilcoxon_test(
            fold_mae_1,
            fold_mae_2,
            self.model_name,
            hypothesis="two-sided",
            debug=debug,
        )

        bright_median = float(np.median(fold_mae_1))
        dark_median = float(np.median(fold_mae_2))

        # delta_median = bright_median - dark_median
        median_differences = np.median(fold_mae_1 - fold_mae_2)

        illu = Illumination_metrics(
            model=metrics_wilk.model,
            effect_size_r=metrics_wilk.effect_size_r,
            p_value=metrics_wilk.p_value,
            t_statistic=metrics_wilk.t_statistic,
            test_Used=metrics_wilk.test_Used,
            bright_median=bright_median,
            dark_median=dark_median,
            delta_median=median_differences,
        )

        return illu, fold_mae_1, fold_mae_2

    def paired_less_wilcoxen_test_on_hr_groups(
        self, debug: Optional[bool] = False, plot: Optional[bool] = False
    ) -> Tuple[metrics, pd.Series, pd.Series]:

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
        grouped_df = df_s1_grouped.merge(
            df_s2_grouped, on="Participant", how="inner"
        ).reset_index()

        df_s1 = grouped_df[["Error_1", "GT_HR_1"]].rename(
            columns={"Error_1": "Error", "GT_HR_1": "GT_HR"}
        )
        df_s2 = grouped_df[["Error_2", "GT_HR_2"]].rename(
            columns={"Error_2": "Error", "GT_HR_2": "GT_HR"}
        )

        if debug:
            print(df_s1)
            print(df_s2)

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_scatterplot(df_s1.copy(), df_s2.copy())
            plotter.plot_box(df_s1["Error"], df_s2["Error"])
            # self.plot_box(fold_mae_1, fold_mae_2)
            if self.save_dir:
                plotter.rain_plot(
                    df_s1["Error"],
                    df_s2["Error"],
                    model_name=self.model_name,
                    save_dir=self.save_dir,
                )
            else:
                plotter.rain_plot(
                    df_s1["Error"], df_s2["Error"], model_name=self.model_name
                )

        metrics_wilk = statistical_test.wilcoxon_test(
            df_s1["Error"].copy(),
            df_s2["Error"].copy(),
            self.model_name,
            hypothesis="less",
            debug=debug,
        )

        low_hr = float(np.median(df_s1["Error"].copy()))
        high_hr = float(np.median(df_s2["Error"].copy()))

        median_differences = np.median(df_s1["Error"] - df_s2["Error"])

        # delta_median = low_hr - high_hr

        hr = HR_metrics(
            model=metrics_wilk.model,
            effect_size_r=metrics_wilk.effect_size_r,
            p_value=metrics_wilk.p_value,
            t_statistic=metrics_wilk.t_statistic,
            test_Used=metrics_wilk.test_Used,
            delta_median=median_differences,
            highhr_median=high_hr,
            lowhr_median=low_hr,
        )

        return hr, df_s1["Error"].copy(), df_s2["Error"].copy()

    def paired_wilcoxen_test_on_hr_groups(
        self, debug: Optional[bool] = False, plot: Optional[bool] = False
    ) -> metrics:

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
        grouped_df = df_s1_grouped.merge(
            df_s2_grouped, on="Participant", how="inner"
        ).reset_index()

        df_s1 = grouped_df[["Error_1", "GT_HR_1"]].rename(
            columns={"Error_1": "Error", "GT_HR_1": "GT_HR"}
        )
        df_s2 = grouped_df[["Error_2", "GT_HR_2"]].rename(
            columns={"Error_2": "Error", "GT_HR_2": "GT_HR"}
        )

        if debug:
            print(df_s1)
            print(df_s2)

        if plot:
            # self.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            # self.plot_error_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_scatterplot(df_s1.copy(), df_s2.copy())
            plotter.plot_box(df_s1["Error"], df_s2["Error"])
            # self.plot_box(fold_mae_1, fold_mae_2)
            if self.save_dir:
                plotter.rain_plot(
                    df_s1["Error"],
                    df_s2["Error"],
                    model_name=self.model_name,
                    save_dir=self.save_dir,
                )
            else:
                plotter.rain_plot(
                    df_s1["Error"], df_s2["Error"], model_name=self.model_name
                )

        metrics_wilk = statistical_test.wilcoxon_test(
            df_s1["Error"],
            df_s2["Error"],
            self.model_name,
            hypothesis="two-sided",
            debug=debug,
        )

        low_hr = float(np.median(df_s1["Error"].copy()))
        high_hr = float(np.median(df_s2["Error"].copy()))

        median_differences = np.median(df_s1["Error"] - df_s2["Error"])
        # delta_median = low_hr - high_hr

        hr = HR_metrics(
            model=metrics_wilk.model,
            effect_size_r=metrics_wilk.effect_size_r,
            p_value=metrics_wilk.p_value,
            t_statistic=metrics_wilk.t_statistic,
            test_Used=metrics_wilk.test_Used,
            delta_median=median_differences,
            highhr_median=high_hr,
            lowhr_median=low_hr,
        )

        return hr

    def perform_Mann_Whitney_test(self, debug=False) -> metrics:

        print("This is performed on HR ")

        df_s1, df_s2 = self._split_based_on_hr()

        if debug:
            plotter.plot_gt_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_distribution(df_s1.copy(), df_s2.copy())
            plotter.plot_error_scatterplot(df_s1.copy(), df_s2.copy())

        fold_mae_1, fold_mae_2 = df_s1["Error"], df_s2["Error"]

        metrics = self.Mann_Whitney_test(
            fold_mae_1, fold_mae_2, self.model_name, debug=debug
        )

        return metrics

    @classmethod
    def batch_process_models(
        cls,
        model_dfs: dict,
        dataset: Literal["CHILL", "COHFACE", "PURE"],
        save_dir: Optional[str] = None,
        plot: Optional[bool] = False,
    ) -> dict:
        """Process multiple models and create individual and composite analyses."""
        all_results = {}

        for i, (model_name, df) in enumerate(model_dfs.items()):
            # Individual model analysis
            #
            analyzer = cls(df, model_name, dataset, save_dir=save_dir)

            # Get results
            # The two-sided wilcocen rank test is performed on illumination splits
            illu_results, bright_df, dark_df = (
                analyzer.paired_wilcoxen_test_on_illumination_groups(plot=plot)
            )

            # One sided wilcoxen ranked test is performed on the HR splits
            if analyzer.has_hr_groups:
                hr_results, low_hr, high_hr = (
                    analyzer.paired_less_wilcoxen_test_on_hr_groups(plot=False)
                )
            else:
                hr_results, low_hr, high_hr = None, None, None

            if dataset == "CHILL":
                # Store results
                all_results[model_name] = {
                    "illumination": {
                        "metrics": illu_results,
                        "Bright": bright_df,
                        "Dark": dark_df,
                    },
                    "HR": {
                        "metrics": hr_results,
                        "Low-HR": low_hr,
                        "High-HR": high_hr,
                    },
                }
            elif dataset == "COHFACE":
                # Store results
                all_results[model_name] = {
                    "illumination": {
                        "metrics": illu_results,
                        "Studio": bright_df,
                        "Natural": dark_df,
                    },
                    "HR": {
                        "metrics": hr_results,
                        "Low-HR": low_hr,
                        "High-HR": high_hr,
                    },
                }

        return all_results


# Function to calculate MAE
def calculate_mae(gt_list, pred_list):
    return sum(abs(gt - pred) for gt, pred in zip(gt_list, pred_list)) / len(gt_list)


def plot_mae_per_fold(mae_dict):
    # convert to numpy array
    sorted_mae_values = [mae_dict[key] for key in sorted(mae_dict.keys())]

    mae_array = np.array(sorted_mae_values)

    # calculate mean and standard error

    mean_mae = np.mean(mae_array)
    median_mae = np.median(mae_array)

    # plot the mae values
    plt.figure(figsize=(8, 5))

    plt.bar(
        sorted(mae_dict.keys()),
        mae_array,
        color="skyblue",
        edgecolor="black",
        label="fold mae",
    )

    plt.axhline(
        mean_mae, color="red", linestyle="--", label=f"mean mae = {mean_mae:.2f}"
    )
    plt.axhline(
        median_mae, color="blue", linestyle="--", label=f"median mae = {median_mae:.2f}"
    )

    plt.title("mae per fold with mean line")

    plt.xlabel("fold")

    plt.ylabel("mae")

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

    # Plot the MAE values

    plt.figure(figsize=(8, 5))

    plt.bar(
        sorted(mae_dict.keys()),
        mae_array,
        color="skyblue",
        edgecolor="black",
        label="Scenario MAE",
    )

    plt.axhline(
        mean_mae, color="red", linestyle="--", label=f"Mean MAE = {mean_mae:.2f}"
    )
    plt.axhline(
        median_mae, color="blue", linestyle="--", label=f"Median MAE = {median_mae:.2f}"
    )

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
            scenario_key = scenario.split("_")[-1][
                -1
            ]  # Extract the last part (e.g., '901')

            mae_per_scenario[scenario_key].append(
                abs(values["GT_HR"] - values["Pred_HR"])
            )
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

    print(
        "\nOverall MAE: ", sum(list(mae_per_fold.values())) / len(mae_per_fold.values())
    )
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert nested dict to tidy DataFrame for plotting
def extract_data(data, scenario, series_labels):
    rows = []
    for model, model_data in data.items():
        scenario_data = model_data[scenario]
        for label in series_labels:
            values = scenario_data[label]
            for v in values:
                rows.append({
                    'Model': model,
                    'Condition': label,
                    'Error': v,
                    'p_value': scenario_data['metrics'].p_value
                })
    return pd.DataFrame(rows)

# -- Plotting function with simulated broken axis using subplots --
def plot_with_significance(df, title, y_offset=5, save_path=None, lims=None):

    label_size = 16
    # Set default limits if none provided
    if lims is None:
        lims = ((0, 15), (65, 75))
    
    # Light color palette for points
    point_palette = {
        'Bright': '#FFCCCC',  # Light red
        'Dark': '#CCE5CC',    # Light green
        'Low-HR': '#CCE5CC',   # Light green
        'High-HR': '#FFB3B3'   # Light red
    }
    
    # Set seaborn context
    sns.set_context("paper", rc={"font.size": 24, "axes.labelsize": 5})   
    
    # Create figure with two subplots (top and bottom)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), 
                                           gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05}, dpi=600)
    
    # Plot on top subplot (upper range)
    sns.boxplot(
        x='Model', 
        y='Error', 
        hue='Condition', 
        data=df, 
        palette='light:#d3d3d3', 
        width=0.6, 
        boxprops={'facecolor': 'none'}, 
        showcaps=True, 
        medianprops={'color': 'black'}, 
        fliersize=0.0,
        ax=ax_top
    )
    
    sns.stripplot(
        x='Model', 
        y='Error', 
        hue='Condition', 
        data=df, 
        dodge=True,
        jitter=0.2,
        palette=point_palette,
        alpha=0.6,
        size=4,
        linewidth=0.8,
        edgecolor='black',
        ax=ax_top
    )
    
    # Plot on bottom subplot (lower range)
    sns.boxplot(
        x='Model', 
        y='Error', 
        hue='Condition', 
        data=df, 
        palette='light:#d3d3d3', 
        width=0.6, 
        boxprops={'facecolor': 'none'}, 
        showcaps=True, 
        medianprops={'color': 'black'}, 
        fliersize=0.0,
        ax=ax_bottom
    )
    
    sns.stripplot(
        x='Model', 
        y='Error', 
        hue='Condition', 
        data=df, 
        dodge=True,
        jitter=0.2,
        palette=point_palette,
        alpha=0.6,
        size=4,
        linewidth=0.8,
        edgecolor='black',
        ax=ax_bottom
    )
    
    # Set y-axis limits for each subplot
    ax_top.set_ylim(lims[1])    # Upper range
    ax_bottom.set_ylim(lims[0]) # Lower range
    
    # Hide x-axis labels and ticks for top plot
    ax_top.set_xlabel('')
    ax_top.set_xticklabels([])
        
    ax_top.set_ylabel('')
    # ax_top.set_ylim(lims[1])
    ax_top.yaxis.set_major_locator(plt.MultipleLocator(5)) 
    ax_top.set_ylim(top=lims[1][1] + 2)
    # ax_top.set_ylim(top=lims[1][0] - 1)


    ax_top.tick_params(axis='x', which='both', bottom=False, top=False)
    
    # Set tick parameters
    ax_top.tick_params(axis='y', labelsize=12)
    ax_bottom.tick_params(axis='x', labelsize=12)
    ax_bottom.tick_params(axis='y', labelsize=12)
    ax_bottom.yaxis.set_major_locator(plt.MultipleLocator(5)) 
    ax_bottom.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=label_size)

    ax_bottom.set_ylabel('')
    ax_bottom.set_ylim(bottom=lims[0][0] - 1) 
    # ax_bottom.set_ylim(bottom=lims[0][1] + 1) 

    # Handle legends - remove from top, keep only on bottom
    
    handles, labels = ax_top.get_legend_handles_labels()
    n_conditions = len(df['Condition'].unique())
    ax_top.legend(handles[-n_conditions:], labels[-n_conditions:], 
                    title='Condition')
    ax_bottom.legend().remove()
    
    # Add significance annotations - always visible
    models = df['Model'].unique()
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        y_max = model_df['Error'].max()
        p_val = model_df['p_value'].values[0]
        
        # Significance stars
        if p_val < 0.005:
            sig_label = "**"
        elif p_val < 0.05:
            sig_label = "*"
        else:
            sig_label = "n.s."
        
        if sig_label != "n.s.":
            x1 = i - 0.2
            x2 = i + 0.2
            y = y_max + y_offset
            h = y_offset * 0.4
            
            # Determine which subplot(s) to use based on y position and ranges
            bottom_range = lims[0]
            top_range = lims[1]
            
            # Check if significance line fits in bottom subplot
            if y <= bottom_range[1]:
                # Line fits in bottom subplot
                ax = ax_bottom
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
                ax.text((x1 + x2) / 2, y + h + 0.5, sig_label, ha='center', va='bottom', 
                       color='black', fontsize=label_size)
            
            # Check if significance line fits in top subplot  
            elif y >= top_range[0]:
                # Line fits in top subplot
                ax = ax_top
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
                ax.text((x1 + x2) / 2, y + h + 0.5, sig_label, ha='center', va='bottom', 
                       color='black', fontsize=label_size)
            
            else:
                # Significance line would be in the gap - place it at top of bottom subplot
                # or bottom of top subplot, whichever is closer to the data
                bottom_distance = abs(y - bottom_range[1])
                top_distance = abs(y - top_range[0])
                
                if bottom_distance <= top_distance:
                    # Place at top of bottom subplot
                    ax = ax_bottom
                    y_pos = bottom_range[1] - (y_offset + h + 1)  # Position near top of bottom subplot
                    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, c='black')
                    ax.text((x1 + x2) / 2, y_pos + h + 0.5, sig_label, ha='center', va='bottom', 
                           color='black', fontsize=label_size)
                else:
                    # Place at bottom of top subplot
                    ax = ax_top
                    y_pos = top_range[0] + y_offset  # Position near bottom of top subplot
                    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, c='black')
                    ax.text((x1 + x2) / 2, y_pos + h + 0.5, sig_label, ha='center', va='bottom', 
                           color='black', fontsize=label_size)
    
    # Customize spines to create broken axis effect
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    
    # # # Add diagonal lines to indicate break
    # d = 0.015  # Size of diagonal lines
    # kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    # ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    # d = 0.015  # Size of diagonal lines

    # kwargs.update(transform=ax_bottom.transAxes)  # switch to bottom axes
    # ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)   # bottom-left diagonal
    # ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    d = 0.010  # Diagonal length in figure coordinates

    # Convert axes coordinates to figure coordinates
    fig_coords = fig.transFigure
    
    kwargs_top = dict(transform=fig_coords, color='k', clip_on=False)
    kwargs_bottom = dict(transform=fig_coords, color='k', clip_on=False)
    
    # Get bounding boxes for axes in figure coordinates
    bbox_top = ax_top.get_position()
    bbox_bottom = ax_bottom.get_position()
    
    # Top diagonals
    ax_top.plot(
        [bbox_top.x0 - d, bbox_top.x0 + d],
        [bbox_top.y0 - d, bbox_top.y0 + d],
        **kwargs_top
    )
    ax_top.plot(
        [bbox_top.x1- d, bbox_top.x1 + d],
        [bbox_top.y0 -d , bbox_top.y0 +d ],
        **kwargs_top
    )
    
    # Bottom diagonals
    ax_bottom.plot(
        [bbox_bottom.x0-d, bbox_bottom.x0 + d],
        [bbox_bottom.y1 -d , bbox_bottom.y1 + d],
        **kwargs_bottom
    )
    ax_bottom.plot(
        [bbox_bottom.x1 - d, bbox_bottom.x1 +d ],
        [bbox_bottom.y1 -d , bbox_bottom.y1 +d ],
        **kwargs_bottom
    )
    
    # Set labels
    ax_bottom.set_xlabel('Model', fontsize=label_size + 2)
    fig.text(0.04, 0.5, 'Error', va='center', rotation='vertical', fontsize=label_size+2)
    
    # Set title
    # if title:
    #     fig.suptitle(title, fontsize=18, y=0.98)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()