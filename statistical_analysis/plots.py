import scipy.stats as stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from dataclasses import dataclass
from metrics import metrics
import ptitprince as pt


@dataclass
class PlotConfig:
    """Configuration for plots with common parameters."""

    figsize: Tuple[int, int] = (10, 6)
    save_dir: Optional[str] = None
    title_prefix: str = ""
    show: bool = True


class plotter:

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

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
        sns.histplot(
            data=df_cat, x="GT_HR", hue="source", bins=40, kde=True, multiple="stack"
        )

        plt.show()

    @staticmethod
    def rain_plot(
        df1: pd.Series, df2: pd.Series, save_dir: Optional[str] = None, model_name=None
    ) -> None:

        # Sample data structure (adapt to your actual DataFrames)
        data = {
            "Groups": ["Group1"] * len(df1) + ["Group2"] * len(df2),
            "HR": list(df1) + list(df2),  # Your HR values
        }

        # adding color
        f, ax = plt.subplots(figsize=(7, 5))

        pt.RainCloud(x="Groups", y="HR", data=data, ax=ax)
        # ax.axhline(80, color='red', linestyle='--', alpha=0.7, label='High HR Threshold')
        # ax.legend(loc='upper right', frameon=False)  # Moved inside
        # ax.axis('off')  # Hide axes
        if model_name is not None:
            plt.title("Error Distribution between groups: {}".format(model_name))
        else:
            plt.title("Error Distribution between groups")

        plt.ylabel("Error (MAE)")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f"{save_dir}/{model_name}.png")
        else:
            plt.show()

    @staticmethod
    def plot_error_distribution(df1: pd.DataFrame, df2: pd.DataFrame) -> None:

        df1["source"] = "source 1"
        df2["source"] = "source 2"

        df_cat = pd.concat([df1, df2], ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_cat, x="error", hue="source", bins=40, kde=True, multiple="stack"
        )
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

    @staticmethod
    def plot_spearman_correlations(
        series1: pd.Series, series2: pd.Series, name: str, debug: Optional[bool] = False
    ) -> metrics:

        return_metrics = metrics(
            model="spearman",
            test_Used="spearman",
            t_statistic=0.0,
            p_value=0.0,
            effect_size_r=0.0,
            median_samples1=0.0,
            median_samples2=0.0,
            delta_median=0.0,
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
        plt.title(
            f"Setting-{name}:{corr_method} Correlation: r = {corr_coef:.3f}, p = {p_value:.3f}"
        )
        plt.xlabel("Series 1")
        plt.ylabel("Series 2")

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