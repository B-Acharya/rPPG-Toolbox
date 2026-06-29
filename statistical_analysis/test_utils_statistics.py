import unittest
import numpy as np
import pandas as pd
import math
from unittest.mock import patch, MagicMock
from utils import (
    statistical_test,
    mbp_datasets_statistical_tests,
    calculate_mae,
    plot_mae_per_fold,
    plot_mae_per_scenario,
)


class TestStatisticalTest(unittest.TestCase):
    """Test cases for the statistical_test class."""

    def setUp(self):
        """Set up test data for statistical tests."""
        # Create sample data for testing
        np.random.seed(42)
        self.samples1 = pd.Series(np.random.normal(5, 1, 20))
        self.samples2 = pd.Series(np.random.normal(6, 1, 20))
        self.model_name = "test_model"

    def test_wilcoxon_test_two_sided(self):
        """Test Wilcoxon test with two-sided hypothesis."""
        result = statistical_test.wilcoxon_test(
            self.samples1,
            self.samples2,
            self.model_name,
            hypothesis="two-sided",
            debug=False,
        )

        # Check if the test returns the expected object with correct properties
        self.assertEqual(result.model, self.model_name)
        self.assertEqual(result.test_Used, "Wilcoxon Signed-Rank Test")
        self.assertIsInstance(result.t_statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size_r, float)
        self.assertLess(result.p_value, 0.05)  # Check for significance

    def test_wilcoxon_test_less(self):
        """Test Wilcoxon test with less hypothesis."""
        result = statistical_test.wilcoxon_test(
            self.samples1,
            self.samples2,
            self.model_name,
            hypothesis="less",
            debug=False,
        )

        # Check basic properties
        self.assertEqual(result.model, self.model_name)
        self.assertEqual(result.test_Used, "Wilcoxon Signed-Rank Test")

        # Verify effect_size_r calculation is reasonable
        self.assertLessEqual(abs(result.effect_size_r), 1.0)

        self.assertEqual(result.model, self.model_name)
        self.assertEqual(result.test_Used, "Wilcoxon Signed-Rank Test")
        self.assertIsInstance(result.t_statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size_r, float)
        self.assertLess(result.p_value, 0.05)  # Check for significance

    def test_wilcoxon_test_greater(self):
        """Test Wilcoxon test with less hypothesis."""
        result = statistical_test.wilcoxon_test(
            self.samples2,
            self.samples1,
            self.model_name,
            hypothesis="greater",
            debug=False,
        )

        # Check basic properties
        self.assertEqual(result.model, self.model_name)
        self.assertEqual(result.test_Used, "Wilcoxon Signed-Rank Test")

        # Verify effect_size_r calculation is reasonable
        self.assertLessEqual(abs(result.effect_size_r), 1.0)

        self.assertEqual(result.model, self.model_name)
        self.assertEqual(result.test_Used, "Wilcoxon Signed-Rank Test")
        self.assertIsInstance(result.t_statistic, float)
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.effect_size_r, float)
        self.assertLess(result.p_value, 0.05)  # Check for significance

    def test_effect_size_small_n(self):
        """Test effect_size with smallest valid n."""
        w_stat = 0.5
        n = 1
        result = statistical_test.effect_size(w_stat, n)
        self.assertTrue(np.isfinite(result))

    def test_effect_size_know_value(self):
        w_stat = 10
        n = 5
        expected_mu = n * (n + 1) / 4
        expected_sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        expected_z = (w_stat - expected_mu) / expected_sigma
        expected_r = expected_z / math.sqrt(n)

        result = statistical_test.effect_size(w_stat, n)
        self.assertAlmostEqual(result, expected_r, places=4)


class TestMbpDatasetsStatisticalTests(unittest.TestCase):
    """Test cases for the mbp_datasets_statistical_tests class."""

    def setUp(self):
        """Set up test data for dataset tests."""
        # Create a mock dataframe for testing
        np.random.seed(42)
        participants = [f"P{i}" for i in range(10)]
        scenarios = [0, 1, 2, 3]

        data = []
        for p in participants:
            for s in scenarios:
                # Create different HR values and errors for different scenarios
                if s in [0, 1]:  # Low HR scenarios
                    hr = np.random.normal(70, 30)
                else:  # High HR scenarios
                    hr = np.random.normal(90, 30)

                if s in [0, 3]:  # Bright scenarios
                    error = np.random.normal(2, 1)
                else:  # Dark scenarios
                    error = np.random.normal(4, 1)

                pred_hr = hr + error

                data.append(
                    {
                        "Participant": p,
                        "scenario": s,
                        "GT_HR": hr,
                        "Pred_HR": pred_hr,
                        "Error": error,
                    }
                )

        self.df = pd.DataFrame(data)
        self.model_name = "test_model"

    def test_init_cohface(self):
        """Test initialization with COHFACE dataset."""
        test_obj = mbp_datasets_statistical_tests(
            dataframe=self.df, model_name=self.model_name, dataset="COHFACE"
        )

        self.assertEqual(test_obj.dark_scenarios, [2, 3])
        self.assertEqual(test_obj.bright_scenarios, [0, 1])
        self.assertFalse(test_obj.has_hr_groups)

    def test_init_chill(self):
        """Test initialization with CHILL dataset."""
        test_obj = mbp_datasets_statistical_tests(
            dataframe=self.df, model_name=self.model_name, dataset="CHILL"
        )

        self.assertEqual(test_obj.high_hr_scenarios, [2, 3])
        self.assertEqual(test_obj.low_hr_scenarios, [0, 1])
        self.assertEqual(test_obj.dark_scenarios, [1, 2])
        self.assertEqual(test_obj.bright_scenarios, [0, 3])
        self.assertTrue(test_obj.has_hr_groups)

    def test_init_pure(self):
        """Test initialization with PURE dataset."""
        with self.assertRaises(NotImplementedError):
            mbp_datasets_statistical_tests(
                dataframe=self.df, model_name=self.model_name, dataset="PURE"
            )

    def test_split_based_on_hr(self):
        """Test splitting based on HR."""
        test_obj = mbp_datasets_statistical_tests(
            dataframe=self.df, model_name=self.model_name, dataset="CHILL"
        )

        low_hr, high_hr = test_obj._split_based_on_hr()

        # Check that the split worked correctly
        self.assertTrue(all(hr < test_obj.hr_threshold for hr in low_hr["GT_HR"]))
        self.assertTrue(all(hr >= test_obj.hr_threshold for hr in high_hr["GT_HR"]))

    def test_split_based_on_illumination(self):
        """Test splitting based on illumination."""
        test_obj = mbp_datasets_statistical_tests(
            dataframe=self.df, model_name=self.model_name, dataset="CHILL"
        )

        bright, dark = test_obj._split_based_on_illumination()

        # Check that the split worked correctly
        self.assertTrue(all(s in test_obj.bright_scenarios for s in bright["scenario"]))
        self.assertTrue(all(s in test_obj.dark_scenarios for s in dark["scenario"]))

    def test_fold_level_mae_per_participant(self):
        """Test fold-level MAE calculation per participant."""
        test_obj = mbp_datasets_statistical_tests(
            dataframe=self.df, model_name=self.model_name, dataset="CHILL"
        )

        # Split into two groups
        df1 = self.df[self.df["scenario"].isin([0, 1])]
        df2 = self.df[self.df["scenario"].isin([2, 3])]

        mae1, mae2 = test_obj._fold_level_mae_per_participant(df1, df2)

        # Check that MAEs are calculated and have expected properties
        self.assertEqual(len(mae1), len(set(self.df["Participant"])))
        self.assertEqual(len(mae2), len(set(self.df["Participant"])))
        self.assertTrue(all(isinstance(v, float) for v in mae1))
        self.assertTrue(all(v >= 0 for v in mae1))  # MAE should be non-negative

    @patch("utils.plotter")
    @patch("utils.statistical_test.wilcoxon_test")
    def test_paired_wilcoxen_test_on_illumination_groups(
        self, mock_wilcoxon, mock_plotter
    ):
        """Test paired Wilcoxon test on illumination groups."""
        # Mock the return value of wilcoxon_test
        mock_wilcoxon.return_value = MagicMock(
            model=self.model_name,
            effect_size_r=0.5,
            p_value=0.02,
            t_statistic=30.0,
            test_Used="Wilcoxon Signed-Rank Test",
        )

        test_obj = mbp_datasets_statistical_tests(
            dataframe=self.df, model_name=self.model_name, dataset="CHILL"
        )

        result, _, _ = test_obj.paired_wilcoxen_test_on_illumination_groups()

        # Verify the result
        self.assertEqual(result.model, self.model_name)
        self.assertEqual(result.test_Used, "Wilcoxon Signed-Rank Test")
        self.assertEqual(result.effect_size_r, 0.5)
        self.assertEqual(result.p_value, 0.02)

        # Check that wilcoxon_test was called once
        mock_wilcoxon.assert_called_once()


class TestCalculationFunctions(unittest.TestCase):
    """Test cases for calculation functions."""

    def test_calculate_mae(self):
        """Test the calculate_mae function."""
        gt_list = [1, 2, 3, 4, 5]
        pred_list = [1.1, 2.2, 2.8, 4.1, 5.4]

        mae = calculate_mae(gt_list, pred_list)

        # Expected MAE: (0.1 + 0.2 + 0.2 + 0.1 + 0.4) / 5 = 0.2
        self.assertAlmostEqual(mae, 0.2, places=5)

    @patch("utils.plt")
    def test_plot_mae_per_fold(self, mock_plt):
        """Test the plot_mae_per_fold function."""
        mae_dict = {"fold1": 0.5, "fold2": 0.7, "fold3": 0.6}

        plot_mae_per_fold(mae_dict)

        # Check that plt.figure was called (indicating plot was created)
        mock_plt.figure.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("utils.plt")
    def test_plot_mae_per_scenario(self, mock_plt):
        """Test the plot_mae_per_scenario function."""
        mae_dict = {"scenario1": 0.5, "scenario2": 0.7, "scenario3": 0.6}

        plot_mae_per_scenario(mae_dict)

        # Check that plt.figure was called (indicating plot was created)
        mock_plt.figure.assert_called_once()
        mock_plt.show.assert_called_once()


if __name__ == "__main__":
    unittest.main()