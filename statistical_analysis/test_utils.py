import pytest
import pandas as pd
from utils import (
    statistical_test,
)  # Assuming your utils.py file contains the statistical_test class

# --------------------------------------------------------------------------------------
#  Tests for the statistical_test class methods
# --------------------------------------------------------------------------------------


def test_wilcoxon_test_significant():
    """Tests wilcoxon_test when a significant difference is expected."""
    # sample data from https://datatab.net/statistics-calculator/hypothesis-test/wilcoxon-test-calculator?example=Wilcoxon_Test
    samples1 = pd.Series(
        [18, 22, 20, 25, 21, 24, 19, 23, 22, 20, 23, 24, 22, 21, 26, 24, 19, 20, 23, 21]
    )
    samples2 = pd.Series(
        [25, 27, 20, 29, 26, 24, 22, 28, 24, 23, 26, 28, 21, 27, 30, 26, 20, 22, 25, 23]
    )
    result = statistical_test.wilcoxon_test(
        samples1, samples2, "TestModel", "two-sided", debug=False
    )
    assert result.p_value < 0.05  # Check for significance
    assert result.test_Used == "Wilcoxon Signed-Rank Test"


def test_wilcoxon_test_no_significant():
    """Tests wilcoxon_test when no significant difference is expected."""
    samples1 = pd.Series([25, 28, 32, 31, 35])
    samples2 = pd.Series([24, 29, 31, 30, 33])
    result = statistical_test.wilcoxon_test(
        samples1, samples2, "TestModel", "two-sided", debug=False
    )
    assert result.p_value > 0.05  # Check for non-significance


def test_wilcoxon_test_less_hypothesis():
    """Tests wilcoxon_test with the 'less' alternative hypothesis."""
    samples1 = pd.Series([20, 22, 29, 24, 30])
    samples2 = pd.Series([25, 28, 32, 31, 35])
    result = statistical_test.wilcoxon_test(
        samples1, samples2, "TestModel", "less", debug=False
    )
    assert (
        result.p_value < 0.05
    )  # Expecting a significant result for "less" in this case


def test_wilcoxon_test_greater_hypothesis():
    """Tests wilcoxon_test with the 'greater' alternative hypothesis."""
    samples1 = pd.Series([25, 28, 32, 31, 35])
    samples2 = pd.Series([20, 22, 29, 24, 30])
    result = statistical_test.wilcoxon_test(
        samples1, samples2, "TestModel", "greater", debug=False
    )
    assert (
        result.p_value < 0.05
    )  # Expecting a significant result for "greater" in this case
