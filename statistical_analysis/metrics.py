from dataclasses import dataclass


@dataclass
class metrics:
    model: str
    test_Used: str
    t_statistic: float
    p_value: float
    effect_size_r: float


@dataclass
class Illumination_metrics(metrics):
    dark_median: float
    bright_median: float
    delta_median: float


@dataclass
class HR_metrics(metrics):
    highhr_median: float
    lowhr_median: float
    delta_median: float
