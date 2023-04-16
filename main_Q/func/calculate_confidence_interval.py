import numpy as np
from scipy.stats import t


def cal_CI(data,conf_level):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sem = sample_std / np.sqrt(len(data))
    df = len(data) - 1
    alpha = 1 - conf_level
    t_crit = t.ppf(1 - alpha / 2, df)

    lower_bound = sample_mean - t_crit * sem
    upper_bound = sample_mean + t_crit * sem

    return lower_bound,upper_bound
