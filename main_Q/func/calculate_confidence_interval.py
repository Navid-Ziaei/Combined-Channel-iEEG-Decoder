from scipy import stats


def cal_CI(data,conf_level):
    sem = stats.sem(data,axis=0)
    t_crit = sem*stats.t.ppf((1 + conf_level) / 2,  len(data) - 1)

    return t_crit
