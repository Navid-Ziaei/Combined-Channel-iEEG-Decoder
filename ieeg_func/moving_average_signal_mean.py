from ieeg_func.mov_avg import moving_avg
import numpy as np
import matplotlib.pyplot as plt


def moving_average_signalmean(plot_elec_avg, elec_com_fifteen,final_time_show, fs, window_size, plot_output):
    avg_win = {}
    for i in range(len(elec_com_fifteen)):
        avg_win[elec_com_fifteen[i]] = []

    if plot_output:
        time_dash = np.arange(0, final_time_show + 30, 30)
        fig, ax = plt.subplots(len(avg_win) + 1, 1, figsize=(30, 20))
        j = 0

    for key in plot_elec_avg.keys():
        signal = plot_elec_avg[key]
        avg_win[key]=moving_avg(signal,window_size)
        if plot_output:
            j=j+1
            time = np.arange(0, len(avg_win[key])) / (fs)
            ax[j].plot(time[:final_time_show * fs], avg_win[key][:final_time_show * fs])
            ax[j].title.set_text(str(key))
            ax[j].vlines(time_dash, ymin=-1 * np.ones(time_dash.shape), ymax=np.ones(time_dash.shape), colors='red', ls='--',
                   lw=2, label='vline_multiple - partial height')

    return avg_win


