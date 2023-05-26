import numpy as np
import matplotlib.pyplot as plt
import os


def plot_comm_elec(elec_com_fifteen, band_all_patient, channel_names_list, final_time, fs, path, freq_band='gamma'):
    os.makedirs(path + 'plot_comm_elec')
    p2 = os.path.join(path, 'plot_comm_elec' + '/')
    plot_elec_avg = {}
    for i in range(len(elec_com_fifteen)):
        plot_elec_avg[elec_com_fifteen[i]] = 0

    time = np.arange(0, band_all_patient[0][freq_band].shape[0]) / fs
    time_dash = np.arange(0, final_time + 30, 30)
    plt.figure(figsize=(30, 10))
    for key in plot_elec_avg.keys():
        n = 0
        for patient_idx in range(len(band_all_patient)):
            electrodes = channel_names_list[patient_idx]
            if key in electrodes:
                n = n + 1
                num_electrode = channel_names_list[patient_idx].index(key)
                plot_elec_avg[key] = plot_elec_avg[key] + band_all_patient[patient_idx][freq_band][:, num_electrode]

        plot_elec_avg[key] = plot_elec_avg[key] / n
        plt.figure(figsize=(30, 10))
        plt.plot(time[:final_time * fs], plot_elec_avg[key][:final_time * fs])
        plt.vlines(time_dash, ymin=-1 * np.ones(time_dash.shape), ymax=np.ones(time_dash.shape), colors='red', ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.title(str(key), fontsize=25)
        plt.savefig(p2 + key)

    return plot_elec_avg
