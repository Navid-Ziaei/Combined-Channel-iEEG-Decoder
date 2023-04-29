import numpy as np
import matplotlib.pyplot as plt
import os

def plot_temporal_signal(raw_car_all,band_all_patient,onset_question,offset_question,fs,path,electrode,patient,final_time):
    os.makedirs(path + 'plot_temporal_signal')
    p2 = os.path.join(path, 'plot_temporal_signal' + '/')
    num_electrode = raw_car_all[patient].ch_names.index(electrode)
    signal = band_all_patient[patient][:, num_electrode]
    time = np.arange(0, signal.shape[0]) / fs
    time_dash = onset_question + offset_question
    time_dash.sort()
    time_dash2 = []
    for i in time_dash:
        if i < final_time:
            time_dash2.append(i)

    time_dash_music = np.arange(0, final_time + 30, 30)

    plt.figure(figsize=(80, 10))
    plt.plot(time[:final_time * fs], signal[:final_time * fs])
    plt.vlines(time_dash2, ymin=-4 * np.ones(len(time_dash2)), ymax=4 * np.ones(len(time_dash2)), colors='red', ls='--',
               lw=2, label='vline_multiple - partial height')
    plt.vlines(time_dash_music, ymin=-4 * np.ones(len(time_dash_music)), ymax=8 * np.ones(len(time_dash_music)),
               colors='black', ls='--', lw=2, label='vline_multiple - partial height')

    #plt.show()
    plt.xlabel('time', fontsize=15)
    plt.ylabel('temporal_signal', fontsize=15)
    plt.title('temporal signal of patient='+str(patient)+'_electrode ='+electrode, fontsize=15)
    plt.savefig(p2+"temporal_signal")
