import numpy as np
import matplotlib.pyplot as plt

def plot_comm_elec(elec_com_fifteen,band_all_patient,raw_car_all,elec,final_time,fs):
    plot_elec_avg = {}
    for i in range(len(elec_com_fifteen)):
        plot_elec_avg[elec_com_fifteen[i]] = 0

    time = np.arange(0, band_all_patient[0].shape[0]) / (fs)

    time_dash = np.arange(0, final_time+30, 30)
    j=0
    for key in plot_elec_avg.keys():
        for i in elec[key]:
            num_electrode = raw_car_all[i].ch_names.index(key)
            plot_elec_avg[key] = plot_elec_avg[key] + band_all_patient[i][:, num_electrode]
        j=j+1
        plot_elec_avg[key] = plot_elec_avg[key] / len(elec[key])
        plt.figure(figsize=(30, 10))
        plt.plot(time[:final_time*fs], plot_elec_avg[key][:final_time*fs])
        plt.vlines(time_dash, ymin=-1 * np.ones(time_dash.shape), ymax=np.ones(time_dash.shape), colors='red', ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.title(str(key), fontsize=25)


    plt.show()
    #plt.savefig('test2.png')

    return plot_elec_avg


