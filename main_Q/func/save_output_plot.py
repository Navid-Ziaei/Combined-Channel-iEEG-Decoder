import matplotlib.pyplot as plt
import numpy as np
import os
from main_Q.func.calculate_confidence_interval import cal_CI

def save_plot(raw_car_all,signal_q,signal_a,path,fs):
    time = np.arange(0,3,1/fs)
    for patient in range(len(signal_q)):
        os.makedirs(path + 'patient_'+str(patient))
        p2=os.path.join(path,'patient_'+str(patient)+'/')
        electrodes = raw_car_all[patient].ch_names
        i=0

        for electrode in electrodes:
            os.makedirs(p2 + 'electrode_' + electrode)
            p3=os.path.join(p2 , 'electrode_' + electrode+'/')
            plt.figure()
            plt.plot(time, signal_q[patient][i])
            pos_dash = cal_CI(signal_q[patient][i], 0.95)
            plt.hlines(y=pos_dash, xmin=0, xmax=3, colors='red', ls='--', lw=2)
            plt.title('patient=' + str(patient) + '  electrode=' + electrode, fontsize=15)
            plt.savefig(p3 +'question')

            plt.figure()
            plt.plot(time, signal_a[patient][i])
            pos_dash = cal_CI(signal_a[patient][i], 0.95)
            plt.hlines(y=pos_dash, xmin=0, xmax=3, colors='red', ls='--', lw=2)
            plt.title('patient=' + str(patient) + '  electrode=' + electrode, fontsize=15)
            plt.savefig(p3 + 'answer')
            i=i+1




    #

