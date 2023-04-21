import numpy as np
import matplotlib.pyplot as plt
from main_Q.func.calculate_confidence_interval import cal_CI
from ieeg_func.histogram_elec import hist_elec
import os

def synch_ci(onset_q,onset_a,signal,fs,time):
    trial_q = np.zeros((len(onset_q), time.shape[0]))
    for i in range(len(onset_q)):
        start_sample = int(onset_q[i] - 0.5) * fs
        end_sample = int(onset_q[i] + 2.5) * fs
        trial_q[i, :] = signal[start_sample:end_sample]
    synch_avg_q = np.mean(trial_q, axis=0)
    ci = cal_CI(trial_q, 0.95)
    ci_l_q = synch_avg_q - ci
    ci_h_q = synch_avg_q + ci

    trial_a = np.zeros((len(onset_a), time.shape[0]))
    for i in range(len(onset_a)):
        start_sample = int(onset_a[i] - 0.5) * fs
        end_sample = int(onset_a[i] + 2.5) * fs
        trial_a[i, :] = signal[start_sample:end_sample]
    synch_avg_a = np.mean(trial_a, axis=0)
    ci = cal_CI(trial_a, 0.95)
    ci_l_a = synch_avg_a - ci
    ci_h_a = synch_avg_a + ci

    return ci_l_q,ci_h_q,ci_l_a,ci_h_a,synch_avg_q,synch_avg_a



def calculate_synchronous_avg(raw_car_all,band_all_patient,onset_q,onset_a,path,fs):
    time = np.arange(0, 3, 1 / fs)
    for patient in range(len(band_all_patient)):
        os.makedirs(path + 'patient_' + str(patient))
        p2 = os.path.join(path, 'patient_' + str(patient) + '/')
        electrodes = raw_car_all[patient].ch_names

        for electrode in electrodes:
            os.makedirs(p2 + 'electrode_' + electrode)
            p3 = os.path.join(p2, 'electrode_' + electrode + '/')
            num_electrode = raw_car_all[patient].ch_names.index(electrode)
            signal=band_all_patient[patient][:,num_electrode]

            ci_l_q,ci_h_q,ci_l_a,ci_h_a,synch_avg_q,synch_avg_a=synch_ci(onset_q,onset_a,signal,fs,time)

            plt.figure()
            plt.plot(time,synch_avg_q,color='blue',linewidth=2, label='question')
            plt.fill_between(time, ci_l_q, ci_h_q,color='blue', alpha=0.2)
            plt.plot(time, synch_avg_a, color='red', linewidth=2, label='answer')
            plt.fill_between(time, ci_l_a, ci_h_a, color='red',alpha=0.2)
            plt.title('patient=' + str(patient) + '  electrode=' + electrode, fontsize=15)
            plt.legend()
            plt.savefig(p3+'p_' + str(patient) + '_elec_' + electrode)


def calculate_synch_avg_common_electrode(raw_car_all,band_all_patient,onset_q,onset_a,path,fs):
    hist,elc_com=hist_elec(raw_car_all, print_analyze=False)
    time = np.arange(0, 3, 1 / fs)
    os.makedirs(path + 'patient_common_electrode' )
    p2 = os.path.join(path, 'patient_common_electrode' + '/')
    for key in elc_com:
        j=0
        plt.figure(figsize=(5,20))
        for patient in range(len(band_all_patient)):
            electrodes = raw_car_all[patient].ch_names
            if key in electrodes:
                num_electrode = raw_car_all[patient].ch_names.index(key)
                signal = band_all_patient[patient][:, num_electrode]
                ci_l_q, ci_h_q, ci_l_a, ci_h_a, synch_avg_q, synch_avg_a = synch_ci(onset_q, onset_a, signal, fs, time)
                plt.plot(time, synch_avg_q+j, color='blue', linewidth=2)
                plt.fill_between(time, ci_l_q+j, ci_h_q+j, color='blue', alpha=0.2)
                plt.plot(time, synch_avg_a+j, color='red', linewidth=2)
                plt.fill_between(time, ci_l_a+j, ci_h_a+j, color='red', alpha=0.2)
                plt.title('patient_' + str(patient))
                j = j + 3
        plt.legend(['question','answer'])
        plt.title('electrode_' + key)
        plt.savefig(p2 + 'elec_' + key)











