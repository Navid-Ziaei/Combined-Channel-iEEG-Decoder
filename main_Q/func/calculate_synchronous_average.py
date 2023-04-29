import numpy as np
import matplotlib.pyplot as plt
from ieeg_func.histogram_elec import hist_elec
from scipy import stats
import os



class synchronous_avg():
    def __init__(self,raw_car_all,band_all_patient,onset_q,onset_a,path,fs):
        self.raw_car_all = raw_car_all
        self.band_all_patient = band_all_patient
        self.onset_q = onset_q
        self.onset_a = onset_a
        self.fs = fs
        os.makedirs(path + 'synchronous_average')
        p2 = os.path.join(path, 'synchronous_average' + '/')
        self.path = p2
        self.time = np.arange(0, 3, 1 /fs)

    def cal_CI(self,data, conf_level):
        sem = stats.sem(data, axis=0)
        t_crit = sem * stats.t.ppf((1 + conf_level) / 2, len(data) - 1)
        return t_crit

    def synch_ci(self,signal):
        trial_q = np.zeros((len(self.onset_q), self.time.shape[0]))
        for i in range(len(self.onset_q)):
            start_sample = int(self.onset_q[i] - 0.5) * self.fs
            end_sample = int(self.onset_q[i] + 2.5) * self.fs
            trial_q[i, :] = signal[start_sample:end_sample]
        synch_avg_q = np.mean(trial_q, axis=0)
        ci = self.cal_CI(trial_q, 0.95)
        ci_l_q = synch_avg_q - ci
        ci_h_q = synch_avg_q + ci

        trial_a = np.zeros((len(self.onset_a), self.time.shape[0]))
        for i in range(len(self.onset_a)):
            start_sample = int(self.onset_a[i] - 0.5) * self.fs
            end_sample = int(self.onset_a[i] + 2.5) * self.fs
            trial_a[i, :] = signal[start_sample:end_sample]
        synch_avg_a = np.mean(trial_a, axis=0)
        ci = self.cal_CI(trial_a, 0.95)
        ci_l_a = synch_avg_a - ci
        ci_h_a = synch_avg_a + ci

        return ci_l_q,ci_h_q,ci_l_a,ci_h_a,synch_avg_q,synch_avg_a



    def calculate_synchronous_avg(self):
        for patient in range(len(self.band_all_patient[:1])):
            os.makedirs(self.path + 'patient_' + str(patient))
            p3 = os.path.join(self.path, 'patient_' + str(patient) + '/')
            electrodes = self.raw_car_all[patient].ch_names

            for electrode in electrodes:
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal=self.band_all_patient[patient][:,num_electrode]

                ci_l_q,ci_h_q,ci_l_a,ci_h_a,synch_avg_q,synch_avg_a=self.synch_ci(signal)

                plt.figure()
                plt.plot(self.time,synch_avg_q,color='blue',linewidth=2, label='question')
                plt.fill_between(self.time, ci_l_q, ci_h_q,color='blue', alpha=0.2)
                plt.plot(self.time, synch_avg_a, color='red', linewidth=2, label='answer')
                plt.fill_between(self.time, ci_l_a, ci_h_a, color='red',alpha=0.2)
                plt.title('patient=' + str(patient) + '  electrode=' + electrode, fontsize=15)
                plt.legend()
                plt.savefig(p3+'p_' + str(patient) + '_elec_' + electrode)


    def calculate_synch_avg_common_electrode(self):
        hist,elc_com=hist_elec(self.raw_car_all, print_analyze=False)
        time = np.arange(0, 3, 1 / self.fs)
        os.makedirs(self.path + 'patient_common_electrode' )
        p3 = os.path.join(self.path, 'patient_common_electrode' + '/')
        for key in elc_com[:1]:
            j=0
            plt.figure(figsize=(5,20))
            for patient in range(len(self.band_all_patient)):
                electrodes = self.raw_car_all[patient].ch_names
                if key in electrodes:
                    num_electrode = self.raw_car_all[patient].ch_names.index(key)
                    signal = self.band_all_patient[patient][:, num_electrode]
                    ci_l_q, ci_h_q, ci_l_a, ci_h_a, synch_avg_q, synch_avg_a = self.synch_ci(signal)
                    plt.plot(time, synch_avg_q+j, color='blue', linewidth=2)
                    plt.fill_between(time, ci_l_q+j, ci_h_q+j, color='blue', alpha=0.2)
                    plt.plot(time, synch_avg_a+j, color='red', linewidth=2)
                    plt.fill_between(time, ci_l_a+j, ci_h_a+j, color='red', alpha=0.2)
                    plt.title('patient_' + str(patient))
                    j = j + 3
            plt.legend(['question','answer'])
            plt.title('electrode_' + key)
            plt.savefig(p3 + 'elec_' + key)











