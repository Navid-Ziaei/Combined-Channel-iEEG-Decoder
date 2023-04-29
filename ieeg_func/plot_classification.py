import numpy as np
import math
import matplotlib.pyplot as plt
from ieeg_func.mov_avg import moving_avg
import pandas as pd
import os


class classification():
    def __init__(self,raw_car_all,band_all_patient,band_all_patient_nohil,onset_q,onset_a,fs,path,t_min,step,num_patient):
        self.raw_car_all=raw_car_all
        self.band_all_patient=band_all_patient
        self.band_all_patient_nohil=band_all_patient_nohil
        self.onset_q=onset_q
        self.onset_a=onset_a
        self.fs=fs
        os.makedirs(path + 'classification')
        p2 = os.path.join(path, 'classification' + '/')
        self.path=p2
        self.num_patient=num_patient
        self.t_min=t_min
        self.step = step


    def power_2(self,my_list, p):
        return [x ** p for x in my_list]

    def plot_class(self,s, ax,num_electrode):
        # self.label[0] (red)/ self.label[1](green)
        label = ['0'] * len(self.onset_q) + ['1'] * len(self.onset_a)
        y = [num_electrode] * len(label)

        df = pd.DataFrame(dict(x=y, y=s, label=label))
        colors = {'0': 'red', '1': 'green'}
        ax.scatter(df['x'], df['y'], c=df['label'].map(colors), marker='o')
        return ax

    def class_avg( self,window_size):
        os.makedirs(self.path + 'AVG')
        p_avg = os.path.join(self.path, 'AVG' + '/')
        for patient in range(len(self.band_all_patient[:self.num_patient])):
            electrodes = self.raw_car_all[patient].ch_names
            fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient[patient][:, num_electrode]
                signal = moving_avg(signal, window_size)
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    s.append(np.mean(signal[start_sample:end_sample]))
                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] +self.step) * self.fs
                    s.append(np.mean(signal[start_sample:end_sample]))
                ax=self.plot_class(s,ax,num_electrode)
            ax.set_xlabel('num_electrode', fontsize=40)
            ax.set_ylabel('average', fontsize=40)
            ax.set_title('patient='+str(patient), fontsize=40)
            fig.savefig(p_avg+'patient_'+str(patient))

    def class_rms(self):
        os.makedirs(self.path + 'RMS')
        p_avg = os.path.join(self.path, 'RMS' + '/')
        for patient in range(len(self.band_all_patient_nohil[:self.num_patient])):
            electrodes = self.raw_car_all[patient].ch_names
            fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient_nohil[patient][:, num_electrode]
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    e = sum(self.power_2(signal[start_sample:end_sample], 2))
                    e2 = (1 / len(signal[start_sample:end_sample])) * math.sqrt(e)
                    s.append(e2)
                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] -self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    e = sum(self.power_2(signal[start_sample:end_sample], 2))
                    e2 = (1 / len(signal[start_sample:end_sample])) * math.sqrt(e)
                    s.append(e2)
                ax = self.plot_class(s, ax, num_electrode)
            ax.set_xlabel('num_electrode', fontsize=40)
            ax.set_ylabel('RMS', fontsize=40)
            ax.set_title('patinet=' + str(patient), fontsize=40)
            fig.savefig(p_avg + 'patinet_' + str(patient))

    def class_max_peak(self):
        os.makedirs(self.path + 'max_peak')
        p_avg = os.path.join(self.path, 'max_peak' + '/')
        for patient in range(len(self.band_all_patient[:self.num_patient])):
            electrodes = self.raw_car_all[patient].ch_names
            fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient[patient][:, num_electrode]
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    s.append(np.max(signal[start_sample:end_sample]))
                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    s.append(np.max(signal[start_sample:end_sample]))
                ax = self.plot_class(s, ax, num_electrode)
            ax.set_xlabel('num_electrode', fontsize=40)
            ax.set_ylabel('max_peak', fontsize=40)
            ax.set_title('patinet=' + str(patient), fontsize=40)
            fig.savefig(p_avg + 'patinet_' + str(patient))

    def class_variance(self):
        os.makedirs(self.path + 'variance')
        p_avg = os.path.join(self.path, 'variance' + '/')
        for patient in range(len(self.band_all_patient[:self.num_patient])):
            electrodes = self.raw_car_all[patient].ch_names
            fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient[patient][:, num_electrode]
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    s.append(np.var(signal[start_sample:end_sample]))
                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    s.append(np.var(signal[start_sample:end_sample]))
                ax=self.plot_class(s,ax,num_electrode)
            ax.set_xlabel('num_electrode', fontsize=40)
            ax.set_ylabel('variance', fontsize=40)
            ax.set_title('patinet='+str(patient), fontsize=40)
            fig.savefig(p_avg+'patinet_'+str(patient))



