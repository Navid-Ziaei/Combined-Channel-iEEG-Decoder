import numpy as np
import math
import matplotlib.pyplot as plt
from ieeg_func.mov_avg import moving_avg
import pandas as pd
import os


class EEGClassifier:
    def __init__(self, raw_car_all, band_all_patient, band_all_patient_nohil, onset_q, onset_a, fs, path, num_patient,
                 t_min, step, allow_plot):
        self.raw_car_all = raw_car_all
        self.band_all_patient = band_all_patient
        self.band_all_patient_nohil = band_all_patient_nohil
        self.onset_q = onset_q
        self.onset_a = onset_a
        self.fs = fs
        self.path = path
        self.num_patient = num_patient
        self.t_min = t_min
        self.step = step
        self.allow_plot = allow_plot

    def power_2(self, my_list, p):
        return [x ** p for x in my_list]

    def plot_class(self, s, ax, num_electrode):
        # self.label[0] (red)/ self.label[1](green)
        label = ['0'] * len(self.onset_q) + ['1'] * len(self.onset_a)
        y = [num_electrode] * len(label)

        df = pd.DataFrame(dict(x=y, y=s, label=label))
        colors = {'0': 'red', '1': 'green'}
        ax.scatter(df['x'], df['y'], c=df['label'].map(colors), marker='o')
        return ax

    def class_avg(self, window_size):
        os.makedirs(self.path + 'AVG')
        p_avg = os.path.join(self.path, 'AVG' + '/')
        self.avg_all_patient = []
        for patient in range(len(self.band_all_patient[:self.num_patient])):
            k = -1
            electrodes = self.raw_car_all[patient].ch_names
            avg_one = np.zeros((len(electrodes), 67, 1))
            if self.allow_plot:
                fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient[patient][:, num_electrode]
                signal = moving_avg(signal, window_size)
                s = []
                k = k + 1
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    s.append(np.mean(signal[start_sample:end_sample]))
                    avg_one[k, i, 0] = np.mean(signal[start_sample:end_sample])

                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    s.append(np.mean(signal[start_sample:end_sample]))
                    avg_one[k, i + len(self.onset_q), 0] = np.mean(signal[start_sample:end_sample])
                if self.allow_plot:
                    ax = self.plot_class(s, ax, num_electrode)
            if self.allow_plot:
                ax.set_xlabel('num_electrode', fontsize=40)
                ax.set_ylabel('average', fontsize=40)
                ax.set_title('patient=' + str(patient), fontsize=40)
                fig.savefig(p_avg + 'patient_' + str(patient))
            self.avg_all_patient.append(avg_one)
        return self.avg_all_patient

    def class_rms(self):
        os.makedirs(self.path + 'RMS')
        p_avg = os.path.join(self.path, 'RMS' + '/')
        self.rms_all_patient = []
        for patient in range(len(self.band_all_patient_nohil[:self.num_patient])):
            k = -1
            electrodes = self.raw_car_all[patient].ch_names
            rms_one = np.zeros((len(electrodes), 67, 1))
            if self.allow_plot:
                fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                k = k + 1
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient_nohil[patient][:, num_electrode]
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    e = sum(self.power_2(signal[start_sample:end_sample], 2))
                    e2 = (1 / len(signal[start_sample:end_sample])) * math.sqrt(e)
                    s.append(e2)
                    rms_one[k, i, 0] = e2
                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    e = sum(self.power_2(signal[start_sample:end_sample], 2))
                    e2 = (1 / len(signal[start_sample:end_sample])) * math.sqrt(e)
                    s.append(e2)
                    rms_one[k, i + len(self.onset_q), 0] = e2
                if self.allow_plot:
                    ax = self.plot_class(s, ax, num_electrode)
            if self.allow_plot:
                ax.set_xlabel('num_electrode', fontsize=40)
                ax.set_ylabel('RMS', fontsize=40)
                ax.set_title('patinet=' + str(patient), fontsize=40)
                fig.savefig(p_avg + 'patinet_' + str(patient))
            self.rms_all_patient.append(rms_one)
        return self.rms_all_patient

    def class_max_peak(self):
        os.makedirs(self.path + 'max_peak')
        p_avg = os.path.join(self.path, 'max_peak' + '/')
        self.max_peak_all_patient = []
        for patient in range(len(self.band_all_patient[:self.num_patient])):
            k = -1
            electrodes = self.raw_car_all[patient].ch_names
            max_peak_one = np.zeros((len(electrodes), 67, 1))
            if self.allow_plot:
                fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                k = k + 1
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient[patient][:, num_electrode]
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    s.append(np.max(signal[start_sample:end_sample]))
                    max_peak_one[k, i, 0] = np.max(signal[start_sample:end_sample])

                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    s.append(np.max(signal[start_sample:end_sample]))
                    max_peak_one[k, i + len(self.onset_q), 0] = np.max(signal[start_sample:end_sample])

                if self.allow_plot:
                    ax = self.plot_class(s, ax, num_electrode)
            if self.allow_plot:
                ax.set_xlabel('num_electrode', fontsize=40)
                ax.set_ylabel('max_peak', fontsize=40)
                ax.set_title('patinet=' + str(patient), fontsize=40)
                fig.savefig(p_avg + 'patinet_' + str(patient))
            self.max_peak_all_patient.append(max_peak_one)
        return self.max_peak_all_patient

    def class_variance(self):
        os.makedirs(self.path + 'variance')
        p_avg = os.path.join(self.path, 'variance' + '/')
        self.variance_all_patient = []
        for patient in range(len(self.band_all_patient[:self.num_patient])):
            k = -1
            electrodes = self.raw_car_all[patient].ch_names
            variance_one = np.zeros((len(electrodes), 67, 1))
            if self.allow_plot:
                fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                k = k + 1
                num_electrode = self.raw_car_all[patient].ch_names.index(electrode)
                signal = self.band_all_patient[patient][:, num_electrode]
                s = []
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    s.append(np.var(signal[start_sample:end_sample]))
                    variance_one[k, i, 0] = np.var(signal[start_sample:end_sample])

                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    s.append(np.var(signal[start_sample:end_sample]))
                    variance_one[k, i + len(self.onset_q), 0] = np.var(signal[start_sample:end_sample])

                if self.allow_plot:
                    ax = self.plot_class(s, ax, num_electrode)
            if self.allow_plot:
                ax.set_xlabel('num_electrode', fontsize=40)
                ax.set_ylabel('variance', fontsize=40)
                ax.set_title('patinet=' + str(patient), fontsize=40)
                fig.savefig(p_avg + 'patinet_' + str(patient))
            self.variance_all_patient.append(variance_one)
        return self.variance_all_patient

    def create_feature_matrix(self):
        featue_matrix_all = []
        for patient in range(len(self.rms_all_patient)):
            featue_matrix = np.zeros((self.rms_all_patient[patient].shape[0], 67, 4))
            for electrode in range(self.rms_all_patient[patient].shape[0]):
                featue_matrix[electrode, :, 0] = self.avg_all_patient[patient][electrode, :, 0]
                featue_matrix[electrode, :, 1] = self.rms_all_patient[patient][electrode, :, 0]
                featue_matrix[electrode, :, 2] = self.max_peak_all_patient[patient][electrode, :, 0]
                featue_matrix[electrode, :, 3] = self.variance_all_patient[patient][electrode, :, 0]
            featue_matrix_all.append(featue_matrix)
        return featue_matrix_all
