import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import moving_avg
import pandas as pd
import os


class QAVisualizer:
    def __init__(self, channel_names, onset_q, onset_a, fs, path, num_patient,t_min, step, allow_plot):
        self.feature_all_patient = dict()
        self.channel_names = channel_names
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

    def get_feature(self, signal, feature_type):
        if feature_type == 'avg':
            feature = np.mean(signal)
        elif feature_type == 'rms':
            e = sum(self.power_2(signal, 2))
            feature = (1 / len(signal)) * math.sqrt(e)
        elif feature_type == 'max_peak':
            feature = np.max(signal)
        elif feature_type == 'variance':
            feature = np.var(signal)
        else:
            raise ValueError("Feature extraction not implemented")
        return feature

    def plot_class_conditional_average(self, data, window_size=250, frq_band_name='gamma', feature_type='avg'):
        """

        :param frq_band_name:
        :param feature_type: can be chosen from 'avg' , 'rms', 'max_peak'
        :return:
        """
        os.makedirs(self.path + feature_type)
        path = os.path.join(self.path, feature_type + '/')

        self.feature_all_patient[feature_type] = []
        for patient in range(len(data[:self.num_patient])):
            k = -1
            electrodes = self.channel_names[patient]
            single_patient_feature = np.zeros((len(electrodes), 67, 1))
            if self.allow_plot:
                fig, ax = plt.subplots(figsize=(60, 40))
            for electrode in electrodes:
                num_electrode = self.channel_names[patient].index(electrode)
                signal = data[patient][frq_band_name][:, num_electrode]
                signal = moving_avg(signal, window_size)
                s = []
                k = k + 1
                for i in range(len(self.onset_q)):
                    start_sample = int(self.onset_q[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_q[i] + self.step) * self.fs
                    feature = self.get_feature(signal[start_sample:end_sample], feature_type)
                    s.append(feature)
                    single_patient_feature[k, i, 0] = feature

                for i in range(len(self.onset_a)):
                    start_sample = int(self.onset_a[i] - self.t_min) * self.fs
                    end_sample = int(self.onset_a[i] + self.step) * self.fs
                    feature = self.get_feature(signal[start_sample:end_sample], feature_type)
                    s.append(feature)
                    single_patient_feature[k, i + len(self.onset_q), 0] = feature
                if self.allow_plot:
                    ax = self.plot_class(s, ax, num_electrode)
            if self.allow_plot:
                ax.set_xlabel('num_electrode', fontsize=40)
                ax.set_ylabel('', fontsize=40)
                ax.set_title('patient=' + str(patient), fontsize=40)
                fig.savefig(path + 'patient_' + str(patient))
            self.feature_all_patient[feature_type].append(single_patient_feature)

    def create_feature_matrix(self):
        featue_matrix_all = []
        for patient in range(len(self.rms_all_patient)):
            featue_matrix = np.zeros((self.rms_all_patient[patient].shape[0], 67, 4))
            for electrode in range(self.rms_all_patient[patient].shape[0]):
                featue_matrix[electrode, :, 0] = self.feature_all_patient[patient][electrode, :, 0]
                featue_matrix[electrode, :, 1] = self.rms_all_patient[patient][electrode, :, 0]
                featue_matrix[electrode, :, 2] = self.max_peak_all_patient[patient][electrode, :, 0]
                featue_matrix[electrode, :, 3] = self.variance_all_patient[patient][electrode, :, 0]
            featue_matrix_all.append(featue_matrix)
        return featue_matrix_all


def plot_comm_elec(common_electrodes, band_all_patient, channel_names_list, final_time, fs, path, freq_band='gamma'):
    """

    :param common_electrodes:
    :param band_all_patient:
    :param channel_names_list:
    :param final_time:
    :param fs:
    :param path:
    :param freq_band:
    :return:
    """
    os.makedirs(path + 'plot_comm_elec')
    p2 = os.path.join(path, 'plot_comm_elec' + '/')
    common_electrode_avg = {}
    for i in range(len(common_electrodes)):
        common_electrode_avg[common_electrodes[i]] = 0

    time = np.arange(0, band_all_patient[0][freq_band].shape[0]) / fs
    time_dash = np.arange(0, final_time + 30, 30)
    plt.figure(figsize=(30, 10))
    for key in common_electrode_avg.keys():
        n = 0
        for patient_idx in range(len(band_all_patient)):
            electrodes = channel_names_list[patient_idx]
            if key in electrodes:
                n = n + 1
                num_electrode = channel_names_list[patient_idx].index(key)
                common_electrode_avg[key] = common_electrode_avg[key] + band_all_patient[patient_idx][freq_band][:,
                                                                        num_electrode]

        common_electrode_avg[key] = common_electrode_avg[key] / n
        plt.figure(figsize=(30, 10))
        plt.plot(time[:final_time * fs], common_electrode_avg[key][:final_time * fs])
        plt.vlines(time_dash, ymin=-1 * np.ones(time_dash.shape), ymax=np.ones(time_dash.shape), colors='red', ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.title(str(key), fontsize=25)
        plt.savefig(p2 + key)

    return common_electrode_avg
