import numpy as np
import math
import matplotlib.pyplot as plt
from utils import moving_avg
import pandas as pd
import pickle as pkl
from scipy import stats
from sklearn.decomposition import PCA
from collections import Counter
from multitaper_spectrogram_python import multitaper_spectrogram


class QAVisualizer:
    def __init__(self, channel_names, onset_1, onset_0, path, settings):
        print("\n =================================== \n"
              "extract feature")
        self.feature_all_patient = []
        self.channel_names = channel_names
        self.onset_1 = onset_1
        self.onset_0 = onset_0
        self.fs = settings['fs']
        self.path = path
        self.settings = settings
        self.feature_list = settings['feature_list']

    def plot_class(self, s, ax, num_electrode):
        # self.label[0] (green)/ self.label[1](red)
        label = ['1'] * len(self.onset_1) + ['0'] * len(self.onset_0)
        y = [num_electrode] * len(label)

        df = pd.DataFrame(dict(x=y, y=s, label=label))
        colors = {'0': 'green', '1': 'red'}
        ax.scatter(df['x'], df['y'], c=df['label'].map(colors), marker='o')
        return ax

    def get_feature(self, signal_with_hilbert, signal_without_hilbert, signal_with_hilbert_movavg, feature):
        if self.feature_list['AVG']:
            feature['AVG'].append(np.mean(signal_with_hilbert_movavg))
        if self.feature_list['RMS']:
            e = sum([x ** 2 for x in signal_without_hilbert])
            feature['RMS'].append(1 / len(signal_without_hilbert) * math.sqrt(e))
        if self.feature_list['max_peak']:
            feature['max_peak'].append(np.max(signal_with_hilbert))
        if self.feature_list['variance']:
            feature['variance'].append(np.var(signal_with_hilbert))

        return feature

    def get_feature_all(self, data_with_hilbert, data_without_hilbert):
        """
        param data_with_hilbert:
        param data_without_hilbert:
        return:
        """
        if self.settings['load_feature_matrix'] is False:
            num_patient = self.settings['parameter_get_feature']['num_patient_get_feature']
            t_min = self.settings['parameter_get_feature']['t_min']
            step = self.settings['parameter_get_feature']['step']
            window_size = self.settings['parameter_get_feature']['window_size']
            frq_band_name = self.settings['band']

            for patient in range(num_patient):
                electrodes = self.channel_names[patient]
                single_patient_feature = []
                for electrode in electrodes:
                    feature = dict()
                    for key in self.feature_list.keys():
                        if self.feature_list[key]:
                            feature[key] = []
                    num_electrode = self.channel_names[patient].index(electrode)
                    signal_with_hilbert = data_with_hilbert[patient][frq_band_name][:, num_electrode]
                    signal_without_hilbert = data_without_hilbert[patient][frq_band_name][:, num_electrode]
                    signal_with_hilbert_movavg = moving_avg(signal_with_hilbert, window_size)
                    for i in range(len(self.onset_1)):
                        start_sample = int(self.onset_1[i] - t_min) * self.fs
                        end_sample = int(self.onset_1[i] + step) * self.fs
                        feature = self.get_feature(signal_with_hilbert[start_sample:end_sample],
                                                   signal_without_hilbert[start_sample:end_sample],
                                                   signal_with_hilbert_movavg[start_sample:end_sample],
                                                   feature)
                    for i in range(len(self.onset_0)):
                        start_sample = int(self.onset_0[i] - t_min) * self.fs
                        end_sample = int(self.onset_0[i] + step) * self.fs
                        feature = self.get_feature(signal_with_hilbert[start_sample:end_sample],
                                                   signal_without_hilbert[start_sample:end_sample],
                                                   signal_with_hilbert_movavg[start_sample:end_sample],
                                                   feature)
                    feature = pd.DataFrame(feature)
                    single_patient_feature.append(feature)

                self.feature_all_patient.append(single_patient_feature)

            if self.settings['save_feature_matrix']:
                with open(self.path.path_save_data + '/feature_all_patient_df.pkl', 'wb') as f:
                    pkl.dump(self.feature_all_patient, f)
        else:
            with open(self.path.path_save_data + '/feature_all_patient_df.pkl', 'rb') as f:
                self.feature_all_patient = pkl.load(f)

        return self.feature_all_patient

    def create_feature_matrix(self):
        print("\n =================================== \n"
              "convert features from pandas.Dataframe to matrix")
        if self.settings['load_feature_matrix'] is False:
            feature_matrix_all = []
            num_feature = sum(self.feature_list.values())
            for patient in range(len(self.feature_all_patient)):
                feature_matrix = np.zeros((len(self.channel_names[patient]), len(self.onset_1)+len(self.onset_0), num_feature))
                for electrode in range(len(self.channel_names[patient])):
                    feature_matrix[electrode, :, :] = self.feature_all_patient[patient][electrode].values

                feature_matrix_all.append(feature_matrix)

            if self.settings['save_feature_matrix']:
                with open(self.path.path_save_data + '/feature_matrix_all.pkl', 'wb') as f:
                    pkl.dump(feature_matrix_all, f)
        else:
            with open(self.path.path_save_data + '/feature_matrix_all.pkl', 'rb') as f:
                feature_matrix_all = pkl.load(f)
        return feature_matrix_all

    def plot_class_conditional_average(self):
        print("\n =================================== \n"
              "plot features of trials of all electrode for each patient ")
        num_patient = self.settings['parameter_get_feature']['num_patient_plot_class_conditional_average']
        for feature_type in self.feature_list.keys():
            if self.feature_list[feature_type]:
                for patient in range(num_patient):
                    fig, ax = plt.subplots(figsize=(60, 40))
                    for electrode in range(len(self.channel_names[patient])):
                        s = self.feature_all_patient[patient][electrode][feature_type]
                        ax = self.plot_class(s, ax, electrode)

                    ax.set_xlabel('num_electrode', fontsize=40)
                    ax.set_ylabel(feature_type, fontsize=40)
                    ax.set_title('patient=' + str(patient), fontsize=40)
                    fig.savefig(self.path.path_results_get_feature_features[feature_type] + 'patient_' + str(patient))


def plot_comm_elec(common_electrodes, band_all_patient, channel_names_list, onset_1, offset_1, path, settings):
    print("\n =================================== \n"
          "plot average of signal of shared electrodes of patients")
    final_time = settings['final_time']
    fs = settings['fs']
    freq_band = settings['band']
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
    common_electrode_avg = {}
    for i in range(len(common_electrodes)):
        common_electrode_avg[common_electrodes[i]] = 0

    time = np.arange(0, band_all_patient[0][freq_band].shape[0]) / fs
    time_dash = onset_1 + offset_1
    time_dash.sort()
    time_dash2 = [i for i in time_dash if i < final_time]
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
        plt.vlines(time_dash2, ymin=-1 * np.ones(len(time_dash2)), ymax=np.ones(len(time_dash2)), colors='red', ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.title(str(key), fontsize=25)
        plt.savefig(path.path_results_plot_common_electrodes + key)

    return common_electrode_avg


def plot_temporal_signal(channel_names_list, band_all_patient, onset_1, offset_1, path, settings):
    print("\n =================================== \n"
          "plot temporal signal of each electrode of each patients")
    fs = settings['fs']
    electrode = settings['parameter_plot_temporal_signal']['electrode']
    patient = settings['parameter_plot_temporal_signal']['patient']
    final_time = settings['final_time']
    freq_band = settings['band']

    if electrode in channel_names_list[patient]:
        num_electrode = channel_names_list[patient].index(electrode)
        signal = band_all_patient[patient][freq_band][:, num_electrode]
        time = np.arange(0, signal.shape[0]) / fs
        time_dash = onset_1 + offset_1
        time_dash.sort()
        time_dash2 = [i for i in time_dash if i < final_time]

        time_dash_music = np.arange(0, final_time + 30, 30)

        plt.figure(figsize=(80, 10))
        plt.plot(time[:final_time * fs], signal[:final_time * fs])
        plt.vlines(time_dash2, ymin=-4 * np.ones(len(time_dash2)), ymax=4 * np.ones(len(time_dash2)), colors='red',
                   ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.vlines(time_dash_music, ymin=-4 * np.ones(len(time_dash_music)), ymax=8 * np.ones(len(time_dash_music)),
                   colors='black', ls='--', lw=2, label='vline_multiple - partial height')

        # plt.show()
        plt.xlabel('time', fontsize=15)
        plt.ylabel('temporal_signal', fontsize=15)
        plt.title('temporal signal of patient=' + str(patient) + '_electrode =' + electrode, fontsize=15)
        plt.savefig(path.path_results_temporal_signal + "temporal_signal")

    else:
        raise ValueError(f"{electrode} isn't in channel_names_list of patient_{patient}")


class SynchronousAvg:
    def __init__(self, channel_names_list, band_all_patient, onset_1, onset_0, path, settings):
        print("\n =================================== \n"
              "plot synchronous average between trials for all electrode of each patient")
        self.channel_names_list = channel_names_list
        self.band_all_patient = band_all_patient
        self.onset_1 = onset_1
        self.onset_0 = onset_0
        self.fs = settings['fs']
        self.freq_band = settings['band']
        self.path = path
        self.t_min = settings['parameter_synchronous_average']['t_min']
        self.step = settings['parameter_synchronous_average']['step']
        self.time = np.arange(0, self.step+self.t_min, 1 / self.fs)

    def calculate_confidence_interval(self, data, conf_level):
        sem = stats.sem(data, axis=0)
        t_crit = sem * stats.t.ppf((1 + conf_level) / 2, len(data) - 1)
        return t_crit

    def synch_ci(self, signal):
        trial_q = np.zeros((len(self.onset_1), self.time.shape[0]))
        for i in range(len(self.onset_1)):
            start_sample = int(self.onset_1[i] - self.t_min) * self.fs
            end_sample = int(self.onset_1[i] + self.step) * self.fs
            trial_q[i, :] = signal[start_sample:end_sample]
        synch_avg_q = np.mean(trial_q, axis=0)
        ci = self.calculate_confidence_interval(trial_q, 0.95)
        ci_l_q = synch_avg_q - ci
        ci_h_q = synch_avg_q + ci

        trial_a = np.zeros((len(self.onset_0), self.time.shape[0]))
        for i in range(len(self.onset_0)):
            start_sample = int(self.onset_0[i] - self.t_min) * self.fs
            end_sample = int(self.onset_0[i] + self.step) * self.fs
            trial_a[i, :] = signal[start_sample:end_sample]
        synch_avg_a = np.mean(trial_a, axis=0)
        ci = self.calculate_confidence_interval(trial_a, 0.95)
        ci_l_a = synch_avg_a - ci
        ci_h_a = synch_avg_a + ci

        return ci_l_q, ci_h_q, ci_l_a, ci_h_a, synch_avg_q, synch_avg_a

    def calculate_synchronous_avg(self, num_patient):
        for patient in range(num_patient):
            electrodes = self.channel_names_list[patient]
            for electrode in electrodes:
                num_electrode = self.channel_names_list[patient].index(electrode)
                signal = self.band_all_patient[patient][self.freq_band][:, num_electrode]

                ci_l_q, ci_h_q, ci_l_a, ci_h_a, synch_avg_q, synch_avg_a = self.synch_ci(signal)

                plt.figure()
                plt.plot(self.time, synch_avg_q, color='blue', linewidth=2, label='label_1')
                plt.fill_between(self.time, ci_l_q, ci_h_q, color='blue', alpha=0.2)
                plt.plot(self.time, synch_avg_a, color='red', linewidth=2, label='label_0')
                plt.fill_between(self.time, ci_l_a, ci_h_a, color='red', alpha=0.2)
                plt.title('patient=' + str(patient) + '  electrode=' + electrode, fontsize=15)
                plt.legend()
                plt.savefig(
                    self.path.path_results_synchronous_average[patient] + 'p_' + str(patient) + '_elec_' + electrode)

    def calculate_synch_avg_common_electrode(self, common_electrodes):
        for key in common_electrodes:
            j = 0
            plt.figure(figsize=(5, 20))
            for patient in range(len(self.band_all_patient)):
                electrodes = self.channel_names_list[patient]
                if key in electrodes:
                    num_electrode = self.channel_names_list[patient].index(key)
                    signal = self.band_all_patient[patient][self.freq_band][:, num_electrode]
                    ci_l_q, ci_h_q, ci_l_a, ci_h_a, synch_avg_q, synch_avg_a = self.synch_ci(signal)
                    plt.plot(self.time, synch_avg_q + j, color='blue', linewidth=2)
                    plt.fill_between(self.time, ci_l_q + j, ci_h_q + j, color='blue', alpha=0.2)
                    plt.plot(self.time, synch_avg_a + j, color='red', linewidth=2)
                    plt.fill_between(self.time, ci_l_a + j, ci_h_a + j, color='red', alpha=0.2)
                    plt.title('patient_' + str(patient))
                    j = j + 3
            plt.legend(['question', 'answer'])
            plt.title('electrode_' + key)
            plt.savefig(self.path.path_results_synch_average_common_electrode + 'elec_' + key)


def plot_pca(channel_names_list, path, feature_matrix, settings):
    print("\n =================================== \n"
          "plot PCA for each patient")
    num_patient = settings['parameter_get_pca']['num_patient']
    if settings['task'] == 'question&answer':
        label = np.zeros(67)
        label[0:15] = 1
    if settings['task'] == 'speech&music':
        label = np.zeros(13)
        label[0:6] = 1
    counter = Counter(label)
    for patient in range(num_patient):
        ch_name = channel_names_list[patient]
        for electrode in range(len(ch_name)):
            feature = feature_matrix[patient][electrode, :, :]
            pca = PCA(n_components=2)
            x = pca.fit_transform(feature)
            plt.figure()
            for label2, _ in counter.items():
                row_ix = np.where(label == label2)[0]
                plt.scatter(x[row_ix, 0], x[row_ix, 1], label=str(label2))
            plt.title('patient=' + str(patient) + '_electrode=' + ch_name[electrode], fontsize=15)
            plt.savefig(path.path_results_get_pca[patient] + 'p_' + str(patient) + '_electrode=' + ch_name[electrode])
            plt.legend()


def plot_wavelet(path, raw_car_all, settings):
    print("\n =================================== \n"
          "plot wavelet for each patient")
    patient = settings['parameter_wavelet']['patient']
    electrode = settings['parameter_wavelet']['electrode']
    # Set spectrogram params
    fs = raw_car_all[patient].info['sfreq']  # Sampling Frequency
    frequency_range = [0, 120]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = 3  # Set time-half bandwidth
    num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [4, 1]  # Window size is 4s with step size of 1s
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = True  # use multiprocessing
    cpus = 3  # use 3 cores  in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = True  # plot spectrogram
    return_fig = False  # do not return plotted spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = True  # print extra info
    xyflip = False  # do not transpose spect output matrix

    final_time = 390
    num_electrode = raw_car_all[patient].ch_names.index(electrode)
    data = raw_car_all[patient].get_data().T[:int(390 * fs), num_electrode]
    # Compute the multitaper spectrogram
    spect, stimes, sfreqs, fig = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                        window_params, min_nfft, detrend_opt, multiprocess, cpus,
                                                        weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
    time_dash = np.arange(0, final_time, 30)
    plt.vlines(time_dash, ymin=0, ymax=np.max(frequency_range) * np.ones(time_dash.shape), colors='blue', ls='--', lw=2,
               label='vline_multiple - partial height')
    plt.title('wavelet_raw_data')
    plt.xlabel('time')
    plt.ylabel('frequency')
    fig.savefig(path.path_results_wavelet + 'wavelet_patient' + str(patient) + '_electrode' + electrode)
