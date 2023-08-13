import matplotlib.pyplot as plt
from utils import moving_avg
import pandas as pd
import pickle as pkl
import numpy as np
import math
import scipy
import statsmodels.api as sm
from collections import Counter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class FeatureExtractor:
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

    def sampen(self, signal, m, r):
        N = len(signal)
        # Split time series and save all templates of length m
        xm_A = []
        xm_B = []
        for i in range(N - m):
            xm_B.append(signal[i: i + m])
        m = m + 1
        for i in range(N - m):
            xm_A.append(signal[i: i + m])
        B = 0
        for i in range(len(xm_B)):
            for j in range(len(xm_B)):
                if i != j:
                    if np.max(np.abs(xm_B[i] - xm_B[j])) < r:
                        B = B + 1
        A = 0
        for i in range(len(xm_A)):
            for j in range(len(xm_A)):
                if i != j:
                    if np.max(np.abs(xm_A[i] - xm_A[j])) < r:
                        A = A + 1
        return -np.log(A / B)

    def objective(self, x, a, b):
        return a * x + b

    def hfd(self, signal, kmax):
        N = signal.shape[0]
        L = []

        for k in range(1, kmax):
            Lk = 0
            for m in range(k):
                Lmk = 0
                for i in range(1, int(np.fix((N - m) / k))):
                    Lmk = Lmk + np.abs(signal[m + i * k] - signal[m + (i - 1) * k])
                Lk = Lk + Lmk * ((N - 1) / (np.fix((N - m) / k) * k))
            L.append((1 / k) * Lk)

        lnk = [np.log(1 / k) for k in range(1, kmax)]
        popt, _ = curve_fit(self.objective, lnk, L)
        return popt[0]

    def get_feature(self, signal_with_hilbert, signal_without_hilbert, signal_with_hilbert_movavg, feature):
        if self.feature_list['AVG']:
            feature['AVG'].append(np.mean(signal_with_hilbert_movavg))

        if self.feature_list['RMS']:
            e = sum([x ** 2 for x in signal_without_hilbert])
            feature['RMS'].append(1 / len(signal_without_hilbert) * math.sqrt(e))

        if self.feature_list['Max_peak']:
            feature['Max_peak'].append(np.max(signal_with_hilbert))

        if self.feature_list['Variance']:
            feature['Variance'].append(np.var(signal_with_hilbert))

        if self.feature_list['Coastline']:
            feature['Coastline'].append(sum(np.abs(np.diff(signal_with_hilbert))))

        if self.feature_list['Band_powers']:
            [freq, psd] = scipy.signal.periodogram(signal_without_hilbert, fs=self.fs, scaling='density')
            feature['Band_powers'].append(np.mean(psd))

        if self.feature_list['Spectral_edge_frequency']:
            [freq, psd] = scipy.signal.periodogram(signal_without_hilbert, fs=self.fs, scaling='density')
            total_psd = sum(psd)
            find_edge = 0
            i = 0
            while find_edge < 0.9 * total_psd:
                find_edge = find_edge + psd[i]
                i = i + 1
            feature['Spectral_edge_frequency'].append(freq[np.min([i, len(freq) - 1])])

        if self.feature_list['Skewness']:
            feature['Skewness'].append(scipy.stats.skew(signal_with_hilbert))

        if self.feature_list['Kurtosis']:
            feature['Kurtosis'].append(scipy.stats.kurtosis(signal_with_hilbert))

        if self.feature_list['Autocorrelation_function']:
            acf = sm.tsa.stattools.acf(signal_with_hilbert)
            for i in range(2, len(acf)):
                if acf[i] * acf[i - 1] < 0:
                    zc_acf = i
                    break

            # feature['Autocorrelation_function'].append(zc_acf)
            feature['Autocorrelation_function'].append(acf[10])

        if self.feature_list['Hjorth_mobility']:
            data_diff = np.diff(signal_with_hilbert)
            feature['Hjorth_mobility'].append((np.std(data_diff)) / (np.std(signal_with_hilbert)))

        if self.feature_list['Hjorth_complexity']:
            data_diff = np.diff(signal_with_hilbert)
            data_diff2 = np.diff(data_diff)
            mobility = (np.std(data_diff)) / (np.std(signal_with_hilbert))
            feature['Hjorth_complexity'].append((np.std(data_diff2)) / (mobility * np.std(data_diff)))

        if self.feature_list['Nonlinear_energy']:
            feature['Nonlinear_energy'].append(np.mean(np.abs(signal_with_hilbert)))

        if self.feature_list['Spectral_entropy']:
            [freq, psd] = scipy.signal.periodogram(signal_without_hilbert, fs=self.fs, scaling='density')
            p_low = psd / (sum(psd) + +1e-12)
            feature['Spectral_entropy'].append(-1 * sum(p_low * np.log(p_low + 1e-12)))

        if self.feature_list['Sample_entropy']:
            feature['Sample_entropy'].append(self.sampen(signal=signal_without_hilbert,
                                                         m=1,
                                                         r=0.2 * np.std(signal_without_hilbert)))

        if self.feature_list['Renyi_entropy']:
            alpha = 2
            h = Counter(signal_without_hilbert)
            n = 0
            for key in h.keys():
                h[key] = h[key] / signal_without_hilbert.shape[0]
                n = n + h[key] ** alpha

            feature['Renyi_entropy'].append((1 / 1 - alpha) * np.log(n))

        if self.feature_list['Shannon_entropy']:
            h = Counter(signal_without_hilbert)
            n = 0
            for key in h.keys():
                h[key] = h[key] / signal_without_hilbert.shape[0]
                n = n + h[key] * np.log(h[key] + 1e-12)

            feature['Shannon_entropy'].append(-1 * n)

        if self.feature_list['Spikes']:
            peaks, _ = find_peaks(signal_with_hilbert, distance=40)
            feature['Spikes'].append(len(peaks))

        if self.feature_list['Fractal_dimension']:
            slope = self.hfd(signal_with_hilbert, kmax=20)
            feature['Fractal_dimension'].append(slope)

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
                print('patient_', patient, 'from', num_patient)
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
            with open(self.path.path_processed_data + '/feature_all_patient_df_' + self.settings['task'] + '.pkl',
                      'rb') as f:
                self.feature_all_patient = pkl.load(f)

    def create_feature_matrix(self):
        print("\n =================================== \n"
              "convert features from pandas.Dataframe to matrix")
        if self.settings['load_feature_matrix'] is False:
            feature_matrix_all = []
            num_feature = sum(self.feature_list.values())
            for patient in range(len(self.feature_all_patient)):
                feature_matrix = np.zeros(
                    (len(self.channel_names[patient]), len(self.onset_1) + len(self.onset_0), num_feature))
                for electrode in range(len(self.channel_names[patient])):
                    feature_matrix[electrode, :, :] = self.feature_all_patient[patient][electrode].values

                feature_matrix_all.append(feature_matrix)

            if self.settings['save_feature_matrix']:
                with open(self.path.path_save_data + '/feature_matrix_all.pkl', 'wb') as f:
                    pkl.dump(feature_matrix_all, f)
        else:
            with open(self.path.path_processed_data + '/feature_matrix_all_' + self.settings['task'] + '.pkl',
                      'rb') as f:
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
                    fig.savefig(
                        self.path.path_results_get_feature_features[feature_type] + 'patient_' + str(patient) + '.svg')
