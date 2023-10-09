import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from multitaper_spectrogram_python import multitaper_spectrogram
import numpy as np
from collections import Counter

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
        plt.figure(figsize=(30, 10), dpi=300)
        plt.plot(time[:final_time * fs], common_electrode_avg[key][:final_time * fs])
        plt.vlines(time_dash2, ymin=-1 * np.ones(len(time_dash2)), ymax=np.ones(len(time_dash2)), colors='red', ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.title(str(key), fontsize=25)
        plt.savefig(path.path_results_plot_common_electrodes + key)
        plt.savefig(path.path_results_plot_common_electrodes + key + '.svg')

        # Save data of plot
        data = np.column_stack((time[:final_time * fs], common_electrode_avg[key][:final_time * fs]))
        np.save(path.path_results_plot_common_electrodes + key + '.npy', data)

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

        plt.figure(figsize=(80, 10), dpi=300)
        plt.plot(time[:final_time * fs], signal[:final_time * fs])
        plt.vlines(time_dash2, ymin=-4 * np.ones(len(time_dash2)), ymax=4 * np.ones(len(time_dash2)), colors='red',
                   ls='--',
                   lw=2, label='vline_multiple - partial height')
        plt.vlines(time_dash_music, ymin=-4 * np.ones(len(time_dash_music)), ymax=8 * np.ones(len(time_dash_music)),
                   colors='black', ls='--', lw=2, label='vline_multiple - partial height')

        plt.xlabel('time', fontsize=15)
        plt.ylabel('temporal_signal', fontsize=15)
        plt.title('temporal signal of patient=' + str(patient) + '_electrode =' + electrode, fontsize=15)
        plt.savefig(path.path_results_temporal_signal + "temporal_signal")
        plt.savefig(path.path_results_temporal_signal + "temporal_signal.svg")

        # Save data of plot
        data = np.column_stack((time[:final_time * fs], signal[:final_time * fs]))
        np.save(path.path_results_temporal_signal + 'temporal_signal' + '.npy', data)

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
        self.time = np.arange(0, self.step + self.t_min, 1 / self.fs)
        self.task = settings['task']

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
        ci_q = self.calculate_confidence_interval(trial_q, 0.95)

        trial_a = np.zeros((len(self.onset_0), self.time.shape[0]))
        for i in range(len(self.onset_0)):
            start_sample = int(self.onset_0[i] - self.t_min) * self.fs
            end_sample = int(self.onset_0[i] + self.step) * self.fs
            trial_a[i, :] = signal[start_sample:end_sample]
        synch_avg_a = np.mean(trial_a, axis=0)
        ci_a = self.calculate_confidence_interval(trial_a, 0.95)

        return ci_q, ci_a, synch_avg_q, synch_avg_a

    def calculate_synchronous_avg(self, num_patient):
        for patient in range(num_patient):
            electrodes = self.channel_names_list[patient]
            for electrode in electrodes:
                num_electrode = self.channel_names_list[patient].index(electrode)
                signal = self.band_all_patient[patient][self.freq_band][:, num_electrode]

                ci_q, ci_a, synch_avg_q, synch_avg_a = self.synch_ci(signal)
                ci_l_q = synch_avg_q - ci_q
                ci_h_q = synch_avg_q + ci_q
                ci_l_a = synch_avg_a - ci_a
                ci_h_a = synch_avg_a + ci_a

                plt.figure(dpi=300)
                if self.task == 'question&answer':
                    label_1 = 'question'
                    label_0 = 'answer'
                if self.task == 'speech&music':
                    label_1 = 'music'
                    label_0 = 'speech'
                plt.plot(self.time, synch_avg_q, color='blue', linewidth=2, label=label_1)
                plt.fill_between(self.time, ci_l_q, ci_h_q, color='blue', alpha=0.2)
                plt.plot(self.time, synch_avg_a, color='red', linewidth=2, label=label_0)
                plt.fill_between(self.time, ci_l_a, ci_h_a, color='red', alpha=0.2)
                plt.title('patient=' + str(patient) + '  electrode=' + electrode, fontsize=15)
                plt.legend()
                plt.savefig(
                    self.path.path_results_synchronous_average[patient] + 'p_' + str(patient) + '_elec_' + electrode)
                plt.savefig(
                    self.path.path_results_synchronous_average[patient] + 'p_' + str(
                        patient) + '_elec_' + electrode + '.svg')

                # Save data of plot
                data = np.column_stack((self.time, synch_avg_a, synch_avg_q, ci_a, ci_q))
                np.save(self.path.path_results_synchronous_average[patient] + 'p_' + str(patient) + '_elec_' + electrode
                        + '.npy', data)

    def calculate_synch_avg_common_electrode(self, common_electrodes):
        for key in common_electrodes:
            j = 0
            plt.figure(figsize=(5, 20), dpi=300)
            if self.task == 'question&answer':
                label_legend = ['question', 'answer']
            if self.task == 'speech&music':
                label_legend = ['music', 'speech']
            for patient in range(len(self.band_all_patient)):
                electrodes = self.channel_names_list[patient]
                if key in electrodes:
                    num_electrode = self.channel_names_list[patient].index(key)
                    signal = self.band_all_patient[patient][self.freq_band][:, num_electrode]
                    ci_q, ci_a, synch_avg_q, synch_avg_a = self.synch_ci(signal)
                    ci_l_q = synch_avg_q - ci_q
                    ci_h_q = synch_avg_q + ci_q
                    ci_l_a = synch_avg_a - ci_a
                    ci_h_a = synch_avg_a + ci_a
                    plt.plot(self.time, synch_avg_q + j, color='blue', linewidth=2)
                    plt.fill_between(self.time, ci_l_q + j, ci_h_q + j, color='blue', alpha=0.2)
                    plt.plot(self.time, synch_avg_a + j, color='red', linewidth=2)
                    plt.fill_between(self.time, ci_l_a + j, ci_h_a + j, color='red', alpha=0.2)
                    plt.title('patient_' + str(patient))
                    j = j + 3
            plt.legend(label_legend)
            plt.title('electrode_' + key)
            plt.savefig(self.path.path_results_synch_average_common_electrode + 'elec_' + key)
            plt.savefig(self.path.path_results_synch_average_common_electrode + 'elec_' + key + '.svg')


def plot_pca(channel_names_list, path, feature_matrix, settings):
    print("\n =================================== \n"
          "plot PCA for each patient")
    num_patient = settings['parameter_get_pca']['num_patient']
    if settings['task'] == 'question&answer':
        label = np.zeros(67)
        label[0:15] = 1
        label_legend = ['question', 'answer']
    if settings['task'] == 'speech&music':
        label = np.zeros(13)
        label[0:6] = 1
        label_legend = ['music', 'speech']
    counter = Counter(label)

    for patient in range(num_patient):
        ch_name = channel_names_list[patient]
        for electrode in range(len(ch_name)):
            feature = feature_matrix[patient][electrode, :, :]
            pca = PCA(n_components=2)
            x = pca.fit_transform(feature)
            plt.figure(dpi=300)
            for label2, _ in counter.items():
                row_ix = np.where(label == label2)[0]
                plt.scatter(x[row_ix, 0], x[row_ix, 1], label=str(label2))
                plt.legend(label_legend)
            plt.title('patient=' + str(patient) + '_electrode=' + ch_name[electrode], fontsize=15)
            plt.savefig(path.path_results_get_pca[patient] + 'p_' + str(patient) + '_electrode=' + ch_name[electrode])
            plt.savefig(
                path.path_results_get_pca[patient] + 'p_' + str(patient) + '_electrode=' + ch_name[electrode] + '.svg')


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
    fig.savefig(path.path_results_wavelet + 'wavelet_patient' + str(patient) + '_electrode' + electrode + '.svg')


def bar_plot_mean_patient(settings, path):
    color_list = ['orangered', 'yellow', 'cyan', 'deeppink', 'lime', 'steelblue', 'purple', 'pink', 'darkgray']
    type_ensemble = 'Max_voting'
    barWidth = 0.5
    bar_pos = 3.5
    num_patient=settings['bar_plot_mean_patient']['num_patient_avg']
    j = 0
    plt.figure(dpi=300)
    for type_balancing in settings['list_type_balancing']:
        for type_classification in settings['list_type_classification']:
            if settings['list_type_classification'][type_classification] & settings['list_type_balancing'][type_balancing]:
                data_ensemble = np.load(path.path_results_classifier[
                                            type_classification + type_balancing + type_ensemble] + 'f_measure_all_ensemble.npy')
                pos = np.argsort(data_ensemble)[-1*num_patient:]
                data = np.load(path.path_results_classifier[
                                   type_classification + type_balancing + type_ensemble] + 'max_performance_all.npy')
                j = j + 1
                mean = np.mean([data[i] for i in pos])
                error = np.std([data[i] for i in pos])
                mean_ensemble = np.mean([data_ensemble[i] for i in pos])
                bar_pos=bar_pos+barWidth
                br = plt.bar(bar_pos, mean, color=color_list[j-1], yerr=error, capsize=2, width=barWidth,
                             edgecolor='gray', label=type_classification + type_balancing, alpha=0.7)
                for bar in br:
                    plt.text(bar.get_x() + bar.get_width() / 2, mean_ensemble, '*', ha='center', va='bottom',
                             fontsize='20', color='red')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks([r + barWidth for r in 10 * np.arange(2) + 2.25], ['', ''])
    plt.legend(loc='upper right', fontsize='5')
    plt.ylabel('Accuracy')
    plt.title('mean performance of single_channel across the ' +str(num_patient)+ ' best patients', fontsize='7')
    plt.savefig(path.path_results_bar_plot+'average_patients')

def bar_plot_best_electrode(settings, paths):
    mean = {}
    error = {}
    barWidth = 20
    type_ensemble = 'Max_voting'
    num_elec=settings['bar_plot_best_electrode']['num_best_electrode']
    num_patient=settings['bar_plot_best_electrode']['num_best_patient']
    br = np.arange(num_elec) * barWidth
    path = paths.path_results_classifier[settings['bar_plot_best_electrode']['type_classification'] +
                                        settings['bar_plot_best_electrode']['type_balancing'] + type_ensemble]
    color = ['deeppink', 'slategray', 'maroon', 'lime', 'royalblue', 'tomato']
    data = np.load(path + 'f_measure_all_ensemble.npy')
    pos_best_patient = np.argsort(data)[-1*num_patient:]
    fig, ax = plt.subplots(figsize=(50, 20), dpi=300)
    k = 0
    for patient in pos_best_patient:
        br = br + 200
        data_patient = np.load(path + 'patient_' + str(patient) + '.npy')
        pos_best_electrode = np.argsort(data_patient[:, 1])[-1*num_elec:]
        for i in range(num_elec):
            bar1 = ax.bar(br[i], float(data_patient[pos_best_electrode[i], 1]), color=color[i],
                          yerr=float(data_patient[pos_best_electrode[i], 2]), capsize=10, width=barWidth,
                          edgecolor='grey')
            for bar in bar1:
                ax.text(bar.get_x() + bar.get_width() / 2, 0, data_patient[pos_best_electrode[i], 0], ha='center',
                        va='bottom', fontsize='20')
    plt.xticks([r + barWidth for r in (200 * (np.arange(5) + 1.15))],
               ['P' + str(pos_best_patient[0]), 'P' + str(pos_best_patient[1]),
                'P' + str(pos_best_patient[2]), 'P' + str(pos_best_patient[3]),
                'P' + str(pos_best_patient[4])], fontsize=40)

    plt.yticks(fontsize=40)
    plt.ylabel('Accuracy', fontsize=40)
    plt.xlabel('patient_id', fontsize=40)
    plt.title(' performance of the'+ str(num_elec)+ 'best channel for'+str(num_patient)+ 'best patients', fontsize='40')
    plt.savefig(paths.path_results_bar_plot+'best_elec_best_patients')