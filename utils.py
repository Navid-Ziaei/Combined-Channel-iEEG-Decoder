import os
import json
from pathlib import Path
import datetime
import numpy as np
from collections import Counter
import pandas as pd


def time_ann(path):
    r = pd.read_csv(path, sep=";")
    onset = []
    offset = []
    for i in range(len(r.index)):
        d = r.iloc[i, 0]
        pos1 = d.find('\t')
        pos2 = d.rfind('\t')
        onset.append(eval(d[pos1 + 1:pos2]))
        offset.append(eval(d[pos2 + 1:]))
    return onset, offset


def read_time(task, t_min, paths):
    if task == 'question&answer':
        onset_1, offset_1 = time_ann(
            path=paths.path_dataset + "/stimuli/annotations/sound/sound_annotation_questions.tsv")
        onset_0, offset_0 = time_ann(
            path=paths.path_dataset + "/stimuli/annotations/sound/sound_annotation_sentences.tsv")

        # remove onset of question from onset of answer
        onset_1_int = [int(x) for x in onset_1]
        offset_1_int = [int(x) for x in offset_1]

        for i in onset_0:
            if int(i) in onset_1_int:
                onset_0.remove(i)

        for i in onset_0:
            if i in onset_1:
                onset_0.remove(i)

        for i in offset_0:
            if int(i) in offset_1_int:
                offset_0.remove(i)

        for i in offset_0:
            if i in offset_1:
                offset_0.remove(i)

    if task == 'speech&music':
        onset_1 = [i for i in np.arange(0, 390, 60)]
        offset_1 = [i for i in np.arange(30, 420, 60)]
        onset_0 = [i for i in np.arange(30, 390, 60)]
        offset_0 = [i for i in np.arange(60, 390, 60)]
        onset_1[0] = onset_1[0] + t_min

    return onset_1, offset_1, onset_0, offset_0


def moving_avg(signal, window_size):
    i = 0
    avg_win = []
    while i < len(signal) - window_size + 1:
        win = signal[i:i + window_size]
        avg_win.append(np.mean(win))
        i = i + 1
    return avg_win


def electrode_histogram(channel_names_list, print_analyze):
    print("\n =================================== \n"
          "analyzing histogram of electrodes")
    all_electrodes_names = []
    for i in range(len(channel_names_list)):
        all_electrodes_names.extend(channel_names_list[i])
    h = Counter(all_electrodes_names)

    elec_more_one = []
    elec_more_ten = []
    elec_more_fifteen = []
    elec_more_twenty = []
    for key in h.keys():
        if h[key] > 1:
            elec_more_one.append(key)
        if h[key] > 10:
            elec_more_ten.append(key)
        if h[key] > 15:
            elec_more_fifteen.append(key)
        if h[key] > 20:
            elec_more_twenty.append(key)
    if print_analyze:
        print('number of unique electrodes is =', len(h), '\n max number of electrode repetition=', 23)
        print('number of shared electrode in more than one patient = ', len(elec_more_one))
        print('number of shared electrode in more than ten patient = ', len(elec_more_ten))
        print('number of shared electrode in more than fifteen patient = ', len(elec_more_fifteen))
        print('number of shared electrode in more than twenty patient = ', len(elec_more_twenty))
        print('\n\n', elec_more_fifteen, 'Electrodes shared between more the 15 patients')
        print('', elec_more_twenty, 'Electrodes shared between more the 20 patients')
    return h, elec_more_fifteen


class Paths:
    def __init__(self, settings):
        self.path_results_plot_common_electrodes = None
        self.path_results_temporal_signal = None
        self.path_results_synchronous_average = {}
        self.path_results_synch_average_common_electrode = None
        self.path_results_get_feature_features = {}
        self.path_results_classification = None
        self.path_results_classifier = {}
        self.path_results_get_pca = {}
        self.path_results_wavelet = None
        self.path_processed_data = ''
        self.path_dataset = ''
        self.path_store_best_model = ''
        self.path_best_result = ''
        self.settings = settings
        self.path_results = ''
        self.path_error_analysis = ''
        self.path_store_model = ''
        self.path_save_data = ''

    def create_path(self, path_dataset, path_processed_data, settings):
        """
        This function creates a path for saving the best models and results
        :param settings: settings of the project
        :param path_dataset:
        :param path_processed_data:
        :returns:
            path_results: the path for saving results'
            path_saved_models: the path for saving the trained models
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        base_path = dir_path + '/results/' + settings['task'] + '/' + datetime.datetime.now().strftime(
            '%Y-%m-%d-%H-%M-%S') + '/'

        self.path_dataset = path_dataset
        # Place where we save preprocessed data
        self.path_processed_data = path_processed_data
        Path(self.path_processed_data).mkdir(parents=True, exist_ok=True)
        # Place where we save features
        self.path_save_data = dir_path + '/data/'
        Path(self.path_save_data).mkdir(parents=True, exist_ok=True)
        # Place where we save figures
        self.path_results = base_path + 'plots/'
        Path(self.path_results).mkdir(parents=True, exist_ok=True)
        # Place where we save model hyperparameters
        self.path_store_model = base_path + 'hyper_param_set/saved_models/'
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)
        


        if settings['plot_common_electrodes_sync_average']:
            self.path_results_plot_common_electrodes = self.path_results + '/plot_common_electrodes/'
            Path(self.path_results_plot_common_electrodes).mkdir(parents=True, exist_ok=True)

        if settings['temporal_signal']:
            self.path_results_temporal_signal = self.path_results + '/temporal_signal/'
            Path(self.path_results_temporal_signal).mkdir(parents=True, exist_ok=True)

        if settings['synchronous_average']:
            path_results_synch_avg = self.path_results + '/synchronous_average/'
            Path(path_results_synch_avg).mkdir(parents=True, exist_ok=True)
            for patient in range(settings['parameter_synchronous_average']['num_patient']):
                self.path_results_synchronous_average[patient] = path_results_synch_avg + 'patient_' + str(
                    patient) + '/'
                Path(self.path_results_synchronous_average[patient]).mkdir(parents=True, exist_ok=True)

            self.path_results_synch_average_common_electrode = path_results_synch_avg + '/patient_common_electrode/'
            Path(self.path_results_synch_average_common_electrode).mkdir(parents=True, exist_ok=True)

        if settings['plot_class_conditional_average']:
            path_results_get_feature = self.path_results + '/features/'
            Path(path_results_get_feature).mkdir(parents=True, exist_ok=True)

            for feature_name in settings['feature_list'].keys():
                if settings['feature_list'][feature_name]:
                    self.path_results_get_feature_features[feature_name] = path_results_get_feature + feature_name + '/'
                    Path(self.path_results_get_feature_features[feature_name]).mkdir(parents=True, exist_ok=True)

        if settings['classification']:
            self.path_results_classification = self.path_results + '/classification/'
            Path(self.path_results_classification).mkdir(parents=True, exist_ok=True)

            for type_classification in settings['list_type_classification'].keys():
                for type_balancing in settings['list_type_balancing'].keys():
                    if settings['list_type_classification'][type_classification] \
                            & settings['list_type_balancing'][type_balancing]:
                        self.path_results_classifier[type_classification + type_balancing] = \
                            self.path_results_classification + '/' + type_classification + '_' + type_balancing + '/'
                        Path(self.path_results_classifier[type_classification + type_balancing]).mkdir(parents=True,
                                                                                                       exist_ok=True)

        if settings['get_pca']:
            path_results_pca = self.path_results + '/PCA/'
            Path(path_results_pca).mkdir(parents=True, exist_ok=True)
            for patient in range(settings['parameter_get_pca']['num_patient']):
                self.path_results_get_pca[patient] = path_results_pca + 'patient_' + str(
                    patient) + '/'
                Path(self.path_results_get_pca[patient]).mkdir(parents=True, exist_ok=True)

        if settings['wavelet']:
            self.path_results_wavelet = self.path_results + '/wavelet/'
            Path(self.path_results_wavelet).mkdir(parents=True, exist_ok=True)

        self.save_settings()

    def save_settings(self):
        """ working directory """
        """ write settings to the json file """
        with open(self.path_store_model + 'settings.json', 'w') as fp:
            json.dump(self.settings, fp)
