import os
import json
from pathlib import Path
import datetime
import numpy as np
from collections import Counter


def moving_avg(signal,window_size):
    i = 0
    avg_win=[]
    while i < len(signal) - window_size + 1:
        win = signal[i:i + window_size]
        avg_win.append(np.mean(win))
        i = i + 1
    return avg_win



def electrode_histogram(channel_names_list, print_analyze):
    all_electrodes_names = []
    for i in range(len(channel_names_list)):
        all_electrodes_names.extend(channel_names_list[i])
    h = Counter(all_electrodes_names)

    elec_more_one = []
    elec_more_ten = []
    elec_more_fifteen = []
    elec_more_tweny = []
    for key in h.keys():
        if h[key] > 1:
            elec_more_one.append(key)
        if h[key] > 10:
            elec_more_ten.append(key)
        if h[key] > 15:
            elec_more_fifteen.append(key)
        if h[key] > 20:
            elec_more_tweny.append(key)
    if print_analyze:
        print('number of unique electrodes is =', len(h), '\n max number of electrode repetition=', 23)
        print('number of shared electrode in more than one patient = ', len(elec_more_one))
        print('number of shared electrode in more than ten patient = ', len(elec_more_ten))
        print('number of shared electrode in more than fifteen patient = ', len(elec_more_fifteen))
        print('number of shared electrode in more than twenty patient = ', len(elec_more_tweny))
        print('\n\n', elec_more_fifteen, 'Electrodes shared between more the 15 patients')
        print('', elec_more_tweny, 'Electrodes shared between more the 20 patients')
    return h, elec_more_fifteen


class Paths:
    def __init__(self, settings):
        self.path_results_classification = None
        self.path_processed_data = ''
        self.path_dataset = ''
        self.path_store_best_model = ''
        self.path_best_result = ''
        self.settings = settings
        self.path_results = ''
        self.path_error_analysis = ''
        self.path_store_model = ''
        self.path_save_data = ''

    def create_path(self, path_dataset, path_processed_data, task, settings):
        """
        This function creates a path for saving the best models and results
        :param settings: settings of the project
        :returns:
            path_results: the path for saving results
            path_saved_models: the path for saving the trained models
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        base_path = dir_path + '/results/' + task + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'

        self.path_processed_data = path_processed_data
        self.path_dataset = path_dataset
        self.path_save_data = dir_path + '/data/'
        self.path_results = base_path + 'plots/'
        self.path_store_model = base_path + 'hayper_param_set/saved_models/'
        Path(self.path_results).mkdir(parents=True, exist_ok=True)
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)
        Path(self.path_processed_data).mkdir(parents=True, exist_ok=True)

        if settings['classification']:
            self.path_results_classification = self.path_results + '/classification/'
            Path(self.path_results_classification).mkdir(parents=True, exist_ok=True)

        self.save_settings()

    def save_settings(self):
        """ working directory """
        """ write settings to the json file """
        with open(self.path_store_model + 'settings.json', 'w') as fp:
            json.dump(self.settings, fp)
