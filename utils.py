import os
import json
from pathlib import Path
import datetime


class Paths:
    def __init__(self, settings):
        self.path_store_best_model = ''
        self.path_best_result = ''
        self.settings = settings
        self.path_results = ''
        self.path_error_analysis = ''
        self.path_store_model = ''

    def create_path(self):
        """
        This function creates a path for saving the best models and results
        :param settings: settings of the project
        :returns:
            path_results: the path for saving results
            path_saved_models: the path for saving the trained models
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        base_path = dir_path + '/results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'

        self.path_results = base_path + 'hayper_param_set/'
        self.path_store_model = base_path + 'hayper_param_set/saved_models/'
        Path(self.path_results).mkdir(parents=True, exist_ok=True)
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)

        self.save_settings()

    def save_settings(self):
        """ working directory """
        """ write settings to the json file """
        with open(self.path_store_model + 'settings.json', 'w') as fp:
            json.dump(self.settings, fp)
