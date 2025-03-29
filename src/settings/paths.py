import os
import datetime
import yaml
from pathlib import Path


class Paths:
    def __init__(self, settings):
        self.settings = settings
        self.path_dataset = ''
        self.path_processed_data = ''
        self.path_save_data = ''
        self.path_results = ''
        self.path_store_model = ''
        self.path_results_classifier = {}
        self.path_results_ensemble_classifier = {}

    def load_device_paths(self):
        """
        Load path_dataset and path_processed_data from device_path.yaml.
        """
        working_folder = Path(__file__).resolve().parents[2]
        config_folder = working_folder / 'configs'

        try:
            with open(config_folder / "device_path.yaml", "r") as file:
                device = yaml.safe_load(file)
        except Exception as e:
            raise Exception('Could not load device_path.yaml from the working directory!') from e

        for key, value in device.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Paths class!'.format(key))

        self.create_path(self.path_dataset, self.path_processed_data, self.settings)

    def create_path(self, path_dataset, path_processed_data, settings):
        """
        Creates necessary directories for saving results and models.
        """
        dir_path = Path(__file__).resolve().parents[2]
        base_path = os.path.join(dir_path, 'results', settings.task,
                                 datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

        self.path_dataset = path_dataset
        self.path_processed_data = path_processed_data
        Path(self.path_processed_data).mkdir(parents=True, exist_ok=True)

        self.path_save_data = os.path.join(base_path, 'data')
        Path(self.path_save_data).mkdir(parents=True, exist_ok=True)

        self.path_store_model = os.path.join(base_path, 'hyper_param_set', 'saved_models')
        Path(self.path_store_model).mkdir(parents=True, exist_ok=True)

        for type_classification, is_enabled_cls in settings.list_type_classification.items():
            for type_balancing, is_enabled_bal in settings.list_type_balancing.items():
                if is_enabled_cls and is_enabled_bal:
                    key = f'{type_classification}{type_balancing}'
                    self.path_results_classifier[key] = os.path.join(base_path,
                                                                     f'{type_classification}_{type_balancing}/')
                    Path(self.path_results_classifier[key]).mkdir(parents=True, exist_ok=True)

                    self.path_results_ensemble_classifier[key] = os.path.join(self.path_results_classifier[key],
                                                                              'ensemble_classifier/')
                    Path(self.path_results_ensemble_classifier[key]).mkdir(parents=True, exist_ok=True)
