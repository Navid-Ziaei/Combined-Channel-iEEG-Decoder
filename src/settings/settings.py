import yaml
from pathlib import Path


class Settings:
    def __init__(self):
        # Settings
        self.__supported_tasks = ['Singing_Music', 'Question_Answer', 'Speech_Music', 'Move_Rest']
        self.__task = None
        self.__num_patient = None
        self.__balance_sample = None
        self.__fs = None

        # Feature Extraction
        self.__feature_list = {
            'AVG': None,
            'RMS': None,
            'Max_peak': None,
            'Variance': None,
            'Coastline': None,
            'Band_powers': None,
            'Spectral_edge_frequency': None,
            'Skewness': None,
            'Kurtosis': None,
            'Autocorrelation_function': None,
            'Hjorth_mobility': None,
            'Hjorth_complexity': None,
            'Nonlinear_energy': None,
            'Spectral_entropy': None,
            'Sample_entropy': None,
            'Renyi_entropy': None,
            'Shannon_entropy': None,
            'Spikes': None,
            'Fractal_dimension': None
        }
        self.__parameter_get_feature = {
            'num_patient_get_feature': None
        }
        self.__save_feature_matrix = None
        self.__load_feature_matrix = None

        # Balancing and Classification Types
        self.__list_type_balancing = {
            'Without_balancing': None,
            'over_sampling': None
        }
        self.__list_type_classification = {
            'Logistic_regression': None,
            'SVM': None,
            'Naive_bayes': None,
            'XGBoost': None,
            'RandomForest': None
        }

    def load_settings(self, file_path=None):
        """
        Load settings from a YAML file.

        :param file_path: Path to the YAML file. If None, tries to find settings.yaml in configs folder.
        """
        if file_path is None:
            working_folder = Path(__file__).resolve().parents[2]
            file_path = working_folder / 'configs' / 'settings.yaml'

        try:
            with open(file_path, "r") as file:
                settings_yaml = yaml.safe_load(file)
        except Exception as e:
            raise Exception(f'Could not load settings from {file_path}!') from e

        # Load main settings
        self.task = settings_yaml.get('task')
        self.num_patient = settings_yaml.get('num_patient')
        self.balance_sample = settings_yaml.get('balance_sample')
        self.fs = settings_yaml.get('fs')

        # Load feature list
        if 'feature_list' in settings_yaml:
            for feature, value in settings_yaml['feature_list'].items():
                if feature in self.__feature_list:
                    self.__feature_list[feature] = value
                else:
                    raise ValueError(f"Unexpected feature: {feature}")

        # Load parameter get feature
        if 'parameter_get_feature' in settings_yaml:
            for key, value in settings_yaml['parameter_get_feature'].items():
                if key in self.__parameter_get_feature:
                    self.__parameter_get_feature[key] = value
                else:
                    raise ValueError(f"Unexpected parameter: {key}")

        # Load feature matrix settings
        self.save_feature_matrix = settings_yaml.get('save_feature_matrix')
        self.load_feature_matrix = settings_yaml.get('load_feature_matrix')

        # Load balancing types
        if 'list_type_balancing' in settings_yaml:
            for key, value in settings_yaml['list_type_balancing'].items():
                if key in self.__list_type_balancing:
                    self.__list_type_balancing[key] = value
                else:
                    raise ValueError(f"Unexpected balancing type: {key}")

        # Load classification types
        if 'list_type_classification' in settings_yaml:
            for key, value in settings_yaml['list_type_classification'].items():
                if key in self.__list_type_classification:
                    self.__list_type_classification[key] = value
                else:
                    raise ValueError(f"Unexpected classification type: {key}")

    def save_settings(self, file_path):
        """
        Save all settings to a YAML file.

        :param file_path: Path where the YAML file will be saved.
        """
        settings_dict = {
            'task': self.__task,
            'num_patient': self.__num_patient,
            'balance_sample': self.__balance_sample,
            'fs': self.__fs,
            'feature_list': self.__feature_list,
            'parameter_get_feature': self.__parameter_get_feature,
            'save_feature_matrix': self.__save_feature_matrix,
            'load_feature_matrix': self.__load_feature_matrix,
            'list_type_balancing': self.__list_type_balancing,
            'list_type_classification': self.__list_type_classification
        }

        with open(file_path, 'w') as yaml_file:
            yaml.dump(settings_dict, yaml_file, default_flow_style=False)

    # Property setters and getters
    @property
    def task(self):
        return self.__task

    @task.setter
    def task(self, value):
        if value in self.__supported_tasks:
            self.__task = value
        else:
            raise ValueError(f"Task should be one of {self.__supported_tasks}")

    @property
    def num_patient(self):
        return self.__num_patient

    @num_patient.setter
    def num_patient(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_patient = value
        else:
            raise ValueError("num_patient must be a positive integer")

    @property
    def balance_sample(self):
        return self.__balance_sample

    @balance_sample.setter
    def balance_sample(self, value):
        if isinstance(value, bool):
            self.__balance_sample = value
        else:
            raise ValueError("balance_sample must be a boolean")

    @property
    def fs(self):
        return self.__fs

    @fs.setter
    def fs(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self.__fs = value
        else:
            raise ValueError("fs must be a positive number")

    @property
    def feature_list(self):
        return self.__feature_list

    @property
    def parameter_get_feature(self):
        return self.__parameter_get_feature

    @property
    def save_feature_matrix(self):
        return self.__save_feature_matrix

    @save_feature_matrix.setter
    def save_feature_matrix(self, value):
        if isinstance(value, bool):
            self.__save_feature_matrix = value
        else:
            raise ValueError("save_feature_matrix must be a boolean")

    @property
    def load_feature_matrix(self):
        return self.__load_feature_matrix

    @load_feature_matrix.setter
    def load_feature_matrix(self, value):
        if isinstance(value, bool):
            self.__load_feature_matrix = value
        else:
            raise ValueError("load_feature_matrix must be a boolean")

    @property
    def list_type_balancing(self):
        return self.__list_type_balancing

    @property
    def list_type_classification(self):
        return self.__list_type_classification
