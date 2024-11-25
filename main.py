from utils import *
from src.data import load_data
from model import *
from import_data import *
from feature_extraction import *

# set device
device = 'system_lab'
if device.lower() == 'navid':
    dataset_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
    processed_data_path = 'F:/maryam_sh/load_data/'
elif device.lower() == 'maryam':
    dataset_path = 'E:/Thesis/dataset/dataset/'
    processed_data_path = 'E:/Thesis/derived data/'
elif device.lower() == 'system_lab':
    dataset_path = 'F:/maryam_sh/Datasets/Raw_data/Audio_visual/'
    processed_data_path = 'F:/maryam_sh/Datasets/Raw_data/'
elif device.lower() == 'navid_lab':
    dataset_path = 'D:/Navid/Dataset/AudioVisualiEEG/'
    processed_data_path = 'D:/Navid/Dataset/AudioVisualiEEG/processed_data/'
else:
    dataset_path = ''
    processed_data_path = ''

# settings
# first determine which task you want
settings = {
    # task : 'Speech_Music' , 'Question_Answer' for dataset: 'audio_visual'
    # task : 'Singing_Music' for dataset: 'music_reconstruction'
    # task : 'Move_Rest' for dataset: 'upper_limb_movement'
    'task': 'Singing_Music',
    'num_patient': 29,
    # for task :'Question_Answer', 'singing&music'
    'balance_sample': False,
    'generate_electrode_histogram': False,
    # Get histogram of electrodes of patients
    'print_analyze_electrode_histogram': False,
    # Sampling frequency for dataset: 'audio_visual':25Hz,dataset: 'music_reconstruction':100Hz
    'fs': 250,
    # Get feature and visualize features for all electrode of each patient
    # feature 'Sample_entropy' must be False for task:'Speech_Music'
    'feature_list': {'AVG': True, 'RMS': True, 'Max_peak': True, 'Variance': True, 'Coastline': True,
                     'Band_powers': True, 'Spectral_edge_frequency': True, 'Skewness': True, 'Kurtosis': True,
                     'Autocorrelation_function': True, 'Hjorth_mobility': True, 'Hjorth_complexity': True,
                     'Nonlinear_energy': True, 'Spectral_entropy': True, 'Sample_entropy': False, 'Renyi_entropy': True,
                     'Shannon_entropy': True, 'Spikes': True, 'Fractal_dimension': True},

    # for task:'Speech_Music', step=29.5 , for task:'Question_Answer', step=2.5
    # for task:'Speech_Music', window_size=20 , for task:'Question_Answer', window_size=20
    # Notice that 't_min'+'step' must be integer
    # number of patient for dataset: 'audio_visual':51, for dataset: 'music_reconstruction':29
    'parameter_get_feature': {'num_patient_get_feature': 12},
    'save_feature_matrix': True,
    'load_feature_matrix': True,
    # Model
    # Specify type_classification : 'Logistic_regression' or 'SVM' or 'Naive_bayes'
    # Specify type_balancing :  'over_sampling' or 'under_sampling'  or 'over&down_sampling' or 'weighted_losfunc'
    # Notice that for classification Naive_bayes don't use 'weighted_losfunc' way for balancing
    'list_type_balancing': {'Without_balancing': False,
                            'over_sampling': True},
    'list_type_classification': {'Logistic_regression': True,
                                 'SVM': True,
                                 'Naive_bayes': True,
                                 'XGBoost': True,
                                 'RandomForest': True},
}

# Create data_path, save_paths and load_paths
paths = Paths(settings)
paths.create_path(path_dataset=dataset_path,
                  path_processed_data=processed_data_path,
                  settings=settings)

# Load and preprocess data
data_all_patient, channel_names_list, labels = load_data(settings, paths)

feature_ex = FeatureExtractor(channel_names=channel_names_list,
                              path=paths,
                              settings=settings)

feature_ex.get_feature_all(data_with_hilbert=band_all_patient_with_hilbert,
                           data_without_hilbert=band_all_patient_without_hilbert)

feature_all_matrix = feature_ex.create_feature_matrix()

# type_classification : 'log_reg' or 'SVM' or 'Naive_bayes'
# type_balancing :  'over_sampling' or 'under_sampling'  or 'over&down_sampling' or 'weighted_losfunc'
# notice that for classification Naive_bayes don't use 'weighted_losfunc' way for balancing
model = ModelSinglePatient(feature_matrix=feature_all_matrix,
                           path=paths,
                           settings=settings,
                           channel_names_list=channel_names_list)
model.create_model()
