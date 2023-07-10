from utils import *
from visualiztion_utils import *
from model import *
from import_data import *

# set device
device = 'system_lab'
if device.lower() == 'navid':
    dataset_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
    processed_data_path = 'F:/maryam_sh/load_data/'
elif device.lower() == 'maryam':
    dataset_path = 'E:/Thesis/dataset/dataset/'
    processed_data_path = 'E:/Thesis/derived data/'
elif device.lower() == 'system_lab':
    dataset_path = 'F:/maryam_sh/dataset/'
    processed_data_path = 'F:/maryam_sh/load_data/'
elif device.lower() == 'navid_lab':
    dataset_path = 'D:/Navid/Dataset/AudioVisualiEEG/'
    processed_data_path = 'D:/Navid/Dataset/AudioVisualiEEG/processed_data/'
else:
    dataset_path = ''
    processed_data_path = ''

# settings
# first determine which task you want
settings = {
    'band': 'gamma',  # Specify frequency band
    'task': 'question&answer',  # task : 'speech&music' , 'question&answer'
    'generate_electrode_histogram': True,  # Get histogram of electrodes of patients
    'print_analyze_electrode_histogram': True,
    'fs': 25,  # Sampling frequency
    'final_time': 150,
    # Get the synchronous average for common average between 15 patient
    # Notice that it just use for 'speech&music' task
    'plot_common_electrodes_sync_average': False,
    'temporal_signal': False,  # Plot temporal signal of one patient
    'parameter_plot_temporal_signal': {'patient': 2,
                                       'electrode': 'T13'},
    # for task:'speech&music', step=29.5 , for task:'question&answer', step=2.5
    # Get the synchronous average between trials for all electrode of each patient
    # Notice that 't_min'+'step' must be integer
    'synchronous_average': False,
    'parameter_synchronous_average': {'num_patient': 3,
                                      't_min': 0.5,
                                      'step': 2.5},
    # Plot wavelet of raw data of signal each patient
    'wavelet': False,
    'parameter_wavelet': {'patient': 2,
                          'electrode': 'T13'},
    # Get feature and visualize features for all electrode of each patient
    'get_feature': True,
    'feature_list': {'AVG': True, 'RMS': True, 'Max_peak': True, 'Variance': True, 'Coastline': True,
                     'Band_powers': True, 'Spectral_edge_frequency': True, 'Skewness': True, 'Kurtosis': True,
                     'Autocorrelation_function': True, 'Hjorth_mobility': True, 'Hjorth_complexity': True,
                     'Nonlinear_energy': True, 'Spectral_entropy': True, 'Sample_entropy': True, 'Renyi_entropy': True,
                     'Shannon_entropy': True, 'Spikes': True, 'Fractal_dimension': True},

    'plot_class_conditional_average': False,
    # for task:'speech&music', step=29.5 , for task:'question&answer', step=2.5
    # for task:'speech&music', window_size=200 , for task:'question&answer', window_size=20
    # Notice that 't_min'+'step' must be integer
    'parameter_get_feature': {'num_patient_get_feature': 47,
                              'num_patient_plot_class_conditional_average': 2,
                              'window_size': 20,
                              't_min': 0.5,
                              'step': 2.5},
    'save_feature_matrix': True,
    'load_feature_matrix': True,
    # Model
    # Specify type_classification : 'Logistic_regression' or 'SVM' or 'Naive_bayes'
    # Specify type_balancing :  'over_sampling' or 'under_sampling'  or 'over&down_sampling' or 'weighted_losfunc'
    # Notice that for classification Naive_bayes don't use 'weighted_losfunc' way for balancing
    'classification': True,
    'list_type_balancing': {'over_sampling': True,
                            'under_sampling': False,
                            'over&down_sampling': False},
    'list_type_classification': {'Logistic_regression': False,
                                 'SVM': True,
                                 'Naive_bayes': False},
    'parameter_classification': {'num_patient': 47},
    # Get Principal Component Analysis
    'get_pca': True,
    'parameter_get_pca': {'num_patient': 10}
}

load_data_settings = {
    'number_of_patients': 63,
    # if 'load_preprocessed_data':False, function create preprocessed_data, else it just load data
    'load_preprocessed_data': True,
    'save_preprocessed_data': True
}

# Create data_path, save_paths and load_paths
paths = Paths(settings)
paths.create_path(path_dataset=dataset_path,
                  path_processed_data=processed_data_path,
                  settings=settings)

" -------------------------------------------------  Load data -------------------------------------------------"
# Load and preprocess data
band_all_patient_with_hilbert, band_all_patient_without_hilbert, channel_names_list = \
    get_data(paths, settings=load_data_settings)

" ------------------------------------------------- analyze data -------------------------------------------------"
# Find the number of common electrodes between patients
if settings['generate_electrode_histogram']:
    hist, share_electrodes_15p = electrode_histogram(channel_names_list,
                                                     print_analyze=settings['print_analyze_electrode_histogram'])

# find onset & offset of trial for specific task
# if task is 'speech&music' : (onset_1,offset_1) refer to music / (onset_0,offset_0) refer to speech
# if task is 'question&answer' : (onset_1,offset_1) refer to question / (onset_0,offset_0) refer to answer
onset_1, offset_1, onset_0, offset_0 = read_time(task=settings['task'],
                                                 t_min=settings['parameter_synchronous_average']['t_min'],
                                                 paths=paths)

" -------------------------------------------------  analyze signal -------------------------------------------------"

# Get the average for common average between 15 patient
if settings['plot_common_electrodes_sync_average']:
    common_electrode_avg = plot_comm_elec(common_electrodes=share_electrodes_15p,
                                          band_all_patient=band_all_patient_with_hilbert,
                                          channel_names_list=channel_names_list,
                                          onset_1=onset_1,
                                          offset_1=offset_1,
                                          path=paths,
                                          settings=settings)

if settings['temporal_signal']:
    plot_temporal_signal(channel_names_list=channel_names_list,
                         band_all_patient=band_all_patient_with_hilbert,
                         onset_1=onset_1,
                         offset_1=offset_1,
                         path=paths,
                         settings=settings)

# Get the synchronous average between trials for all electrode of each patient
if settings['synchronous_average']:
    synch_avg = SynchronousAvg(channel_names_list=channel_names_list,
                               band_all_patient=band_all_patient_with_hilbert,
                               onset_1=onset_1,
                               onset_0=onset_0,
                               path=paths,
                               settings=settings)
    # calculate and plot synch_avg for all electrodes of all patients
    synch_avg.calculate_synchronous_avg(num_patient=settings['parameter_synchronous_average']['num_patient'])
    # calculate and plot synch_avg for patients have common electrode
    synch_avg.calculate_synch_avg_common_electrode(common_electrodes=share_electrodes_15p)

if settings['wavelet']:
    raw_car_all = load_raw_data(paths, load_data_settings)
    plot_wavelet(path=paths,
                 raw_car_all=raw_car_all,
                 settings=settings)

" ------------------------------------------------- Feature extraction ---------------------------------------------"
if settings['get_feature']:
    class_viz = QAVisualizer(channel_names=channel_names_list,
                             onset_1=onset_1,
                             onset_0=onset_0,
                             path=paths,
                             settings=settings)
    class_viz.get_feature_all(data_with_hilbert=band_all_patient_with_hilbert,
                              data_without_hilbert=band_all_patient_without_hilbert)

    feature_all_matrix = class_viz.create_feature_matrix()

    if settings['plot_class_conditional_average'] and settings['load_feature_matrix'] is False:
        class_viz.plot_class_conditional_average()

" ------------------------------------------------- Classification ---------------------------------------------"

if settings['classification']:
    # type_classification : 'log_reg' or 'SVM' or 'Naive_bayes'
    # type_balancing :  'over_sampling' or 'under_sampling'  or 'over&down_sampling' or 'weighted_losfunc'
    # notice that for classification Naive_bayes don't use 'weighted_losfunc' way for balancing
    model = ModelSinglePatient(feature_matrix=feature_all_matrix,
                               path=paths,
                               settings=settings,
                               channel_names_list=channel_names_list)
    model.create_model()

" ------------------------------------------------- PCA ---------------------------------------------"

if settings['get_pca']:
    plot_pca(channel_names_list=channel_names_list,
             path=paths,
             feature_matrix=feature_all_matrix,
             settings=settings)

" ------------------------------------------------- The end :)) ---------------------------------------------"
