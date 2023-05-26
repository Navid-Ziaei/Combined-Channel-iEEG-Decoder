from utils import *
from visualiztion_utils import *
from model import *
from ieeg_func.analyze_mean_signal import analyze_signal_mean_patient
from ieeg_func.plot_wavelete_new import plot_wavelet2
from import_data import *

# set device
device = 'navid_lab'
if device.lower() == 'navid':
    dataset_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
    processed_data_path = 'F:/maryam_sh/load_data/'
elif device.lower() == 'maryam':
    dataset_path = 'E:/Thesis/dataset/dataset/'
    processed_data_path = 'F:/maryam_sh/load_data/'
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
    'generate_electrode_histogram': True,  # calculate histogram of electrodes of patients
    'plot_common_electrodes_sync_average': True,  # plot mean patient of output of common electrode
    'classification': True,  # find mean_30sec moving average of signal_common electrode
    'analyze_signal_mean_patient': True,
    'wavelete': True,  # plot wavelet of raw data of signal each patient
    'fs': 25,
    'final_time': 120
}

load_data_settings = {
    'number_of_patients': 2,
    'use_only_gamma_band': False,
    'apply_hilbert': False,
    # if 'subject_list':True, function just calculate subject_list without calculate raw_data and band_all note that
    # we want subject_list because index of patient in common electrode list doesn't define number of each patient
    # but is index of subject_list
    'subject_list': False,
    'save_preprocessed_data': True,
    'load_preprocessed_data': False
}

# Create data_path, save_paths and load_paths
paths = Paths(settings)
paths.create_path(path_dataset=dataset_path,
                  path_processed_data=processed_data_path,
                  task='task_speech&music',
                  settings=settings)

# Load and preprocess data
band_all_patient_with_hilbert, band_all_patient_without_hilbert, subject_list, channel_names_list = \
    get_data(paths, settings=load_data_settings)

# Find the number of common electrodes between patients
if settings['generate_electrode_histogram']:
    hist, share_electrodes_15p = electrode_histogram(channel_names_list,
                                                     print_analyze=True)

# Get the synchronus average for common average between 15 patient
if settings['plot_common_electrodes_sync_average']:
    common_electrode_avg = plot_comm_elec(common_electrodes=share_electrodes_15p,
                                          band_all_patient=band_all_patient_without_hilbert,
                                          channel_names_list=channel_names_list,
                                          final_time=settings['final_time'],
                                          fs=settings['fs'],
                                          path=paths.path_results,
                                          freq_band='gamma')

" -------------------------------------------------  Classification -------------------------------------------------"
if settings['classification']:
    onset_music = np.arange(0, 390, 60)
    onset_speech = np.arange(30, 390, 60)
    feature_set = {'AVG': True, 'RMS': True}
    num_patient = 6

    class_viz = QAVisualizer(channel_names_list,
                             onset_music,
                             onset_speech,
                             settings['fs'],
                             paths.path_results_classification,
                             num_patient, t_min=0.5, step=30, allow_plot=True)

    if feature_set['RMS']:
        class_viz.plot_class_conditional_average(data=band_all_patient_without_hilbert,
                                                 window_size=250,
                                                 frq_band_name='gamma',
                                                 feature_type='rms')
    if feature_set['AVG']:
        class_viz.plot_class_conditional_average(data=band_all_patient_without_hilbert,
                                                 window_size=250,
                                                 frq_band_name='gamma',
                                                 feature_type='avg')

if settings['analyze_signal_mean_patient']:
    analyze = analyze_signal_mean_patient(settings['fs'], paths.path_results, window_size=200)
    # output of calssification of mean_signal on patient for each common electrode
    signal_mean_patient = True
    # output of calssification of signal of each patient for each common electrode
    signal_each_patient = True

    if signal_mean_patient:
        electrode_for_classification = 'T14'
        avg_win = analyze.moving_average_signalmean(common_electrode_avg,
                                                    share_electrodes_15p, settings['final_time'],
                                                    electrode_for_classification, plot_output=True)

    if signal_each_patient:
        common_electrode = 'T13'
        avg_win2 = analyze.moving_average_signal_each_patient(channel_names_list,
                                                              band_all_patient_with_hilbert,
                                                              band_all_patient_without_hilbert,
                                                              common_electrode, feature='rms')

if settings['wavelete']:
    raw_car_all = load_raw_data(paths, settings)
    plot_wavelet2(paths.path_results, raw_car_all, patient=2, electrode='T13')
