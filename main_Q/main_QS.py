from main_Q.func.plot_temporal import plot_temporal_signal
from ieeg_func.plot_classification import classification
from main_Q.func.read_time_annotation import read_time
from main_Q.func.calculate_synchronous_average import synchronous_avg
from main_Q.func.create_model import create_model
from main_Q.func.PCA import plot_PCA
from utils import *
import pickle

# set device
device = 'maryam_laptop'
if device.lower() == 'navid':
    data_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
elif device.lower() == 'maryam_laptop':
    data_path = 'E:/Thesis/dataset/dataset'
elif device.lower() == 'system_lab':
    data_path = 'F:/maryam_sh/dataset'
else:
    data_path = ''

# raed annotation of question and answer
onset_question, offset_question, onset_answer, offset_answer = read_time()
fs = 25

# settings
# first determine which task you want
settings = {
    # load data "raw_data and data of bands"
    'load_data': True,
    # plot temporal signal of one patient
    'temporal_signal': False,
    # find synchronous average
    'sync_AVG': False,
    # find output_classification of signal_common electrode
    'output_classification': False,
    # create model for single patient
    'create_model_single': False,
    'PCA': True
}

paths = Paths(settings)
paths.create_path()
os.makedirs(paths.path_results + 'task_question&answer')
path_save_result = os.path.join(paths.path_results, 'task_question&answer' + '/')

if settings['load_data']:
    # TRUE:if your data have all band   False:if your data have just GAMMA band
    all_band = False
    with open(paths.processed_data_path + 'raw_car_all.txt', 'rb') as f:
        raw_car_all = pickle.load(f)
    with open(paths.processed_data_path + 'band_all_patient.txt', 'rb') as f:
        band_all_patient = pickle.load(f)
    with open(paths.processed_data_path + 'band_all_patient_nohil.txt', 'rb') as f:
        band_all_patient_nohil = pickle.load(f)

if all_band:
    band = 'alpha'
    band_all_patient2 = []
    band_all_patient_nohil2 = []
    for i in range(len(band_all_patient)):
        band_all_patient2.append(band_all_patient[i][band])
        band_all_patient_nohil2.append(band_all_patient_nohil[i][band])
else:
    band_all_patient2 = band_all_patient
    band_all_patient_nohil2 = band_all_patient_nohil

if settings['temporal_signal']:
    plot_temporal_signal(raw_car_all, band_all_patient2, onset_question, offset_question, fs, path_save_result,
                         electrode='T13', patient=3, final_time=150)

if settings['sync_AVG']:
    synch_avg = synchronous_avg(raw_car_all, band_all_patient2, onset_question, onset_answer, path_save_result, fs)
    # calculate and plot synch_avg for all electrodes of all patients
    synch_avg.calculate_synchronous_avg()
    # calculate and plot synch_avg for patients have common electrode
    synch_avg.calculate_synch_avg_common_electrode()

if settings['output_classification']:
    feature_set = {'AVG': True, 'RMS': True, 'max_peak': True, 'variance': True}
    num_patient = 47
    cls = classification(raw_car_all, band_all_patient2, band_all_patient_nohil2, onset_question, onset_answer
                         , fs, path_save_result, num_patient, t_min=0.5, step=2.5, allow_plot=False)

    if feature_set['RMS']:
        rms = cls.class_rms()
    if feature_set['AVG']:
        avg = cls.class_avg(window_size=20)
    if feature_set['max_peak']:
        max_peak = cls.class_max_peak()
    if feature_set['variance']:
        variance = cls.class_variance()

    feature_matrix = cls.create_feature_matrix()
    # with open(paths.path_save_data + 'feature_matrix.txt', 'wb') as f:
    #    pickle.dump(feature_matrix, f)

if settings['create_model_single']:
    with open(paths.processed_data_path + 'feature_matrix.txt', 'rb') as f:
        feature_matrix = pickle.load(f)
    # type_classification : 'log_reg' or 'SVM' or 'Naive_bayes'
    # type_balancing :  'over_sampling' or 'under_sampling'  or 'over&down_sampling' or 'weighted_losfunc'
    # notice that for classification Naive_bayes don't use 'weighted_losfunc' way for balancing
    list_type_balancing = ['over_sampling', 'under_sampling', 'over&down_sampling', 'weighted_losfunc']
    list_type_classification = ['log_reg', 'SVM', 'Naive_bayes']
    num_patient = 47

    for type_balancing in list_type_balancing:
        for type_classification in list_type_classification:
            model = create_model(feature_matrix, num_patient, path_save_result, type_balancing, type_classification)
            f_measure_all, precision_all, recall_all = model.model_single()
            model.save_plot_result(raw_car_all)

if settings['PCA']:
    with open(paths.processed_data_path + 'feature_matrix.txt', 'rb') as f:
        feature_matrix = pickle.load(f)
    plot_PCA(raw_car_all, path_save_result, feature_matrix)

print('end')
