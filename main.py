from utils import *
import numpy as np
from ieeg_func.get_data import gdata
from ieeg_func.histogram_elec import hist_elec
from ieeg_func.plot_avg_patient_comelec import plot_comm_elec
from ieeg_func.analyze_mean_signal import analyze_signal_mean_patient
from ieeg_func.plot_wavelete_new import plot_wavelet2
from ieeg_func.plot_classification import classification
import pickle


# set device
device = 'system_lab'
if device.lower() == 'navid':
    data_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
elif device.lower() == 'maryam':
    data_path = 'E:\Thesis\dataset\dataset'
elif device.lower() == 'system_lab':
    data_path = 'F:/maryam_sh/dataset'
else:
    data_path = ''


# settings
# first determine which task you want
settings = {
    # get raw_data and data of bands
    'get_data':True,
    # save data "raw_data and data of bands"
    'save_data':False,
    # load data "raw_data and data of bands"
    'load_data':True,
    #calculate histogram of electrodes of patients
    'histogram_elec':True,
    # plot mean patient of output of common electrode
    'plot_commelec':True,
    # find mean_30sec moving average of signal_common electrode
    'classification': True,
    'analyze_signal_mean_patient':True,
    #plot wavelet of raw data of signal each patient
    'wavelete':True,
}

paths = Paths(settings)
paths.create_path()
os.makedirs(paths.path_results + 'task_speech&music')
path_save_result= os.path.join(paths.path_results, 'task_speech&music' + '/')
setting_parameter={}
fs = 25
final_time=120


setting_parameter.update({
        'number_of_patients': 63,
        'just_gamma': True,
        'hilbert': False,
        # if 'subject_list':True, function just calculate subject_list without calculate raw_data and band_all
        #note that we want subject_list because index of patient in common electrode list doesn't define number of each patient but is index of subject_list
        'subject_list':False
    })

if settings['get_data']:
    if setting_parameter['hilbert']:
        band_all_patient, raw_car_all,subject_list = gdata(data_path,setting_parameter['subject_list'], setting_parameter['number_of_patients'],setting_parameter['just_gamma'], setting_parameter['hilbert'])
    else:
        band_all_patient_nohil, raw_car_all_nohil,subject_list = gdata(data_path,setting_parameter['subject_list'], setting_parameter['number_of_patients'],setting_parameter['just_gamma'], setting_parameter['hilbert'])



if settings['save_data']:
    #with open(paths.path_save_data + 'raw_car_all.txt', 'wb') as f:
        #pickle.dump(raw_car_all_nohil, f)
    with open(paths.path_save_data + 'band_all_patient_nohil_allband.txt', 'wb') as f:
        pickle.dump(band_all_patient_nohil, f)

if settings['load_data']:
    # TRUE:if your data have all band   False:if your data have just GAMMA band
    with open(paths.path_load_data + 'raw_car_all.txt', 'rb') as f:
        raw_car_all=pickle.load(f)
    with open(paths.path_load_data + 'band_all_patient.txt', 'rb') as f:
        band_all_patient=pickle.load(f)
    with open(paths.path_load_data + 'band_all_patient_nohil.txt', 'rb') as f:
        band_all_patient_nohil=pickle.load(f)


if setting_parameter['just_gamma'] :
    band_all_patient2=band_all_patient
    band_all_patient_nohil2 = band_all_patient_nohil
else :
    band = 'gamma'
    band_all_patient2 = []
    band_all_patient_nohil2=[]
    for i in range(len(band_all_patient)):
        band_all_patient2.append(band_all_patient[i][band])
        band_all_patient_nohil2.append(band_all_patient_nohil[i][band])




if settings['histogram_elec']:
    hist,elec_morecommon_fifteen=hist_elec(raw_car_all,print_analyze=True)


if settings['plot_commelec']:
        plot_elec_avg=plot_comm_elec(elec_morecommon_fifteen,band_all_patient2,raw_car_all,final_time,fs,path_save_result)


if settings['classification']:
    onset_music=np.arange(0,390,60)
    onset_speech=np.arange(30,390,60)
    feature_set = {'AVG': True, 'RMS': True}

    cls = classification(raw_car_all, band_all_patient2, band_all_patient_nohil2, onset_music, onset_speech
                         , fs, path_save_result,t_min=0.5,step=30, num_patient=6)

    if feature_set['RMS']:
        cls.class_rms()
    if feature_set['AVG']:
        cls.class_avg(window_size=200)


if settings['analyze_signal_mean_patient']:
    analyze=analyze_signal_mean_patient(fs,path_save_result, window_size=200)
    # output of calssification of mean_signal on patient for each common electrode
    signal_mean_patient = True
    # output of calssification of signal of each patient for each common electrode
    signal_each_patient = True

    if signal_mean_patient:
        electrode_for_classification = 'T14'
        avg_win = analyze.moving_average_signalmean(plot_elec_avg, elec_morecommon_fifteen, final_time,
                                                    electrode_for_classification,plot_output=True)

    if signal_each_patient:
        common_electrode = 'T13'
        avg_win2 = analyze.moving_average_signal_each_patient(raw_car_all, band_all_patient2,band_all_patient_nohil2,
                                                          common_electrode,feature='rms')

if settings['wavelete']:
    plot_wavelet2(path_save_result,raw_car_all,patient=2,electrode='T13')


