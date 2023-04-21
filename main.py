from utils import *
from ieeg_func.get_data import gdata
from ieeg_func.histogram_elec import hist_elec
from ieeg_func.patient_comelec import com_elec
from ieeg_func.plot_avg_patient_comelec import plot_comm_elec
from ieeg_func.moving_average_signal_mean import moving_average_signalmean
from ieeg_func.output_for_class import output_classification
from ieeg_func.movavg_each_patient import moving_average_signal_each_patient
from ieeg_func.movavg_each_patient_all import moving_average_each_patient_allelec
from ieeg_func.rms_signal_comm import RMS_signal_comm
from ieeg_func.plot_wavlete_patient import plot_wavelet




import matplotlib.pyplot as plt
import pickle


# set device
device = 'system_lab'
if device.lower() == 'navid':
    data_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
elif device.lower() == 'maryam_laptop':
    data_path = 'E:\Thesis\dataset\dataset'
elif device.lower() == 'system_lab':
    data_path = 'F:/maryam_sh/dataset'
else:
    data_path = ''


# settings
# first determine which task you want
settings = {
    # get raw_data and data of bands
    'get_data':False,
    # save data "raw_data and data of bands"
    'save_data':False,
    # load data "raw_data and data of bands"
    'load_data':True,
    #calculate histogram of electrodes of patients
    'histogram_elec':True,
    # print number of patient have common electrode
    'patient_com_elec':False,
    # plot mean patient of output of common electrode
    'plot_commelec':True,
    # find mean_30sec moving average of signal_common electrode
    'output_classification_movAVG': False,
    # find RMS of signal_common electrode
    'output_classification_RMS': False,
    #plot wavelet of raw data of signal each patient
    'wavelete':False,
}

paths = Paths(settings)
paths.create_path()

setting_parameter={}
fs = 25
final_time=120


setting_parameter.update({
        'number_of_patients': 63,
        'just_gamma': False,
        'hilbert': True,
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
    with open(paths.path_save_data + 'band_all_patient_allband.txt', 'wb') as f:
        pickle.dump(band_all_patient, f)


if settings['load_data']:
    # TRUE:if your data have all band   False:if your data have just GAMMA band
    all_band=False
    with open(paths.path_load_data + 'raw_car_all.txt', 'rb') as f:
        raw_car_all=pickle.load(f)
    with open(paths.path_load_data + 'band_all_patient.txt', 'rb') as f:
        band_all_patient=pickle.load(f)
    #with open(path_load_data + 'raw_car_all_nohil.txt', 'rb') as f:
        #raw_car_all_nohil = pickle.load(f)
    #with open(path_load_data + 'band_all_patient_nohil.txt', 'rb') as f:
        #band_all_patient_nohil=pickle.load(f)


if settings['histogram_elec']:
    hist,elec_morecommon_fifteen=hist_elec(raw_car_all,print_analyze=True)

if settings['patient_com_elec']:
    patient_common_elec=com_elec(raw_car_all,elec_morecommon_fifteen,print_patient= True)


if settings['plot_commelec']:
    if setting_parameter['just_gamma'] or ~all_band:
        plot_elec_avg=plot_comm_elec(elec_morecommon_fifteen,band_all_patient,raw_car_all,final_time,fs,paths.path_results)
    else:
        band = 'gamma'
        plot_elec_avg=plot_comm_elec(elec_morecommon_fifteen,band_all_patient[:][band],raw_car_all,patient_common_elec,final_time,fs)

if settings['output_classification_movAVG']:
    # output of calssification of mean_signal on patient for each common electrode
    signal_mean_patient=True
    # output of calssification of signal of each patient for each common electrode
    signal_each_patient=True
    # output of calssification of signal of each patient for all electrode
    signal_each_patient_all_elec=True

    if signal_mean_patient:
        electrode = 'T14'
        avg_win=moving_average_signalmean(plot_elec_avg,elec_morecommon_fifteen,final_time,fs,window_size=200,plot_output=True)
        fig, ax = plt.subplots()
        out_class_one_electrode=output_classification(avg_win[electrode],ax,step=30*fs,ax_x=1,AVG=True,RMS=False)
        plt.show()

    if signal_each_patient:
        electrode='T14'
        if setting_parameter['just_gamma'] or ~all_band:
            avg_win2=moving_average_signal_each_patient(raw_car_all,band_all_patient,patient_common_elec[electrode],electrode,window_size=200)
        else:
            band = 'gamma'
            avg_win2=moving_average_signal_each_patient(raw_car_all,band_all_patient[:][band],patient_common_elec[electrode],electrode,window_size=200)
        plt.show()


    if signal_each_patient_all_elec:
        patient = 5
        if setting_parameter['just_gamma'] or ~all_band:
            avg_win3=moving_average_each_patient_allelec(raw_car_all[patient],band_all_patient[patient],window_size=200)
        else:
            band = 'gamma'
            avg_win3=moving_average_each_patient_allelec(raw_car_all[patient],band_all_patient[patient][band],window_size=200)
        plt.show()

if settings['output_classification_RMS']:
    hist, elec_morecommon_fifteen = hist_elec(raw_car_all_nohil, print_analyze=True)
    patient_common_elec = com_elec(raw_car_all_nohil, elec_morecommon_fifteen, print_patient=True)
    electrode='T13'
    if setting_parameter['just_gamma'] or ~all_band:
        rms=RMS_signal_comm(raw_car_all_nohil, band_all_patient_nohil, patient_common_elec[electrode], electrode)
    else:
        band = 'gamma'
        rms=RMS_signal_comm(raw_car_all_nohil, band_all_patient_nohil[:][band], patient_common_elec[electrode], electrode)

if settings['wavelete']:
    plot_wavelet(raw_car_all,patient=2,electrode='T13')


