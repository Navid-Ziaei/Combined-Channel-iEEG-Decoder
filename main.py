from ieeg_func.utils import *
from ieeg_func.get_data import gdata
from ieeg_func.histogram_elec import hist_elec
from ieeg_func.patient_comelec import com_elec
from ieeg_func.plot_avg_patient_comelec import plot_comm_elec

import pickle


# set device
device = 'maryam_laptop'
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
    'patient_com_elec':True,
    # plot mean patient of output of common electrode
    'plot_commelec':True
}

paths = Paths(settings)
paths.create_path()

setting_parameter={}

if settings['get_data']:
    setting_parameter.update({
        'number_of_patients': 1,
        'just_gamma': True,
        'hilbert': True,
        # if 'subject_list':True, function just calculate subject_list without calculate raw_data and band_all
        #note that we want subject_list because index of patient in common electrode list doesn't define number of each patient but is index of subject_list
        'subject_list':True
    })
    band_all_patient, raw_car_all,subject_list = gdata(data_path,setting_parameter['subject_list'], setting_parameter['number_of_patients'],setting_parameter['just_gamma'], setting_parameter['hilbert'])



if settings['save_data']:
    with open(paths.path_results + 'raw_car_all.txt', 'wb') as f:
        pickle.dump(raw_car_all, f)
    with open(paths.path_results + 'band_all_patient.txt', 'wb') as f:
        pickle.dump(band_all_patient, f)


if settings['load_data']:
    path_load_data='E:\\Thesis\\notebook\\data_load\\'
    with open(path_load_data + 'raw_car_home.txt', 'rb') as f:
        raw_car_all=pickle.load(f)
    with open(path_load_data + 'band_all_home.txt', 'rb') as f:
        band_all_patient=pickle.load(f)

if settings['histogram_elec']:
    hist,elec_morecommon_fifteen=hist_elec(raw_car_all,print_analyze=True)

if settings['patient_com_elec']:
    patient_common_elec=com_elec(raw_car_all,elec_morecommon_fifteen,print_patient= True)


if settings['plot_commelec']:
    plot_comm_elec(elec_morecommon_fifteen,band_all_patient,raw_car_all,patient_common_elec,final_time=120,fs=25)


# fig = plt.figure()
#plt.plot([1,2,3],[1,2,3])
#fig.savefig(paths.path_results + 'test.png')
#plt.show()
#print(settings['number_of_patients'])
