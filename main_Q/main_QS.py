from func.plot_temporal import plot_temporal_signal
from func.plot_classification import output_classification
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

onset_question=[30.7361,36.601,44.003,44.679,48.935,53.58,100.823,155.133,219.345,220.178,238.553,273.766 ,278.646,281.898,296.384]
offset_question=[32.49,37.862,44.395,45.88,50.222,55.9107,102.334,157.156,219.971,221.502,239.62,274.745,281.821,282.893,297.588]
fs=25

# settings
# first determine which task you want
settings = {
    # load data "raw_data and data of bands"
    'load_data':True,
    #plot temporal signal of one patient
    'temporal_signal':True,
    # find output_classification of signal_common electrode
    'output_classification': True,
}

if settings['load_data']:
    path_load_data='E:\\Thesis\\notebook\\data_load\\'
    with open(path_load_data + 'r.txt', 'rb') as f:
        raw_car_all=pickle.load(f)
    with open(path_load_data + 'b.txt', 'rb') as f:
        band_all_patient=pickle.load(f)
    #with open(path_load_data + 'r2.txt', 'rb') as f:
        #raw_car_all_nohil = pickle.load(f)
    #with open(path_load_data + 'b2.txt', 'rb') as f:
        #band_all_patient_nohil=pickle.load(f)

if settings['temporal_signal']:
    plot_temporal_signal(raw_car_all,band_all_patient,onset_question,offset_question,fs,band='gamma',
                         electrode='T13',patient=0,final_time=150)

if settings['output_classification']:
    output=output_classification(raw_car_all,band_all_patient,onset_question,fs,
                                 window_size=50,band='gamma',patient=1,all_time=True,AVG=False,RMS=True)


