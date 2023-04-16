from main_Q.func.plot_temporal import plot_temporal_signal
from main_Q.func.plot_classification import output_classification
from main_Q.func.read_time_annotation import read_time
from main_Q.func.calculate_synchronous_average import calculate_synchronous_avg
from main_Q.func.save_output_plot import save_plot
#from main_Q.func.classification_max import cal_max
from ieeg_func.utils import *
import pickle
import matplotlib.pyplot as plt


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

onset_question,offset_question=read_time(path="F:/maryam_sh/dataset/stimuli/annotations/sound/sound_annotation_questions.tsv")
onset_answer,offset_answer=read_time(path="F:/maryam_sh/dataset/stimuli/annotations/sound/sound_annotation_sentences.tsv")

onset_question_2=[int(x) for x in onset_question]
onset_answer_2=[int(x) for x in onset_answer]
offset_question_2=[int(x) for x in offset_question]
offset_answer_2=[int(x) for x in offset_answer]

for i in onset_answer:
    if(int(i) in onset_question_2):
        onset_answer.remove(i)

for i in onset_answer:
    if(i in onset_question):
        onset_answer.remove(i)

for i in offset_answer:
    if(int(i) in offset_question_2):
        offset_answer.remove(i)

for i in offset_answer:
    if(i in offset_question):
        offset_answer.remove(i)
fs=25

# settings
# first determine which task you want
settings = {
    # load data "raw_data and data of bands"
    'load_data':True,
    #plot temporal signal of one patient
    'temporal_signal':False,
    # find output_classification of signal_common electrode
    'output_classification': True,
    #find synchronous average
    'sync_AVG': False
}


paths = Paths(settings)
paths.create_path()

if settings['load_data']:
    path_load_data = 'F:/maryam_sh/'
    all_band = False
    with open(path_load_data + 'raw_car_all.txt', 'rb') as f:
        raw_car_all=pickle.load(f)
    with open(path_load_data + 'band_all_patient.txt', 'rb') as f:
        band_all_patient=pickle.load(f)
    #with open(path_load_data + 'r2.txt', 'rb') as f:
        #raw_car_all_nohil = pickle.load(f)
    #with open(path_load_data + 'b2.txt', 'rb') as f:
        #band_all_patient_nohil=pickle.load(f)

#patient=0
#cal_max(raw_car_all,band_all_patient,patient,onset_question,onset_answer,fs)

if settings['temporal_signal']:
    if ~all_band:
        plot_temporal_signal(raw_car_all,band_all_patient,onset_question,offset_question,fs,
                             electrode='T13',patient=3,final_time=150)
    else:
        band = 'gamma'
        plot_temporal_signal(raw_car_all, band_all_patient[:][band], onset_question, offset_question, fs,
                             electrode='T13', patient=3, final_time=150)


if settings['output_classification']:
    if ~all_band:
        output=output_classification(raw_car_all,band_all_patient,onset_question,onset_answer,fs,
                                     window_size=20,patient=1,AVG=True, RMS=False,max_peak=False)
    else:
        band = 'gamma'
        output = output_classification(raw_car_all, band_all_patient[:][band], onset_question,onset_answer, fs,
                                       window_size=20, patient=1, AVG=True, RMS=False,max_peak=False)


if settings['sync_AVG']:
    save_output=True
    if ~all_band:
        sync_AVG_question=calculate_synchronous_avg(raw_car_all,band_all_patient,onset_question,offset_question,fs)
        sync_AVG_answer = calculate_synchronous_avg(raw_car_all,band_all_patient,onset_answer,offset_answer,fs)

    else:
        band = 'gamma'
        sync_AVG_question = calculate_synchronous_avg(raw_car_all, band_all_patient[:][band], onset_question, offset_question,fs)
        sync_AVG_answer = calculate_synchronous_avg(raw_car_all, band_all_patient[:][band], onset_answer, offset_answer,fs)

    with open('sync_AVG_question.txt', 'wb') as f:
        pickle.dump(sync_AVG_question, f)
    with open('sync_AVG_answer.txt', 'wb') as f:
        pickle.dump(sync_AVG_answer, f)

    if save_output:
        save_plot(raw_car_all,sync_AVG_question,sync_AVG_answer,paths.path_results,fs)



print('end')





