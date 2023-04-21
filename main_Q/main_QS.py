from main_Q.func.plot_temporal import plot_temporal_signal
from main_Q.func.plot_classification import classification
from main_Q.func.read_time_annotation import read_time
from main_Q.func.calculate_synchronous_average import calculate_synchronous_avg,calculate_synch_avg_common_electrode
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

#raed annotation of question and answer
onset_question,offset_question=read_time(path="F:/maryam_sh/dataset/stimuli/annotations/sound/sound_annotation_questions.tsv")
onset_answer,offset_answer=read_time(path="F:/maryam_sh/dataset/stimuli/annotations/sound/sound_annotation_sentences.tsv")

# remove onset of question from onset of answer
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
    # TRUE:if your data have all band   False:if your data have just GAMMA band
    all_band = True
    with open(paths.path_load_data + 'raw_car_all.txt', 'rb') as f:
        raw_car_all=pickle.load(f)
    with open(paths.path_load_data + 'band_all_patient_allband.txt', 'rb') as f:
        band_all_patient=pickle.load(f)
    with open(paths.path_load_data + 'band_all_patient_nohil.txt', 'rb') as f:
        band_all_patient_nohil=pickle.load(f)


if settings['temporal_signal']:
    if ~all_band:
        plot_temporal_signal(raw_car_all,band_all_patient,onset_question,offset_question,fs,paths.path_results,
                             electrode='T13',patient=3,final_time=150)
    else:
        band = 'gamma'
        plot_temporal_signal(raw_car_all, band_all_patient[:][band], onset_question, offset_question, fs,paths.path_results,
                             electrode='T13', patient=3, final_time=150)


if settings['output_classification']:
    feature_set={'AVG':True ,'RMS':False , 'max_peak':True , 'variance':True}
    if all_band==False:
        cls=classification(raw_car_all,band_all_patient,band_all_patient_nohil
                           ,onset_question,onset_answer,fs,paths.path_results,num_patient=10)
    if all_band==True:
        band = 'theta'
        x=[]
        for i in  range(len(band_all_patient)):
            x.append(band_all_patient[i][band])
        cls = classification(raw_car_all, x, band_all_patient_nohil
                             , onset_question, onset_answer, fs, paths.path_results, num_patient=10)

    if feature_set['RMS']:
        cls.class_rms()
    if feature_set['AVG']:
        cls.class_avg(window_size=20)
    if feature_set['max_peak']:
        cls.class_max_peak()
    if feature_set['variance']:
        cls.class_variance()


if settings['sync_AVG']:
    if ~all_band:
        #calculate and plot synch_avg for all electrodes of all patients
        calculate_synchronous_avg(raw_car_all,band_all_patient,onset_question,onset_answer,paths.path_results,fs)
        # calculate and plot synch_avg for patients have common electrode
        calculate_synch_avg_common_electrode(raw_car_all,band_all_patient,onset_question,onset_answer,paths.path_results,fs)

    else:
        band = 'gamma'
        # calculate and plot synch_avg for all electrodes of all patients
        calculate_synchronous_avg(raw_car_all, band_all_patient[:][band], onset_question, onset_answer, paths.path_results, fs)
        # calculate and plot synch_avg for patients have common electrode
        calculate_synch_avg_common_electrode(raw_car_all, band_all_patient[:][band], onset_question, onset_answer,paths.path_results, fs)

print('end')





