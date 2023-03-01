import mne_bids
import os
import numpy as np
from import_data import SubjectDataset


# set device
device = 'Navid'
if device.lower() == 'navid':
    data_path = 'F:/Datasets/ieeg_visual/ds003688-download/'
elif device.lower() == 'maryam':
    data_path = 'E:/maryam.sh'
else:
    data_path = ''

# settings
settings = {
    'number_of_patients': 3,
}

subjects = mne_bids.get_entity_vals(data_path, 'subject')
subject = subjects[0]
acqs = mne_bids.get_entity_vals(os.path.join(data_path, 'sub-' + subject, 'ses-iemu', 'ieeg'), 'acquisition')
acq = acqs[0]

# Load all data
band_all_patient = []
raw_car_all = []
subject_list = subjects[:settings['number_of_patients']]
i = 0
for subject in subject_list:
    if 'iemu' in mne_bids.get_entity_vals(os.path.join(data_path, 'sub-' + subject), 'session'):
        i = i + 1
        s = SubjectDataset(data_path, subject, acquisition=acq)
        raw_car = s.preprocess()
        events, events_id = s.extract_events()
        bands2, bands3, bands4, band5 = s.extract_bands()
        band_all_patient.append(band5['gamma'])
        raw_car_all.append(raw_car)


# find common electrodes
elec_more_one = []
elec_more_ten = []
elec_more_fifteen = []
elec_more_tweny = []
elec_more_thirty = []

for key in h.keys():
    if (h[key] > 1):
        elec_more_one.append(key)
    if (h[key] > 10):
        elec_more_ten.append(key)
    if (h[key] > 15):
        elec_more_fifteen.append(key)
    if (h[key] > 20):
        elec_more_tweny.append(key)

print('number of total electrode is =', len(h), '\nmax number of electrod is same=', 23)

print('number of electrod that are same in more than one patient = ', len(elec_more_one))
print('number of electrod that are same in more than ten patient = ', len(elec_more_ten))
print('number of electrod that are same in more than fifteen patient = ', len(elec_more_fifteen))
print('number of electrod that are same in more than tweny patient = ', len(elec_more_tweny))

print('\n\n', elec_more_fifteen, 'elec_more_fifteen\n\n')

print(elec_more_tweny, 'elec_more_tweny')


elec= {'F01':[] , 'F03':[] , 'F05': [] , 'F06':[] , 'F07':[] , 'F08':[] , 'T01':[] , 'T02':[] , 'T03':[] , 'T04':[] , 'T05':[] , 'T06':[], 'T07':[], 'T09':[], 'T10':[], 'T11':[], 'T12':[], 'T13':[], 'T14':[], 'T08':[], 'T15':[]}

for i in range(len(raw_car_all)):
    for key in elec.keys():
        if key in raw_car_all[i].ch_names:
            elec[key].append(i)

for key in elec.keys():
    print('\n  ' ,key, '=', elec[key])


plot_elec_avg= {'F01':0 , 'F03':0 , 'F05': 0 , 'F06':0 , 'F07':0 , 'F08':0 , 'T01':0 , 'T02':0 , 'T03':0 , 'T04':0 , 'T05':0 , 'T06':0, 'T07':0, 'T09':0, 'T10':0, 'T11':0, 'T12':0 , 'T13':0 , 'T14':0, 'T08':0, 'T15':0}
time=np.arange(0, band_all_patient[0].shape[0])/(25)


for key in plot_elec_avg.keys():
    for i in elec[key]:
        num_electrode=raw_car_all[i].ch_names.index(key)
        plot_elec_avg[key]=plot_elec_avg[key]+band_all_patient[i][:,num_electrode]
    plot_elec_avg[key]=plot_elec_avg[key]/len(elec[key])
    plt.figure(figsize=(30,10))
    plt.plot(time[:120*25],plot_elec_avg[key][:120*25])
    plt.title(str(key),fontsize=25)
    plt.show()