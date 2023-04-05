from ieeg_func.mov_avg import moving_avg
from ieeg_func.output_for_class import output_classification
import matplotlib.pyplot as plt

def moving_average_signal_each_patient(raw_car_all,band_all_patient,elec_com,electrode,window_size):
    fs=25
    avg_win2 = {}
    for i in range(len(elec_com)):
        avg_win2[elec_com[i]] = []

    fig, ax = plt.subplots()
    for patient in elec_com:
        num_electrode = raw_car_all[patient].ch_names.index(electrode)
        signal = band_all_patient[patient][:, num_electrode]
        s= moving_avg(signal, window_size)
        avg_win2[patient]=output_classification(s,ax, step=30 * fs, ax_x=patient,AVG=True,RMS=False)


