from ieeg_func.mov_avg import moving_avg
from ieeg_func.output_for_class import output_classification
import matplotlib.pyplot as plt

def moving_average_each_patient_allelec(raw_car_one_patient,band_all_one_patient,window_size,band):
    fs=25
    electrode_of_patient=raw_car_one_patient.ch_names

    avg_win3 = {}
    for i in range(len(electrode_of_patient)):
        avg_win3[electrode_of_patient[i]] = []

    fig, ax = plt.subplots()
    for electrode in electrode_of_patient:
        num_electrode = raw_car_one_patient.ch_names.index(electrode)
        signal=band_all_one_patient[band][:,num_electrode]
        s = moving_avg(signal, window_size)
        avg_win3[electrode] = output_classification(s,ax, step=30 * fs, ax_x=num_electrode, AVG=True, RMS=False)