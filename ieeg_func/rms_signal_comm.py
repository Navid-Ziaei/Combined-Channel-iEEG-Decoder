from ieeg_func.output_for_class import output_classification
from ieeg_func.calculate_rms import cal_rms
import matplotlib.pyplot as plt


def RMS_signal_comm(raw_car_all_nohil,band_all_patient_nohil,elec_com,electrode,band):
    fs = 25
    rms = {}
    for i in range(len(elec_com)):
        rms[elec_com[i]] = []

    for patient in elec_com:
        num_electrode = raw_car_all_nohil[patient].ch_names.index(electrode)
        signal = band_all_patient_nohil[patient][band][:, num_electrode]
        rms[patient] = cal_rms(signal, step=30 * fs)
        output_classification(rms[patient], step=30*fs, ax_x=patient,AVG=False,RMS=True)

    plt.show()

