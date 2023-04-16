import numpy as np
import matplotlib.pyplot as plt

def calculate_synchronous_avg(raw_car_all,band_all_patient,onset,offset,fs):
    sync_avg=[]
    for patient in range(len(band_all_patient)):
        electrodes = raw_car_all[patient].ch_names
        sync_avg_onepatient = []

        for electrode in electrodes:
            num_electrode = raw_car_all[patient].ch_names.index(electrode)
            signal=band_all_patient[patient][:,num_electrode]
            s=0
            for i in range(len(onset)):
                start_sample=int(onset[i]-0.5)*fs
                end_sample= int(onset[i]+2.5)*fs
                s=s+signal[start_sample:end_sample]
            sync_avg_onepatient.append(s/len(onset))

        sync_avg.append(sync_avg_onepatient)

    return sync_avg







