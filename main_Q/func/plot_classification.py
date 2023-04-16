import numpy as np
import math
import matplotlib.pyplot as plt
from ieeg_func.mov_avg import moving_avg
import pandas as pd

def power_2(my_list,p):
    return [ x**p for x in my_list ]

def output_classification(raw_car_all,band_all_patient,onset_q,onset_a,fs,window_size,patient,AVG,RMS,max_peak):
    electrodes = raw_car_all[patient].ch_names
    fig, ax = plt.subplots(figsize=(80, 40))
    #fig1, ax1 = plt.subplots(figsize=(80, 40))

    for electrode in electrodes:
        num_electrode = raw_car_all[patient].ch_names.index(electrode)
        if RMS:
            signal = band_all_patient[patient][:, num_electrode]
        if AVG:
            c = band_all_patient[patient][:, num_electrode]
            signal=moving_avg(c, window_size)
        if max_peak:
            signal = band_all_patient[patient][:, num_electrode]

        s = []
        for i in range(len(onset_q)):
            start_sample = int(onset_q[i] - 0.5) * fs
            end_sample = int(onset_q[i] + 2.5) * fs
            if RMS:
                e = sum(power_2(signal[start_sample:end_sample], 2))
                e2 = (1 / len(signal[start_sample:end_sample])) * math.sqrt(e)
                s.append(e2)
            if AVG:
                s.append(np.mean(signal[start_sample:end_sample]))
            if max_peak:
                s.append(np.max(signal[start_sample:end_sample]))

        for i in range(len(onset_a[:15])):
            start_sample = int(onset_a[i] - 0.5) * fs
            end_sample = int(onset_a[i] + 2.5) * fs
            if RMS:
                e = sum(power_2(signal[start_sample:end_sample], 2))
                e2 = (1 / len(signal[start_sample:end_sample])) * math.sqrt(e)
                s.append(e2)
            if AVG:
                s.append(np.mean(signal[start_sample:end_sample]))
            if max_peak:
                s.append(np.max(signal[start_sample:end_sample]))


        # question=0(red)  answer=1(green)
        label = ['0'] * len(onset_q) + ['1'] * len(onset_a[:15])
        y = [num_electrode] * len(label)

        df = pd.DataFrame(dict(x=y , y=s, label=label))
        colors = {'0': 'red', '1': 'green'}
        ax.scatter(df['x'], df['y'], c=df['label'].map(colors), marker='o')

    fig.savefig('class_q')


