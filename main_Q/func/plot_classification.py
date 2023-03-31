from ieeg_func.mov_avg import moving_avg
from main_Q.func.out_class import plot_output_classification

import numpy as np
import math

def power_2(my_list,p):
    return [ x**p for x in my_list ]

def output_classification(raw_car_all,band_all_patient,onset_question,fs,window_size,band,patient,all_time,AVG,RMS):
    step = 2 * fs
    onset_question_corr = [int(x) * fs for x in onset_question]
    electrodes = raw_car_all[patient].ch_names

    avg_win = {}
    label_all = {}

    for electrode in electrodes:
        num_electrode = raw_car_all[patient].ch_names.index(electrode)
        signal = band_all_patient[patient][band][:, num_electrode]
        if AVG:
            one_elec=moving_avg(signal, window_size)
        if RMS:
            one_elec=signal

        i = 0
        s = []
        label = []
        j = 1
        val = 0
        while i < len(one_elec) - step + 1:
            if j * 30 * fs <= i <= (j + 1) * 30 * fs:
                val = 0
                for m in range(int(step / fs)):
                    if i + m * (fs) in onset_question_corr:
                        i = i + m * (fs)
                        label.append('1')
                        if AVG:
                            s.append(np.mean(one_elec[int(i):int(i + step)]))
                        if RMS:
                            e = sum(power_2(one_elec[int(i):int(i + step)], 2))
                            e2 = (1 / step) * math.sqrt(e)
                            s.append(e2)
                        val = 1

                if val == 0:
                    label.append('0')
                    if AVG:
                        s.append(np.mean(one_elec[int(i):int(i + step)]))
                    if RMS:
                        e = sum(power_2(one_elec[int(i):int(i + step)], 2))
                        e2 = (1 / step) * math.sqrt(e)
                        s.append(e2)
                if all_time:
                    if i + step > (j + 1) * 30 * fs:
                        j = j + 2

            i = i + step

        avg_win[electrode] = s
        label_all[electrode] = label
    plot_output_classification(electrodes,avg_win,label_all,label,AVG,RMS)
