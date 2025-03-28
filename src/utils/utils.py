import os
import json
from pathlib import Path
import datetime
import numpy as np
from collections import Counter
import pandas as pd


def time_ann(path):
    r = pd.read_csv(path, sep=";")
    onset = []
    offset = []
    for i in range(len(r.index)):
        d = r.iloc[i, 0]
        pos1 = d.find('\t')
        pos2 = d.rfind('\t')
        onset.append(eval(d[pos1 + 1:pos2]))
        offset.append(eval(d[pos2 + 1:]))
    return onset, offset


def read_time(task, t_min, paths):
    if task == 'Question_Answer':
        onset_1, offset_1 = time_ann(
            path=paths.path_dataset + "/stimuli/annotations/sound/sound_annotation_questions.tsv")
        onset_0, offset_0 = time_ann(
            path=paths.path_dataset + "/stimuli/annotations/sound/sound_annotation_sentences.tsv")

        # remove onset of question from onset of answer
        onset_1_int = [int(x) for x in onset_1]
        offset_1_int = [int(x) for x in offset_1]

        for i in onset_0:
            if int(i) in onset_1_int:
                onset_0.remove(i)

        for i in onset_0:
            if i in onset_1:
                onset_0.remove(i)

        for i in offset_0:
            if int(i) in offset_1_int:
                offset_0.remove(i)

        for i in offset_0:
            if i in offset_1:
                offset_0.remove(i)

    if task == 'Speech_Music':
        """onset_1 = [i for i in np.arange(0, 390, 60)]
        offset_1 = [i for i in np.arange(30, 420, 60)]
        onset_0 = [i for i in np.arange(30, 390, 60)]
        offset_0 = [i for i in np.arange(60, 390, 60)]
        onset_1[0] = onset_1[0] + t_min"""
        time_1 = np.arange(0, 30, 2)
        time_0 = np.arange(30, 60, 2)
        onset_1 = []
        onset_0 = []
        for i in range(7):
            onset_1.extend((time_1 + i * 60).tolist())
            if i < 6:
                onset_0.extend((time_0 + i * 60).tolist())


    if task == 'Singing_Music':
        onset_0 = [14, 16, 24, 26, 28, 34, 36, 43, 45, 47, 56, 57, 64, 66, 73, 75]
        onset_1 = [i for i in np.arange(0, 14, 2)] + [19, 21, 30, 32] + [40] + [i for i in np.arange(50, 56, 2)] + \
                  [60, 62, 69, 71] + [i for i in np.arange(81, 189, 2)]
        onset_1[0] = onset_1[0] + 0.5
    else:
        onset_1 = np.zeros(150)
        onset_0 = np.ones(150)

    return onset_1, onset_0


def moving_avg(signal, window_size):
    i = 0
    avg_win = []
    while i < len(signal) - window_size + 1:
        win = signal[i:i + window_size]
        avg_win.append(np.mean(win))
        i = i + 1
    return avg_win


def electrode_histogram(channel_names_list, print_analyze):
    print("\n =================================== \n"
          "analyzing histogram of electrodes")
    all_electrodes_names = []
    for i in range(len(channel_names_list)):
        all_electrodes_names.extend(channel_names_list[i])
    h = Counter(all_electrodes_names)

    elec_more_one = []
    elec_more_ten = []
    elec_more_fifteen = []
    elec_more_twenty = []
    for key in h.keys():
        if h[key] > 1:
            elec_more_one.append(key)
        if h[key] > 10:
            elec_more_ten.append(key)
        if h[key] > 15:
            elec_more_fifteen.append(key)
        if h[key] > 20:
            elec_more_twenty.append(key)
    if print_analyze:
        print('number of unique electrodes is =', len(h), '\n max number of electrode repetition=', 23)
        print('number of shared electrode in more than one patient = ', len(elec_more_one))
        print('number of shared electrode in more than ten patient = ', len(elec_more_ten))
        print('number of shared electrode in more than fifteen patient = ', len(elec_more_fifteen))
        print('number of shared electrode in more than twenty patient = ', len(elec_more_twenty))
        print('\n\n', elec_more_fifteen, 'Electrodes shared between more the 15 patients')
        print('', elec_more_twenty, 'Electrodes shared between more the 20 patients')
    return h, elec_more_fifteen


