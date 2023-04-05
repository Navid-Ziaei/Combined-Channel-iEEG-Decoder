import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def output_classification(signal,ax,step,ax_x,AVG,RMS):

    if AVG:
        i = 0
        s = []
        while i < len(signal) - step + 1:
            s.append(np.mean(signal[i:i + step]))
            i = i + step


    if RMS:
        s=signal


    # music=0 &&  speech=1
    label = ['0', '1'] * 6
    y = [ax_x] * 12
    df = pd.DataFrame(dict(x=y, y=s, label=label))
    colors = {'0': 'red', '1': 'green'}
    ax.scatter(df['x'], df['y'], c=df['label'].map(colors), marker='o')
    if AVG:
        return s