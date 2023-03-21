
import numpy as np

def moving_avg(signal,window_size):
    i = 0
    avg_win=[]
    while i < len(signal) - window_size + 1:
        win = signal[i:i + window_size]
        avg_win.append(np.mean(win))
        i = i + 1
    return avg_win
