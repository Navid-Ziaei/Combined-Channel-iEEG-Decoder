import math

def power_2(my_list,p):
    return [ x**p for x in my_list ]

def cal_rms(signal,step):
    i = 0
    s_rms = []
    while i < len(signal) - step + 1:
        e = (1 / step) * sum(power_2(signal[i:i + step], 2))
        e2 = math.sqrt(e)
        s_rms.append(e2)
        i = i + step

    return s_rms
