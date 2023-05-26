from ieeg_func.mov_avg import moving_avg
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math

class analyze_signal_mean_patient():
    def __init__(self,fs,path,window_size):
        self.window_size=window_size
        self.fs=fs
        self.path=path


    def average(self,signal,step):
        i = 0
        s = []
        while i < len(signal) - step + 1:
            s.append(np.mean(signal[i:i + step]))
            i = i + step

        return s

    def power_2(self,my_list, p):
        return [x ** p for x in my_list]

    def cal_rms(self,signal, step):
        i = 0
        s_rms = []
        while i < len(signal) - step + 1:
            e = sum(self.power_2(signal[i:i + step], 2))
            e2 = (1 / step) * math.sqrt(e)
            s_rms.append(e2)
            i = i + step
        return s_rms

    def plot_class(self,s,ax_x,ax):
        label = ['0', '1'] * 6
        y = [ax_x] * 12
        df = pd.DataFrame(dict(x=y, y=s, label=label))
        colors = {'0': 'red', '1': 'green'}
        ax.scatter(df['x'], df['y'], c=df['label'].map(colors), marker='o')
        return ax


    def moving_average_signalmean(self,plot_elec_avg, elec_com_fifteen,final_time_show ,electrode,plot_output):
        avg_win = {}
        for i in range(len(elec_com_fifteen)):
            avg_win[elec_com_fifteen[i]] = []

        if plot_output:
            os.makedirs(self.path + 'output_moving_average_common_electrode')
            p2 = os.path.join(self.path, 'output_moving_average_common_electrode' + '/')
            time_dash = np.arange(0, final_time_show + 30, 30)

        for key in plot_elec_avg.keys():
            signal = plot_elec_avg[key]
            avg_win[key]=moving_avg(signal,self.window_size)
            if plot_output:
                fig, ax = plt.subplots(figsize=(30, 20))
                time = np.arange(0, len(avg_win[key])) / (self.fs)
                ax.plot(time[:final_time_show * self.fs], avg_win[key][:final_time_show * self.fs])
                ax.vlines(time_dash, ymin=-1 * np.ones(time_dash.shape), ymax=np.ones(time_dash.shape), colors='red', ls='--',
                       lw=2, label='vline_multiple - partial height')
                ax.set_xlabel('time', fontsize=40)
                ax.set_ylabel('average_patients', fontsize=40)
                ax.set_title(str(key),fontsize=40)
                fig.savefig(p2 + str(key))

        s=self.average(avg_win[electrode],30*self.fs)
        fig, ax = plt.subplots()
        ax=self.plot_class(s,1,ax)
        ax.set_xlabel('electrode', fontsize=40)
        ax.set_ylabel('average_patients', fontsize=40)
        ax.set_title(electrode,fontsize=40)
        fig.savefig(p2 + 'calssification_elec_'+electrode)

        return avg_win

    def moving_average_signal_each_patient(self,raw_car_all, band_all_patient, band_all_patient_nohil,electrode,feature):

        os.makedirs(self.path + 'output_classification_common_electrode_all_patient')
        p2 = os.path.join(self.path, 'output_classification_common_electrode_all_patient' + '/')
        fig, ax = plt.subplots(figsize=(60, 40))
        for patient in range(len(band_all_patient)):
            electrodes = raw_car_all[patient].ch_names
            if electrode in electrodes:
                num_electrode = raw_car_all[patient].ch_names.index(electrode)
                signal = band_all_patient[patient][:, num_electrode]
                if feature=='avg':
                    s1 = moving_avg(signal, self.window_size)
                    s = self.average(s1, 30 * self.fs)
                if feature=='rms':
                    signal2=band_all_patient_nohil[patient][:, num_electrode]
                    s = self.cal_rms(signal2, step=30 * self.fs)


                ax = self.plot_class(s[:12], patient, ax)
        ax.set_xlabel('patient', fontsize=40)
        ax.set_ylabel(feature+'_patients', fontsize=40)
        ax.set_title(electrode,fontsize=40)
        fig.savefig(p2 + 'calssification_elec_' + electrode)


