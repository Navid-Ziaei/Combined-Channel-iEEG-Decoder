from ieeg_func.multitaper_spectrogram_python import multitaper_spectrogram
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_wavelet2(path,raw_car_all,patient,electrode):
    os.makedirs(path + 'Wavelet')
    p2 = os.path.join(path, 'Wavelet' + '/')
    # Set spectrogram params
    fs = raw_car_all[patient].info['sfreq']  # Sampling Frequency
    frequency_range = [0, 120]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = 3  # Set time-half bandwidth
    num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [4, 1]  # Window size is 4s with step size of 1s
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = True  # use multiprocessing
    cpus = 3  # use 3 cores  in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = True  # plot spectrogram
    return_fig = False  # do not return plotted spectrogram
    clim_scale = False  # do not auto-scale colormap
    verbose = True  # print extra info
    xyflip = False  # do not transpose spect output matrix

    final_time = 390
    num_electrode = raw_car_all[patient].ch_names.index(electrode)
    data = raw_car_all[patient].get_data().T[:int(390 * fs), num_electrode]
    # Compute the multitaper spectrogram
    spect, stimes, sfreqs, fig = multitaper_spectrogram(data, fs, frequency_range, time_bandwidth, num_tapers,
                                                        window_params, min_nfft, detrend_opt, multiprocess, cpus,
                                                        weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
    time_dash = np.arange(0, final_time, 30)
    plt.vlines(time_dash, ymin=0, ymax=np.max(frequency_range) * np.ones(time_dash.shape), colors='blue', ls='--', lw=2,
               label='vline_multiple - partial height')
    plt.title('wavelet_raw_data')
    plt.xlabel('time')
    plt.ylabel('frequency')
    fig.savefig(p2+'wavelet')

