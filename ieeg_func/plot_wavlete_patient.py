import numpy as np
import matplotlib.pyplot as plt
import mne

def plot_wavelet(raw_car_all,patient,electrode):
    time_dash = np.arange(0, 390, 30)
    sfreq = raw_car_all[patient].info['sfreq']
    num_electrode = raw_car_all[patient].ch_names.index(electrode)
    data = raw_car_all[patient].get_data().T[:int(sfreq) * 390, num_electrode]
    data2 = data.reshape((1, 1, data.shape[0]))
    n_cycles = 7.0
    freqs = np.arange(1, 120, 1)
    out = mne.time_frequency.tfr._compute_tfr(data2, freqs, sfreq, method='morlet', n_cycles=7.0)

    single_patient_output = np.abs(out[0, 0, :, :])
    single_patient_output_normalized = single_patient_output / np.max(single_patient_output, axis=-1, keepdims=True)
    time = np.arange(0, 390, 1 / int(sfreq))
    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(time, freqs, single_patient_output_normalized, shading='auto')
    plt.vlines(time_dash, ymin=0, ymax=np.max(freqs) * np.ones(time_dash.shape), colors='red', ls='--', lw=2,label='vline_multiple - partial height')
    fig.savefig('img_wavelet.png')
    plt.show()
