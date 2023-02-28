import mne
import matplotlib.pyplot as plt
import numpy as np

# add a coment

header_file = 'F:/Datasets/ieeg_visual/ds003688-download/sub-03/ses-iemu/ieeg/sub-03_ses-iemu_task-film_acq' \
              '-clinical_run-1_ieeg.vhdr '
marker_file ='F:/Datasets/ieeg_visual/ds003688-download/sub-03/ses-iemu/ieeg/sub-03_ses-iemu_task-film_acq-clinical_run-1_ieeg.vmrk'

filename = 'F:/Datasets/ieeg_visual/ds003688-download/sub-03/ses-iemu/ieeg/sub-03_ses-iemu_task-film_acq-clinical_run' \
           '-1_ieeg.eeg '

# Load the data using MNE-Python's io module
raw = mne.io.read_raw_brainvision(header_file, preload=True, verbose=False)

# Load the events from the marker file
events = mne.events_from_annotations(raw)
data = raw.get_data()
# Print some information about the loaded data
print(raw.info)

fs = raw.info['sfreq']
time = np.arange(0, data.shape[1])/(fs*60)
plt.figure()
plt.plot(time[:1000], data[33,:1000])
plt.show()
# Print some information about the events
print(events[0][:10])


