import mne_bids
import mne
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
import warnings
import pickle as pkl
import re
from fractions import Fraction
from scipy.signal import resample_poly


class SubjectDataset(object):
    def __init__(self, input_root, subject, preload=True, **kwargs):
        self.root = input_root
        self.subject = subject
        self.datatype = 'ieeg'
        self.acquisition = None
        self.__dict__.update(kwargs)
        self.task = 'film'
        self.expected_duration = 390

        if preload:
            self._set_paths()
            self._load_data()
            self._get_bad_electrodes()

    def _set_paths(self):
        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      task=self.task,
                                      suffix='ieeg',
                                      extension='.vhdr',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)

        print(self.task)
        assert len(bids_path.match()) == 1, 'None or more than one run for task is found'

        self.raw_path = str(bids_path.match()[0])
        self.run = mne_bids.get_entities_from_fname(bids_path.match()[0])['run']
        self.session = mne_bids.get_entities_from_fname(bids_path.match()[0])['session']

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      task=self.task,
                                      session=self.session,
                                      suffix='channels', run=self.run,
                                      extension='.tsv',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)
        self.channels_path = str(bids_path)

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      session=self.session, suffix='electrodes',
                                      extension='.tsv',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)
        self.electrodes_path = str(bids_path)

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      suffix='T1w',
                                      extension='.nii.gz',
                                      root=self.root)
        self.anat_path = str(bids_path.match()[0])
        self.anat_session = mne_bids.get_entities_from_fname(bids_path.match()[0])['session']
        print('Picking up BIDS files done')

    def _load_data(self):
        self.raw = mne.io.read_raw_brainvision(self.raw_path,
                                               eog=(['EOG']),
                                               misc=(['OTHER', 'ECG', 'EMG']),
                                               scale=1.0,
                                               preload=False,
                                               verbose=True)
        self.channels = pd.read_csv(self.channels_path, sep='\t', header=0, index_col=None)
        self.electrodes = pd.read_csv(self.electrodes_path, sep='\t', header=0, index_col=None)
        self.raw.set_channel_types({ch_name: str(x).lower()
        if str(x).lower() in ['ecog', 'seeg', 'eeg'] else 'misc'
                                    for ch_name, x in zip(self.raw.ch_names, self.channels['type'].values)})
        self.other_channels = self.channels['name'][~self.channels['type'].isin(['ECOG', 'SEEG'])].tolist()
        self.raw.drop_channels(self.other_channels)
        print('Loading BIDS data done')

    def _get_bad_electrodes(self):
        self.bad_electrodes = self.channels['name'][
            (self.channels['type'].isin(['ECOG', 'SEEG'])) & (self.channels['status'] == 'bad')].tolist()
        self.all_electrodes = self.channels['name'][(self.channels['type'].isin(['ECOG', 'SEEG']))].tolist()
        print('Getting bad electrodes done')

    def sort_nicely(self, l):
        """ Sort the given list in the way that humans expect.
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        l.sort(key=alphanum_key)

    def resample(self, x, sr1, sr2, axis=0):
        '''sr1: target, sr2: source'''
        a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
        return resample_poly(x, a, b, axis).astype(np.float32)

    def smooth_signal(self, y, n):
        box = np.ones(n) / n
        ys = np.convolve(y, box, mode='same')
        return ys

    def zscore(self,x):
        return (x - np.mean(x, 0, keepdims=True)) / np.std(x, 0, keepdims=True)

    def preprocess(self):
        raw_dbe = self._discard_bad_electrodes()
        raw_nf = self._notch_filter()
        raw_car = self._common_average_reference()
        return raw_car

    def _discard_bad_electrodes(self):
        if self.raw is not None:
            [self.bad_electrodes.remove(i) for i in self.bad_electrodes if i not in self.raw.ch_names]
            self.raw.info['bads'].extend([ch for ch in self.bad_electrodes])
            print('Bad channels indicated: ' + str(self.bad_electrodes))
            print('Dropped ' + str(self.raw.info['bads']) + 'channels')
            self.raw.drop_channels(self.raw.info['bads'])
            self.raw.load_data()
            print('Remaining channels ' + str(self.raw.ch_names))
            return self.raw

    def _notch_filter(self):
        if self.raw is not None:
            if np.any(np.isnan(self.raw._data)):
                # self.raw._data[np.isnan(self.raw._data)]=0 # bad hack for nan values
                warnings.warn('There are NaNs in the data, replacing with 0')
            self.raw.notch_filter(freqs=np.arange(50, 251, 50), verbose=False)
            print('Notch filter done')
        return self.raw

    def _common_average_reference(self):
        if self.raw is not None:
            self.raw_car, _ = mne.set_eeg_reference(self.raw.copy(), 'average')
            print('CAR done')
        return self.raw_car

    def extract_events(self, plot=False):
        events, events_id = self._read_events()
        if plot:
            self._plot_events()
        return events, events_id

    def _read_events(self):
        if self.raw is not None:
            if self.task == 'film':
                custom_mapping = {'Stimulus/music': 2,
                                  'Stimulus/speech': 1,
                                  'Stimulus/end task': 5}
            elif self.task == 'rest':
                custom_mapping = {'Stimulus/start task': 1,
                                  'Stimulus/end task': 2}
            else:
                raise NotImplementedError
            self.events, self.event_id = mne.events_from_annotations(self.raw, event_id=custom_mapping,
                                                                     use_rounding=False)
            print('Reading events done')
        return self.events, self.event_id

    def _plot_events(self):
        if self.raw_car is not None:
            self.raw_car.plot(events=self.events,
                              start=0,
                              duration=180,
                              color='gray',
                              event_color={2: 'g', 1: 'r'},
                              bgcolor='w')

    def extract_bands(self, apply_hilbert, smooth=False):
        if self.raw_car is not None:
            self._compute_band_envelopes(apply_hilbert)
            self._crop_band_envelopes()
            self._resample_band_envelopes()
            if smooth:
                self._smooth_band_envelopes()
            bands5, bands6, band_block_means = self._compute_block_means_per_band()
        return bands5

    def _compute_band_envelopes(self, apply_hilbert):
        if self.raw_car is not None:
            bands = {'delta': [1, 4], 'theta': [5, 8], 'alpha': [8, 12], 'beta': [13, 24], 'gamma': [60, 120]}
            self.bands1 = OrderedDict.fromkeys(bands.keys())
            for key in self.bands1.keys():
                if apply_hilbert:
                    self.bands1[key] = self.raw_car.copy().filter(bands[key][0], bands[key][1],
                                                                  verbose=False).apply_hilbert(
                        envelope=True).get_data().T
                else:
                    self.bands1[key] = self.raw_car.copy().filter(bands[key][0], bands[key][1],
                                                                  verbose=False).get_data().T
            print('Extracting band envelopes done')
        return self.bands1

    def _crop_band_envelopes(self):
        if self.bands1 is not None:
            self.bands2 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                self.bands2[key] = self.bands1[key][self.events[0, 0]:self.events[-1, 0]]
        return self.bands2

    def _resample_band_envelopes(self):
        if self.bands1 is not None:
            self.bands3 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                self.bands3[key] = self.resample(self.bands2[key], 25, int(self.raw.info['sfreq']))
        return self.bands3

    def _smooth_band_envelopes(self):
        if self.bands1 is not None:
            for key in self.bands1.keys():
                self.bands[key] = np.apply_along_axis(self.smooth_signal, 0, self.bands[key], 5)

    def _compute_block_means_per_band(self):
        if self.bands1 is not None:
            self.band_block_means = OrderedDict.fromkeys(self.bands1.keys())
            band6 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                band6[key] = self.zscore(self.bands3[key][:self.expected_duration * 25])
                band7 = band6[key].reshape((-1, 750, band6[key].shape[-1]))  # 13 blocks in chill or 6 in rest
                self.band_block_means[key] = np.mean(band7, 1)
        return band6, band7, self.band_block_means


def get_data(paths, settings):
    """
    settings should contain number_of_patients, use_only_gamma_band, hilbert
    :param paths:
    :param settings:
    :return:
    """
    number = settings['number_of_patients']

    # List the selected subjects
    subjects = mne_bids.get_entity_vals(paths.path_dataset, 'subject')
    selected_subjects = subjects[:number]

    # Choose clinical as default aquisition type
    acquisition_types = mne_bids.get_entity_vals(
        os.path.join(paths.path_dataset, 'sub-' + subjects[0], 'ses-iemu', 'ieeg'),
        'acquisition')
    acquisition_type = acquisition_types[0]
    print("Selected Acquisition is {}".format(acquisition_type))

    # Load patients data
    band_all_patient_with_hilbert, band_all_patient_without_hilbert, subject_list, channel_names_list = [], [], [], []
    for subject in selected_subjects:
        if 'iemu' in mne_bids.get_entity_vals(os.path.join(paths.path_dataset, 'sub-' + subject), 'session'):
            print("\n =================================== \n"
                  "Patient {} from {}".format(subject, len(selected_subjects)))
            if settings['load_preprocessed_data'] is False:
                subject_data = SubjectDataset(paths.path_dataset, subject, acquisition=acquisition_type)
                # car: common average reference
                raw_car = subject_data.preprocess()
                events, events_id = subject_data.extract_events()
                band_data_with_hilbert = subject_data.extract_bands(apply_hilbert=True)
                band_data_without_hilbert = subject_data.extract_bands(apply_hilbert=False)
                channel_names = raw_car.ch_names

                if settings['save_preprocessed_data'] is True:
                    file_path_subband_data_with_hilbert = paths.path_processed_data + \
                                                          'sub{}_sub-bands_data_with_hilbert.pkl'.format(subject)
                    file_path_subband_data_without_hilbert = paths.path_processed_data + \
                                                             'sub{}_sub-bands_data_without_hilbert.pkl'.format(subject)
                    file_path_raw_data = paths.path_processed_data + 'sub{}_raw_car.pkl'.format(subject)
                    file_path_channel_names = paths.path_processed_data + 'sub{}_channel_names.pkl'.format(subject)
                    with open(file_path_subband_data_with_hilbert, 'wb') as f:
                        pkl.dump(band_data_with_hilbert, f)
                    with open(file_path_subband_data_without_hilbert, 'wb') as f:
                        pkl.dump(band_data_without_hilbert, f)
                    with open(file_path_raw_data, 'wb') as f:
                        pkl.dump(raw_car, f)
                    with open(file_path_channel_names, 'wb') as f:
                        pkl.dump(channel_names, f)
            else:
                file_path_subband_data_with_hilbert = paths.path_processed_data + \
                                                      'sub{}_sub-bands_data_with_hilbert.pkl'.format(subject)
                file_path_subband_data_without_hilbert = paths.path_processed_data + \
                                                         'sub{}_sub-bands_data_without_hilbert.pkl'.format(subject)
                file_path_raw_data = paths.path_processed_data + 'sub{}_raw_car.pkl'.format(subject)
                file_path_channel_names = paths.path_processed_data + 'sub{}_channel_names.pkl'.format(subject)
                if os.path.isfile(file_path_subband_data_without_hilbert):
                    # Load the data from the pickle file
                    with open(file_path_subband_data_with_hilbert, 'rb') as f:
                        band_data_with_hilbert = pkl.load(f)
                    with open(file_path_subband_data_without_hilbert, 'rb') as f:
                        band_data_without_hilbert = pkl.load(f)
                    # with open(file_path_raw_data, 'rb') as f:
                    # raw_car = pkl.load(f)
                    with open(file_path_channel_names, 'rb') as f:
                        channel_names = pkl.load(f)
                else:
                    raise ValueError(f"The file '{file_path_subband_data_without_hilbert}' does not exist. Put "
                                     f"the data in this directory or"
                                     f"run the code with load_preprocessed_data=False to generate the data")
            band_all_patient_with_hilbert.append(band_data_with_hilbert)
            band_all_patient_without_hilbert.append(band_data_without_hilbert)
            channel_names_list.append(channel_names)
    return band_all_patient_with_hilbert, band_all_patient_without_hilbert, channel_names_list


def load_raw_data(paths, settings):
    """
    Load raw data
    :param paths:
    :param settings:
    :return:
    """
    number = settings['number_of_patients']

    # List the selected subjects
    subjects = mne_bids.get_entity_vals(paths.path_dataset, 'subject')
    selected_subjects = subjects[:number]

    # Choose clinical as default aquisition type
    acquisition_types = mne_bids.get_entity_vals(
        os.path.join(paths.path_dataset, 'sub-' + subjects[0], 'ses-iemu', 'ieeg'),
        'acquisition')
    acquisition_type = acquisition_types[0]
    print("Selected Acquisition is {}".format(acquisition_type))

    # Load patients data
    raw_car_all = []
    for subject in selected_subjects:
        if 'iemu' in mne_bids.get_entity_vals(os.path.join(paths.path_dataset, 'sub-' + subject), 'session'):
            print("\n =================================== \n"
                  "Patient {} from {}".format(subject, len(selected_subjects)))
            file_path_raw_data = paths.path_processed_data + 'sub{}_raw_car.pkl'.format(subject)
            if os.path.isfile(file_path_raw_data):
                # Load the data from the pickle file
                with open(file_path_raw_data, 'rb') as f:
                    raw_car = pkl.load(f)

            else:
                raise ValueError(
                    f"The file '{file_path_raw_data}' does not exist. Put the data in this directory or"
                    f"run the code with load_preprocessed_data=False to generate the data")

            raw_car_all.append(raw_car)
    return raw_car_all
