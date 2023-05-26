import mne_bids
import mne
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
from ieeg_func.classes.cls import resample, smooth_signal, zscore


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
                                      extension='vhdr',
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
                                      extension='tsv',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)
        self.channels_path = str(bids_path)

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      session=self.session, suffix='electrodes',
                                      extension='tsv',
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
            self.raw.notch_filter(freqs=np.arange(50, 251, 50))
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

    def extract_bands(self,hilber, smooth=False):
        if self.raw_car is not None:
            bands2 = self._compute_band_envelopes(hilber)
            bands3 = self._crop_band_envelopes()
            bands4 = self._resample_band_envelopes()
            if smooth: self._smooth_band_envelopes()
            bands5, bands6, band_block_means = self._compute_block_means_per_band()
        return bands5

    def _compute_band_envelopes(self, hilber):
        if self.raw_car is not None:
            bands = {'delta': [1, 4], 'theta': [5, 8], 'alpha': [8, 12], 'beta': [13, 24], 'gamma': [60, 120]}
            self.bands1 = OrderedDict.fromkeys(bands.keys())
            for key in self.bands1.keys():
                if hilber:
                    self.bands1[key] = self.raw_car.copy().filter(bands[key][0], bands[key][1]).apply_hilbert(
                        envelope=True).get_data().T
                else:
                    self.bands1[key] = self.raw_car.copy().filter(bands[key][0], bands[key][1]).get_data().T
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
                self.bands3[key] = resample(self.bands2[key], 25, int(self.raw.info['sfreq']))
        return self.bands3

    def _smooth_band_envelopes(self):
        if self.bands1 is not None:
            for key in self.bands1.keys():
                self.bands[key] = np.apply_along_axis(smooth_signal, 0, self.bands[key], 5)

    def _compute_block_means_per_band(self):
        if self.bands1 is not None:
            self.band_block_means = OrderedDict.fromkeys(self.bands1.keys())
            band6 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                band6[key] = zscore(self.bands3[key][:self.expected_duration * 25])
                band7 = band6[key].reshape((-1, 750, band6[key].shape[-1]))  # 13 blocks in chill or 6 in rest
                self.band_block_means[key] = np.mean(band7, 1)
        return band6, band7, self.band_block_means


def gdata(bids_dir,sub_list,number,just_gamma,hilber):
    subjects = mne_bids.get_entity_vals(bids_dir, 'subject')
    subjects2 = subjects[:number]
    acqs = mne_bids.get_entity_vals(os.path.join(bids_dir, 'sub-' + subjects[0], 'ses-iemu', 'ieeg'), 'acquisition')
    acq = acqs[0]
    band_all_patient=[]
    raw_car_all=[]
    subject_list = []
    for subject in subjects2:
        if 'iemu' in mne_bids.get_entity_vals(os.path.join(bids_dir, 'sub-' + subject), 'session'):
            if sub_list==False:
                s = SubjectDataset(bids_dir, subject, acquisition=acq)
                raw_car = s.preprocess()
                events, events_id = s.extract_events()
                band= s.extract_bands(hilber)
                if just_gamma:
                    band_all_patient.append(band['gamma'])
                else:
                    band_all_patient.append(band)
                raw_car_all.append(raw_car)
            subject_list.append(subject)
    return band_all_patient,raw_car_all,subject_list