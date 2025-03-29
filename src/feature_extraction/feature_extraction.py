import numpy as np
import pandas as pd
import scipy
import math
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class FeatureExtractor:
    """
    A class for extracting various signal features from biomedical data.

    Extracts time-domain, frequency-domain, and non-linear features from signals
    and organizes them into feature matrices for further analysis.
    """

    def __init__(self, channel_names, path, settings):
        """
        Initialize the FeatureExtractor.

        Args:
            channel_names: Names of channels for each patient
            onset_1: Indices of positive class onsets
            onset_0: Indices of negative class onsets
            path: Path object containing directory paths
            settings: Dictionary containing configuration parameters
        """
        print("\n==================================="
              "\nExtracting features")

        self.channel_names = channel_names
        self.onset_1 = 0
        self.onset_0 = 0
        self.fs = settings.fs  # Sampling frequency
        self.path = path
        self.settings = settings
        self.feature_list = settings.feature_list
        self.feature_all_patient = []

    @staticmethod
    def sampen(signal, m, r):
        """
        Calculate the sample entropy for a given signal.

        Sample entropy measures the complexity of a time series by quantifying
        the probability that similar patterns will remain similar.

        Args:
            signal: Input signal array
            m: Template length (embedding dimension)
            r: Similarity threshold

        Returns:
            Sample entropy value
        """
        n = len(signal)
        xm_A, xm_B = [], []

        # Create template vectors
        for i in range(n - m):
            xm_B.append(signal[i:i + m])
        for i in range(n - (m + 1)):
            xm_A.append(signal[i:i + (m + 1)])

        # Count matches
        B, A = 0, 0
        for i in range(len(xm_B)):
            for j in range(len(xm_B)):
                if i != j and np.max(np.abs(np.array(xm_B[i]) - np.array(xm_B[j]))) < r:
                    B += 1

        for i in range(len(xm_A)):
            for j in range(len(xm_A)):
                if i != j and np.max(np.abs(np.array(xm_A[i]) - np.array(xm_A[j]))) < r:
                    A += 1

        # Avoid division by zero
        if B == 0:
            return np.nan

        return -np.log(A / B)

    def _linear_function(self, x, a, b):
        return a * x + b

    def hfd(self, signal, kmax=20):
        """
        Calculate the Higuchi Fractal Dimension (HFD) of a signal.

        HFD quantifies the complexity of a time series by estimating its
        fractal dimension in the time domain.

        Args:
            signal: Input signal array
            kmax: Maximum k value for curve fitting

        Returns:
            Higuchi Fractal Dimension
        """
        N = len(signal)
        L = []

        for k in range(1, kmax + 1):
            Lk = 0

            # Calculate the normalized length for k-delay
            for m in range(k):
                # Create a new subsequence
                indices = np.arange(m, N, k)
                subsequence = signal[indices]

                # Calculate the length
                if len(subsequence) > 1:
                    L_m = np.sum(np.abs(np.diff(subsequence)))
                    L_m = L_m * (N - 1) / ((N - m) // k * k)
                    Lk += L_m

            L.append(Lk / k)

        # Curve fitting
        x = np.log(1 / np.arange(1, kmax + 1))
        y = np.log(np.array(L) + 1e-10)  # Add small constant to avoid log(0)

        try:
            popt, _ = curve_fit(self._linear_function, x, y)
            return popt[0]  # Slope is the fractal dimension
        except RuntimeError:
            return np.nan

    def get_feature(self, signal_with_hilbert, feature):
        """
        Extract features from a single signal segment.

        Args:
            signal_with_hilbert: Signal processed with Hilbert transform
            feature: Dictionary to store extracted features

        Returns:
            Updated feature dictionary
        """
        # Time-domain features
        if self.feature_list['AVG']:
            feature['AVG'].append(np.mean(signal_with_hilbert))

        if self.feature_list['RMS']:
            rms_value = np.sqrt(np.mean(np.square(signal_with_hilbert)))
            feature['RMS'].append(rms_value)

        if self.feature_list['Max_peak']:
            feature['Max_peak'].append(np.max(signal_with_hilbert))

        if self.feature_list['Variance']:
            feature['Variance'].append(np.var(signal_with_hilbert))

        if self.feature_list['Coastline']:
            feature['Coastline'].append(np.sum(np.abs(np.diff(signal_with_hilbert))))

        # Frequency-domain features
        if self.feature_list['Band_powers'] or self.feature_list['Spectral_edge_frequency'] or self.feature_list[
            'Spectral_entropy']:
            freq, psd = scipy.signal.periodogram(signal_with_hilbert, fs=self.fs, scaling='density')

        if self.feature_list['Band_powers']:
            feature['Band_powers'].append(np.mean(psd))

        if self.feature_list['Spectral_edge_frequency']:
            total_psd = np.sum(psd)
            cumulative_psd = np.cumsum(psd)
            edge_idx = np.where(cumulative_psd >= 0.9 * total_psd)[0]
            edge_idx = edge_idx[0] if len(edge_idx) > 0 else len(freq) - 1
            feature['Spectral_edge_frequency'].append(freq[edge_idx])

        # Statistical features
        if self.feature_list['Skewness']:
            feature['Skewness'].append(scipy.stats.skew(signal_with_hilbert))

        if self.feature_list['Kurtosis']:
            feature['Kurtosis'].append(scipy.stats.kurtosis(signal_with_hilbert))

        # Auto-correlation features
        if self.feature_list['Autocorrelation_function']:
            acf = sm.tsa.stattools.acf(signal_with_hilbert, nlags=40)
            zc_acf = 0
            for i in range(2, len(acf)):
                if acf[i] * acf[i - 1] < 0:
                    zc_acf = i
                    break

            feature['Autocorrelation_function'].append(acf[10])

        # Hjorth parameters
        if self.feature_list['Hjorth_mobility'] or self.feature_list['Hjorth_complexity']:
            data_diff = np.diff(signal_with_hilbert)

        if self.feature_list['Hjorth_mobility']:
            mobility = np.std(data_diff) / np.std(signal_with_hilbert)
            feature['Hjorth_mobility'].append(mobility)

        if self.feature_list['Hjorth_complexity']:
            data_diff2 = np.diff(data_diff)
            mobility = np.std(data_diff) / np.std(signal_with_hilbert)
            complexity = np.std(data_diff2) / (mobility * np.std(data_diff))
            feature['Hjorth_complexity'].append(complexity)

        # Nonlinear features
        if self.feature_list['Nonlinear_energy']:
            feature['Nonlinear_energy'].append(np.mean(np.abs(signal_with_hilbert)))

        if self.feature_list['Spectral_entropy']:
            p_norm = psd / (np.sum(psd) + 1e-12)
            spectral_entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
            feature['Spectral_entropy'].append(spectral_entropy)

        if self.feature_list['Sample_entropy']:
            se = self.sampen(
                signal=signal_with_hilbert,
                m=1,
                r=0.2 * np.std(signal_with_hilbert)
            )
            feature['Sample_entropy'].append(se)

        if self.feature_list['Renyi_entropy']:
            alpha = 2
            # Use binning to reduce computational complexity
            hist, _ = np.histogram(signal_with_hilbert, bins=20)
            hist = hist / np.sum(hist)
            non_zero = hist > 0

            if np.any(non_zero):
                sum_prob = np.sum(hist[non_zero] ** alpha)
                renyi = (1 / (1 - alpha)) * np.log(sum_prob)
                feature['Renyi_entropy'].append(renyi)
            else:
                feature['Renyi_entropy'].append(0)

        if self.feature_list['Shannon_entropy']:
            # Use binning for Shannon entropy calculation
            hist, _ = np.histogram(signal_with_hilbert, bins=20)
            hist = hist / np.sum(hist)
            non_zero = hist > 0

            if np.any(non_zero):
                shannon = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
                feature['Shannon_entropy'].append(shannon)
            else:
                feature['Shannon_entropy'].append(0)

        # Spike features
        if self.feature_list['Spikes']:
            peaks, _ = find_peaks(signal_with_hilbert, distance=40)
            feature['Spikes'].append(len(peaks))

        # Fractal dimension
        if self.feature_list['Fractal_dimension']:
            fd = self.hfd(signal_with_hilbert)
            feature['Fractal_dimension'].append(fd)

        return feature

    def get_feature_all(self, data):
        """
        Extract features from all patients and all electrodes.

        Args:
            data: List of raw data arrays
        """
        if not self.settings.load_feature_matrix:
            num_patient = self.settings.num_patient

            for patient in range(num_patient):
                print(f'Processing patient {patient + 1}/{num_patient}')
                single_patient_feature = []

                # for num_electrode in range(data[patient].shape[1]):
                for num_electrode in range(2):
                    feature = {key: [] for key in self.feature_list.keys() if self.feature_list[key]}

                    for i in range(data[patient].shape[0]):
                        feature = self.get_feature(data[patient][i, num_electrode, :], feature)

                    feature_df = pd.DataFrame(feature)
                    single_patient_feature.append(feature_df)

                self.feature_all_patient.append(single_patient_feature)

            if self.settings.save_feature_matrix:
                with open(f"{self.path.path_save_data}/feature_all_patient_df.pkl", 'wb') as f:
                    pickle.dump(self.feature_all_patient, f)
        else:
            # Load precomputed features
            task = self.settings.task
            if task == 'Singing_Music':
                path_feature = f"{self.path.path_processed_data}Music_Reconstruction"
            elif task == 'Move_Rest':
                path_feature = f"{self.path.path_processed_data}Upper_Limb_Movement"
            else:
                path_feature = f"{self.path.path_processed_data}Audio_visual"

            file_path = f"{path_feature}/feature_all_patient_df_{task}.pkl"
            with open(file_path, 'rb') as f:
                self.feature_all_patient = pickle.load(f)

    def create_feature_matrix(self):
        """
        Convert feature DataFrames to feature matrices.

        Returns:
            List of feature matrices for all patients
        """
        print("\n==================================="
              "\nConverting features from pandas.DataFrame to matrix")

        if not self.settings.load_feature_matrix:
            feature_matrix_all = []
            num_features = sum(self.feature_list.values())

            for patient in range(len(self.feature_all_patient)):
                num_channels = len(self.channel_names[patient])
                num_samples = self.feature_all_patient[patient][0].shape[0]

                feature_matrix = np.zeros((num_channels, num_samples, num_features))

                # for electrode in range(num_channels):
                for electrode in range(2):
                    feature_matrix[electrode, :, :] = self.feature_all_patient[patient][electrode].values

                feature_matrix_all.append(feature_matrix)

            if self.settings.save_feature_matrix:
                with open(f"{self.path.path_save_data}/feature_matrix_all.pkl", 'wb') as f:
                    pickle.dump(feature_matrix_all, f)
        else:
            # Load precomputed feature matrices
            task = self.settings.task
            if task == 'Singing_Music':
                path_feature = f"{self.path.path_processed_data}Music_Reconstruction"
            elif task == 'Move_Rest':
                path_feature = f"{self.path.path_processed_data}Upper_Limb_Movement"
            else:
                path_feature = f"{self.path.path_processed_data}Audio_visual"

            file_path = f"{path_feature}/feature_matrix_all_{task}.pkl"
            with open(file_path, 'rb') as f:
                feature_matrix_all = pickle.load(f)

        return feature_matrix_all
