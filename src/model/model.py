import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pickle as pkl
from collections import Counter
from xgboost import XGBClassifier
from src.visualization.visualization_utils import (_save_plot_data,
                                                   _plot_save_ensemble_performance,
                                                   _plot_max_performance,
                                                   _write_max_performance_csv,
                                                   _plot_ensemble_f_measure,
                                                   _plot_patient_f_measure,
                                                   _save_patient_results_to_csv)


class ModelSinglePatient:
    """Class for training and evaluating classification models on single patient data."""

    def __init__(self, feature_matrix, path, settings, channel_names_list, label):
        """
        Initialize the ModelSinglePatient class.

        Parameters:
        -----------
        feature_matrix : numpy.ndarray
            Feature matrix for each patient
        path : object
            Object containing path information for saving results
        settings : object
            Object containing settings for classification and balancing
        channel_names_list : list
            List of channel names for each patient
        label : numpy.ndarray
            Labels for classification
        """
        self.feature_matrix = feature_matrix
        self.label = label
        self.settings = settings
        self.num_patient = settings.num_patient
        self.path = path
        self.channel_names_list = channel_names_list
        self.classifiers = {
            'Logistic_regression': LogisticRegression,
            'SVM': SVC,
            'Naive_bayes': GaussianNB,
            'XGBoost': XGBClassifier,
            'RandomForest': RandomForestClassifier
        }

        # Set up professional plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

    def create_model(self):
        """Create and evaluate models for all combinations of balancing and classification types."""
        list_type_balancing = self.settings.list_type_balancing.keys()
        list_type_classification = self.settings.list_type_classification.keys()

        for type_balancing in list_type_balancing:
            for type_classification in list_type_classification:
                if (self.settings.list_type_classification[type_classification] and
                        self.settings.list_type_balancing[type_balancing]):
                    print(f"\n =================================== \n"
                          f"\n classifier:{type_classification} , balancing:{type_balancing}")

                    results = self.classifier(
                        type_classification=type_classification,
                        type_balancing=type_balancing
                    )

                    self.save_plot_result(
                        ch_name_list=self.channel_names_list,
                        type_classification=type_classification,
                        type_balancing=type_balancing,
                        **results
                    )

    def normalize(self, x):
        """
        Normalize features using MinMaxScaler.

        Parameters:
        -----------
        x : numpy.ndarray
            Features to normalize

        Returns:
        --------
        numpy.ndarray
            Normalized features
        """
        scaler = MinMaxScaler()
        scaler.fit(x)
        return scaler.transform(x)

    def resample_data(self, x, y, type_balancing):
        """
        Resample data using SMOTE to address class imbalance.

        Parameters:
        -----------
        x : numpy.ndarray
            Features
        y : numpy.ndarray
            Labels
        type_balancing : str
            Type of balancing to use

        Returns:
        --------
        tuple
            Resampled features and labels
        """
        oversample = SMOTE()
        x, y = oversample.fit_resample(x, y)
        return x, y

    def get_classifier(self, type_classification):
        """
        Get classifier instance based on classification type.

        Parameters:
        -----------
        type_classification : str
            Type of classifier to use

        Returns:
        --------
        object
            Classifier instance
        """
        if type_classification == 'XGBoost':
            return XGBClassifier(eval_metric='mlogloss')
        elif type_classification == 'RandomForest':
            return RandomForestClassifier(n_estimators=100)
        else:
            return self.classifiers[type_classification]()

    def balance_learn_model(self, x_train, y_train, type_balancing, type_classification):
        """
        Balance data if needed and train a classifier.

        Parameters:
        -----------
        x_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        type_balancing : str
            Type of balancing to use
        type_classification : str
            Type of classifier to use

        Returns:
        --------
        object
            Trained classifier
        """
        if not self.settings.list_type_balancing['Without_balancing']:
            x_train, y_train = self.resample_data(x_train, y_train, type_balancing)

        cls = self.get_classifier(type_classification)
        cls.fit(x_train, y_train)
        return cls

    def max_voting(self, y_pre_matrix, y_test_all_fold, f_measure_val, num_classifier_ensemble):
        """
        Perform max voting ensemble across classifiers.

        Parameters:
        -----------
        y_pre_matrix : numpy.ndarray
            Predictions from all classifiers
        y_test_all_fold : list
            Test labels for all folds
        f_measure_val : numpy.ndarray
            F-measure values for validation set
        num_classifier_ensemble : int
            Number of classifiers to include in ensemble

        Returns:
        --------
        tuple
            F-measure for ensemble and positions of classifiers
        """
        num_folds = y_pre_matrix.shape[1]
        f_measure_ensemble = np.zeros((num_folds, num_classifier_ensemble))
        pos_classifier_all_fold = []

        for fold_num in range(num_folds):
            # Sort classifiers by performance
            f_measure_sort = sorted(np.unique(f_measure_val[:, fold_num]), reverse=True)
            pos_classifier = []
            for k in range(len(f_measure_sort)):
                pos = np.where(f_measure_val[:, fold_num] == f_measure_sort[k])
                pos_classifier.extend([int(x) for x in pos[0]])

            # Evaluate different ensemble sizes
            for i in range(num_classifier_ensemble):
                y_pre_voting = []
                for j in range(i + 1):
                    y_pre_voting.append(
                        y_pre_matrix[pos_classifier[j], fold_num, :len(y_test_all_fold[fold_num])]
                    )

                # Perform majority voting
                y_pre_ensemble = []
                for trial_num in range(len(y_pre_voting[0])):
                    y_pre_one_trial = [y_pre_voting[classifier_num][trial_num]
                                       for classifier_num in range(len(y_pre_voting))]
                    majority_label, _ = Counter(y_pre_one_trial).most_common()[0]
                    y_pre_ensemble.append(majority_label)

                # Calculate metrics
                out = precision_recall_fscore_support(
                    y_test_all_fold[fold_num],
                    y_pre_ensemble,
                    average='weighted',
                    zero_division=0
                )
                f_measure_ensemble[fold_num, i] = out[2]

            pos_classifier_all_fold.append(pos_classifier)

        return f_measure_ensemble, pos_classifier_all_fold

    def check_nan(self, feature_matrix):
        """
        Remove features with NaN values.

        Parameters:
        -----------
        feature_matrix : numpy.ndarray
            Feature matrix

        Returns:
        --------
        numpy.ndarray
            Feature matrix without NaN features
        """
        nan_feature_indices = np.where(np.any(np.isnan(feature_matrix), axis=(0, 1)))[0]

        if nan_feature_indices.size > 0:
            cleaned_feature_matrix = np.delete(feature_matrix, nan_feature_indices, axis=2)
        else:
            cleaned_feature_matrix = feature_matrix
        return cleaned_feature_matrix

    def classifier(self, type_classification, type_balancing):
        """
        Train and evaluate classifiers for each patient.

        Parameters:
        -----------
        type_classification : str
            Type of classifier to use
        type_balancing : str
            Type of balancing to use

        Returns:
        --------
        dict
            Dictionary containing all evaluation metrics
        """
        f_measure_all = []
        f_measure_bestchannel = np.zeros(self.num_patient)
        precision_all = []
        recall_all = []
        num_classifier_ensemble = 30
        f_measure_all_ensemble = np.zeros((self.num_patient, 5, num_classifier_ensemble))
        pos_classifier_all = []

        for patient in range(self.num_patient):
            print(f'\n patient_{patient} from {self.num_patient}')
            feature = self.check_nan(self.feature_matrix[patient])

            num_electrodes = self.feature_matrix[patient].shape[0]
            f_measure = np.zeros((num_electrodes, 5))
            f_measure_val = np.zeros((num_electrodes, 5))
            precision = np.zeros((num_electrodes, 5))
            recall = np.zeros((num_electrodes, 5))
            y_pre_matrix = np.zeros((num_electrodes, 5, int(0.2 * len(self.label)) + 1))

            # Process each electrode
            for electrode in range(num_electrodes):
                x = feature[electrode, :, :]
                x[np.isinf(x)] = 0
                data_norm = self.normalize(x)
                data_all_fold = {'y_test': [], 'y_val_fold': []}

                # Cross-validation
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for i, (train_index, test_index) in enumerate(kf.split(data_norm, self.label)):
                    x_train, x_test = data_norm[train_index], data_norm[test_index]
                    y_train, y_test = self.label[train_index], self.label[test_index]
                    data_all_fold['y_test'].append(y_test)

                    # Split training data into train and validation
                    (x_train_fold, x_val_fold,
                     y_train_fold, y_val_fold,
                     indices_train, indices_val) = train_test_split(
                        x_train, y_train, range(len(x_train)),
                        test_size=0.2, random_state=42, stratify=y_train
                    )
                    val_ind = [train_index[i] for i in indices_val]
                    data_all_fold['y_val_fold'].append(y_val_fold)

                    # Train and evaluate model
                    cls = self.balance_learn_model(
                        x_train_fold, y_train_fold,
                        type_balancing, type_classification
                    )

                    # Evaluate on validation set
                    y_pre_fold = cls.predict(x_val_fold)
                    out_val = precision_recall_fscore_support(
                        y_val_fold, y_pre_fold,
                        average='weighted', zero_division=0
                    )
                    f_measure_val[electrode, i] = out_val[2]

                    # Evaluate on test set
                    y_pre = cls.predict(x_test)
                    y_pre_matrix[electrode, i, :len(y_pre)] = y_pre

                    out_test = precision_recall_fscore_support(
                        y_test, y_pre,
                        average='weighted', zero_division=0
                    )
                    precision[electrode, i] = out_test[0]
                    recall[electrode, i] = out_test[1]
                    f_measure[electrode, i] = out_test[2]

            # Calculate ensemble performance
            f_measure_all_ensemble[patient, :, :], pos_classifier = self.max_voting(
                y_pre_matrix, data_all_fold['y_test'],
                f_measure_val, num_classifier_ensemble
            )
            pos_classifier_all.append(pos_classifier)

            # Find best channel performance
            best_channel_f_measures = []
            for k in range(5):
                pos = np.argmax(f_measure_val[:, k])
                best_channel_f_measures.append(f_measure[pos, k])
            f_measure_bestchannel[patient] = np.mean(best_channel_f_measures)

            f_measure_all.append(f_measure)
            precision_all.append(precision)
            recall_all.append(recall)

        return {
            'f_measure_all': f_measure_all,
            'precision_all': precision_all,
            'recall_all': recall_all,
            'f_measure_all_ensemble': f_measure_all_ensemble,
            'f_measure_bestchannel': f_measure_bestchannel,
            'pos_classifier_all_fold': pos_classifier_all
        }

    def save_plot_result(self, ch_name_list, f_measure_all, f_measure_bestchannel, precision_all,
                         recall_all, f_measure_all_ensemble, pos_classifier_all_fold,
                         type_classification, type_balancing):
        """
        Save and plot results.

        Parameters:
        -----------
        ch_name_list : list
            List of channel names for each patient
        f_measure_all : list
            F-measures for all patients
        f_measure_bestchannel : numpy.ndarray
            Best F-measure for each patient
        precision_all : list
            Precision values for all patients
        recall_all : list
            Recall values for all patients
        f_measure_all_ensemble : numpy.ndarray
            F-measures for ensemble classifiers
        pos_classifier_all_fold : list
            Positions of classifiers for all folds
        type_classification : str
            Type of classifier used
        type_balancing : str
            Type of balancing used
        """
        max_performance_all = []
        result_path = self.path.path_results_classifier[type_classification + type_balancing]
        ensemble_path = self.path.path_results_ensemble_classifier[type_classification + type_balancing]

        # Process each patient's results
        for patient in range(self.num_patient):
            # Calculate statistics
            f_measure_mean = np.mean(f_measure_all[patient], axis=1)
            f_measure_var = np.var(f_measure_all[patient], axis=1)
            precision_mean = np.mean(precision_all[patient], axis=1)
            precision_var = np.var(precision_all[patient], axis=1)
            recall_mean = np.mean(recall_all[patient], axis=1)
            recall_var = np.var(recall_all[patient], axis=1)

            # Store best performance
            best_electrode_idx = f_measure_mean.argmax()
            max_performance = {
                'max_f_measure': np.max(f_measure_mean),
                'best_electrode': ch_name_list[patient][best_electrode_idx],
                'num_best_electrode': best_electrode_idx
            }
            max_performance_all.append(max_performance)

            # Save patient results to CSV
            _save_patient_results_to_csv(
                patient, ch_name_list[patient],
                f_measure_mean, f_measure_var,
                precision_mean, precision_var,
                recall_mean, recall_var,
                result_path
            )

            # Plot F-measure for all channels
            _plot_patient_f_measure(
                patient, f_measure_mean, f_measure_var,
                result_path
            )

            # Save data for plotting
            _save_plot_data(
                patient, ch_name_list[patient],
                f_measure_mean, f_measure_var,
                result_path
            )

            # Plot ensemble F-measure
            _plot_ensemble_f_measure(
                patient, f_measure_all_ensemble,
                ensemble_path
            )

        # Save classifier positions
        with open(f"{result_path}patient_pos_classifier.pkl", 'wb') as f:
            pkl.dump(pos_classifier_all_fold, f)

        # Plot max performance across patients
        _plot_max_performance(f_measure_bestchannel, result_path)

        # Save max performance data
        np.save(f"{result_path}max_performance_all.npy", f_measure_bestchannel)

        # Plot and save ensemble performance
        _plot_save_ensemble_performance(
            f_measure_all_ensemble,
            result_path
        )

        # Write summary CSV for all patient
        _write_max_performance_csv(
            max_performance_all, f_measure_all_ensemble,
            type_classification, type_balancing,
            result_path
        )
