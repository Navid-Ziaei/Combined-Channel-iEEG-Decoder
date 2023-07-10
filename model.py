import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import csv
from collections import Counter
from sklearn.model_selection import train_test_split


class ModelSinglePatient:
    def __init__(self, feature_matrix, path, settings, channel_names_list):

        self.feature_matrix = feature_matrix
        self.settings = settings
        self.num_patient = settings['parameter_classification']['num_patient']
        self.path = path
        self.channel_names_list = channel_names_list

    def create_model(self):
        list_type_balancing = self.settings['list_type_balancing'].keys()
        list_type_classification = self.settings['list_type_classification'].keys()

        for type_balancing in list_type_balancing:
            for type_classification in list_type_classification:
                if self.settings['list_type_classification'][type_classification] & \
                        self.settings['list_type_balancing'][type_balancing]:
                    print("\n =================================== \n"
                          f"\n classifier:{type_classification} , balancing:{type_balancing} is running ...")
                    f_measure_all, precision_all, recall_all, f_measure_all_ensemble = self.classifier(
                        type_classification=type_classification,
                        type_balancing=type_balancing)
                    self.save_plot_result(ch_name_list=self.channel_names_list,
                                          f_measure_all=f_measure_all,
                                          precision_all=precision_all,
                                          recall_all=recall_all,
                                          f_measure_all_ensemble=f_measure_all_ensemble,
                                          type_classification=type_classification,
                                          type_balancing=type_balancing)

    def normalize(self, x):
        scaler = MinMaxScaler()
        scaler.fit(x)
        return scaler.transform(x)

    def resample_data(self, x, y, type_balancing):

        if type_balancing == 'over&down_sampling':
            # define pipeline
            over = SMOTE(sampling_strategy=0.58)
            under = RandomUnderSampler(sampling_strategy=0.66)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            # transform the dataset
            x, y = pipeline.fit_resample(x, y)

        if type_balancing == 'over_sampling':
            # transform the dataset
            oversample = SMOTE()
            x, y = oversample.fit_resample(x, y)

        if type_balancing == 'under_sampling':
            # define pipeline
            under = RandomUnderSampler()
            x, y = under.fit_resample(x, y)

        return x, y

    def combine_classifier(self, y_pre_matrix, y_validation_all_fold, f_measure, num_classifier_ensemble):
        f_measure_ensemble = np.zeros((5, num_classifier_ensemble))

        for fold_num in range(y_pre_matrix.shape[1]):
            f_measure_sort = sorted(np.unique(f_measure[:, fold_num]), reverse=True)
            pos_classifier = []
            for k in range(len(f_measure_sort)):
                pos = np.where(f_measure[:, fold_num] == f_measure_sort[k])
                pos_classifier = pos_classifier + [int(x) for x in pos[0]]

            for i in range(num_classifier_ensemble):
                y_pre_voting = []
                for j in range(i + 1):
                    y_pre_voting.append(y_pre_matrix[pos_classifier[j], fold_num, :len(y_validation_all_fold[fold_num])])

                y_pre_ensemble = []
                for trial_num in range(len(y_pre_voting[0])):
                    y_pre_one_trial = []
                    for classifier_num in range(len(y_pre_voting)):
                        y_pre_one_trial.append(y_pre_voting[classifier_num][trial_num])
                    majority_label, _ = Counter(y_pre_one_trial).most_common()[0]
                    y_pre_ensemble.append(majority_label)
                out = precision_recall_fscore_support(y_validation_all_fold[fold_num], y_pre_ensemble,
                                                      average='weighted',
                                                      zero_division=0)
                f_measure_ensemble[fold_num, i] = out[2]

        return f_measure_ensemble

    def classifier(self, type_classification, type_balancing):
        # question_label=1  answer_label=0
        if self.settings['task'] == 'question&answer':
            label = np.zeros(67)
            label[0:15] = 1
        if self.settings['task'] == 'speech&music':
            label = np.zeros(13)
            label[0:6] = 1
        f_measure_all = []
        precision_all = []
        recall_all = []
        num_classifier_ensemble = 30
        f_measure_all_ensemble = np.zeros((self.num_patient, 5, num_classifier_ensemble))
        for patient in range(self.num_patient):
            print(f'\n patient_{patient} from {self.num_patient}')
            feature = self.feature_matrix[patient]
            f_measure = np.zeros((self.feature_matrix[patient].shape[0], 5))
            precision = np.zeros((self.feature_matrix[patient].shape[0], 5))
            recall = np.zeros((self.feature_matrix[patient].shape[0], 5))
            y_pre_matrix = np.zeros((self.feature_matrix[patient].shape[0], 5, 14))
            for electrode in range(self.feature_matrix[patient].shape[0]):
                x = feature[electrode, :, :]
                x_norm = self.normalize(x)
                y_validation_all_fold = []
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for i, (train_index, test_index) in enumerate(kf.split(x_norm, label)):
                    x_train, x_validation = x_norm[train_index], x_norm[test_index]
                    y_train, y_validation = label[train_index], label[test_index]
                    y_validation_all_fold.append(y_validation)
                    x_train_fold, x_test_fold, y_train_fold, y_test_fold = train_test_split(x_train,
                                                                                            y_train,
                                                                                            test_size=0.2,
                                                                                            random_state=42)
                    if self.settings['task'] == 'question&answer':
                        x_train_fold_balance, y_train_fold_balance = self.resample_data(x_train_fold, y_train_fold,
                                                                                        type_balancing)

                    if type_classification == 'Logistic_regression':
                        cls = LogisticRegression().fit(x_train_fold_balance, y_train_fold_balance)
                    if type_classification == 'SVM':
                        cls = SVC().fit(x_train_fold_balance, y_train_fold_balance)
                    if type_classification == 'Naive_bayes':
                        cls = GaussianNB().fit(x_train_fold_balance, y_train_fold_balance)

                    y_pre_fold = cls.predict(x_test_fold)
                    out = precision_recall_fscore_support(y_test_fold, y_pre_fold, average='weighted', zero_division=0)
                    precision[electrode, i] = out[0]
                    recall[electrode, i] = out[1]
                    f_measure[electrode, i] = out[2]
                    y_pre = cls.predict(x_validation)
                    y_pre_matrix[electrode, i, :len(y_pre)] = y_pre
            f_measure_all_ensemble[patient, :, :] = self.combine_classifier(y_pre_matrix, y_validation_all_fold,
                                                                            f_measure, num_classifier_ensemble)
            f_measure_all.append(f_measure)
            precision_all.append(precision)
            recall_all.append(recall)
        return f_measure_all, precision_all, recall_all, f_measure_all_ensemble

    def save_plot_result(self, ch_name_list, f_measure_all, precision_all, recall_all, f_measure_all_ensemble,
                         type_classification,
                         type_balancing):
        max_performance_all = []
        for patient in range(self.num_patient):
            max_performance = {'max_f_measure': 0, 'best_electrode': 0}
            f_measure_mean = np.mean(f_measure_all[patient], axis=1)
            f_measure_var = np.var(f_measure_all[patient], axis=1)
            precision_mean = np.mean(precision_all[patient], axis=1)
            precision_var = np.var(precision_all[patient], axis=1)
            recall_mean = np.mean(recall_all[patient], axis=1)
            recall_var = np.var(recall_all[patient], axis=1)
            max_performance['max_f_measure'] = np.max(f_measure_mean)
            max_performance['best_electrode'] = ch_name_list[patient][f_measure_mean.argmax()]
            max_performance['num_best_electrode'] = f_measure_mean.argmax()
            max_performance_all.append(max_performance)

            # write csv file for each patient for each classifier
            header = ['num_channel', 'channel', '', 'f_measure', '', 'precision', '', 'recall']
            ch_name = ch_name_list[patient]
            with open(self.path.path_results_classifier[type_classification + type_balancing] +
                      'patient_' + str(patient) + '.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                for i in range(f_measure_all[patient].shape[0]):
                    writer.writerow([i, ch_name[i], ' ', str(f_measure_mean[i]) + '±' + str(f_measure_var[i]), ' ',
                                     str(precision_mean[i]) + '±' + str(precision_var[i]), ' ',
                                     str(recall_mean[i]) + '±' + str(recall_var[i])])

            # plot f_measure for all channel
            channel_index = np.arange(0, f_measure_all[patient].shape[0])
            plt.figure(dpi=300)
            plt.plot(channel_index, f_measure_mean, color='blue', linewidth=2)
            plt.fill_between(channel_index, f_measure_mean - f_measure_var, f_measure_mean + f_measure_var, color='red',
                             alpha=0.2)
            plt.title('patient=' + str(patient), fontsize=15)
            plt.ylabel('f_measure')
            plt.xlabel('channel')
            plt.savefig(self.path.path_results_classifier[type_classification + type_balancing] +
                        'patient_' + str(patient))
            plt.savefig(self.path.path_results_classifier[type_classification + type_balancing] +
                        'patient_' + str(patient)+'.svg')

        # write csv file for max performance all patient for each classifier
        header = ['num_patient', '', 'num_best_channel', '', 'best_channel', '', 'max_f_measure', '',
                  'f_measure_ensemble']
        with open(self.path.path_results_classification + 'max_performance__' + type_classification + '_' +
                  type_balancing + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            for patient in range(self.num_patient):
                writer.writerow([patient, ' ', max_performance_all[patient]['num_best_electrode'], ' ',
                                 max_performance_all[patient]['best_electrode'], ' ',
                                 max_performance_all[patient]['max_f_measure'], ' ',
                                 f_measure_all_ensemble[patient]])

        # plot max performance all patient
        plt.figure(dpi=300)
        max_performance_each_patient = []
        for patient in range(self.num_patient):
            max_performance_each_patient.append(max_performance_all[patient]['max_f_measure'])
        plt.plot(np.arange(0, self.num_patient), max_performance_each_patient, color='blue', linewidth=2)
        plt.title('max_performance_all')
        plt.ylabel('f_measure')
        plt.xlabel('num_patient')
        plt.savefig(self.path.path_results_classification + 'max_performance_all')
        plt.savefig(self.path.path_results_classification + 'max_performance_all.svg')

        # plot f_measure_ensemble for all patient
        f_measure_all_ensemble_mean = np.mean(f_measure_all_ensemble, axis=1)
        f_measure_all_ensemble_std = np.std(f_measure_all_ensemble, axis=1)
        for patient in range(self.num_patient):
            plt.figure(dpi=300)
            plt.plot(f_measure_all_ensemble_mean[patient, :], color='red', linewidth=2)
            plt.fill_between(np.arange(0, f_measure_all_ensemble_mean.shape[1]),
                             f_measure_all_ensemble_mean[patient, :] - f_measure_all_ensemble_std[patient, :],
                             f_measure_all_ensemble_mean[patient, :] + f_measure_all_ensemble_std[patient, :],
                             color='green',
                             alpha=0.2)
            plt.title('f_measure of combining classifier_patient_' + str(patient), fontsize=15)
            plt.ylabel('f_measure')
            plt.xlabel('number of classifier')
            plt.savefig(self.path.path_results_ensemble_classifier + 'patient_' + str(patient))
            plt.savefig(self.path.path_results_ensemble_classifier + 'patient_' + str(patient)+'.svg')

        plt.figure(dpi=300)
        plt.plot(np.max(f_measure_all_ensemble_mean, axis=1), color='red', linewidth=2)
        plt.title('max f_measure of combining classifier')
        plt.ylabel('f_measure')
        plt.xlabel('num_patient')
        plt.savefig(self.path.path_results_classification + 'f_measure_all_ensemble')
        plt.savefig(self.path.path_results_classification + 'f_measure_all_ensemble.svg')
