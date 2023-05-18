import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import csv
import os

class create_model():
    def __init__(self,feature_matrix,num_patient,path,type_balancing,type_classification):
        self.feature_matrix=feature_matrix
        self.type_classification=type_classification
        self.num_patient=num_patient
        self.type_balancing=type_balancing

        os.makedirs(path + 'create_model_'+type_classification+'_'+type_balancing)
        p2 = os.path.join(path, 'create_model_'+type_classification+'_'+type_balancing + '/')
        self.path = p2


    def norm(self,x):
        scaler = MinMaxScaler()
        scaler.fit(x)
        return scaler.transform(x)

    def resample_data(self,X, y):

        if self.type_balancing=='over&down_sampling' :
            # define pipeline
            over = SMOTE(sampling_strategy=0.58)
            under = RandomUnderSampler(sampling_strategy=0.66)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            # transform the dataset
            X, y = pipeline.fit_resample(X, y)


        if self.type_balancing == 'over_sampling':
            # transform the dataset
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)


        if self.type_balancing == 'under_sampling':
            # define pipeline
            under = RandomUnderSampler()
            X, y = under.fit_resample(X, y)

        return X, y


    def model_single(self):
        #questin_label=1  answer_label=0
        label=np.zeros(67)
        label[0:15]=1
        self.f_measure_all=[]
        self.precision_all=[]
        self.recall_all=[]
        for patient in range(len(self.feature_matrix[:self.num_patient])):
            feature=self.feature_matrix[patient]
            f_measure = np.zeros((self.feature_matrix[patient].shape[0],5))
            precision = np.zeros((self.feature_matrix[patient].shape[0], 5))
            recall = np.zeros((self.feature_matrix[patient].shape[0], 5))
            for electrode in range(self.feature_matrix[patient].shape[0]):
                x=feature[electrode,:,:]
                x_norm=self.norm(x)
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for i, (train_index, test_index) in enumerate(kf.split(x,label)):
                    x_train2,x_test=x_norm[train_index],x_norm[test_index]
                    y_train2, y_test=label[train_index],label[test_index]
                    x_train, y_train = self.resample_data(x_train2, y_train2)
                    if self.type_classification=='log_reg':
                        if self.type_balancing=='weighted_losfunc':
                            cls = LogisticRegression(class_weight='balanced').fit(x_train, y_train)
                        else:
                            cls= LogisticRegression().fit(x_train, y_train)
                    if self.type_classification == 'SVM':
                        if self.type_balancing == 'weighted_losfunc':
                            cls = SVC(class_weight='balanced').fit(x_train, y_train)
                        else:
                            cls = SVC().fit(x_train, y_train)
                    if self.type_classification == 'Naive_bayes':
                        cls = GaussianNB().fit(x_train, y_train)
                    y_pre=cls.predict(x_test)
                    out= precision_recall_fscore_support(y_test, y_pre, average='weighted')
                    precision[electrode,i]=out[0]
                    recall[electrode, i] =out[1]
                    f_measure[electrode, i] =out[2]
            self.f_measure_all.append(f_measure)
            self.precision_all.append(precision)
            self.recall_all.append(recall)
        return self.f_measure_all,self.precision_all,self.recall_all

    def save_plot_result(self,raw_car_all):
        for patient in range(len(self.f_measure_all)):
            os.makedirs(self.path + 'patient_' + str(patient))
            p3 = os.path.join(self.path, 'patient_' + str(patient) + '/')
            f_measure_mean=np.mean(self.f_measure_all[patient],axis=1)
            f_measure_var = np.var(self.f_measure_all[patient], axis=1)
            precision_mean=np.mean(self.precision_all[patient],axis=1)
            precision_var = np.var(self.precision_all[patient], axis=1)
            recall_mean=np.mean(self.recall_all[patient],axis=1)
            recall_var = np.var(self.recall_all[patient], axis=1)

            #write csv file
            header=['num_channel','channel','','f_measure','','precision','','recall']
            ch_name=raw_car_all[patient].ch_names
            with open(p3+'p_' + str(patient)+'.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                for i in range(self.f_measure_all[patient].shape[0]):
                    writer.writerow([i,ch_name[i], ' ', str(f_measure_mean[i]) + '±' + str(f_measure_var[i])
                                    , ' ', str(precision_mean[i]) + '±' + str(precision_var[i])
                                    ,' ', str(recall_mean[i]) + '±' + str(recall_var[i])])


            #plot f_measure for all channel
            channel_index=np.arange(0,self.f_measure_all[patient].shape[0])
            plt.figure()
            plt.plot(channel_index, f_measure_mean, color='blue', linewidth=2)
            plt.fill_between(channel_index, f_measure_mean-f_measure_var, f_measure_mean+f_measure_var, color='red', alpha=0.2)
            plt.title('patient=' + str(patient) , fontsize=15)
            plt.ylabel('f_measure')
            plt.xlabel('channel')
            plt.savefig(p3 + 'p_' + str(patient))


















