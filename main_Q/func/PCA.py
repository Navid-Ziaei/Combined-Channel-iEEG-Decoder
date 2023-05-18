import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_PCA(raw_car_all,path, feature_matrix):
    os.makedirs(path + 'PCA')
    p2 = os.path.join(path, 'PCA' + '/')
    for patient in range(len(feature_matrix[:2])):
        os.makedirs(p2 + 'patient_' + str(patient))
        p3 = os.path.join(p2, 'patient_' + str(patient) + '/')
        ch_name = raw_car_all[patient].ch_names
        for electrode in range(feature_matrix[patient].shape[0]):
            d = feature_matrix[patient][electrode, :, :]
            pca = PCA(n_components=2)
            x = pca.fit_transform(d)
            y = np.zeros(67)
            y[0:15] = 1
            counter = Counter(y)
            plt.figure()
            for label, _ in counter.items():
                row_ix = np.where(y == label)[0]
                plt.scatter(x[row_ix, 0], x[row_ix, 1], label=str(label))
            plt.title('patient=' + str(patient)+'_electrode='+ ch_name[electrode], fontsize=15)
            plt.savefig(p3 + 'p_' + str(patient)+'_electrode='+ ch_name[electrode])
            plt.legend()
