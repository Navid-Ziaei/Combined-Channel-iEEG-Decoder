import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#TASK : 'singing_music' or 'speech_music' or 'question_answer'
task = 'singing_music'
path = f'E:/Thesis/Single_patient_paper/{task}/'
classifier_list = ['Logistic_regression', 'Naive_bayes', 'RandomForest', 'SVM', 'XGBoost']
num_patient = 29
f1_score_max = {}
f1_score_ensemble = {}

for cls in classifier_list:
    if task == 'move_rest':
        f1_score_ensemble[cls] = np.load(path + cls + '_Without_balancing/Max_voting/f_measure_all_ensemble.npy')
        f1_score_max[cls] = np.load(path + cls + '_Without_balancing/Max_voting/max_performance_all.npy')
    else:
        f1_score_ensemble[cls] = np.load(path + cls + '_over_sampling/Max_voting/f_measure_all_ensemble.npy')
        f1_score_max[cls] = np.load(path + cls + '_over_sampling/Max_voting/max_performance_all.npy')


# Prepare data for Seaborn
data = []
for cls in classifier_list:
    data.append({"Classifier": cls, "F1 Score": np.mean(f1_score_max[cls]), "Mode": "Best channel"})
    data.append({"Classifier": cls, "F1 Score": np.mean(f1_score_ensemble[cls]), "Mode": "Combined channel"})

# Convert data to DataFrame
df = pd.DataFrame(data)

# Shortened names for classifiers
classifier_short = {'Logistic_regression': 'LR', 'Naive_bayes': 'NB', 'RandomForest': 'RF', 'SVM': 'SVM', 'XGBoost': 'XGB'}
df['Classifier'] = df['Classifier'].map(classifier_short)

# Create the bar plot
sns.set_theme(style="whitegrid")
plt.figure(dpi=300)
ax = sns.barplot(data=df, x="Classifier", y="F1 Score", hue="Mode", palette={"Best channel": "darkred", "Combined channel": "teal"})

# Customize the plot
ax.set_ylabel('F1 Score', fontsize=15)
ax.set_xlabel('')

if task == 'singing_music':
    ax.set_ylim(0.5, 0.88)
elif task == 'speech_music':
    ax.set_ylim(0.5, 0.9)
else:
    ax.set_ylim(0.5, 0.92)

ax.set_xticklabels(df['Classifier'].unique(), fontsize=15)
ax.legend(title="Mode", fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig(path + f'Comparison_bar_{task}_sb.png')
plt.savefig(path + f'Comparison_bar_{task}_sb.svg')



