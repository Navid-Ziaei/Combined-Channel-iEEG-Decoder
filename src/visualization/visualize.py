import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#TASK : 'singing_music' or 'speech_music' or 'question_answer' or 'move_rest'
task = 'singing_music'
path = f'E:/Thesis/Single_patient_paper/{task}/'
classifier_list = ['Logistic_regression', 'Naive_bayes', 'RandomForest', 'SVM', 'XGBoost']
num_patient = 51
f1_score = {}
for cls in classifier_list:
    if task == 'move_rest':
        f1_score[cls] = np.load(path + cls + '_Without_balancing/Max_voting/f_measure_all_ensemble.npy')
    else:
        f1_score[cls] = np.load(path + cls + '_over_sampling/Max_voting/f_measure_all_ensemble.npy')


data = []
for cls in classifier_list:
    for score in f1_score[cls]:
        data.append({'Classifier': cls, 'F1 Score': score})
df = pd.DataFrame(data)

# Initialize the Seaborn plot
plt.figure(dpi=300)
palette = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']
sns.boxplot(data=df, x='Classifier', y='F1 Score', palette=palette, width=0.6, showfliers=False,
            boxprops=dict(edgecolor='black'), medianprops=dict(color='black'),
            whiskerprops=dict(color='black'), capprops=dict(color='black'))

# Add individual data points
sns.stripplot(data=df, x='Classifier', y='F1 Score', color='black', size=3, jitter=True, alpha=0.7)

# Customize the plot
plt.xticks(ticks=range(len(classifier_list)), labels=['LR', 'NB', 'RF', 'SVM', 'XGB'], fontsize=15)
plt.ylabel('F1 Score', fontsize=15)
plt.xlabel('')
if task == 'singing_music':
    plt.ylim(0.5, 1)
    plt.yticks(np.arange(0.5, 1, 0.05), fontsize=12)
elif task == 'speech_music':
    plt.ylim(0.5, 1)
    plt.yticks(np.arange(0.5, 1.05, 0.05), fontsize=12)
else:
    plt.ylim(0.48, 1)
    plt.yticks(np.arange(0.48, 1.05, 0.05), fontsize=12)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig(path + f'Comparison_sb_{task}.png')
plt.savefig(path + f'Comparison_sb_{task}.svg')

# Open a text file in write mode to save the output
with open(path + "output.txt", "w") as file:
    # Loop through f1_score dictionary and write the output to the file
    for cls in f1_score.keys():
        file.write(f'{cls} : {round(np.mean(f1_score[cls]), 2)} ± {round(np.std(f1_score[cls]), 2)}\n')

    # Print the end message to the file
    file.write('end\n')

print('end')
