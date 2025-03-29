import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_models_bar(tasks, classifier_list, path_result):
    # Shortened names for classifiers
    classifier_short = {
        'Logistic_regression': 'LR',
        'Naive_bayes': 'NB',
        'RandomForest': 'RF',
        'SVM': 'SVM',
        'XGBoost': 'XGB'
    }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300)

    for idx, task in enumerate(tasks):
        path = path_result + f'{task}/'
        f1_score_max = {}
        f1_score_ensemble = {}

        for cls in classifier_list:
            if task == 'move_rest':
                f1_score_ensemble[cls] = np.load(
                    path + cls + '_Without_balancing/Max_voting/f_measure_all_ensemble.npy')
                f1_score_max[cls] = np.load(path + cls + '_Without_balancing/Max_voting/max_performance_all.npy')
            else:
                f1_score_ensemble[cls] = np.load(path + cls + '_over_sampling/Max_voting/f_measure_all_ensemble.npy')
                f1_score_max[cls] = np.load(path + cls + '_over_sampling/Max_voting/max_performance_all.npy')

        # Prepare data for Seaborn
        data = []
        for cls in classifier_list:
            data.append(
                {"Classifier": classifier_short[cls], "F1 Score": np.mean(f1_score_max[cls]), "Mode": "Best channel"})
            data.append({"Classifier": classifier_short[cls], "F1 Score": np.mean(f1_score_ensemble[cls]),
                         "Mode": "Combined channel"})

        df = pd.DataFrame(data)

        ax = sns.barplot(data=df, x="Classifier", y="F1 Score", hue="Mode",
                         palette={"Best channel": "darkred", "Combined channel": "teal"}, ax=axes[idx])
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_xlabel('')
        ax.set_title(f'Task: {task.replace("_", " ").title()}', fontsize=14)

        if task == 'singing_music':
            ax.set_ylim(0.5, 0.88)
        elif task == 'speech_music':
            ax.set_ylim(0.5, 0.9)
        else:
            ax.set_ylim(0.5, 0.92)

        ax.set_xticks(range(len(df['Classifier'].unique())))
        ax.set_xticklabels(df['Classifier'].unique(), fontsize=12)
        ax.legend(title="Mode", fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('F:/Thesis Master/single_participant_code/src/visualization/Comparison_bar_all_tasks.png')
    # plt.savefig('F:/Thesis Master/Single_patient_paper/Comparison_bar_all_tasks.svg')
    plt.show()


def compare_models_box(tasks, classifier_list, path_result):
    classifier_short = {
        'Logistic_regression': 'LR',
        'Naive_bayes': 'NB',
        'RandomForest': 'RF',
        'SVM': 'SVM',
        'XGBoost': 'XGB'
    }

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    palette = ['#E91E63', '#3357FF', 'cyan', '#8E44AD', 'limegreen']

    for idx, task in enumerate(tasks):
        path = path_result + f'{task}/'
        f1_score = {}

        for cls in classifier_list:
            if task == 'move_rest':
                f1_score[cls] = np.load(path + cls + '_Without_balancing/Max_voting/f_measure_all_ensemble.npy')
            else:
                f1_score[cls] = np.load(path + cls + '_over_sampling/Max_voting/f_measure_all_ensemble.npy')

        # Prepare data for Seaborn
        data = []
        for cls in classifier_list:
            for score in f1_score[cls]:
                data.append({'Classifier': classifier_short[cls], 'F1 Score': score})
        df = pd.DataFrame(data)

        # Create boxplot with individual data points
        sns.boxplot(data=df, x='Classifier', y='F1 Score', hue='Classifier', palette=palette, width=0.6,
                    showfliers=False,
                    boxprops=dict(edgecolor='black'), medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'), ax=axes[idx], legend=False)
        sns.stripplot(data=df, x='Classifier', y='F1 Score', color='black', size=3, jitter=True, alpha=0.7,
                      ax=axes[idx])

        axes[idx].set_title(f'Task: {task.replace("_", " ").title()}', fontsize=14)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('F1 Score', fontsize=12)
        axes[idx].set_xticks(range(len(df['Classifier'].unique())))
        axes[idx].set_xticklabels(df['Classifier'].unique(), fontsize=12)

        if task == 'singing_music':
            axes[idx].set_ylim(0.5, 1)
            axes[idx].set_yticks(np.arange(0.5, 1, 0.05))
        elif task == 'speech_music':
            axes[idx].set_ylim(0.5, 1)
            axes[idx].set_yticks(np.arange(0.5, 1.05, 0.05))
        else:
            axes[idx].set_ylim(0.48, 1)
            axes[idx].set_yticks(np.arange(0.48, 1.05, 0.05))

        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('F:/Thesis Master/single_participant_code/src/visualization/Comparison_boxplot_all_tasks.png')
    # plt.savefig('E:/Thesis/Single_patient_paper/Comparison_boxplot_all_tasks.svg')
    plt.show()



