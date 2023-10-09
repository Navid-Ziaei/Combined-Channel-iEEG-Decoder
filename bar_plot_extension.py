import numpy as np
import matplotlib.pyplot as plt


def bar_plot_mean_patient(settings, path):
    color_list = ['orangered', 'yellow', 'cyan', 'deeppink', 'lime', 'steelblue', 'purple', 'pink', 'darkgray']
    type_ensemble = 'Max_voting'
    barWidth = 0.5
    bar_pos = 3.5
    num_patient = settings['bar_plot_mean_patient']['num_patient_avg']
    j = 0
    plt.figure(dpi=300)
    for type_balancing in settings['list_type_balancing']:
        for type_classification in settings['list_type_classification']:
            if settings['list_type_classification'][type_classification] & settings['list_type_balancing'][
                type_balancing]:
                data_ensemble = np.load(
                    path + type_classification + '_' + type_balancing + '/' + type_ensemble + '/' + 'f_measure_all_ensemble.npy')
                pos = np.argsort(data_ensemble)[-1 * num_patient:]
                data = np.load(
                    path + type_classification + '_' + type_balancing + '/' + type_ensemble + '/' + 'max_performance_all.npy')
                j = j + 1
                mean = np.mean([data[i] for i in pos])
                error = np.std([data[i] for i in pos])
                mean_ensemble = np.mean([data_ensemble[i] for i in pos])
                bar_pos = bar_pos + barWidth
                br = plt.bar(bar_pos, mean, color=color_list[j - 1], yerr=error, capsize=2, width=barWidth,
                             edgecolor='gray', label=type_classification + type_balancing, alpha=0.7)
                for bar in br:
                    plt.text(bar.get_x() + bar.get_width() / 2, mean_ensemble, '*', ha='center', va='bottom',
                             fontsize='20', color='red')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks([r + barWidth for r in 10 * np.arange(2) + 2.25], ['', ''])
    plt.legend(loc='upper right', fontsize='5')
    plt.ylabel('Accuracy')
    plt.title('mean performance of single_channel across the ' + str(num_patient) + ' best patients', fontsize='7')
    plt.savefig(path + 'average_patients')


def bar_plot_best_electrode(settings, paths):
    mean = {}
    error = {}
    barWidth = 20
    type_ensemble = 'Max_voting'
    num_elec = settings['bar_plot_best_electrode']['num_best_electrode']
    num_patient = settings['bar_plot_best_electrode']['num_best_patient']
    br = np.arange(num_elec) * barWidth
    path = paths + settings['bar_plot_best_electrode']['type_classification'] + '_' \
           + settings['bar_plot_best_electrode']['type_balancing'] + '/' + type_ensemble+'/'

    color = ['deeppink', 'slategray', 'maroon', 'lime', 'royalblue', 'tomato']
    data = np.load(path + 'f_measure_all_ensemble.npy')
    pos_best_patient = np.argsort(data)[-1 * num_patient:]
    fig, ax = plt.subplots(figsize=(50, 20), dpi=300)
    k = 0
    for patient in pos_best_patient:
        br = br + 200
        data_patient = np.load(path + 'patient_' + str(patient) + '.npy')
        pos_best_electrode = np.argsort(data_patient[:, 1])[-1 * num_elec:]
        for i in range(num_elec):
            bar1 = ax.bar(br[i], float(data_patient[pos_best_electrode[i], 1]), color=color[i],
                          yerr=float(data_patient[pos_best_electrode[i], 2]), capsize=10, width=barWidth,
                          edgecolor='grey')
            for bar in bar1:
                ax.text(bar.get_x() + bar.get_width() / 2, 0, data_patient[pos_best_electrode[i], 0], ha='center',
                        va='bottom', fontsize='20')
    plt.xticks([r + barWidth for r in (200 * (np.arange(5) + 1.15))],
               ['P' + str(pos_best_patient[0]), 'P' + str(pos_best_patient[1]),
                'P' + str(pos_best_patient[2]), 'P' + str(pos_best_patient[3]),
                'P' + str(pos_best_patient[4])], fontsize=40)

    plt.yticks(fontsize=40)
    plt.ylabel('Accuracy', fontsize=40)
    plt.xlabel('patient_id', fontsize=40)
    plt.title(' performance of the' + str(num_elec) + 'best channel for' + str(num_patient) + 'best patients',
              fontsize='40')
    plt.savefig(paths+ 'best_elec_best_patients')


"__________________________________settings______________________________________________"

settings = {
    'list_type_balancing': {'Without_balancing': True,
                            'over_sampling': False,
                            'under_sampling': False,
                            'over&down_sampling': False},

    'list_type_classification': {'Logistic_regression': True,
                                 'SVM': True,
                                 'Naive_bayes': True},
    # Bar plot as final result
    'bar_plot_mean_patient': {'plot': True,
                              'num_patient_avg': 10},

    'bar_plot_best_electrode': {'plot': True,
                                'type_classification': 'SVM',
                                'type_balancing': 'Without_balancing',
                                'num_best_patient': 5,
                                'num_best_electrode': 6}
}

path = 'F:/maryam_sh/ziaei_github/iEEG_fMRI_audiovisual/results/question&answer/2023-10-09-11-43-05' \
       '/plots/classification/'

bar_plot_mean_patient(settings, path)
bar_plot_best_electrode(settings, path)
