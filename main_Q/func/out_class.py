import matplotlib.pyplot as plt
import pandas as pd

def plot_output_classification(electrodes,avg_win,label_all,label2,AVG,RMS):
    fig, ax = plt.subplots(figsize=(80, 60))
    i = 0
    for electrode in electrodes:
        i = i + 5
        y = [i] * len(label2)
        df2 = pd.DataFrame(dict(y=avg_win[electrode], x=y, label=label_all[electrode]))
        colors = {'0': 'red', '1': 'green'}
        ax.scatter(df2['x'], df2['y'], c=df2['label'].map(colors), marker='o')
    if AVG:
        plt.ylabel('AVG', fontsize=50)
        plt.savefig("AVG")
    if RMS:
        plt.ylabel('RMS', fontsize=50)
        plt.savefig("RMS")

    #plt.show()