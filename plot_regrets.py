import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_cumulative_regrets(df_regrets, label, T=200):
    print('dddd')
    plt.plot(df_regrets.mean(), label=label)
    plt.fill_between(df_regrets.columns, df_regrets.mean() - df_regrets.std()/np.sqrt(len(df_regrets)), df_regrets.mean() + df_regrets.std()/np.sqrt(len(df_regrets)), alpha=0.3)

def plot_regrets(file_list, T, save_path):
    for file in file_list:
        df = pd.read_csv(file)
        print(df.shape)
        label = file.split('/')[-1].split('.')[0].strip('_regrets')
        plot_cumulative_regrets(df, label, T)
    plt.xlabel('T')
    plt.ylabel('Cumulative Regret')
    plt.xticks(np.arange(1, T, 50))
    plt.legend()
    plt.show()
    plt.savefig(save_path)

if __name__ == '__main__':
    file_list = []
    file_list.append('data/linucb_regrets.csv')
    file_list.append('data/Qtransformer_linucb_1_t_gamma_1_regrets.csv')
    T = 200
    save_path = 'figs/regrets.png'
    plot_regrets(file_list, T, save_path)



    