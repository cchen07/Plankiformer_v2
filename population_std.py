import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

import argparse

def read_population_xlsx(population_xlsx_path):
    df_population = pd.read_excel(population_xlsx_path, index_col=0)
    df_population = df_population.drop(df_population[df_population['Ground_truth'] == 0].index)

    return df_population

def plot_population_count_std(models, population_xlsx_paths, outpath):
    df = pd.DataFrame()
    for i, j in zip(models, population_xlsx_paths):
        df_population = read_population_xlsx(j)
        df.index = df_population.index
        df['Ground_truth'] = df_population['Ground_truth']
        df[i] = df_population['Predict']

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Class')
    plt.ylabel('Count')
    width = 0.5
    x = np.arange(0, len(df) * 2, 2)
    x1 = x - width / 2
    x2 = x + width / 2
    plt.bar(x1, df['Ground_truth'], width=width, label='Ground_truth')
    plt.bar(x2, np.mean(df[models], axis=1), yerr=np.std(df[models], axis=1), width=width, ecolor='black', capsize=3, label='Prediction')
    plt.xticks(x, df.index, rotation=45, rotation_mode='anchor', ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath + 'Population_count_std.png', dpi=300)
    plt.close()
    

parser = argparse.ArgumentParser(description='Plot population count with error bar')
parser.add_argument('-models', nargs='*', help='name of models')
parser.add_argument('-population_xlsx_paths', nargs='*', help='paths of population xlsx file')
parser.add_argument('-outpath')
args = parser.parse_args()

if __name__ == '__main__':
    plot_population_count_std(args.models, args.population_xlsx_paths, args.outpath)