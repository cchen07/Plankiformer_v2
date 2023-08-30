import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem


def plot_performance_overview(model_name, test_dataset, accuracy, f1_score, outpath, remove_0):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('Accuracy')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(accuracy[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(accuracy, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('F1-score')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(f1_score[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(f1_score, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    plt.colorbar()

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance_overview.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_overview_rm_0.png', dpi=300)
    plt.close()


def read_test_report(test_report_file):
    test_report = pd.read_csv(test_report_file)
    accuracy_value = format(float(test_report.iloc[0].item()), '.3f')
    f1_value = format(float(test_report.iloc[2].item()), '.3f')

    return accuracy_value, f1_value


def performance_matrix(model_performance_paths, testsets, finetuned, ensemble, remove_0):
    n_model = len(model_performance_paths)
    # n_dataset = len(os.listdir(model_performance_paths[0]))
    n_dataset = len(testsets)
    # test_dataset = os.listdir(model_performance_paths[0])
    test_dataset = testsets
    # test_dataset.sort()

    accuracy = np.zeros([n_model, n_dataset])
    f1_score = np.zeros([n_model, n_dataset])

    for i, imodel_path in enumerate(model_performance_paths):
        # dataset_names = os.listdir(imodel_path)
        dataset_names = testsets
        # dataset_names.sort()
        for j, idataset in enumerate(dataset_names):
            test_report_path = imodel_path + '/' + idataset + '/'

            if remove_0 == 'no':
                if finetuned == 0:
                    if ensemble == 0:
                        test_report_file = test_report_path + 'Single_test_report_original.txt'
                    elif ensemble == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_original.txt'
                if finetuned == 1:
                    if ensemble == 0:
                        test_report_file = test_report_path + 'Single_test_report_tuned.txt'
                    elif ensemble == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_tuned.txt'
                if finetuned == 2:
                    if ensemble == 0:
                        test_report_file = test_report_path + 'Single_test_report_finetuned.txt'
                    elif ensemble == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_finetuned.txt'
            elif remove_0 == 'yes':
                if finetuned == 0:
                    if ensemble == 0:
                        test_report_file = test_report_path + 'Single_test_report_rm_0_original.txt'
                    elif ensemble == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_rm_0_geo_mean_original.txt'
                if finetuned == 1:
                    if ensemble == 0:
                        test_report_file = test_report_path + 'Single_test_report_rm_0_tuned.txt'
                    elif ensemble == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_rm_0_geo_mean_tuned.txt'
                if finetuned == 2:
                    if ensemble == 0:
                        test_report_file = test_report_path + 'Single_test_report_rm_0_finetuned.txt'
                    elif ensemble == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_rm_0_geo_mean_finetuned.txt'

            accuracy_value, f1_value = read_test_report(test_report_file)
            accuracy[i, j], f1_score[i, j] = accuracy_value, f1_value
    
    return accuracy, f1_score, test_dataset


def plot_performance_curve(model_name, test_dataset, accuracy, f1_score, outpath, remove_0):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Accuracy')
    for i, j in zip(accuracy, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('F1-score')
    for i, j in zip(f1_score, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance_curves.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_curves_rm_0.png', dpi=300)
    plt.close()


def plot_performance(aug_types, test_dataset, accuracy, f1_score, outpath, remove_0):
    n_aug = len(aug_types)
    n_model_aug = len(accuracy)/n_aug
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Accuracy')
    for i, aug_type in enumerate(aug_types):
        accuracy_aug_type = accuracy[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_accuracy = np.mean(accuracy_aug_type, axis=0)
        sem_accuracy = sem(accuracy_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)), y=mean_accuracy, yerr=sem_accuracy, fmt='s', capsize=5, label=aug_type)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('F1-score')
    for i, aug_type in enumerate(aug_types):
        f1_aug_type = f1_score[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_f1 = np.mean(f1_aug_type, axis=0)
        sem_f1 = sem(f1_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)), y=mean_f1, yerr=sem_f1, fmt='s', capsize=5, label=aug_type)
    plt.legend()

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_rm_0.png', dpi=300)
    plt.close()


parser = argparse.ArgumentParser(description='Generate an overview of test performance')
parser.add_argument('-model_names', nargs='*', help='list of model names')
parser.add_argument('-aug_types', nargs='*', help='list of augmentation types')
parser.add_argument('-testsets', nargs='*', help='list of test datasets')
parser.add_argument('-model_performance_paths', nargs='*', help='list of performance paths for each model')
parser.add_argument('-outpath', help='path for saving the overview')
parser.add_argument('-finetuned', type=int, help='0 for original, 1 for tuned, 2 for finetuned')
parser.add_argument('-ensemble', type=int, help='0 for single model, 1 for ensembled model')
parser.add_argument('-remove_0', choices=['yes', 'no'], default='no', help='remove 0 support classes or not')
args = parser.parse_args()

if __name__ == '__main__':
    accuracy, f1_score, test_dataset = performance_matrix(args.model_performance_paths, args.testsets, args.finetuned, args.ensemble, args.remove_0)
    model_name = args.model_names
    outpath = args.outpath
    plot_performance_overview(model_name, test_dataset, accuracy, f1_score, outpath, args.remove_0)
    plot_performance_curve(model_name, test_dataset, accuracy, f1_score, outpath, args.remove_0)
    plot_performance(args.aug_types, test_dataset, accuracy, f1_score, outpath, args.remove_0)
