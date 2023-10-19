import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from distinctipy import distinctipy
import argparse
import random
from pathlib import Path
from scipy.stats import sem, pearsonr
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor


random.seed(100)

def preprocess_distance_txt(distance_txt_path):
    distance_txt = pd.read_csv(distance_txt_path, header=None)
    distance = distance_txt.drop(distance_txt.index[-1])
    df_distance = pd.DataFrame(columns=['class', 'distance'], index=range(len(distance)))

    for i in range(len(distance)):
        df_distance['class'][i] = distance.iloc[i].item()[0:20].strip()
        df_distance['distance'][i] = float(distance.iloc[i].item()[20:28])

    return df_distance

def preprocess_bias_xlsx(bias_xlsx_path):
    df_bias = pd.read_excel(bias_xlsx_path)
    df_bias = df_bias.rename(columns={df_bias.columns[0]: 'class'})
    df_bias['APE'] = abs(np.divide(df_bias['Bias'], df_bias['Ground_truth']))
    df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] == 0].index)
    df_bias = df_bias.reset_index(drop=True)

    return df_bias

def plot_distance_bias_scatter(testsets, distance_txt_paths, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['distance'] = df_distance['distance']
        df_distance_bias['APE'] = df_bias['APE']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)

    plt.figure(figsize=(10, 10))
    plt.suptitle(model)
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    plt.ylabel('Absolute Percentage Error')
    # colors = cm.nipy_spectral(np.linspace(0, 1, len(np.unique(df['class']))))
    random.seed(100)
    colors = distinctipy.get_colors(len(np.unique(df['class'])), pastel_factor=0.7)
    for iclass, c in zip(np.unique(df['class']), colors):
        plt.scatter(df[df['class'] == iclass].distance, df[df['class'] == iclass].APE, label=iclass, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance(testsets, distance_txt_paths, outpath, feature_or_pixel, PCA, threshold, distance_type):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])    
        df_distance_testset = pd.DataFrame(columns=['class', 'testset', 'distance'], index=range(len(df_distance)))
        df_distance_testset['class'] = df_distance['class']
        df_distance_testset['testset'] = testsets[i]
        df_distance_testset['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_testset])
    df = df.reset_index(drop=True)

    classes = np.unique(df['class'])
    df_plot = pd.DataFrame(columns=['ID_test', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5'], index=classes)

    for index, row in df.iterrows():
        df_plot.loc[row['class'], row['testset']] = row['distance']
    
    plt.figure(figsize=(10, 8))
    plt.xlabel('class')
    plt.ylabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    plt.grid(alpha=0.5)
    plt.xticks(range(len(classes)), labels=classes, rotation=45, rotation_mode='anchor', ha='right')
    for column in df_plot:
        plt.scatter(x=range(len(df_plot)), y=df_plot[column], label=column, s=10)
    plt.scatter(x=range(len(df_plot)), y=df_plot.mean(axis=1), label='average', s=50, c='red')
    plt.legend(loc=8, bbox_to_anchor=(0.5, -0.3), fancybox=True, ncol=7)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + feature_or_pixel + '_PCA_' + distance_type + '_distance_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + feature_or_pixel + '_' + distance_type + '_distance_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_err(testsets, distance_txt_paths, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['distance'] = df_distance['distance']
        df_distance_bias['APE'] = df_bias['APE']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)

    bias_mean = []
    bias_std = []
    bias_sem = []
    distance_mean = []
    distance_std = []
    distance_sem = []

    for i in testsets:
        bias_mean_testset = np.average(df[df['testset'] == i]['APE'])
        bias_mean.append(bias_mean_testset)
        bias_std_testset = np.std(df[df['testset'] == i]['APE'])
        bias_std.append(bias_std_testset)
        bias_sem_testset = sem(df[df['testset'] == i]['APE'])
        bias_sem.append(bias_sem_testset)

        distance_mean_testset = np.average(df[df['testset'] == i]['distance'])
        distance_mean.append(distance_mean_testset)
        distance_std_testset = np.std(df[df['testset'] == i]['distance'])
        distance_std.append(distance_std_testset)
        distance_sem_testset = sem(df[df['testset'] == i]['distance'])
        distance_sem.append(distance_sem_testset)


    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Testset')
    ax.set_ylabel('Absolute Percentage Error')
    x = range(len(testsets))
    plt.xticks(x, labels=testsets)
    plt.ylim(-0.1, 1.1)

    plt.errorbar(x, bias_mean, yerr=bias_sem, fmt='-s', color='g', capsize=5, label='Absolute Percentage Error')

    ax_2 = ax.twinx()
    ax_2.set_ylabel('1 - ' + distance_type + ' Distance to training set')
    plt.errorbar(x, np.subtract(1, distance_mean), yerr=distance_sem, fmt='-s', color='r', capsize=5, label='1 - ' + distance_type+' Distance')
    # plt.errorbar(x, distance_mean, yerr=distance_sem, fmt='-s', color='r', capsize=5, label=distance_type+' Distance')
    plt.ylim(0.95*np.min(np.subtract(1, distance_mean)), 1.05*np.max(np.subtract(1, distance_mean)))
    # plt.ylim(0.4, 1)

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()
    ax.clear()

def plot_distance_bias_testset(testsets, distance_txt_paths, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting global correlation of distance and bias (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['distance'] = df_distance['distance']
        df_distance_bias['APE'] = df_bias['APE']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)

    bias_mean = []
    bias_sem = []
    distance_mean = []
    distance_sem = []
    for i in testsets:
        bias_mean_testset = np.average(df[df['testset'] == i]['APE'])
        bias_mean.append(bias_mean_testset)
        if len(df[df['testset'] == i]['APE']) > 1:
            bias_sem_testset = sem(df[df['testset'] == i]['APE'])
        else:
            bias_sem_testset = 0
        bias_sem.append(bias_sem_testset)
        distance_mean_testset = np.average(df[df['testset'] == i]['distance'])
        distance_mean.append(distance_mean_testset)
        if len(df[df['testset'] == i]['distance']) > 1:
            distance_sem_testset = sem(df[df['testset'] == i]['distance'])
        else:
            distance_sem_testset = 0
        distance_sem.append(distance_sem_testset)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_mean[i], xerr=distance_sem[i], yerr=bias_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(bias_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()  
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    x = np.array([])
    y = np.array([])
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_mean[i], xerr=distance_sem[i], yerr=bias_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].APE, label=itestset, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].APE)), axis=None)
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')
    
    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_mean[i], xerr=distance_sem[i], yerr=bias_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].APE, label=itestset, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_class(testsets, distance_txt_paths, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting global correlation of distance and bias (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['distance'] = df_distance['distance']
        df_distance_bias['APE'] = df_bias['APE']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
    classes = np.unique(df['class']).tolist()

    bias_mean = []
    bias_sem = []
    distance_mean = []
    distance_sem = []
    for i in classes:
        bias_mean_class = np.average(df[df['class'] == i]['APE'])
        bias_mean.append(bias_mean_class)
        if len(df[df['class'] == i]['APE']) > 1:
            bias_sem_class = sem(df[df['class'] == i]['APE'])
        else:
            bias_sem_class = 0
        bias_sem.append(bias_sem_class)
        distance_mean_class = np.average(df[df['class'] == i]['distance'])
        distance_mean.append(distance_mean_class)
        if len(df[df['class'] == i]['distance']) > 1:
            distance_sem_class = sem(df[df['class'] == i]['distance'])
        else:
            distance_sem_class = 0
        distance_sem.append(distance_sem_class)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_mean[i], xerr=distance_sem[i], yerr=bias_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(bias_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    x = np.array([])
    y = np.array([])

    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_mean[i], xerr=distance_sem[i], yerr=bias_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].APE, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['class'] == iclass]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['class'] == iclass].APE)), axis=None)
    # correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_mean[i], xerr=distance_sem[i], yerr=bias_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].APE, label=iclass, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].APE, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.array([])
        y = np.array([])
        for itestset in df[df['class'] == iclass].testset.values:
            x = np.concatenate((x, df[(df['class'] == iclass) & (df['testset'] == itestset)].distance), axis=None)
            y = np.concatenate((y, df[(df['class'] == iclass) & (df['testset'] == itestset)].APE), axis=None)
        # correlation, p_value = pearsonr(x, y)
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        y_fit = reg.predict(x.reshape(-1, 1))
        plt.plot(x, y_fit, color=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_rise_testset(testsets, distance_txt_paths, train_bias_xlsx_path, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting global correlation of distance and bias drop (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        df_bias_train = preprocess_bias_xlsx(train_bias_xlsx_path)
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
            df_bias_train = df_bias_train.drop(df_bias_train[df_bias_train['Ground_truth'] < threshold].index)
            df_bias_train = df_bias_train.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['APE'] = df_bias['APE']
        dict_test = {key:bias for key, bias in zip(df_bias['class'], df_bias['APE'])}
        dict_train = {key:bias for key, bias in zip(df_bias_train['class'], df_bias_train['APE'])}
        bias_rise = []
        for i in dict_test.keys():
            bias_rise_class = 1 - np.divide(dict_test[i], dict_train[i])
            bias_rise.append(bias_rise_class)
        df_distance_bias['APE_rise'] = bias_rise
        
        df_distance_bias['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)
    
    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)

    bias_rise_mean = []
    bias_rise_sem = []
    distance_mean = []
    distance_sem = []
    for i in testsets:
        bias_rise_mean_testset = np.average(df[df['testset'] == i]['APE_rise'])
        bias_rise_mean.append(bias_rise_mean_testset)
        if len(df[df['testset'] == i]['APE_rise']) > 1:
            bias_rise_sem_testset = sem(df[df['testset'] == i]['APE_rise'])
        else:
            bias_rise_sem_testset = 0
        bias_rise_sem.append(bias_rise_sem_testset)

        distance_mean_testset = np.average(df[df['testset'] == i].distance)
        distance_mean.append(distance_mean_testset)
        if len(df[df['testset'] == i].distance) > 1:
            distance_sem_testset = sem(df[df['testset'] == i].distance)
        else:
            distance_sem_testset = 0
        distance_sem.append(distance_sem_testset)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))

    # correlation, p_value = pearsonr(np.array(distance_mean), np.array(bias_rise_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(bias_rise_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    x = np.array([])
    y = np.array([])
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].APE_rise, label=itestset)
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].APE_rise)), axis=None)
    # correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    plt.plot(x, y_fit, color='red', label='regression line')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset)
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].APE_rise, label=itestset, alpha=0.4)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].APE_rise, label=itestset, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_rise_class(testsets, distance_txt_paths, train_bias_xlsx_path, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting global correlation of distance and bias drop (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        df_bias_train = preprocess_bias_xlsx(train_bias_xlsx_path)
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
            df_bias_train = df_bias_train.drop(df_bias_train[df_bias_train['Ground_truth'] < threshold].index)
            df_bias_train = df_bias_train.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['APE'] = df_bias['APE']
        dict_test = {key:bias for key, bias in zip(df_bias['class'], df_bias['APE'])}
        dict_train = {key:bias for key, bias in zip(df_bias_train['class'], df_bias_train['APE'])}
        bias_rise = []
        for i in dict_test.keys():
            bias_rise_class = 1 - np.divide(dict_test[i], dict_train[i])
            bias_rise.append(bias_rise_class)
        df_distance_bias['APE_rise'] = bias_rise
        
        df_distance_bias['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
    
    classes = np.unique(df['class']).tolist()
    
    bias_rise_mean = []
    bias_rise_sem = []
    distance_mean = []
    distance_sem = []
    for i in classes:
        bias_rise_mean_class = np.average(df[df['class'] == i]['APE_rise'])
        bias_rise_mean.append(bias_rise_mean_class)
        if len(df[df['class'] == i]['APE_rise']) > 1:
            bias_rise_sem_class = sem(df[df['class'] == i]['APE_rise'])
        else:
             bias_rise_sem_class = 0
        bias_rise_sem.append(bias_rise_sem_class)


        distance_mean_class = np.average(df[df['class'] == i].distance)
        distance_mean.append(distance_mean_class)
        if len(df[df['class'] == i].distance) > 1:
            distance_sem_class = sem(df[df['class'] == i].distance)
        else:
            distance_sem_class = 0
        distance_sem.append(distance_sem_class)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
    
    # correlation, p_value = pearsonr(np.array(distance_mean), np.array(bias_rise_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(bias_rise_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    x = np.array([])
    y = np.array([])
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].APE_rise, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['class'] == iclass]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['class'] == iclass].APE_rise)), axis=None)
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=bias_rise_mean[i], xerr=distance_sem[i], yerr=bias_rise_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].APE_rise, label=iclass, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].APE_rise, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.array([])
        y = np.array([])
        for itestset in df[df['class'] == iclass].testset.values:
            x = np.concatenate((x, df[(df['class'] == iclass) & (df['testset'] == itestset)].distance), axis=None)
            y = np.concatenate((y, df[(df['class'] == iclass) & (df['testset'] == itestset)].APE_rise), axis=None)
        # correlation, p_value = pearsonr(x, y)
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        y_fit = reg.predict(x.reshape(-1, 1))
        plt.plot(x, y_fit, color=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_rise(testsets, distance_dfs, train_bias_xlsx_path, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and bias drop per component (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = distance_dfs[i]
        df_distance = df_distance.reset_index(level=0)
        df_distance.columns.values[0] = 'class'
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        df_bias_train = preprocess_bias_xlsx(train_bias_xlsx_path)
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
            df_bias_train = df_bias_train.drop(df_bias_train[df_bias_train['Ground_truth'] < threshold].index)
            df_bias_train = df_bias_train.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['APE'] = df_bias['APE']
        dict_test = {key:bias for key, bias in zip(df_bias['class'], df_bias['APE'])}
        dict_train = {key:bias for key, bias in zip(df_bias_train['class'], df_bias_train['APE'])}
        bias_rise = []
        for i in dict_test.keys():
            bias_rise_class = 1 - np.divide(dict_test[i], dict_train[i])
            bias_rise.append(bias_rise_class)
        df_distance_bias['APE_rise'] = bias_rise
        for idistance in df_distance.columns.values[1:]:
            df_distance_bias[idistance] = df_distance[idistance]
        # df_distance_bias['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)
    df['mean_distance'] = df.iloc[:, 6:].mean(axis=1)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        n_datapoint = df['testset'].value_counts()
        dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        for itestset in dropped_testsets:
            testsets.remove(itestset)
            df = df.drop(df[df['testset'] == itestset].index)
            df = df.reset_index(drop=True)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('APE rise ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for ifeature, c in zip(df.columns.values[6:-1], colors):
        bias_rise_mean = []
        bias_rise_sem = []
        distance_mean = []
        distance_sem = []
        for i, itestset in enumerate(testsets):
            bias_rise_mean_testset = np.average(df[df['testset'] == itestset]['APE_rise'])
            bias_rise_mean.append(bias_rise_mean_testset)
            bias_rise_sem_testset = sem(df[df['testset'] == itestset]['APE_rise'])
            bias_rise_sem.append(bias_rise_sem_testset)

            distance_mean_testset = np.average(df[df['testset'] == itestset][ifeature])
            distance_mean.append(distance_mean_testset)
            distance_sem_testset = sem(df[df['testset'] == itestset][ifeature])
            distance_sem.append(distance_sem_testset)
        
        weights = 1 / np.sqrt(np.array(distance_sem)**2 + np.array(bias_rise_sem)**2)
        params, covariance = curve_fit(linear_func, distance_mean, bias_rise_mean, sigma=weights)
        a = params[0]
        b = params[1]
        bias_rise_mean_pred = a * np.array(distance_mean) + b
        plt.errorbar(distance_mean, bias_rise_mean, xerr=distance_sem, yerr=bias_rise_sem, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
        # plt.errorbar(distance_mean, bias_rise_mean, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
#         plt.plot(distance_mean, bias_rise_mean_pred, color=np.array([c]), label=ifeature)
        if a < -1:
            plt.axline((distance_mean[0], bias_rise_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
        else:
            plt.axline((distance_mean[0], bias_rise_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        # plt.axline((distance_mean[0], bias_rise_mean_pred[0]), (distance_mean[-1], bias_rise_mean_pred[-1]), color=np.array([c]), label=ifeature)
    plt.legend(ncol=2)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_per_feature_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_per_feature_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias(testsets, distance_dfs, train_bias_xlsx_path, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and bias per component (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = distance_dfs[i]
        df_distance = df_distance.reset_index(level=0)
        df_distance.columns.values[0] = 'class'
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        df_bias_train = preprocess_bias_xlsx(train_bias_xlsx_path)
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
            df_bias_train = df_bias_train.drop(df_bias_train[df_bias_train['Ground_truth'] < threshold].index)
            df_bias_train = df_bias_train.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['APE'] = df_bias['APE']
        dict_test = {key:bias for key, bias in zip(df_bias['class'], df_bias['APE'])}
        dict_train = {key:bias for key, bias in zip(df_bias_train['class'], df_bias_train['APE'])}
        bias_rise = []
        for i in dict_test.keys():
            bias_rise_class = 1 - np.divide(dict_test[i], dict_train[i]+1e-8)
            bias_rise.append(bias_rise_class)
        df_distance_bias['APE_rise'] = bias_rise
        for idistance in df_distance.columns.values[1:]:
            df_distance_bias[idistance] = df_distance[idistance]
        # df_distance_bias['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)
    df['mean_distance'] = df.iloc[:, 6:].mean(axis=1)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        n_datapoint = df['testset'].value_counts()
        dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        for itestset in dropped_testsets:
            testsets.remove(itestset)
            df = df.drop(df[df['testset'] == itestset].index)
            df = df.reset_index(drop=True)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set')
    plt.ylabel('Absolute Percentage Error')
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for ifeature, c in zip(df.columns.values[6:-1], colors):
        bias_mean = []
        bias_sem = []
        distance_mean = []
        distance_sem = []
        for i, itestset in enumerate(testsets):
            bias_mean_testset = np.average(df[df['testset'] == itestset]['APE'])
            bias_mean.append(bias_mean_testset)
            bias_sem_testset = sem(df[df['testset'] == itestset]['APE'])
            bias_sem.append(bias_sem_testset)

            distance_mean_testset = np.average(df[df['testset'] == itestset][ifeature])
            distance_mean.append(distance_mean_testset)
            distance_sem_testset = sem(df[df['testset'] == itestset][ifeature])
            distance_sem.append(distance_sem_testset)
        
        weights = 1 / np.sqrt(np.array(distance_sem)**2 + np.array(bias_sem)**2)
        params, covariance = curve_fit(linear_func, distance_mean, bias_mean, sigma=weights)
        a = params[0]
        b = params[1]
        bias_mean_pred = a * np.array(distance_mean) + b
        plt.errorbar(distance_mean, bias_mean, xerr=distance_sem, yerr=bias_sem, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
        # plt.errorbar(distance_mean, bias_mean, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
#         plt.plot(distance_mean, bias_mean_pred, color=np.array([c]), label=ifeature)
        if a > 1:
            plt.axline((distance_mean[0], bias_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
        else:
            plt.axline((distance_mean[0], bias_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        # plt.axline((distance_mean[0], bias_mean_pred[0]), (distance_mean[-1], bias_mean_pred[-1]), color=np.array([c]), label=ifeature)
    plt.legend(ncol=2)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_per_feature_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_per_feature_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_rise_per_class_per_component(testsets, distance_dfs, train_bias_xlsx_path, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and bias rise per component per class (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = distance_dfs[i]
        df_distance = df_distance.reset_index(level=0)
        df_distance.columns.values[0] = 'class'
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        df_bias_train = preprocess_bias_xlsx(train_bias_xlsx_path)
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
            df_bias_train = df_bias_train.drop(df_bias_train[df_bias_train['Ground_truth'] < threshold].index)
            df_bias_train = df_bias_train.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['APE'] = df_bias['APE']
        dict_test = {key:bias for key, bias in zip(df_bias['class'], df_bias['APE'])}
        dict_train = {key:bias for key, bias in zip(df_bias_train['class'], df_bias_train['APE'])}
        bias_rise = []
        for i in dict_test.keys():
            bias_rise_class = 1 - np.divide(dict_test[i], dict_train[i])
            bias_rise.append(bias_rise_class)
        df_distance_bias['APE_rise'] = bias_rise
        for idistance in df_distance.columns.values[1:]:
            df_distance_bias[idistance] = df_distance[idistance]
        # df_distance_bias['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)
    df['mean_distance'] = df.iloc[:, 6:].mean(axis=1)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)

    df_slopes = pd.DataFrame(index=np.unique(df['class'].values), columns=df.columns.values[6:-1])
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for iclass in np.unique(df['class'].values):
        testsets_class = df[df['class'] == iclass].testset.values
        if len(testsets_class) < 2:
            continue
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.xlabel('Distance to training set')
        plt.ylabel('APE rise ratio')

        for ifeature,c in zip(df.columns.values[6:-1], colors):
            bias_rise_mean = []
            distance_mean = []
            for i, itestset in enumerate(testsets_class):
                bias_rise_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)]['APE_rise'])
                bias_rise_mean.append(bias_rise_mean_testset)
                
                distance_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)][ifeature])
                distance_mean.append(distance_mean_testset)
            params, covariance = curve_fit(linear_func, distance_mean, bias_rise_mean)
            a = params[0]
            b = params[1]
#             a, b = np.polyfit(distance_mean, bias_rise_mean, deg=1)
            bias_rise_mean_pred = a * np.array(distance_mean) + b
            # if a >= -5 and a <= 5:
            df_slopes.loc[iclass, ifeature] = a
            
            plt.scatter(distance_mean, bias_rise_mean, c=np.array([c]), alpha=0.3)
            if a < -1:
                plt.axline((distance_mean[0], bias_rise_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
            else:
                plt.axline((distance_mean[0], bias_rise_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        
        plt.title(iclass)
        plt.legend(ncol=2)
        plt.tight_layout()

        if PCA == 'yes':
            Path(outpath + 'per_PC_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_PC_per_class/' + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        elif PCA == 'no':
            Path(outpath + 'per_feature_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_feature_per_class/' + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        plt.close()

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 1, 1)
    plt.bar(x=range(len(df_slopes.columns)), height=df_slopes.mean(), color='royalblue')
    plt.xticks(range(len(df_slopes.columns)), labels=df_slopes.columns, rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Average slope')
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_rise_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_rise_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_bias_per_class_per_component(testsets, distance_dfs, train_bias_xlsx_path, bias_xlsx_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and bias per component per class (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = distance_dfs[i]
        df_distance = df_distance.reset_index(level=0)
        df_distance.columns.values[0] = 'class'
        df_bias = preprocess_bias_xlsx(bias_xlsx_paths[i])
        df_bias_train = preprocess_bias_xlsx(train_bias_xlsx_path)
        if threshold > 0:
            df_bias = df_bias.drop(df_bias[df_bias['Ground_truth'] < threshold].index)
            df_bias = df_bias.reset_index(drop=True)
            df_bias_train = df_bias_train.drop(df_bias_train[df_bias_train['Ground_truth'] < threshold].index)
            df_bias_train = df_bias_train.reset_index(drop=True)
        df_distance_bias = pd.DataFrame(columns=['class', 'testset', 'distance', 'APE'], index=range(len(df_distance)))
        df_distance_bias['class'] = df_distance['class']
        df_distance_bias['testset'] = testsets[i]
        df_distance_bias['APE'] = df_bias['APE']
        dict_test = {key:bias for key, bias in zip(df_bias['class'], df_bias['APE'])}
        dict_train = {key:bias for key, bias in zip(df_bias_train['class'], df_bias_train['APE'])}
        bias_rise = []
        for i in dict_test.keys():
            bias_rise_class = 1 - np.divide(dict_test[i], dict_train[i]+1e-8)
            bias_rise.append(bias_rise_class)
        df_distance_bias['APE_rise'] = bias_rise
        for idistance in df_distance.columns.values[1:]:
            df_distance_bias[idistance] = df_distance[idistance]
        # df_distance_bias['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_bias])
    df = df.reset_index(drop=True)
    df['mean_distance'] = df.iloc[:, 6:].mean(axis=1)

    if class_filter == 'yes':
        # df_ID = df[df['testset'] == 'ID_test']
        # dropped_classes_1 = df_ID[df_ID['APE'] > 0.7]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        # dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        # for iclass in dropped_classes:
        for iclass in dropped_classes_2:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)

    df_slopes = pd.DataFrame(index=np.unique(df['class'].values), columns=df.columns.values[6:-1])
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for iclass in np.unique(df['class'].values):
        testsets_class = df[df['class'] == iclass].testset.values
        if len(testsets_class) < 2:
            continue
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.xlabel('Distance to training set')
        plt.ylabel('Absolute Percentage Error')
        
        for ifeature,c in zip(df.columns.values[6:-1], colors):
            bias_mean = []
            distance_mean = []
            for i, itestset in enumerate(testsets_class):
                bias_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)]['APE'])
                bias_mean.append(bias_mean_testset)
                
                distance_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)][ifeature])
                distance_mean.append(distance_mean_testset)
            params, covariance = curve_fit(linear_func, distance_mean, bias_mean)
            a = params[0]
            b = params[1]
#             a, b = np.polyfit(distance_mean, bias_mean, deg=1)
            bias_mean_pred = a * np.array(distance_mean) + b
            # if a >= -5 and a <= 5:
            df_slopes.loc[iclass, ifeature] = a
            
            plt.scatter(distance_mean, bias_mean, c=np.array([c]), alpha=0.3)
            if a > 1:
                plt.axline((distance_mean[0], bias_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
            else:
                plt.axline((distance_mean[0], bias_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        
        plt.title(iclass)
        plt.legend(ncol=2)
        plt.tight_layout()

        if PCA == 'yes':
            Path(outpath + 'per_PC_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_PC_per_class/' + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        elif PCA == 'no':
            Path(outpath + 'per_feature_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_feature_per_class/' + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        plt.close()

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 1, 1)
    plt.bar(x=range(len(df_slopes.columns)), height=df_slopes.mean(), color='royalblue')
    plt.xticks(range(len(df_slopes.columns)), labels=df_slopes.columns, rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Average slope')
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_bias_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_bias_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def linear_func(x, a, b):
    return a * x + b


parser = argparse.ArgumentParser(description='Plot Domain distance versus prediction bias')
parser.add_argument('-testsets', nargs='*', help='list of testset names')
parser.add_argument('-distance_class_xlsx_paths', nargs='*', help='list of distance xlsx files')
parser.add_argument('-distance_txt_paths', nargs='*', help='list of distance txt files')
parser.add_argument('-train_bias_xlsx_path', help='bias xlsx file of training set')
parser.add_argument('-bias_xlsx_paths', nargs='*', help='list of bias xlsx files')
parser.add_argument('-outpath', help='path for saving the figure')
parser.add_argument('-model', help='model that being tested')
parser.add_argument('-feature_or_pixel', choices=['feature', 'pixel'], help='using feature distance or pixel distance')
parser.add_argument('-PCA', choices=['yes', 'no'], help='the distance is calculated with PCA or not')
parser.add_argument('-threshold', type=int, default=0, help='threshold of image number to select analysed classes')
parser.add_argument('-distance_type', choices=['Hellinger', 'Wasserstein', 'KL', 'Theta', 'Chi', 'I', 'Imax', 'Imagewise', 'Mahalanobis'], help='type of distribution distance')
parser.add_argument('-class_filter', choices=['yes', 'no'], help='whether to filter out less important classes')
args = parser.parse_args()

if __name__ == '__main__':
    # plot_distance_bias_scatter(args.testsets, args.distance_txt_paths, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type)
    # plot_distance(args.testsets, args.distance_txt_paths, args.outpath, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type)
    # plot_distance_bias_err(args.testsets, args.distance_txt_paths, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type)
    plot_distance_bias_testset(args.testsets, args.distance_txt_paths, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
    plot_distance_bias_class(args.testsets, args.distance_txt_paths, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
    # plot_distance_bias_rise_testset(args.testsets, args.distance_txt_paths, args.train_bias_xlsx_path, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
    # plot_distance_bias_rise_class(args.testsets, args.distance_txt_paths, args.train_bias_xlsx_path, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)

    if args.PCA == 'no':
        distance_dfs = []
        for i in args.distance_class_xlsx_paths:
            df_distance = pd.read_excel(i, index_col=0)
            distance_dfs.append(df_distance)
        plot_distance_bias(args.testsets, distance_dfs, args.train_bias_xlsx_path, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
        # plot_distance_bias_rise(args.testsets, distance_dfs, args.train_bias_xlsx_path, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
        plot_distance_bias_per_class_per_component(args.testsets, distance_dfs, args.train_bias_xlsx_path, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
        # plot_distance_bias_rise_per_class_per_component(args.testsets, distance_dfs, args.train_bias_xlsx_path, args.bias_xlsx_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)