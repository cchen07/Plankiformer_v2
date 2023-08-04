import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import utils_analysis.lib.distance_metrics as dm

plt.rcParams['figure.dpi'] = 300
plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15)
plt.rc('figure', titlesize=15) 

def PlotPixelDistribution_PCA(PCA_files_pixel, outpath, selected_components_pixel, n_bins_pixel, data_labels, image_threshold, distance_type):

    print('-----------------Now plotting PCA pixel distribution for each class and each selected component.-----------------')
    
    df_pca_1 = pd.read_csv(PCA_files_pixel[0], sep=',', index_col=0)
    df_pca_2 = pd.read_csv(PCA_files_pixel[1], sep=',', index_col=0)

    list_class_1 = np.unique(df_pca_1['class'])
    list_class_2 = np.unique(df_pca_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    if selected_components_pixel == ['all']:
        selected_components_pixel = df_pca_1.columns.values[:-1]

    for iclass in list_class_rep:
        df_pca_1_class = df_pca_1[df_pca_1['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)
        df_pca_2_class = df_pca_2[df_pca_2['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)

        if not (df_pca_1_class.shape[0] >= image_threshold and df_pca_2_class.shape[0] >= image_threshold):
            continue

        for ipc in selected_components_pixel:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ipc + ' (normalized)')
            ax.set_ylabel('Density')

            component_1 = df_pca_1_class[ipc]
            component_2 = df_pca_2_class[ipc]
            components = [component_1, component_2]

            min_bin = np.min([min(component_1), min(component_2)])
            max_bin = np.max([max(component_1), max(component_2)])

            normalized_component = np.divide((np.array(components, dtype=object) - min_bin), (max_bin - min_bin))

            histogram = plt.hist(normalized_component, histtype='stepfilled', bins=n_bins_pixel, range=(0, 1), density=True, alpha=0.5, label=data_labels)
            density_1 = histogram[0][0]
            density_2 = histogram[0][1]
            
            if distance_type == 'Hellinger':
                distance = dm.HellingerDistance(density_1, density_2)
            elif distance_type == 'Wasserstein':
                distance = dm.WassersteinDistance(density_1, density_2)
            elif distance_type == 'KL':
                distance = dm.KullbackLeibler(density_1, density_2)
            elif distance_type == 'Theta':
                distance = dm.ThetaDistance(density_1, density_2)
            elif distance_type == 'Chi':
                distance = dm.ChiDistance(density_1, density_2)
            elif distance_type == 'I':
                distance = dm.IDistance(density_1, density_2)
            elif distance_type == 'Imax':
                distance = dm.ImaxDistance(density_1, density_2)
            
            plt.title(distance_type + ' Distance = %.3f' % distance)
            plt.legend()
            plt.tight_layout()

            outpath_component = outpath + ipc + '/'
            Path(outpath_component).mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath_component + iclass + '_' + distance_type + '.png')
            plt.close()
            ax.clear()


def PlotGlobalDistanceversusBin_pixel_PCA(PCA_files_pixel, outpath, explained_variance_ratio_pixel, image_threshold, distance_type):

    print('-----------------Now plotting global distances of PCA pixel v.s. numbers of bin.-----------------')

    df_pca_1 = pd.read_csv(PCA_files_pixel[0], sep=',', index_col=0)
    df_pca_2 = pd.read_csv(PCA_files_pixel[1], sep=',', index_col=0)

    # list_class_1 = np.unique(df_pca_1['class'])
    # list_class_2 = np.unique(df_pca_2['class'])
    # list_class_rep = list(set(list_class_1) & set(list_class_2))
    # list.sort(list_class_rep)

    class_1 = df_pca_1['class'].to_list()
    class_2 = df_pca_2['class'].to_list()
    df_count_1 = pd.DataFrame(np.unique(class_1, return_counts=True)).transpose()
    df_count_2 = pd.DataFrame(np.unique(class_2, return_counts=True)).transpose()
    list_class_1 = df_count_1[df_count_1.iloc[:, 1]>=image_threshold].iloc[:, 0].to_list()
    list_class_2 = df_count_2[df_count_2.iloc[:, 1]>=image_threshold].iloc[:, 0].to_list()
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_principal_components = df_pca_1.columns.to_list()[:-1]
    # list_explained_variance_ratio = np.loadtxt(explained_variance_ratio_pixel)
    list_n_bins = [5, 10, 15, 20, 25, 30]

    df_global_distance = pd.DataFrame(columns=[in_bins for in_bins in list_n_bins], index=[iclass for iclass in list_class_rep])
    for in_bins in list_n_bins:
        list_global_distance = []
        for iclass in list_class_rep:
            df_pca_1_class = df_pca_1[df_pca_1['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)
            df_pca_2_class = df_pca_2[df_pca_2['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)

            list_distance = []
            for ipc in list_principal_components:
                component_1 = df_pca_1_class[ipc]
                component_2 = df_pca_2_class[ipc]
                components = [component_1, component_2]

                min_bin = np.min([min(component_1), min(component_2)])
                max_bin = np.max([max(component_1), max(component_2)])

                normalized_component = np.divide((np.array(components, dtype=object) - min_bin), (max_bin - min_bin))

                histogram_1 = np.histogram(normalized_component[0], bins=in_bins, range=(0, 1), density=True)
                histogram_2 = np.histogram(normalized_component[1], bins=in_bins, range=(0, 1), density=True)
                density_1 = histogram_1[0]
                density_2 = histogram_2[0]
                
                if distance_type == 'Hellinger':
                    distance = dm.HellingerDistance(density_1, density_2)
                elif distance_type == 'Wasserstein':
                    distance = dm.WassersteinDistance(density_1, density_2)
                elif distance_type == 'KL':
                    distance = dm.KullbackLeibler(density_1, density_2)
                elif distance_type == 'Theta':
                    distance = dm.ThetaDistance(density_1, density_2)
                elif distance_type == 'Chi':
                    distance = dm.ChiDistance(density_1, density_2)
                elif distance_type == 'I':
                    distance = dm.IDistance(density_1, density_2)
                elif distance_type == 'Imax':
                    distance = dm.ImaxDistance(density_1, density_2)
                list_distance.append(distance)

            # global_distance_each_class = np.average(list_distance, weights=list_explained_variance_ratio)
            global_distance_each_class = np.average(list_distance)
            list_global_distance.append(global_distance_each_class)

        df_global_distance[in_bins] = list_global_distance

    df_global_distance = df_global_distance.transpose()

    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Number of bins')
    ax.set_ylabel(distance_type + ' Distance')
    plt.figure(figsize=(10, 10))
    
    for iclass in list_class_rep:
        plt.plot(list_n_bins, df_global_distance[iclass], label=iclass)

    plt.legend(bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.tight_layout()

    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + distance_type + '_distance_bin_pixel_PCA.png')
    plt.close()
    ax.clear()
    

def GlobalDistance_pixel(PCA_files_pixel, outpath, n_bins_pixel, explained_variance_ratio_pixel, image_threshold, distance_type):

    print('-----------------Now computing global distances on PCA pixel.-----------------')

    df_pca_1 = pd.read_csv(PCA_files_pixel[0], sep=',', index_col=0)
    df_pca_2 = pd.read_csv(PCA_files_pixel[1], sep=',', index_col=0)

    list_class_1 = np.unique(df_pca_1['class'])
    list_class_2 = np.unique(df_pca_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_principal_components = df_pca_1.columns.to_list()[:-1]
    # list_explained_variance_ratio = np.loadtxt(explained_variance_ratio_pixel)

    df_pca = pd.DataFrame()
    list_global_distance = []
    for iclass in list_class_rep:
        df_pca_1_class = df_pca_1[df_pca_1['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)
        df_pca_2_class = df_pca_2[df_pca_2['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)

        if not (df_pca_1_class.shape[0] >= image_threshold and df_pca_2_class.shape[0] >= image_threshold):
            continue
        
        list_distance = []
        for ipc in list_principal_components:
            component_1 = df_pca_1_class[ipc]
            component_2 = df_pca_2_class[ipc]
            components = [component_1, component_2]

            min_bin = np.min([min(component_1), min(component_2)])
            max_bin = np.max([max(component_1), max(component_2)])

            normalized_component = np.divide((np.array(components, dtype=object) - min_bin), (max_bin - min_bin))

            histogram_1 = np.histogram(normalized_component[0], bins=n_bins_pixel, range=(0, 1), density=True)
            histogram_2 = np.histogram(normalized_component[1], bins=n_bins_pixel, range=(0, 1), density=True)
            density_1 = histogram_1[0]
            density_2 = histogram_2[0]

            if distance_type == 'Hellinger':
                distance = dm.HellingerDistance(density_1, density_2)
            elif distance_type == 'Wasserstein':
                distance = dm.WassersteinDistance(density_1, density_2)
            elif distance_type == 'KL':
                distance = dm.KullbackLeibler(density_1, density_2)
            elif distance_type == 'Theta':
                distance = dm.ThetaDistance(density_1, density_2)
            elif distance_type == 'Chi':
                distance = dm.ChiDistance(density_1, density_2)
            elif distance_type == 'I':
                distance = dm.IDistance(density_1, density_2)
            elif distance_type == 'Imax':
                distance = dm.ImaxDistance(density_1, density_2)
            list_distance.append(distance)

        df_pca[iclass] = list_distance

        # global_distance_each_class = np.average(list_distance, weights=list_explained_variance_ratio)
        global_distance_each_class = np.average(list_distance)
        list_global_distance.append(global_distance_each_class)

        Path(outpath).mkdir(parents=True, exist_ok=True)
        with open(outpath + 'Global_' + distance_type + '_Distance_pixel_PCA.txt', 'a') as f:
            f.write('%-20s%-20f\n' % (iclass, global_distance_each_class))

    df_pca = df_pca.transpose()
    df_pca.to_excel(outpath + distance_type + '_Distance_class_pixel_PCA.xlsx', index=True)

    global_distance = np.average(list_global_distance)
    with open(outpath + 'Global_' + distance_type + '_Distance_pixel_PCA.txt', 'a') as f:
        f.write(f'\n Global Distance: {global_distance}\n')

