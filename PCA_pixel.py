import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils_analysis.lib import pixel_extraction as pe


def ConcatAllClasses(datapath, resized_length):

    '''Concatenate pixels of all images (all classes) in a dataset and normalize.'''

    list_class = os.listdir(datapath)
    list.sort(list_class)
    df_all_pixel = pd.DataFrame()

    for iclass in list_class:
        class_datapath = datapath + iclass + '/'
        df_class_pixel = pe.LoadPixels(class_datapath, resized_length)
        df_class_pixel = df_class_pixel / 255
        df_class_pixel['class'] = iclass
        df_all_pixel = pd.concat([df_all_pixel, df_class_pixel], ignore_index=True)

    return df_all_pixel


def PrincipalComponentAnalysis(dataframe, n_components):

    '''Principal component analysis on a dataframe.'''

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(dataframe.iloc[:, :-1].values)
    df_pca = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(n_components)])
    df_pca['class'] = dataframe['class']

    return pca, df_pca


def PCA_train_val_test(dataframe, pca):

    '''Implement PCA on in-distribution datasets.'''

    principal_components = pca.fit_transform(dataframe.iloc[:, :-1].values)
    df_pca_split = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(np.shape(principal_components)[1])])
    df_pca_split['class'] = dataframe['class']

    return df_pca_split


def PCA_OOD(dataframe_OOD, pca):

    '''Implement PCA on out-of-distribution datasets.'''

    principal_components = pca.fit_transform(dataframe_OOD.iloc[:, :-1].values)
    df_pca_OOD = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(np.shape(principal_components)[1])])
    df_pca_OOD['class'] = dataframe_OOD['class']

    return df_pca_OOD


parser = argparse.ArgumentParser(description='Principal component analysis on datasets')
parser.add_argument('-Zoolake2_datapath', help='path of the Zoolake2 dataset')
parser.add_argument('-in_distribution_datapaths', nargs='*', help='paths of the in-domain datasets, in an order of: train_val_test')
parser.add_argument('-OOD_datapaths', nargs='*', help='paths of the out-of-distribution datasets')
parser.add_argument('-outpath', help='path for saving output csv')
parser.add_argument('-n_components', type=int, help='number of principal components')
parser.add_argument('-resized_length', type=int, help='length of resized image')
args = parser.parse_args()


if __name__ == '__main__':

    df = ConcatAllClasses(args.Zoolake2_datapath, args.resized_length)
    pca, df_pca = PrincipalComponentAnalysis(df, n_components=args.n_components)
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    df_pca.to_csv(args.outpath + 'PCA_Zoolake2_pixel.csv')

    np.savetxt(args.outpath + 'PCA_explained_variance_ratio_pixel.txt', pca.explained_variance_ratio_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance ratio')
    plt.grid()
    plt.tight_layout()
    plt.savefig(args.outpath + 'PCA_explained_variance_ratio_pixel.png')


    df_train = ConcatAllClasses(args.in_distribution_datapaths[0], args.resized_length)
    df_val = ConcatAllClasses(args.in_distribution_datapaths[1], args.resized_length)
    df_test = ConcatAllClasses(args.in_distribution_datapaths[2], args.resized_length)

    df_pca_train = PCA_train_val_test(df_train, pca)
    df_pca_val = PCA_train_val_test(df_val, pca)
    df_pca_test = PCA_train_val_test(df_test, pca)
    df_pca_train.to_csv(args.outpath + 'PCA_train_pixel.csv')
    df_pca_val.to_csv(args.outpath + 'PCA_val_pixel.csv')
    df_pca_test.to_csv(args.outpath + 'PCA_test_pixel.csv')

    for i in range(len(args.OOD_datapaths)):
        df_OOD = ConcatAllClasses(args.OOD_datapaths[i], args.resized_length)

        df_pca_OOD = PCA_OOD(df_OOD, pca)
        df_pca_OOD.to_csv(args.outpath + 'PCA_OOD{}_pixel.csv'.format(i + 1))