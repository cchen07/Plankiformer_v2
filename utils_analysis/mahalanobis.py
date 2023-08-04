from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial import distance


def MahalanobisDistance(data_point, df_ID_feature):
    ID_mean = np.mean(df_ID_feature, axis=0)
    cov_matrix = np.cov(df_ID_feature.values, rowvar=False)
    cov_matrix_inverse = np.linalg.inv(cov_matrix)
    mahalanobis = distance.mahalanobis(data_point, ID_mean, cov_matrix_inverse)
    return mahalanobis

def distance_mahalanobis(df_feature_1, df_feature_2, image_threshold):
    list_class_1 = np.unique(df_feature_1['class'])
    list_class_2 = np.unique(df_feature_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)
    
    df_distance = pd.DataFrame(columns=['Mahalanobis'], index=list_class_rep)
    for iclass in list_class_rep:
        df_1_class = df_feature_1[df_feature_1['class']==iclass].drop(['class'], axis=1).dropna(how='all', axis=1).reset_index(drop=True)
        df_2_class = df_feature_2[df_feature_2['class']==iclass].drop(['class'], axis=1).dropna(how='all', axis=1).reset_index(drop=True)

        if not (df_1_class.shape[0] >= image_threshold and df_2_class.shape[0] >= image_threshold):
            continue

        list_distance_class = []
        for i in df_2_class.index:
            mahalanobis = MahalanobisDistance(df_2_class.iloc[i].values, df_1_class)
            list_distance_class.append(mahalanobis)
        distance_class = np.mean(list_distance_class)
        df_distance.loc[iclass, 'Mahalanobis'] = distance_class
    df_distance = df_distance.dropna(axis=0)

    return df_distance

def distance_mahalanobis_mean(df_feature_1, df_feature_2, image_threshold):
    list_class_1 = np.unique(df_feature_1['class'])
    list_class_2 = np.unique(df_feature_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)
    
    df_distance = pd.DataFrame(columns=['Mahalanobis'], index=list_class_rep)
    for iclass in list_class_rep:
        df_1_class = df_feature_1[df_feature_1['class']==iclass].drop(['class'], axis=1).dropna(how='all', axis=1).reset_index(drop=True)
        df_2_class = df_feature_2[df_feature_2['class']==iclass].drop(['class'], axis=1).dropna(how='all', axis=1).reset_index(drop=True)

        if not (df_1_class.shape[0] >= image_threshold and df_2_class.shape[0] >= image_threshold):
            continue

        mahalanobis = MahalanobisDistance(np.mean(df_2_class, axis=0), df_1_class)
        df_distance.loc[iclass, 'Mahalanobis'] = mahalanobis
    df_distance = df_distance.dropna(axis=0)

    return df_distance

def GlobalDistance_feature(feature_files, outpath, PCA, image_threshold, mahal_mean):

    print('-----------------Now computing global distances on feature (threshold: {}, distance: Mahalanobis).-----------------'.format(image_threshold))

    df_1 = pd.read_csv(feature_files[0], index_col=0)
    df_2 = pd.read_csv(feature_files[1], index_col=0)

    if mahal_mean == 'no':
        df_distance = distance_mahalanobis(df_1, df_2, image_threshold)
    elif mahal_mean == 'yes':
        df_distance = distance_mahalanobis_mean(df_1, df_2, image_threshold)

    # if PCA == 'yes':
    #     df_distance.columns = ['PC_' + str(i+1) for i in range(len(df_1.columns.to_list()[:-1]))]

    Path(outpath).mkdir(parents=True, exist_ok=True)

    for iclass in df_distance.index:
        if PCA == 'no':
            with open(outpath + 'Global_Mahalanobis_Distance_feature.txt', 'a') as f:
                f.write('%-20s%-20f\n' % (iclass, df_distance.loc[iclass, 'Mahalanobis']))
        elif PCA == 'yes':
            with open(outpath + 'Global_Mahalanobis_Distance_feature_PCA.txt', 'a') as f:
                f.write('%-20s%-20f\n' % (iclass, df_distance.loc[iclass, 'Mahalanobis']))
    if PCA == 'no':
        df_distance.to_excel(outpath + 'Mahalanobis_Distance_class_feature.xlsx', index=True)
        global_distance = np.average(df_distance)
        with open(outpath + 'Global_Mahalanobis_Distance_feature.txt', 'a') as f:
            f.write(f'\n Global Distance: {global_distance}\n')
    elif PCA == 'yes':
        df_distance.to_excel(outpath + 'Mahalanobis_Distance_class_feature_PCA.xlsx', index=True)
        global_distance = np.average(df_distance)
        with open(outpath + 'Global_Mahalanobis_Distance_feature_PCA.txt', 'a') as f:
            f.write(f'\n Global Distance: {global_distance}\n')