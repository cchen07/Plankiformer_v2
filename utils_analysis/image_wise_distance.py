from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def normalize_feature_df(df_feature_1, df_feature_2):
    df_feature = pd.concat([df_feature_1.drop(['class'], axis=1), df_feature_2.drop(['class'], axis=1)])
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df_feature)
    
    df_feature_1_normalized = pd.DataFrame(normalized_data[:len(df_feature_1)], columns=df_feature_1.columns[:-1])
    df_feature_2_normalized = pd.DataFrame(normalized_data[len(df_feature_1):], columns=df_feature_2.columns[:-1])
    
    df_feature_1_normalized['class'] = df_feature_1['class']
    df_feature_2_normalized['class'] = df_feature_2['class']
    return df_feature_1_normalized, df_feature_2_normalized

def distance_image_wise(df_feature_1, df_feature_2, image_threshold):
    list_class_1 = np.unique(df_feature_1['class'])
    list_class_2 = np.unique(df_feature_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_features_all = df_feature_1.columns.to_list()[:-1]
    
    df_distance = pd.DataFrame(columns=list_features_all, index=list_class_rep)
    list_distance = []
    for ii, iclass in enumerate(list_class_rep):
        df_1_class = df_feature_1[df_feature_1['class']==iclass].drop(['class'], axis=1).dropna(how='all', axis=1).reset_index(drop=True)
        df_2_class = df_feature_2[df_feature_2['class']==iclass].drop(['class'], axis=1).dropna(how='all', axis=1).reset_index(drop=True)
        df_class = pd.concat([df_1_class, df_2_class])
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df_class)
        df_1_class_normalized = pd.DataFrame(normalized_data[:len(df_1_class)], columns=df_1_class.columns)
        df_2_class_normalized = pd.DataFrame(normalized_data[len(df_1_class):], columns=df_2_class.columns)

        list_features = df_1_class_normalized.columns.values

        if not (df_1_class_normalized.shape[0] >= image_threshold and df_2_class_normalized.shape[0] >= image_threshold):
            continue
        
        list_distance_class = []
        for ifeature in list_features:
            feature_1 = df_1_class_normalized[ifeature]
            feature_2 = df_2_class_normalized[ifeature]
            
            list_distance_feature = []
            for i in feature_1:
                for j in feature_2:
                    distance = abs(i - j)
                    list_distance_feature.append(distance)
            distance_feature = np.mean(list_distance_feature)
            list_distance_class.append(distance_feature)
        df_distance.iloc[ii, :len(list_distance_class)] = list_distance_class
        distance_class = np.mean(list_distance_class)
        list_distance.append(distance_class)
    # df_distance['global_distance'] = list_distance
    df_distance = df_distance.dropna(how='all', axis=0)
    
    return df_distance

def GlobalDistance_feature(feature_files, outpath, PCA, image_threshold):

    print('-----------------Now computing global distances on feature (threshold: {}, distance: Imagewise).-----------------'.format(image_threshold))

    df_1 = pd.read_csv(feature_files[0], index_col=0)
    df_2 = pd.read_csv(feature_files[1], index_col=0)

    col_class_1 = df_1.pop('class')
    col_class_2 = df_2.pop('class')
    df_1['class'] = col_class_1
    df_2['class'] = col_class_2

    # df_1_normalized, df_2_normalized = normalize_feature_df(df_1, df_2)

    df_distance = distance_image_wise(df_1, df_2, image_threshold)

    if PCA == 'yes':
        df_distance.columns = ['PC_' + str(i+1) for i in range(len(df_1.columns.to_list()[:-1]))]

    Path(outpath).mkdir(parents=True, exist_ok=True)

    list_mean_distance = []
    for iclass in df_distance.index:
        mean_distance = np.nanmean(df_distance.loc[iclass])
        list_mean_distance.append(mean_distance)
        if PCA == 'no':
            with open(outpath + 'Global_Imagewise_Distance_feature.txt', 'a') as f:
                f.write('%-20s%-20f\n' % (iclass, mean_distance))
        elif PCA == 'yes':
            with open(outpath + 'Global_Imagewise_Distance_feature_PCA.txt', 'a') as f:
                f.write('%-20s%-20f\n' % (iclass, mean_distance))
    if PCA == 'no':
        df_distance.to_excel(outpath + 'Imagewise_Distance_class_feature.xlsx', index=True)
        global_distance = np.nanmean(list_mean_distance)
        with open(outpath + 'Global_Imagewise_Distance_feature.txt', 'a') as f:
            f.write(f'\n Global Distance: {global_distance}\n')
    elif PCA == 'yes':
        df_distance.to_excel(outpath + 'Imagewise_Distance_class_feature_PCA.xlsx', index=True)
        global_distance = np.nanmean(list_mean_distance)
        with open(outpath + 'Global_Imagewise_Distance_feature_PCA.txt', 'a') as f:
            f.write(f'\n Global Distance: {global_distance}\n')