import os
import argparse
from pathlib import Path

import pandas as pd

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


parser = argparse.ArgumentParser(description='extract the pixels of a dataset and save them as a file')
parser.add_argument('-datapaths', nargs='*', help='paths of datasets')
parser.add_argument('-dataset_labels', nargs='*', help='label of each dataset')
parser.add_argument('-outpath', help='path for saving output file')
parser.add_argument('-resized_length', type=int, help='length of resized images')
args = parser.parse_args()

if __name__ == '__main__':

    for idatapath, ilabel in zip(args.datapaths, args.dataset_labels):
        df_all_pixel = ConcatAllClasses(idatapath, args.resized_length)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        df_all_pixel.to_csv(args.outpath + ilabel + '_pixel.csv')