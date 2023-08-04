import os

import pandas as pd
import cv2


def LoadPixels(class_image_datapath, resized_length):
    df_pixels = pd.DataFrame()

    list_image = os.listdir(class_image_datapath)
    for img in list_image:
        if img == 'Thumbs.db':
            continue
        
        image = cv2.imread(class_image_datapath + '/' + img)
        image = cv2.resize(image, (resized_length, resized_length), interpolation=cv2.INTER_LANCZOS4)
        pixels = pd.Series(image.flatten())
        
        df_pixels = pd.concat([df_pixels, pixels], axis=1, ignore_index=True)

    df_pixels = df_pixels.transpose()

    return df_pixels