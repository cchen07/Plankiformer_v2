#!/bin/bash -l

python3 dataset_analysis.py -datapaths /home/EAWAG/chenchen/data/Zooplankton/train_data/splitting_20221212/0_train/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD5/ -datapath_labels Zoolake2_train Zoolake2_OOD5 -outpath /home/EAWAG/chenchen/out/dataset_out/Zooplankton/20230214/abundance/