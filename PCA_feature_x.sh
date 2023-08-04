#!/bin/bash -l
python3 PCA_feature.py -Zoolake2_datapath /home/EAWAG/chenchen/data/Zooplankton/train_data/training_zooplankton_new_220823/ -in_distribution_datapaths /home/EAWAG/chenchen/data/Zooplankton/train_data/splitting_20221212/0_train/ /home/EAWAG/chenchen/data/Zooplankton/train_data/splitting_20221212/0_val/ /home/EAWAG/chenchen/data/Zooplankton/train_data/splitting_20221212/0_test/ -OOD_datapaths /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD1/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD2/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD3/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD4/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD5/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD6/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD7/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD8/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD9/ /home/EAWAG/chenchen/data/Zooplankton/test_data/OOD10/ -outpath /home/EAWAG/chenchen/out/dataset_out/Zooplankton/20230804/feature_PCA_x/ -n_components 0.95 -nice_feature no -global_x yes