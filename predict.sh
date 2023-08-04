#!/bin/bash -l
#test_path=/home/EAWAG/chenchen/data/Zooplankton/train_data/splitting_20221212/0_test/
#test_path=/home/EAWAG/chenchen/data/Zooplankton/train_data/splitting_20221212/0_train/
#test_path=/home/EAWAG/chenchen/data/Zooplankton/test_data/zooformer_5000images/
test_path=/home/EAWAG/chenchen/data/Zooplankton/test_data/OOD1/

out=/home/EAWAG/chenchen/out/predict_out/Zooplankton/20230801/EfficientNetB2/medium_aug_01/OOD1/
param_path=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/EfficientNetB2/
model_path=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/EfficientNetB2/trained_models/medium_aug_01/
ensemble=0
fine=1
threshold=0.0
gpu=yes
gpu_id=1
python predict_labeled.py -test_path $test_path -test_outpath $out -main_param_path $param_path -model_path $model_path -ensemble $ensemble -finetuned $fine -threshold $threshold -use_gpu $gpu -gpu_id $gpu_id

out=/home/EAWAG/chenchen/out/predict_out/Zooplankton/20230801/EfficientNetB2/medium_aug_02/OOD1/
param_path=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/EfficientNetB2/
model_path=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/EfficientNetB2/trained_models/medium_aug_02/
ensemble=0
fine=1
threshold=0.0
gpu=yes
gpu_id=1
python predict_labeled.py -test_path $test_path -test_outpath $out -main_param_path $param_path -model_path $model_path -ensemble $ensemble -finetuned $fine -threshold $threshold -use_gpu $gpu -gpu_id $gpu_id

out=/home/EAWAG/chenchen/out/predict_out/Zooplankton/20230801/EfficientNetB2/medium_aug_03/OOD1/
param_path=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/EfficientNetB2/
model_path=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/EfficientNetB2/trained_models/medium_aug_03/
ensemble=0
fine=1
threshold=0.0
gpu=yes
gpu_id=1
python predict_labeled.py -test_path $test_path -test_outpath $out -main_param_path $param_path -model_path $model_path -ensemble $ensemble -finetuned $fine -threshold $threshold -use_gpu $gpu -gpu_id $gpu_id