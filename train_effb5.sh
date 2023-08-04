#!/bin/bash -l
gpu_id=0
data=/home/EAWAG/chenchen/data/Zooplankton/train_data/training_zooplankton_new_220823/
out=/home/EAWAG/chenchen/out/train_out/Zooplankton/20230120_baseline/EfficientNetB5/
arch=efficientnetb5
name=Init_1
bs=22
epoch=100
fine=1
epoch_fine=100
add_layer=no
last_layer_fine=yes
lr=0.000524272724987394
lr_fine=0.0000524272724987394
wd=0.0167187878541967
do_1=0
do_2=0
fc=0
lr_scheduler=no
es=no
balance=yes
loss_f1_acc=2
warmup=0
resume=no
python3 main.py -datapaths $data -outpath $out -epochs $epoch -finetune $fine -finetune_epochs $epoch_fine -batch_size $bs -init_name $name -architecture $arch -add_layer $add_layer -last_layer_finetune $last_layer_fine -gpu_id $gpu_id -run_lr_scheduler $lr_scheduler -run_early_stopping $es -resume_from_saved $resume -lr $lr -finetune_lr $lr_fine -weight_decay $wd -dropout_1 $do_1 -dropout_2 $do_2 -fc_node $fc -balance_weight $balance -save_best_model_on_loss_or_f1_or_accuracy $loss_f1_acc -warmup $warmup -classifier multi -aug -datakind image -ttkind image -save_data yes -resize_images 1 -L 128 -valid_set yes -test_set yes -dataset_name zoolake -training_data False -run_cnn_or_on_colab yes -use_gpu yes