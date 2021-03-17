#!/bin/bash
cd /home/ys587/tensorflow3/upcall-basic-net/script

#python ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.py -m lenet_dropout_conv -d Data_3 -a
#python ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.py -m lenet_dropout_conv -d Data_2 -a
python ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.py -m lenet_dropout_conv -d Data_1 -a
#python ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.py -m lenet_dropout_conv -d Data_4 -a
#python ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.py -m lenet_dropout_conv -d Data_6 -a
#python ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.py -m lenet_dropout_conv -d Data_5 -a

#nohup ./DCLDE2018_cross_validate_evaluate_full_data_to_large_Virginia_dataset.sh > ../../20190117.log 2>&1 &
