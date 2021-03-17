#!/bin/bash
cd /home/ubuntu3/tensorflow3/upcall-basic-net/script 
#python ./DCLDE2018_cross_validate_augment.py -m lenet
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet
#python ./DCLDE2018_cross_validate_augment.py -m lenet -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet
#python ./DCLDE2018_cross_validate_augment.py -m birdnet -a
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet -a

#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv
#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv -a
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv -a
#python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru 
#python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru
#python ./DCLDE2018_cross_validate_augment.py -m recurr_lstm
##python ./DCLDE2018_cross_validate_evaluate.py -m recurr_lstm

#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv -d Data_1 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv -d Data_1 -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv -d Data_1 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv -d Data_1 -a
#python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru -d Data_1
#python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru -d Data_1

#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv -d Data_3 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv -d Data_3 -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv -d Data_3 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv -d Data_3 -a
#python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru -d Data_3
#python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru -d Data_3

#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv -d Data_4 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv -d Data_4 -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv -d Data_4 -a
#python ./DCLDE2018_cross_validate_evaluate_split_data.py -m birdnet_dropout_conv -d Data_4 -a
#python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru -d Data_3
python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru -d Data_3
python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru -d Data_4
python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru -d Data_4

#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv -d Data_5 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv -d Data_5 -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv -d Data_5 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv -d Data_5 -a
#python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru -d Data_5
#python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru -d Data_5

#python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv -d Data_6 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv -d Data_6 -a
#python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv -d Data_6 -a
#python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv -d Data_6 -a
#python ./DCLDE2018_cross_validate_augment.py -m conv1d_gru -d Data_6
#python ./DCLDE2018_cross_validate_evaluate.py -m conv1d_gru -d Data_6

#aws s3 cp /home/ubuntu/__ExptResult/x-validate-birdnet s3://upcall-net-cv/ --recursive
