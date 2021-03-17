#!/bin/bash
cd /home/ys587/tensorflow3/upcall-basic-net/script 
#python ./DCLDE2018_cross_validate_augment.py -m lenet
python ./DCLDE2018_cross_validate_evaluate.py -m lenet
python ./DCLDE2018_cross_validate_augment.py -m birdnet
python ./DCLDE2018_cross_validate_evaluate.py -m birdnet
python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_conv
python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_conv
python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_conv
python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_conv
python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_input_conv,
python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_input_conv,
python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_input_conv,
python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_input_conv,
python ./DCLDE2018_cross_validate_augment.py -m recurr_lstm
python ./DCLDE2018_cross_validate_evaluate.py -m recurr_lstm
python ./DCLDE2018_cross_validate_augment.py -m lenet_dropout_input
python ./DCLDE2018_cross_validate_evaluate.py -m lenet_dropout_input
python ./DCLDE2018_cross_validate_augment.py -m birdnet_dropout_input 
python ./DCLDE2018_cross_validate_evaluate.py -m birdnet_dropout_input
#python ./DCLDE2018_cross_validate_augment.py -m 
#python ./DCLDE2018_cross_validate_evaluate.py -m 
#aws s3 cp /home/ubuntu/__Paper/x-validate-birdnet s3://upcall-net-cv/ --recursive
