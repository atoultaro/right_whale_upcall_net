#!/bin/bash
cd /home/ys587/tensorflow3/upcall-basic-net/script 
python ./DCLDE2018_cross_validate_augment.py -m recurr_lstm -t
python ./DCLDE2018_cross_validate_evaluate.py -m recurr_lstm -t
#aws s3 cp /home/ubuntu/__Paper/x-validate-birdnet s3://upcall-net-cv/ --recursive
