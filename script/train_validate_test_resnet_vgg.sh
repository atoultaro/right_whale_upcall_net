#!/bin/bash
cd /home/ys587/tensorflow3/upcall-basic-net/script

python ./DCLDE2018_cross_validate_augment.py -m resnet
python ./DCLDE2018_cross_validate_augment.py -m vgg
python ./DCLDE2018_cross_validate_evaluate.py -m resnet
python ./DCLDE2018_cross_validate_evaluate.py -m vgg

