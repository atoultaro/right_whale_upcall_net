# Upcall basic net

Sept 4, 2019
Merge the branch YuShiuMasterWork to master.


Aug 1, 2018
Test Kaitlin's implementation using dsp package on my accuracy_measure.py. Several bugs were found and fixed. 
Precision/recall curve was drawed. It was mostly decent but the curve shows the probability score is lower 
than it's supposed to be using my own detection code (de0bfipstwytection.py).


July 24, 2018
3 main modules: 

train_classifier.py: eveything we need to train a classifier

detection.py: eveything we need to run detection, given a trained classifier

accuracy_measure.py: eveything we need to measure the accuracy by comparing with truth files


2 others:

config: parameter configuration

sound_util: eveything we need to read & process sounds


3 specific scripts to DCLDE 2018:

DCLDE2018_data: configuration to multiple training datasets for DCLDE 2018

DCLDE2018_train_classifier: script to train a classifier

DCLDE2018_run_detection: script to run multiple models and multiple days for DCLDE 2018

=====================
July 3, 2018
Module overview:

UpcallExptV2.py
Classifier training/validation from sound clips and deep neural network model selection


ModelSearchExptV0.py
Detector/Classifier testing on sound stream


NegDataGenerate.py
(1) generate detector performance (false positive number, false negative number and etc.) and 
(2) generate negative sound clips to classifier training dataset


Mar, 2018
Weekend hack-a-thon to create a classification model

