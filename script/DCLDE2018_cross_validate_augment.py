# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:21:46 2018

@author: ys587

Sept 18, 2018
To cross-validate of classification/detection using split training/validation data
To train a classifier, later used for evaluation sound stream, using all training data (no split)


build the convnet model, prepare training data and run the classifier 
training

Args:
    LabelFile: the truth label in txt file
    SoundPath: path to the training sound files
    config (class object): configuration parameters 
    check_path: output path of training results
Returns:
    True if the function runs well
"""
import os, sys
ROOT_DIR = os.path.abspath("../../upcall-basic-net") 
# the path where upcall-basic-net is
sys.path.append(ROOT_DIR)

from upcall.config import Config
#from upcall.train_classifier import *
from upcall.train_classifier import lenet, lenet_dropout_input, \
lenet_dropout_conv, lenet_dropout_input_conv, birdnet, birdnet_dropout_input, \
birdnet_dropout_conv, birdnet_dropout_input_conv, resnet, vgg
from upcall.train_classifier import recurr_lstm, conv1d_gru
from upcall.train_classifier import net_train_cross_validation

from script.DCLDE2018_train_data import prepare_truth_data
import time
from contextlib import redirect_stdout
#import numpy as np

import argparse
# Parse Arguments
parser = argparse.ArgumentParser(description="Experiment A: train on DCLDE 13' clips 4 days & tested on DCLDE 13' sound stream 3 days")
parser.add_argument("-m", "--model_name", type=str, default=None, help="MODEL_NAME: lenet, lenet_dropout_input, lenet_dropout_conv, \
                    lenet_dropout_input_conv, birdnet, birdnet_dropout_input, birdnet_dropout_conv, birdnet_dropout_input_conv, recurr_lstm, \
                    conv1d_gru, resnet, vgg") 
parser.add_argument("-d", "--dataset_name", type=str, default=None, help="DATASET_NAME: Data_1 for Kaggle; Data_2 for DCL13'; Data_3 for BOEM Virginia; Data_4 for Kaggle & DCL13; Data_5 for BOEM & DCL13; Data_6 for BOEM, Kaggle & DCL13") 
parser.add_argument("-a", "--augment", action='store_true', help="Do data augmentation when -a is True; no augmentation otherwise")
parser.add_argument("-t","--test", action='store_true', help = 'Use testing parameters')
args = parser.parse_args()

start_time = time.time() # start counting the time. Will show the run time in the end.

##################################
DATA_DIR = r'/home/ys587/__Data' # Data
TRAIN_RESULT_FOLDER = r'/home/ys587/__ExptResult' # Result
##################################

##################################
# Only parameter to speecify
#model_name = r'lenet'
#model_name = r'birdnet'
#model_name = r'lenet_dropout_conv'
#model_name = r'birdnet_dropout_conv'
if args.model_name is None:
    #model_name = "lenet"
    model_name = 'birdnet'
else:
    model_name = args.model_name
print('Model name: '+model_name)

if args.dataset_name is None:
    dataset_name = "Data_2"
else:
    dataset_name = args.dataset_name
print('Dataset name: '+dataset_name)
##################################
# DATASET & MODEL
config = Config(dataset=dataset_name, model=model_name)
#config = Config(dataset='Data_2', model='lenet')
#config = Config(dataset='Data_2', model='birdnet') # failed a couple of times, probably needs augmentation
#config = Config(dataset='Data_2', model='lenet_dropout_conv')
#config = Config(dataset='Data_2', model='birdnet_dropout_conv')

##################################
if args.augment is True:
    config.DO_AUGMENT = True
    print('Data augmentation is used for conv-based nets.\n')
else:
    config.DO_AUGMENT = False
    print('No data augmentation is used for conv-based nets.\n')

##################################
if (model_name == 'recurr_lstm') or (model_name == r'conv1d_gru'):
    config.RECURR = True
    config.EPOCHS = 200
    #config.DO_AUGMENT = False

##################################
# TEST PARAMETERS
if args.test is True:
    print("Testing parameters are being used.")    
    if False: # quick testing of end-to-end system
        config.NUM_RUNS = 2
        config.NUM_FOLDS = 3
        config.BEST_ACCU_HIST = 0.001    
        config.EPOCHS = 10
    else:
        config.NUM_RUNS = 1
        config.NUM_FOLDS = 2
        config.BEST_ACCU_HIST = 0.3
        config.EPOCHS = 25


print('EPOCHS: '+str(config.EPOCHS))
print('RUNS: '+str(config.NUM_RUNS))
print('FOLDS: '+str(config.NUM_FOLDS))
print('BEST_ACCU_HIST: '+str(config.BEST_ACCU_HIST))
    
# Model of DCLDE 2018 # Build models here
try:
    # use model name in string as function name to generate the model
    func_model_generate = globals()[config.MODEL]
    model = func_model_generate(config)
except:
    print("ModelError: Model name is either incorrect or model is not accepted.")
    sys.exit()

##################################
# OUTPUT PATH
if args.test is True:
    #config.TRAIN_RESULT_PATH = r'/home/ys587/tensorflow3/__ExptResult/cv_validate_test_'+model_name
    if args.augment is True:
        config.TRAIN_RESULT_PATH = os.path.join(TRAIN_RESULT_FOLDER,'cv_TEST_'+model_name+'_train_'+dataset_name+'_augment')
    else:
        config.TRAIN_RESULT_PATH = os.path.join(TRAIN_RESULT_FOLDER,'cv_TEST_'+model_name+'_train_'+dataset_name)
elif args.augment is True:
    config.TRAIN_RESULT_PATH = os.path.join(TRAIN_RESULT_FOLDER,'cv_'+model_name+'_train_'+dataset_name+'_augment')
else:
    #config.TRAIN_RESULT_PATH = r'/home/ys587/tensorflow3/__ExptResult/cv_validate_'+model_name
    config.TRAIN_RESULT_PATH = os.path.join(TRAIN_RESULT_FOLDER,'cv_'+model_name+'_train_'+dataset_name)
    
###################################
## OUTPUT PATH
#if args.test is True:
#    config.TRAIN_RESULT_PATH = r'/home/ys587/tensorflow3/__ExptResult/cv_validate_test_'+model_name
#else:
#    config.TRAIN_RESULT_PATH = r'/home/ys587/tensorflow3/__ExptResult/cv_validate_'+model_name
#    #config.TRAIN_RESULT_PATH = r'/tmp/x-validate' #<<<=== TEST

if not os.path.exists(config.TRAIN_RESULT_PATH):
    os.makedirs(config.TRAIN_RESULT_PATH, exist_ok=True)

##################################       
# prepare training dta
label, feature = prepare_truth_data(config)
##random.seed(a = 29) # seed to randomly sample of training data

# train the classifier
#cvscores = net_train_cross_validation(feature, label, model, config)
average_precision, auc_ROC, F1_score = net_train_cross_validation(feature, label, model, config)
#net_train(feature, label, model, config)

#np.savetxt(os.path.join(config.TRAIN_RESULT_PATH, 'average_precision.txt'), average_precision, delimiter=",", fmt='%.6f')
#np.savetxt(os.path.join(config.TRAIN_RESULT_PATH, 'auc_ROC.txt'), auc_ROC, delimiter=",", fmt='%.6f')
#np.savetxt(os.path.join(config.TRAIN_RESULT_PATH, 'F1_score.txt'), F1_score, delimiter=",", fmt='%.6f')

with open(os.path.join(config.TRAIN_RESULT_PATH,'result.txt'), 'w') as f:
    with redirect_stdout(f):
        print('Average Precision Score:')
        print(average_precision)
        print('mean = '+str(average_precision.mean()))
        print('std = '+str(average_precision.std()))

        print('\nArea under the ROC curve: ')
        print(auc_ROC)
        print('mean = '+str(auc_ROC.mean()))
        print('std = '+str(auc_ROC.std()))
        
        print('\nF1 score for NARW upcalls: ')
        print(F1_score)
        print('mean = '+str(F1_score.mean()))
        print('std = '+str(F1_score.std()))

        print('\nThe run time of training the classifier is '+str(time.time() - start_time)+' Sec')

print('Average Precision Score:')
print(average_precision)
print('mean = '+str(average_precision.mean()))
print('std = '+str(average_precision.std()))

print('\nArea under the ROC curve: ')
print(auc_ROC)
print('mean = '+str(auc_ROC.mean()))
print('std = '+str(auc_ROC.std()))

print('\nF1 score for NARW upcalls: ')
print(F1_score)
print('mean = '+str(F1_score.mean()))
print('std = '+str(F1_score.std()))

print('The run time of training the classifier is '+str(time.time() - start_time)+' Sec')
