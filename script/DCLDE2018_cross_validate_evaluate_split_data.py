# -*- coding: utf-8 -*-
"""
(1) Training classifiers using FULL dataset &
Evaluate the accuracy performance on 3-day DCL13' sound stream.
Goal: observe the effect of random initialization on the accuracy of the testing data
Computation: do this once for each run of K-fold CV

(2) Applying the classifiers from K-fold CV to evaluate the accuracy 
performance on 3-day DCL13' sound stream
Goal: observe the effect of data split on the accuracy of the testing data
Computation: do this once for each run, where 90% of data is used for training

This script is expected to be run after K-fold cross validation. 
It will train N times of classifiers using the full dataset, where N is the number of runs. 
Each time of N, it will load the weights that were used in N runs.
Created on Mon Nov 26 16:27:43 2018

@author: ys587
"""
from __future__ import print_function
import os, sys, glob
ROOT_DIR = os.path.abspath("../../upcall-basic-net") 
# the path where upcall-basic-net is
sys.path.append(ROOT_DIR)
import time
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from upcall.config import Config
from upcall.train_classifier import lenet, lenet_dropout_input, \
lenet_dropout_conv, lenet_dropout_input_conv, birdnet, birdnet_dropout_input, \
birdnet_dropout_conv, birdnet_dropout_input_conv
from upcall.train_classifier import recurr_lstm, conv1d_gru
from upcall.train_classifier import find_best_model
from script.DCLDE2018_train_data import prepare_truth_data

from upcall.accuracy_measure import map_build_day_file
from upcall.run_detector_dsp import make_sound_stream
from upcall.full_process import full_process_interfaceDSP, full_process_interface_dsp

from upcall.train_classifier import F1_Class

# release GPU memory 
from keras import backend as K
import gc
#from numba import cuda # direct control over CUDA
import argparse
import shutil

start_time = time.time()

##################################
DATA_DIR = r'/home/ys587/__Data' # Data
TRAIN_RESULT_FOLDER = r'/home/ys587/__ExptResult' # Result
##################################


class accuracy_history_train_only(keras.callbacks.Callback):
    """ 
    Callback function ro report classifier accuracy on-th-fly
    
    Args:
        keras.callbacks.Callback: keras callback
    """
    def on_train_begin(self, logs={}):
        #self.acc = []
        self.F1_Class = []

    def on_epoch_end(self, epoch, logs={}):
        #self.acc.append(logs.get('acc'))
        #self.val_acc.append(logs.get('val_acc'))
        self.F1_Class.append(logs.get('F1_Class'))

# Parse Arguments
parser = argparse.ArgumentParser(description="Experiment A: train on DCLDE 13' clips 4 days & tested on DCLDE 13' sound stream 3 days")
parser.add_argument("-m", "--model_name", type=str, default=None, help="MODEL_NAME: lenet, lenet_dropout_input, lenet_dropout_conv, \
                    lenet_dropout_input_conv, birdnet, birdnet_dropout_input, birdnet_dropout_conv, birdnet_dropout_input_conv, recurr_lstm, \
                    conv1d_gru")
parser.add_argument("-d", "--dataset_name", type=str, default=None, help="DATASET_NAME: Data_1 for Kaggle; Data_2 for DCL13'; Data_3 for BOEM Virginia; Data_4 for Kaggle & DCL13; Data_5 for BOEM & DCL13; Data_6 for BOEM, Kaggle & DCL13") 
parser.add_argument("-a", "--augment", action='store_true', help="Do data augmentation when -a is True; no augmentation otherwise")
parser.add_argument("-t","--test", action='store_true', help = 'Use testing parameters')

args = parser.parse_args()
##################################
# Only parameter to speecify
#model_name = r'lenet'
#model_name = r'birdnet'
#model_name = r'lenet_dropout_conv'
#model_name = r'birdnet_dropout_conv'
if args.model_name is None:
    model_name = "birdnet"
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
    
##################################
if args.augment is True:
    config.DO_AUGMENT = True
    print('Data augmentation is used.\n')
else:
    config.DO_AUGMENT = False
    print('No data augmentation is used.\n')

##################################
if (model_name == 'recurr_lstm') or (model_name == r'conv1d_gru'):
    config.EPOCHS = 200
    config.RECURR = True
    config.DO_AUGMENT = False
    config.LR = 0.01
    
##################################
# TEST PARAMETERS
if args.test is True:
    print("Testing parameters are being used.")
    config.TEST_MODE = True
    config.NUM_RUNS = 2
    config.NUM_FOLDS = 3
    config.BEST_ACCU_HIST = 0.001    
    config.EPOCHS = 10
    
    print('EPOCHS: '+str(config.EPOCHS))
    print('RUNS: '+str(config.NUM_RUNS))
    print('FOLDS: '+str(config.NUM_FOLDS))
    print('BEST_ACCU_HIST: '+str(config.BEST_ACCU_HIST))
    
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
    

label_in, feature_in = prepare_truth_data(config)
#feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_X, config.IMG_Y, 1)
feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_T, config.IMG_F, 1)
#label0 = keras.utils.to_categorical(label_in, config.NUM_CLASSES)

##################################
if config.RECURR == True:
    feature_in = np.squeeze(feature_in)

func_model_generate = globals()[config.MODEL]
model_name_format = 'epoch_{epoch:02d}_F1_{F1_Class:.4f}.hdf5'

# evaluate data
# truth data:
truth_folder = os.path.join(DATA_DIR, 'DCL_St_Andrew/Sound_3_days_seltab')
###truth_folder = os.path.join(DATA_DIR, '__Large_Test_Dataset/__seltab')
day_file_map = map_build_day_file(truth_folder)

if args.test is False:
    # testing sound stream of St Andrew DCL
    SoundPath = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days')
    ###SoundPath = os.path.join(DATA_DIR, r'__Large_Test_Dataset/__sound_short')
else:
    SoundPath = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days_short_test') ## TEST

day_list = os.listdir(SoundPath)
day_list2 = []
for dd in day_list:
    day_list2.append(SoundPath+'/'+dd)
day_list = day_list2
del day_list2

file_days = [os.path.join(SoundPath , s) for s in day_list]


############################################################
# (1) Training classifiers using FULL dataset
# EFFECT OF RANDOM INITIALIZATION
# create a fold in cross-validation folder to store testing/evalidation results

if False:
    result_path = os.path.join(config.TRAIN_RESULT_PATH, '__full_data')
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)
    
    #for ff in run_folders:
    for rr in range(config.NUM_RUNS):
    #for rr in range(5, config.NUM_RUNS):
    #for rr in range(3):    
        ff = result_path+'/Run'+str(rr)
        os.makedirs(ff, exist_ok=True)
        print(ff)
        
        # Model
        os.makedirs(ff, exist_ok=True)
        check_path = os.path.join(ff, model_name_format)
        checkpoint = ModelCheckpoint(check_path, monitor='F1_Class', verbose=0,
                                     save_best_only=False, mode='max', period=config.EPOCHS)
        model_path = os.path.join(config.TRAIN_RESULT_PATH, 'Run'+str(rr), 'model_weights.h5')
        history = accuracy_history_train_only()
        callbacks_list = [checkpoint, history]
                                    
        trial_count = 0
        print('Run '+str(rr)+'; trial: '+str(trial_count))
        #best_accu_hist = 0.75
        best_accu = 0
        # Every time classifier training gets trapped in local maximum, data are shuffled to re-train it, to avoid the trapping.    
        seed_curr = config.SHUFFLE_SEED1
        while(best_accu < config.BEST_ACCU_HIST):
            # data: training data. using the full dataset
            train_ind = list(range(len(label_in)))
            np.random.seed(seed=seed_curr)
            np.random.shuffle(train_ind) 
            
            X_train = feature_in[train_ind]
            Y_train = keras.utils.to_categorical(label_in[train_ind], config.NUM_CLASSES)
    
            # model
            model = func_model_generate(config)
    
            model.compile(loss=keras.losses.categorical_crossentropy, 
                          optimizer=keras.optimizers.Adam(lr=config.LR, decay=config.DECAY, amsgrad=True), 
                            metrics=[F1_Class])
            
            model.load_weights(model_path)
            # fit the model with or without data augmentation
            if config.DO_AUGMENT == False:
                model.fit(X_train, Y_train, batch_size=config.BATCH_SIZE, 
                          epochs=config.EPOCHS, verbose=1, callbacks=callbacks_list, 
                            class_weight=config.CLASS_WEIGHT
                            )
            else:
                    datagen = ImageDataGenerator( width_shift_range=0.1, # if 0.1 ==> 40*0.1 = 4; -4, ..., 0, 4 are the possible shifts
                                    height_shift_range=0.1)
                    model.fit_generator(datagen.flow(X_train, Y_train, 
                                                     batch_size=config.BATCH_SIZE, 
                                                     ),
                                                     #seed=config.augment_flow_seed), 
                                    epochs=config.EPOCHS, verbose=1, 
                                    steps_per_epoch=int(1.2*len(Y_train)/config.BATCH_SIZE),
                                    #validation_data=datagen.flow(X_test, Y_test , batch_size=config.BATCH_SIZE),
                                    callbacks=callbacks_list, 
                                    class_weight=config.CLASS_WEIGHT,
                                    workers = config.WORKERS,
                                    use_multiprocessing=config.USE_MULTIPROCESS,
                                    max_queue_size = config.MAX_QUEUE_SIZE
                                    )
            best_model_path, best_accu = find_best_model(ff)
            trial_count += 1
            seed_curr += 1
            K.clear_session()
            #del model
            gc.collect()
        
        expt_label = 'Run'+str(rr)
        # Make soundstream of all folders
        #sample_stream = make_sound_stream(file_days)
    
        # If the selcetion table path does not exist create it
    #    if not os.path.exists(seltab_detect_path):
    #        os.makedirs(seltab_detect_path, exist_ok=True)
        print('Detecting:')
        full_process_interface_dsp(expt_label, 
                                      day_list, 
                                      best_model_path, 
                                      ff, 
                                      day_file_map,
                                      config)
        
    #    full_process_interfaceDSP(expt_label, 
    #                                  sample_stream, 
    #                                  best_model_path, 
    #                                  ff, 
    #                                  day_file_map,
    #                                  config)
                                      
    K.clear_session()
    gc.collect()
    #cuda.select_device(0)
    #cuda.close()


############################################################
# (2) Applying the classifiers from K-fold CV to evaluate the accuracy 
# EFFECT OF DATA SPLIT

result_split_path = os.path.join(config.TRAIN_RESULT_PATH, '__split_data')
if os.path.exists(result_split_path):
    shutil.rmtree(result_split_path)
#if not os.path.exists(result_split_path):
os.makedirs(result_split_path, exist_ok=True)

for rr in range(config.NUM_RUNS):
    ff = result_split_path+'/Run'+str(rr)
    os.makedirs(ff, exist_ok=True)
    print(ff)

    expt_label = 'Run'+str(rr)
    # /home/ys587/tensorflow3/tmp/x-validate-test/Run0/Fold0/
    classifier_path = os.path.join(config.TRAIN_RESULT_PATH, expt_label,'Fold1') # always use Fold 0 since we're using only one fold
    
    print('classifier_path: '+classifier_path)
    best_model_path, best_accu = find_best_model(classifier_path)
    
    
    # Make soundstream of all folders
    #sample_stream = make_sound_stream(file_days)
    print('Best model is in:')
    print(best_model_path)
    
#    full_process_interfaceDSP(expt_label, 
#                                  sample_stream, 
#                                  best_model_path, 
#                                  ff, 
#                                  day_file_map,
#                                  config)  
    print('Detecting:')
    full_process_interface_dsp(expt_label, 
                                  day_list, 
                                  best_model_path, 
                                  ff, 
                                  day_file_map,
                                  config)  


print('\nThe run time of training the classifier is '+str(time.time() - start_time)+' Sec')