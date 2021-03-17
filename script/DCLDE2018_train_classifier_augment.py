# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:21:46 2018

@author: ys587
"""
"""
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
from upcall.train_classifier import *
from upcall.train_classifier import lenet, lenet_dropout_input, \
lenet_dropout_conv, lenet_dropout_input_conv, birdnet, birdnet_dropout_input, \
birdnet_dropout_conv, birdnet_dropout_input_conv
from upcall.train_classifier import net_train, model_preprocess, net_train_augment

from script.DCLDE2018_train_data import prepare_truth_data
import time

start_time = time.time()

##################################
# select a dataset to train the classifier and a model to run.
##################################
#config = Config(dataset='Data_2', model='birdnet')
#config = Config(dataset='Data_4', model='birdnet_dropout_input_conv')
config = Config(dataset='Data_2', model='birdnet_dropout_conv')
#config = Config(dataset='Data_2', model='lenet')
#config = Config(dataset='Data_2', model='lenet_dropout_conv')
#config = Config('Data_1', 'Lenet_dropout_conv') # model of the test case 38-0.8361.hdf5


##################################
# Classifier training
##################################
# Model of DCLDE 2018 # Build models here
try:
    # use model name in string as function name to generate the model
    func_model_generate = globals()[config.MODEL]
    model = func_model_generate(config)
except:
    print("ModelError: Model name is either incorrect or model is not accepted.")
    sys.exit()

#config.TRAIN_RESULT_PATH = r'/tmp/Augment/Lenet_Model5' #<<<===
#config.TRAIN_RESULT_PATH = r'/tmp/Augment/Lenet_dropout_conv' #<<<===
#config.TRAIN_RESULT_PATH = r'/tmp/Augment/BirdNet' #<<<===
config.TRAIN_RESULT_PATH = r'/tmp/Augment/birdnet_dropout_conv' #<<<===
# check if config.TRAIN_RESULT_PATH exists
model = model_preprocess(model, config)
        
# prepare training dta
label, feature = prepare_truth_data(config)
##random.seed(a = 29) # seed to randomly sample of training data

# train the classifier
net_train_augment(feature, label, model, config)

print('The run time of training the classifier is '+str(time.time() - start_time)+' Sec')


################################
# Select the classifier of the best accuracy on validation data
################################
import glob
import re
import numpy as np

def find_best_model(classifier_path):
    # list all files ending with .hdf5
    day_list = sorted(glob.glob(os.path.join(classifier_path+'/','*.hdf5')))

    # re the last 4 digits for accuracy
    hdf5_filename = []
    hdf5_accu = np.zeros(len(day_list))
    for dd in range(len(day_list)):
        filename = os.path.basename(day_list[dd])
        hdf5_filename.append(filename)
        m = re.search("_F1_(0.\d{4}).hdf5", filename)
        hdf5_accu[dd] = float(m.groups()[0])
    
    # select the laregest one and write to the variable classifier_file
    ind_max = np.argmax(hdf5_accu)
    return day_list[ind_max] # best model file

classifier_path = config.TRAIN_RESULT_PATH
classifier_file = find_best_model(classifier_path)
print("The best trained classifier is: "+ classifier_file)


if False:
    ################################
    # Detection testing
    ################################
    from upcall.accuracy_measure import map_build_day_file
    from upcall.full_process import full_process_interface
    
    # truth data:
    DATA_DIR = r'/home/ys587/__Data'
    TruthFolder = os.path.join(DATA_DIR, r'__TruthLabelVAPlusStAndrew')
    day_file_map = map_build_day_file(TruthFolder)
    
    # testing sound stream of St Andrew DCL
    test_sound_path = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days')
    ###test_sound_path = r'/mnt/159NAS/users/yu_shiu_ys587/__DeepContext/__NARW_Data/VA_Historical/'
    day_list = sorted(glob.glob(os.path.join(test_sound_path+'/','*')))
    #day_list = day_list[:2]
    
    
    # Test
    expt_label = 'Test detection'
    # detection output path of selection tables
    seltab_detect_path = os.path.join(config.TRAIN_RESULT_PATH,'SelTabOut')
    if not os.path.exists(seltab_detect_path):
        os.makedirs(seltab_detect_path, exist_ok=True)
    full_process_interface(expt_label, day_list, classifier_file, classifier_path, seltab_detect_path, day_file_map, config)    
