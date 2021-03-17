# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:29:40 2018

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
from upcall.train_classifier import lenet, lenet_dropout_input, \
lenet_dropout_conv, lenet_dropout_input_conv, birdnet, birdnet_dropout_input, \
birdnet_dropout_conv, birdnet_dropout_input_conv
from upcall.train_classifier import net_train, model_preprocess

from script.DCLDE2018_train_data import prepare_truth_data
import time

start_time = time.time()

#config = Config(dataset='Data_4', model='birdnet_dropout_input_conv')
config = Config(dataset='Data_1', model='lenet')
#config = Config('Data_1', 'Lenet_dropout_conv') # model of the test case 38-0.8361.hdf5

##################################
# Model of DCLDE 2018 # Build models here
##################################
try:
    # use model name in string as function name to generate the model
    func_model_generate = globals()[config.MODEL]
    model = func_model_generate(config)
except:
    print("ModelError: Model name is either incorrect or model is not accepted.")
    sys.exit()

#if config.MODEL == r'lenet':
#    model = lenet(config)
#elif config.MODEL == r'lenet_dropout_input':
#    model = lenet_dropout_input(config)
#elif config.MODEL == r'lenet_dropout_conv':
#    model = lenet_dropout_conv(config)
#elif config.MODEL == r'lenet_dropout_input_conv':
#    model = lenet_dropout_input_conv(config)
#    
#elif config.MODEL == r'birdnet':
#    model = birdnet(config)
#elif config.MODEL == r'birdnet_dropout_input':
#    model = birdnet_dropout_input(config)
#elif config.MODEL == r'birdnet_dropout_conv':
#    model = birdnet_dropout_conv(config)
#elif config.MODEL == r'birdnet_dropout_input_conv':
#    model = birdnet_dropout_input_conv(config)

model = model_preprocess(model, config)
        
##################################
# Data
##################################
label, feature = prepare_truth_data(config)
##random.seed(a = 29) # seed to randomly sample of training data
        
net_train(feature, label, model, config)

print('The run time of training the classifier is '+str(time.time() - start_time)+' Sec')

