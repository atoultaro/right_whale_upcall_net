# -*- coding: utf-8 -*-
"""
Following PEP style
Created on Jul 17, 2018

@author: ys587
"""
from __future__ import print_function

import keras
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, \
                        GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, \
                        Activation, add
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Input, TimeDistributed
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import LSTM, GRU

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score, \
                            confusion_matrix, classification_report, f1_score

import numpy as np
import os, sys, glob, re
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import precision_recall_curve
from contextlib import redirect_stdout
import math
import gc

#from sound_util import augment_data
#from upcall.config import Config
#from script.DCLDE2018_data import prepare_truth_data

from keras.preprocessing.image import ImageDataGenerator

class accuracy_history(keras.callbacks.Callback):
    """ 
    Callback function ro report classifier accuracy on-th-fly
    
    Args:
        keras.callbacks.Callback: keras callback
    """
    def on_train_begin(self, logs={}):
        #self.acc = []
        self.F1_Class = []
        #self.val_acc = []
        self.val_F1_Class = []

    def on_epoch_end(self, epoch, logs={}):
        #self.acc.append(logs.get('acc'))
        #self.val_acc.append(logs.get('val_acc'))
        self.F1_Class.append(logs.get('F1_Class'))
        self.val_F1_Class.append(logs.get('val_F1_Class'))
    
#    def on_batch_end(self, batch, logs={}):
#        self.F1_Class.append(logs.get('F1_Class'))
#        self.val_F1_Class.append(logs.get('val_F1_Class'))

class NAN_callback(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        loss = logs.get('val_F1_Class')
        if math.isnan(loss):
            self.model.stop_training = True

#INTERESTING_CLASS_ID = 0  # Choose the class of interest
def F1_Class(y_true, y_pred): # keep 80 columns; return precision / recall
    """
    Reference: 
    https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """
    from keras import backend as K
    INTERESTING_CLASS_ID = 1 # right whale upcall
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_pred = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_pred, 
                                   INTERESTING_CLASS_ID), 'int32') # TP & FP
    accuracy_mask2 = K.cast(K.equal(class_id_true, 
                                    INTERESTING_CLASS_ID), 'int32') # TP & FN
    
    class_TP_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32')* \
                    accuracy_mask # TP
    class_FP_tensor = K.cast(K.equal((1-class_id_true), class_id_pred), \
                    'int32') * accuracy_mask # FP
    class_FN_tensor = K.cast(K.equal(class_id_true, (1-class_id_pred)), \
                    'int32') * accuracy_mask2 # FN
    
    Class_TP = K.sum(class_TP_tensor)
    Class_FP = K.sum(class_FP_tensor)
    Class_FN = K.sum(class_FN_tensor)
    Precision = Class_TP / (Class_TP + Class_FP)
    Recall = Class_TP / (Class_TP + Class_FN)
    F1_score = 2* Precision * Recall/(Precision + Recall)
    
    return F1_score
    

def lenet(config):
    """
    Buidling the model of LeNet
    
    Args:
        config: configuration class object
        
        input_shape: (config.IMG_F, config.IMG_T, 1)
        config.NUM_CLASSES: numbwer of classes
    Returns:
        model: built model
    """
    #input_shape = (config.IMG_F, config.IMG_T, 1)
    #num_classes = config.NUM_CLASSES

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                     input_shape=(config.IMG_T, config.IMG_F, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model

    
def lenet_dropout_input(config):
    """
    Buidling the model of LeNet with dropout on the input layer
    
    Args:
        config: configuration class object
        
        input_shape: (config.IMG_F, config.IMG_T, 1)
        config.NUM_CLASSES: numbwer of classes
        config.RATE_DROPOUT_INPUT: dropout rate
    Returns:
        model: built model
    """
    model = Sequential()
    model.add(Dropout(config.RATE_DROPOUT_INPUT, 
                      input_shape=(config.IMG_T, config.IMG_F, 1)))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model
    
def lenet_dropout_conv(config):
    """
    Buidling the model of LeNet with dropout on the convolutional layers
    and full-connected layer
    
    Args:
        config: configuration class object
        
        input_shape: (config.IMG_F, config.IMG_T, 1)
        config.NUM_CLASSES: numbwer of classes
        config.RATE_DROPOUT_CONV
        config.RATE_DROPOUT_FC
    Returns:
        model: built model
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
              input_shape=(config.IMG_T, config.IMG_F, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))    
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(config.RATE_DROPOUT_FC))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model

def lenet_dropout_input_conv(config):
    """
    Buidling the model of LeNet with dropout on the input, both 
    convolutional layers and full-connected layer
    
    Args:
        config: configuration class object
        
        input_shape: (config.IMG_F, config.IMG_T, 1)
        config.NUM_CLASSES: numbwer of classes
        config.RATE_DROPOUT_INPUT: dropout rate
        config.RATE_DROPOUT_CONV
        config.RATE_DROPOUT_FC
    Returns:
        model: built model
    """
    model = Sequential()
    model.add(Dropout(config.RATE_DROPOUT_INPUT, 
                      input_shape=(config.IMG_T, config.IMG_F, 1)))
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(config.RATE_DROPOUT_FC))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model
        
def birdnet(config):
    """
    Building the model of BirdNet convolutional neural network
        
    Args:
        config: configuration class object
    Returns:
        model: built model
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same', 
                     input_shape=(config.IMG_T, config.IMG_F, 1)))

    # Group 1
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 2
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 3
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 4
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 5
    model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    # 1x1 convolution
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal',
                     padding='same'))
    model.add(GlobalMaxPooling2D())     
    #model.add(Dense(config.NUM_CLASSES, activation='sigmoid'))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model

def birdnet_dropout_input(config):
    """
    Building the model of BirdNet convolutional neural network
        
    Args:
        config: configuration class object
    Returns:
        model: built model
    """
    model = Sequential()
    model.add(Dropout(config.RATE_DROPOUT_INPUT, 
                      input_shape=(config.IMG_T, config.IMG_F, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), 
                         activation='relu', kernel_initializer='he_normal', 
                         padding='same'))
    # Group 1
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 2
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 3
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 4
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Group 5
    model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    # 1x1 convolution
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal',
                     padding='same'))
    model.add(GlobalMaxPooling2D())     
    #model.add(Dense(config.NUM_CLASSES, activation='sigmoid'))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model
    
def birdnet_dropout_conv(config):
    """
    Building the model of BirdNet convolutional neural network
        
    Args:
        config: configuration class object
    Returns:
        model: built model
    """
    model = Sequential()    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same', 
                     input_shape=(config.IMG_T, config.IMG_F, 1)))
    # Group 1
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 2
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 3
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 4
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 5
    model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # 1x1 convolution
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), activation='relu', 
                     kernel_initializer='he_normal',padding='same'))
    model.add(GlobalMaxPooling2D())      
    #model.add(Dense(config.NUM_CLASSES, activation='sigmoid'))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model

def birdnet_dropout_input_conv(config):
    """
    Building the model of BirdNet convolutional neural network
        
    Args:
        config: configuration class object
    Returns:
        model: built model
    """
    model = Sequential()    
    model.add(Dropout(config.RATE_DROPOUT_INPUT, 
                      input_shape=(config.IMG_T, config.IMG_F, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_normal', padding='same'))
    # Group 1
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 2
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 3
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 4
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # Group 5
    model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), 
                     activation='relu', kernel_initializer='he_normal', 
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Dropout(config.RATE_DROPOUT_CONV))
    # 1x1 convolution
    model.add(Conv2D(1024, kernel_size=(1, 1), strides=(1, 1), activation='relu', 
                     kernel_initializer='he_normal',padding='same'))
    model.add(GlobalMaxPooling2D())      
    #model.add(Dense(config.NUM_CLASSES, activation='sigmoid'))
    model.add(Dense(config.NUM_CLASSES, activation='softmax'))
    
    return model

def recurr_lstm(config):
    # 3 input dimensions are: samples, time steps, and features
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(config.IMG_T, config.IMG_F) ))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=config.NUM_CLASSES, activation='softmax'))
    
    return model

def conv1d_gru(config):
    """
    """
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1, input_shape=(config.IMG_T, config.IMG_F)))
    
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2))) 
    model.add(Dropout(config.RATE_DROPOUT_CONV))

    model.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.35, return_sequences=True))
    model.add(BatchNormalization())
    #model.add(Dropout(config.RATE_DROPOUT_CONV))

    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.35, return_sequences=False))
    model.add(BatchNormalization())
    #model.add(Dropout(config.RATE_DROPOUT_CONV))

    model.add(Dense(units=config.NUM_CLASSES, activation='softmax'))

    return model
##############################################################################
# Resnet
from keras.layers.advanced_activations import PReLU

def name_builder(type, stage, block, name):
    return "{}{}{}_branch{}".format(type, stage, block, name)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    F1, F2, F3 = filters
    
    def name_fn(type, name):
        return name_builder(type, stage, block, name)
    
    x = Conv2D(F1, (1, 1), name=name_fn('res', '2a'))(input_tensor)
    x = BatchNormalization(name=name_fn('bn', '2a'))(x)
    x = PReLU()(x)
    
    x = Conv2D(F2, kernel_size, padding='same', name=name_fn('res', '2b'))(x)
    x = BatchNormalization(name=name_fn('bn', '2b'))(x)
    x = PReLU()(x)
    
    x = Conv2D(F3, (1, 1), name=name_fn('res', '2c'))(x)
    x = BatchNormalization(name=name_fn('bn', '2c'))(x)
    x = PReLU()(x)
    
    x = add([x, input_tensor])
    x = PReLU()(x)
    
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    def name_fn(type, name):
        return name_builder(type, stage, block, name)

    F1, F2, F3 = filters

    x = Conv2D(F1, (1, 1), strides=strides, name=name_fn("res", "2a"))(input_tensor)
    x = BatchNormalization(name=name_fn("bn", "2a"))(x)
    x = PReLU()(x)

    x = Conv2D(F2, kernel_size, padding='same', name=name_fn("res", "2b"))(x)
    x = BatchNormalization(name=name_fn("bn", "2b"))(x)
    x = PReLU()(x)

    x = Conv2D(F3, (1, 1), name=name_fn("res", "2c"))(x)
    x = BatchNormalization(name=name_fn("bn", "2c"))(x)

    sc = Conv2D(F3, (1, 1), strides=strides, name=name_fn("res", "1"))(input_tensor)
    sc = BatchNormalization(name=name_fn("bn", "1"))(sc)

    x = add([x, sc])
    x = PReLU()(x)

    return x
 
# resnet
def resnet(config):
    input_tensor = Input(shape=(config.IMG_T, config.IMG_F, 1))
    net = ZeroPadding2D((3, 3))(input_tensor)
    net = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(net)
    net = BatchNormalization(name="bn_conv1")(net)
    net = PReLU()(net)
    net = MaxPooling2D((3, 3), strides=(2, 2))(net)
    
    net = conv_block(net, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    net = identity_block(net, 3, [64, 64, 256], stage=2, block='b')
    net = identity_block(net, 3, [64, 64, 256], stage=2, block='c')
    
    net = conv_block(net, 3, [128, 128, 512], stage=3, block='a')
    net = identity_block(net, 3, [128, 128, 512], stage=3, block='b')
    net = identity_block(net, 3, [128, 128, 512], stage=3, block='c')
    net = identity_block(net, 3, [128, 128, 512], stage=3, block='d')
    
    net = conv_block(net, 3, [256, 256, 1024], stage=4, block='a')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='b')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='c')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='d')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='e')
    net = identity_block(net, 3, [256, 256, 1024], stage=4, block='f')
    net = AveragePooling2D((2, 2))(net)
    
    net = Flatten()(net)
    net = Dense(config.NUM_CLASSES, activation="softmax", name="softmax")(net)

    model = Model(inputs=input_tensor, outputs=net)
    
    return model
##############################################################################
# VGG-16
def two_conv_pool(x, F1, F2, name):
    x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

    return x

def three_conv_pool(x, F1, F2, F3, name):
    x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(F3, (3, 3), activation=None, padding='same', name='{}_conv3'.format(name))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

    return x

def vgg(config): # vgg16
    input_tensor = Input(shape=(config.IMG_T, config.IMG_F, 1))
    net = input_tensor

    net = two_conv_pool(net, 64, 64, "block1")
    net = two_conv_pool(net, 128, 128, "block2")
    net = three_conv_pool(net, 256, 256, 256, "block3")
    net = three_conv_pool(net, 512, 512, 512, "block4")
    net = Flatten()(net)
    net = Dense(512, activation='relu', name='fc')(net)
    net = Dense(config.NUM_CLASSES, activation='softmax', name='predictions')(net)

    model = Model(inputs=input_tensor, outputs=net)
    return model

##############################################################################
def net_train(feature_in, label_in, model, config):
    """
    Get the data and model and then run them

    Args:
    feature_in: input feature/spectrogream
    label_in: input truth labels of classes
    model: built deep network in the classifier
    config: configuration
    
    Returns:
    """
    ##################################
    # Data
    #feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_F, config.IMG_T, 1)
    feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_T, config.IMG_F, 1)
    label0 = keras.utils.to_categorical(label_in, config.NUM_CLASSES)

    X_train, X_test, Y_train, Y_test = train_test_split(feature_in, label0, 
                                                    test_size=0.2, 
                                                    random_state=42) # 42

    # Add augmentation data into training, not validation
    if config.AUGMENT_DO is True:
       X_train_aug, Y_train_aug = augment_data(X_train, Y_train, config)
       X_train = np.vstack((X_train, X_train_aug))
       Y_train = np.vstack((Y_train, Y_train_aug))       
       
    
    model_name_format = 'epoch_{epoch:02d}_F1_{val_F1_Class:.4f}.hdf5'
    check_path = os.path.join(config.TRAIN_RESULT_PATH, model_name_format)
    checkpoint = ModelCheckpoint(check_path, monitor='val_F1_Class', verbose=0,
                                 save_best_only=True, mode='max')
    
    # Fit the model   
    history = accuracy_history()
    callbacks_list = [checkpoint, history]
    
    model.fit(X_train, Y_train, batch_size=config.BATCH_SIZE, 
              epochs=config.EPOCHS, verbose=1, 
              validation_data=(X_test, Y_test), callbacks=callbacks_list, 
                class_weight=config.CLASS_WEIGHT)
                
    print('\nNumber of NARW calls in training set: '+
            str(int(Y_train[:,1].sum())))
    print('Number of NARW calls in testing set: '+
            str(int(Y_test[:,1].sum())))

    score = model.evaluate(X_test, Y_test, verbose=0)
    #print(model.metrics_names)
    print('\nTest loss:', score[0])
    print('Test F1 score:', score[1])
    
    class_prob = model.predict(X_test) # predict_proba the same as predict
    print("\nAverage Precision Score: "+
        str(average_precision_score(Y_test, class_prob)))
    print("Area under the ROC curve: "+
        str(roc_auc_score(Y_test, class_prob)))
    
    #Y_pred = ((model.predict(X_test))[:,1]>0.5).astype(int)
    Y_pred = np.argmax(class_prob, axis=1)
    confu_mat = confusion_matrix(Y_pred, Y_test[:,1].astype(int))
    
    print('\nConfuison matrix: ')
    print(confu_mat)
    print(classification_report(Y_test[:,1].astype(int), Y_pred))
    
    with open(os.path.join(config.TRAIN_RESULT_PATH,'ConfusionMatrix.txt'), 'w') as f2:
        with redirect_stdout(f2):
            print('\nTest loss:', score[0])
            print('Test F1 score:', score[1])
            print("Average Precision Score: "+
                str(average_precision_score(Y_test, class_prob)))
            print("Area under the ROC curve: "+
                str(roc_auc_score(Y_test, class_prob)))
            print('Confuison matrix: ')
            print(confu_mat)
            print(classification_report(Y_test[:,1].astype(int), Y_pred))

    Y_test_label = Y_test[:,1].astype(int)
    
    FN_ind = np.where((Y_test_label & (1-Y_pred))>0)[0]
    FP_ind = np.where(((1-Y_test_label) & Y_pred)>0)[0]
    
    # output audio clips of FP & FN
    #ind2sound(FP_ind, SoundPath, '/tmp/Model/FP')
    #ind2sound(FN_ind, SoundPath, '/tmp/Model/FN')

    # classification accuracy over epochs
    plt.figure()
    plt.plot(range(1, config.EPOCHS+1), history.val_F1)
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    #plt.savefig('/tmp/Model/AccuVsEpoch.png',dpi=300)
    plt.savefig(os.path.join(config.TRAIN_RESULT_PATH, 'AccuVsEpoch.png'),dpi=300)
    
    
    # precision recall curve
    plt.figure()
    precision, recall, threshold = precision_recall_curve(np.argmax(Y_test, 
                                    axis=1), class_prob[:,1], pos_label=1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curves ')
    #plt.savefig('/tmp/Model/Precision-Recall.png',dpi=300)
    plt.savefig(os.path.join(config.TRAIN_RESULT_PATH, 'Precision-Recall.png'),dpi=300)
    
    return FP_ind, FN_ind

def net_train_augment(feature_in, label_in, model, config):
    """
    Get the data and model and then run them
    ***Data Augmentation Experiment***

    Args:
    feature_in: input feature/spectrogream
    label_in: input truth labels of classes
    model: built deep network in the classifier
    config: configuration
    
    Returns:
        FP_ind
        FN_ind
    """
    # ImageDataGenerator
    ##################################
    # Data
    feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_T, config.IMG_F, 1)
    label0 = keras.utils.to_categorical(label_in, config.NUM_CLASSES)

    X_train, X_test, Y_train, Y_test = train_test_split(feature_in, label0, 
                                                    test_size=0.2, 
                                                    random_state=42) # 42

    datagen = ImageDataGenerator( width_shift_range=0.1, # if 0.1 ==> 40*0.1 = 4; -4, ..., 0, 4 are the possible shifts
                                    height_shift_range=0.1)

    if False:
        datagen = ImageDataGenerator( zca_whitening = True,
                                     rotation_range=20, # degree
                                     width_shift_range=0.1, # if 0.1 ==> 40*0.1 = 4; -4, ..., 0, 4 are the possible shifts
                                     height_shift_range=0.1,
                                     shear_range = 0.1,
                                     zoom_range = 0.1 )

    datagen.fit(X_train)
       
    
    model_name_format = 'epoch_{epoch:02d}_F1_{val_F1_Class:.4f}.hdf5'
    check_path = os.path.join(config.TRAIN_RESULT_PATH, model_name_format)
    checkpoint = ModelCheckpoint(check_path, monitor='val_F1_Class', verbose=0,
                                 save_best_only=True, mode='max')
    
    # Fit the model   
    history = accuracy_history()
    callbacks_list = [checkpoint, history]
    
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=config.BATCH_SIZE), 
                        epochs=config.EPOCHS, verbose=1, 
                        steps_per_epoch=int(2.0*len(Y_train)/config.BATCH_SIZE),
                        #validation_data=(X_test, Y_test), 
                        validation_data=datagen.flow(X_test, Y_test, batch_size=config.BATCH_SIZE),
                        callbacks=callbacks_list, 
                        class_weight=config.CLASS_WEIGHT)
    if False:
        model.fit(X_train, Y_train, batch_size=config.BATCH_SIZE, 
                  epochs=config.EPOCHS, verbose=1, 
                  validation_data=(X_test, Y_test), callbacks=callbacks_list, 
                    class_weight=config.CLASS_WEIGHT)
                
    print('\nNumber of NARW calls in training set: '+
            str(int(Y_train[:,1].sum())))
    print('Number of NARW calls in testing set: '+
            str(int(Y_test[:,1].sum())))

    score = model.evaluate(X_test, Y_test, verbose=0)
    #print(model.metrics_names)
    print('\nTest loss:', score[0])
    print('Test F1 score:', score[1])
    
    class_prob = model.predict(X_test) # predict_proba the same as predict
    print("\nAverage Precision Score: "+
        str(average_precision_score(Y_test, class_prob)))
    print("Area under the ROC curve: "+
        str(roc_auc_score(Y_test, class_prob)))
    
    Y_pred = ((model.predict(X_test))[:,1]>0.5).astype(int)
    confu_mat = confusion_matrix(Y_pred, Y_test[:,1].astype(int))
    
    print('\nConfuison matrix: ')
    print(confu_mat)
    print(classification_report(Y_test[:,1].astype(int), Y_pred))
    
    with open(os.path.join(config.TRAIN_RESULT_PATH,'ConfusionMatrix.txt'), 'w') as f2:
        with redirect_stdout(f2):
            print('\nTest loss:', score[0])
            print('Test F1 score:', score[1])
            print("Average Precision Score: "+
                str(average_precision_score(Y_test, class_prob)))
            print("Area under the ROC curve: "+
                str(roc_auc_score(Y_test, class_prob)))
            print('Confuison matrix: ')
            print(confu_mat)
            print(classification_report(Y_test[:,1].astype(int), Y_pred))

    Y_test_label = Y_test[:,1].astype(int)
    
    FN_ind = np.where((Y_test_label & (1-Y_pred))>0)[0]
    FP_ind = np.where(((1-Y_test_label) & Y_pred)>0)[0]
    
    # output audio clips of FP & FN
    #ind2sound(FP_ind, SoundPath, '/tmp/Model/FP')
    #ind2sound(FN_ind, SoundPath, '/tmp/Model/FN')

    # classification accuracy over epochs
    plt.figure()
    plt.plot(range(1, config.EPOCHS+1), history.val_F1)
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    #plt.savefig('/tmp/Model/AccuVsEpoch.png',dpi=300)
    plt.savefig(os.path.join(config.TRAIN_RESULT_PATH, 'AccuVsEpoch.png'),dpi=300)
    
    
    # precision recall curve
    plt.figure()
    precision, recall, threshold = precision_recall_curve(np.argmax(Y_test, 
                                    axis=1), class_prob[:,1], pos_label=1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curves ')
    #plt.savefig('/tmp/Model/Precision-Recall.png',dpi=300)
    plt.savefig(os.path.join(config.TRAIN_RESULT_PATH, 'Precision-Recall.png'),dpi=300)
    
    return FP_ind, FN_ind
    
def net_train_cross_validation(feature_in, label_in, model, config):
    #feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_F, config.IMG_T, 1)
    #feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_T, config.IMG_F, 1)
    feature_in = feature_in.reshape(feature_in.shape[0], config.IMG_T, config.IMG_F, 1)
    label0 = keras.utils.to_categorical(label_in, config.NUM_CLASSES)
    
    #seed = 42
    #seed = 2
    #seed = 41
    
    # accuracy loggers
    average_precision_K_fold = np.zeros(config.NUM_RUNS)
    auc_ROC_K_fold = np.zeros(config.NUM_RUNS)
    F1_score_K_fold = np.zeros(config.NUM_RUNS)

    run_ind = 0
    
    # Continue from the run_ind we left behind
    #run_ind = 9
    
    seed_list = list(range(run_ind, config.NUM_RUNS))
    for seed_cv in seed_list:
        
        Y_pred_K_fold = K_fold_cross_validation(feature_in, label_in, seed_cv, run_ind, config)        
        
        # accuracy of the whole K-fold cross-validation
        print('\nThe accuracy of K-fold cross validation for Run '+str(run_ind)+'\n')
        #Y_pred_K_fold0 = (Y_pred_K_fold[:,1]>0.5).astype(int)
        Y_pred_K_fold0 = np.argmax(Y_pred_K_fold, axis=1)
        np.save(os.path.join(config.TRAIN_RESULT_PATH, 'Run'+str(run_ind),'Y_pred_K_fold.npy'), Y_pred_K_fold)
        
        confu_mat = confusion_matrix(Y_pred_K_fold0, label_in)
    
        acc1 = average_precision_score(label0, Y_pred_K_fold, average=None)[1]
        average_precision_K_fold[run_ind] = acc1
        print("Average Precision Score for NARW upcalls: "+ str(acc1))
        
        acc2 = roc_auc_score(label0, Y_pred_K_fold)
        auc_ROC_K_fold[run_ind] = acc2
        print("Area under the ROC curve: "+ str(acc2))
        
        acc3 = f1_score(label0[:,1].astype(int), Y_pred_K_fold0)
        F1_score_K_fold[run_ind] = acc3
        print("F1 score for NARW upcalls: "+ str(acc3))
        
        print('Confuison matrix: ')
        print(confu_mat)
        acc4 = classification_report(label0[:,1].astype(int), Y_pred_K_fold0) 
        print(acc4)
    
        with open(os.path.join(config.TRAIN_RESULT_PATH, 'Run'+str(run_ind), 'ConfusionMatrix.txt'), 'w') as f2:
            with redirect_stdout(f2):
                print('Average Precision Score for NARW upcalls: '+ str(acc1))                
                print('Area under the ROC curve: '+ str(acc2))                
                print('F1 score for NARW upcalls: '+str(acc3))
                print('Confuison matrix: ')
                print(confu_mat)
                print(acc4)

        acc_metrics = {'AP': average_precision_K_fold, 'AUC':auc_ROC_K_fold , 'F1':F1_score_K_fold}
        df_acc = pd.DataFrame(data=acc_metrics)
        df_acc.to_csv(os.path.join(config.TRAIN_RESULT_PATH, 'acc_update.txt'), index=False)
        
        run_ind += 1
        
    #print('F1_cross_validate: '+str(F1_cross_validate))
    #return F1_cross_validate
    return average_precision_K_fold, auc_ROC_K_fold, F1_score_K_fold

def K_fold_cross_validation(feature_in, label_in, seed_cv, run_ind, config):
    from keras import backend as K
    
    if config.RECURR == True:
        feature_in = np.squeeze(feature_in) # Conv: (x, time, freq, x); recurrent: (x, time, freq)
        config.DO_AUGMENT = False
        config.LR = 0.01 # use larger learning rate for recurrent nets
    
    # stratified random shuffling of the K-fold data split
    print('Cross validation K Fold: '+str(config.NUM_FOLDS)+'.')
    rs = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=seed_cv)
    ##rs = StratifiedShuffleSplit(n_splits=config.NUM_FOLDS, test_size=.1, random_state=seed)
    folds = list(rs.split(feature_in, label_in))
        
    # the variable to collect testing results of K-fold. Results will be writtern in one fold at a time
    Y_pred_K_fold = np.zeros((label_in.shape[0],2))        
    
    model_name_format = 'epoch_{epoch:02d}_F1_{val_F1_Class:.4f}.hdf5'
    func_model_generate = globals()[config.MODEL]
    for fold_ind, (train_ind, test_ind) in enumerate(folds):
        # create neural net model        
        model = func_model_generate(config)
        
        if fold_ind == 0:
            run_path = os.path.join(config.TRAIN_RESULT_PATH,  'Run'+str(run_ind))
            os.makedirs(run_path, exist_ok=True)
            with open(os.path.join(run_path, 'net_architecture.txt'), 'w') as f:
                with redirect_stdout(f):
                    model.summary()
            
            # Save the model initial weight so that later each fold in K-fold cross validation will use this identical set of weights
            model.save_weights(os.path.join(config.TRAIN_RESULT_PATH,  'Run'+str(run_ind), 'model_weights.h5'))
        else:
            model.load_weights(os.path.join(config.TRAIN_RESULT_PATH, 'Run'+str(run_ind), 'model_weights.h5'))
        
        #model.compile(loss=keras.losses.categorical_crossentropy, 
        #    optimizer=keras.optimizers.Adam(lr=config.LR, decay=config.DECAY, amsgrad=True), metrics=[F1_Class])

        train_result_path = os.path.join(run_path, 'Fold'+str(fold_ind))
        if not os.path.exists(train_result_path):
            os.makedirs(train_result_path, exist_ok=True)
        
        check_path = os.path.join(train_result_path, model_name_format)
        checkpoint = ModelCheckpoint(check_path, monitor='val_F1_Class', verbose=0,
                                     save_best_only=True, mode='max')

        # Fit the model   
        history = accuracy_history()
        callbacks_list = [checkpoint, history]
        
        trial_count = 0
        #best_accu_hist = 0.75
        #config.BEST_ACCU_HIST # TEST
        best_accu = 0
        seed_curr = config.SHUFFLE_SEED2
        while(best_accu < config.BEST_ACCU_HIST):
            # shuffle the order of the training data and testing data, respectively
            np.random.seed(seed=seed_curr)
            np.random.shuffle(train_ind) #
            np.random.shuffle(test_ind)
            print('Fold '+str(fold_ind)+' ...')
            print('Trial '+str(trial_count))
            
            if True:
                # model
                del model
                model = func_model_generate(config)
                model.compile(loss=keras.losses.categorical_crossentropy, 
                              optimizer=keras.optimizers.Adam(lr=config.LR, 
                              decay=config.DECAY, amsgrad=True), 
                              metrics=[F1_Class])
                #model.compile(loss=keras.losses.categorical_crossentropy, 
                #             optimizer=keras.optimizers.Adam(decay=config.DECAY, 
                #                amsgrad=True), metrics=[F1_Class])
            
            
            model.load_weights(os.path.join(config.TRAIN_RESULT_PATH, 'Run'+str(run_ind), 'model_weights.h5'))
            
            X_train = feature_in[train_ind]
            Y_train = keras.utils.to_categorical(label_in[train_ind], config.NUM_CLASSES)
            X_test = feature_in[test_ind]
            Y_test = keras.utils.to_categorical(label_in[test_ind], config.NUM_CLASSES)
            
            # fit the model with or without data augmentation
            if config.DO_AUGMENT == False:
                model.fit(X_train, Y_train, batch_size=config.BATCH_SIZE, 
                          epochs=config.EPOCHS, verbose=1, 
                          validation_data=(X_test, Y_test), callbacks=callbacks_list, 
                            class_weight=config.CLASS_WEIGHT
                            )
            else:
                    datagen = ImageDataGenerator( width_shift_range=0.1, # if 0.1 ==> 40*0.1 = 4; -4, ..., 0, 4 are the possible shifts
                                    height_shift_range=0.1)
                    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=config.BATCH_SIZE), 
                                    epochs=config.EPOCHS, verbose=1, 
                                    steps_per_epoch=int(1.2*len(Y_train)/config.BATCH_SIZE),
                                    validation_data=(X_test, Y_test), 
                                    #validation_data=datagen.flow(X_test, Y_test, batch_size=config.BATCH_SIZE),
                                    callbacks=callbacks_list, 
                                    class_weight=config.CLASS_WEIGHT,
                                    workers = config.WORKERS,
                                    use_multiprocessing=config.USE_MULTIPROCESS,
                                    max_queue_size = config.MAX_QUEUE_SIZE
                                    )
            
            best_model_path, best_accu = find_best_model(train_result_path)
            
            trial_count += 1
            seed_curr += 1
            #K.clear_session()
            #del model
            #gc.collect()
        
        print('\nNumber of NARW calls in training set: '+
                str(int(Y_train[:,1].sum())))
        print('Number of NARW calls in testing set: '+
                str(int(Y_test[:,1].sum())))
    
        # load the best model of this run & fold
        #best_model_path, best_accu = find_best_model(train_result_path)
        #model = load_model(best_model_path, custom_objects=\
        #    {'F1_Class': F1_Class}) # Temporary for F1_Class metric
        
        
        score = model.evaluate(X_test, Y_test, verbose=0)
        #print(model.metrics_names)
        print('\nTest loss:', score[0])
        print('Test F1 score:', score[1])
        
        class_prob = model.predict(X_test) # predict_proba the same as predict
        Y_pred_K_fold[test_ind] = class_prob
        # The sigmoid activation is outputing values between 0 and 1 
        # independently from one another. If you want probabilities outputs 
        # that sums up to 1, use the softmax activation on your last layer, 
        # it will normalize the output to sum up to 1. 
        
        print("\nAverage Precision Score: "+
            str(average_precision_score(Y_test, class_prob)))
        print("Area under the ROC curve: "+
            str(roc_auc_score(Y_test, class_prob)))
        
        #Y_pred = (class_prob[:,1]>0.5).astype(int)
        Y_pred = np.argmax(class_prob, axis=1)
        confu_mat = confusion_matrix(Y_pred, Y_test[:,1].astype(int))
        
        print('\nConfuison matrix: ')
        print(confu_mat)
        print(classification_report(Y_test[:,1].astype(int), Y_pred))
        
        with open(os.path.join(train_result_path,'ConfusionMatrix.txt'), 'w') as f2:
            with redirect_stdout(f2):
                print('\nTest loss:', score[0])
                print('Test F1 score:', score[1])
                print("Average Precision Score: "+
                    str(average_precision_score(Y_test, class_prob)))
                print("Area under the ROC curve: "+
                    str(roc_auc_score(Y_test, class_prob)))
                print('Confuison matrix: ')
                print(confu_mat)
                print(classification_report(Y_test[:,1].astype(int), Y_pred))
    
        # Turn interactive plotting off
        plt.ioff()
        
        # classification accuracy over epochs
        fig1 = plt.figure()
        plt.plot(range(1, config.EPOCHS+1), history.val_F1_Class)
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        #plt.savefig('/tmp/Model/AccuVsEpoch.png',dpi=300)
        plt.savefig(os.path.join(train_result_path, 'AccuVsEpoch.png'),dpi=300)
        plt.close(fig1)
        
        # precision recall curve
        fig2 = plt.figure()
        precision, recall, threshold = precision_recall_curve(np.argmax(Y_test, 
                                        axis=1), class_prob[:,1], pos_label=1)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curves ')
        #plt.savefig('/tmp/Model/Precision-Recall.png',dpi=300)
        plt.savefig(os.path.join(train_result_path, 'Precision-Recall.png'),dpi=300)
        plt.close(fig2)
        #plt.show()
        
        K.clear_session()
        #del model
        gc.collect()        
        
    return Y_pred_K_fold

def model_preprocess(model, config):
    """ Compile the model and display model summary
    """
    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=keras.optimizers.Adam(), metrics=[F1_Class])
    model.summary() # print to a file

    #if not os.path.isdir(config.TRAIN_RESULT_PATH):
    #    os.makedirs(config.TRAIN_RESULT_PATH)

    # Check if the folder of traiining result exists
    if not os.path.exists(config.TRAIN_RESULT_PATH):
        os.makedirs(config.TRAIN_RESULT_PATH, exist_ok=True)
    else:
        print("\nERROR: The folder to hold trained models is not empty. Please delete it.\n")
        sys.exit()        
        
    with open(os.path.join(config.TRAIN_RESULT_PATH, 'net_architecture.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    return model
    
def find_best_model(classifier_path):
    """
    Return the path to the model with the best accuracy, given the path to 
    all the trained classifiers
    Args:
        classifier_path: path to all the trained classifiers
    Return:
        the path of the model with the best accuracy
    """
    # list all files ending with .hdf5
    day_list = sorted(glob.glob(os.path.join(classifier_path+'/','*.hdf5')))

    # re the last 4 digits for accuracy
    hdf5_filename = []
    hdf5_accu = np.zeros(len(day_list))
    for dd in range(len(day_list)):
        filename = os.path.basename(day_list[dd])
        hdf5_filename.append(filename)
        #m = re.search("_F1_(0.\d{4}).hdf5", filename)
        m = re.search("_F1_([0-1].\d{4}).hdf5", filename)
        hdf5_accu[dd] = float(m.groups()[0])
    
    # select the laregest one and write to the variable classifier_file
    if len(hdf5_accu)==0:
        best_model_path = ''
        best_accu = 0
    else:
        ind_max = np.argmax(hdf5_accu)
        best_model_path = day_list[ind_max]
        best_accu = hdf5_accu[ind_max]        
        # purge all model files except the best_model
        for ff in day_list:
            if ff != best_model_path:
                os.remove(ff)

    return best_model_path, best_accu      
    #return day_list[ind_max], hdf5_accu[ind_max] # best model file
