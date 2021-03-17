# -*- coding: utf-8 -*-
"""
Deep Context classifier training / detection testing
Base Configurations class.

Created on Thu Jul 19 10:01:59 2018
@author: ys587
"""
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    #NAME = None  # Overrid in sub-classes
    
    #############################
    # models available
    #############################
    MODEL_POSS = ['lenet', 'lenet_dropout_input', 'lenet_dropout_conv',
                  'lenet_dropout_input_conv', 'birdnet', 
                  'birdnet_dropout_input', 'birdnet_dropout_conv',
                  'birdnet_dropout_input_conv']

    #############################
    # datasets available
    #############################
    DATA_POSS = ['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6']
    # Data_1: Kaggle
    # Data_2: DCL
    # Data3: BOEM
    # Data_4: Kaggle + DCL
    # Data_5: BOEM + DCL
    # Data_6: Kaggle + BOEM + DCL
    
    #############################
    # full process control
    #############################
    # ACCU_ONLY: 
    # False: run detection and generate accuracy performance
    # True: when no need to run detection and generate only accuracy performance
    ACCU_ONLY = False # default: False
    # ACCU_ONLY = True # evaluate accuracy only; no detection
    
    #############################
    # Testing mode
    #############################
    TEST_MODE = False

    #############################
    # classifier training
    #############################
    # FFT parameters
    FFT_SIZE = 256
    WIN_SIZE = FFT_SIZE
    HOP_SIZE = 100 # 50 msec
    IMG_F = 40 # 40, freq, IMG_X or 35
    # IMG_F = 35
    IMG_T = 40 # time, IMG_Y
    IMG_F_START, IMG_T_START = 5, 0

    # Net
    BATCH_SIZE = 1000
    #BATCH_SIZE = 500
    EPOCHS = 100

    NUM_CLASSES = 2 # binary classiciation
    
    # optimizer decay
    #DECAY = 0.01 # used for Adam
    DECAY = 0.005 # used for Adam
        
    # dropout
    RATE_DROPOUT_INPUT = 0.2
    RATE_DROPOUT_CONV = 0.2
    RATE_DROPOUT_FC = 0.2

    # initial conditions
    TRAIN_RESULT_PATH = r'/tmp/result/Model'
    
    # optimizer learning rate
    LR = 0.001
    
    # class weight
    CLASS_WEIGHT = {0: 1.,
                1: 3.}
    #############################
    # detection testing
    #############################
    SAMPLE_RATE = 2000
    FRAME_STEP_SEC = 0.1 # sec
    FRAME_SIZE_SEC = 2.0 # each window is 2 sec long
    
    OVERLAP_RATIO = 1.0 # non-max suppression
    
    TIME_FORMAT = "%Y%m%d_%H%M%S"
    
    BOX_OUT_F1 = 50.0
    BOX_OUT_F2 = 350.0
    #############################
    # Augmentation Arguments
    #############################
    # Do augmentation when AUGMENT_DO is true; not do it when False
    DO_AUGMENT = True
    #DO_AUGMENT = False
    
    # Proportion - proportion of the dataset to augment and add
    # Prop_of_image - proportion of the image to roll along x and y axis
    # Dims - which axes to roll. Must be same length as prop_of_image
    AUGMENT_ARGS= {'proportion':0.2,
                    'prop_of_image' : [.2, .2],
                    'dims' : [0, 1]}
                    
    #############################
    # K-fold cross validation
    #############################
    #NUM_RUNS = 30 # the number of k-fold cross validation runs
    # NUM_RUNS = 1
    NUM_RUNS = 10
    NUM_FOLDS = 10 
    
    BEST_ACCU_HIST = 0.75
    #############################
    # accuracy measurement
    #############################
    # start threshold and step for precision-recall curve
    SCORE_THRE = 0.05  # 0.2
    STEP_THRE = 0.025
    SEP_THRE = 1.0 # sec
    
    #############################
    # control over 
    SOUND_DATA_SEED = 1  # randomly ssampling from dataset
    SHUFFLE_SEED1 = 4  # shuffle the training dataset for evulation/testing
    SHUFFLE_SEED2 = 101  # shuffle the training dataset for K-fold cross-validation
    
    #############################
    # Is recurrent net?
    RECURR = False
    
    #############################
    # Speed up fit_generator
    WORKERS = 4
    USE_MULTIPROCESS = True
    MAX_QUEUE_SIZE = 20
    
    #############################
    # Speed up Detector_dsp
    MAX_STREAMS = 18000 # .5 hour
    #MAX_STREAMS = 36000 # 1 hour
    #MAX_STREAMS = 864000 # 86400 sec per day / .1 (=> 0.1 sec hop size)
    CV_REUSE_DATA = True
    USE_SAVED_FEATURE = False # use saved features; will not use saved features 
    # when saved numpy file is gigantic. e.g. Maryland project has 12-channel 
    # sounds, which will result in a numpy file more than 100 GByte. It's more 
    # than the amount of RAM in the computer.

    #############################
    # Parallel
    # 0: no parallel computing is used
    # 1: parallel computing using multiple cores of CPU
    # 2: parallel computing using GPU
    PARALLEL_NUM = 1
    # Number of processes: the maximum is: 2 x number of CPUs - 1.
    NUM_CORE = 7
                
    def __init__(self, dataset='Data_2', model='Lenet'):
        self.MODEL = model        
        self.DATASET = dataset
        
        self.FRAME_STEP = int(self.FRAME_STEP_SEC*self.SAMPLE_RATE)
        self.FRAME_SIZE = int(self.FRAME_SIZE_SEC*self.SAMPLE_RATE)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    def display_model_poss(self):
        """Display supported models."""
        print('possible conv models are:\n')
        for m in self.MODEL_POSS:
            print(m+'\n')

        
