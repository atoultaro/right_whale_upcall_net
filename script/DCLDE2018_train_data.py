# -*- coding: utf-8 -*-
"""
prepare the truth data including both sound and truth label

For our work for DCLDE 2018 and the following paper, the data have 
several different formats and thus are processed in their own way
here.

Created on Mon Jul 23 13:27:11 2018

@author: ys587
"""
from __future__ import print_function
import numpy as np
from upcall.sound_util import sound2fea, sound2fea_clips
import os

##################################
DATA_DIR = r'/home/ys587/__Data'
#DATA_DIR = '/cache/kpalmer/quick_ssd/data/YS File Structure'
##################################

def prepare_truth_data(config):
    if config.DATASET == 'Data_1': # Kaggle only
        print('Extract features from Kaggle:')
        label_file = os.path.join(DATA_DIR, r'Kaggle2013/TrainList.csv')
        sound_path = os.path.join(DATA_DIR, r'Kaggle2013/train')
        
        label = np.loadtxt(label_file, delimiter=',')            
        feature = sound2fea(sound_path, label.shape[0], 
                                config)

    elif config.DATASET == 'Data_2': # DCL13
        print('Extract features from DCL:')
        # Class right whale
        FeaSpectroP1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_TP/'), 1.0, config) # 800
        FeaSpectro0 = FeaSpectroP1
        Label0 = np.ones(FeaSpectroP1.shape[0])
        del FeaSpectroP1

        FeaSpectroP2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_TP/'), 1.0, config) # 2,300
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP2))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP2.shape[0])))
        del FeaSpectroP2
        
        FeaSpectroP3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_TP/'), 1.0, config) # 1,700
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP3))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP3.shape[0])))
        del FeaSpectroP3
        
        FeaSpectroP4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_TP/'), 1.0, config) # 2,200
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP4))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP4.shape[0])))
        del FeaSpectroP4
            
        # Class Non-right whale
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Training/'), 1.0, config) # 2,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Gunshot/'), 0.5, config) #1,500
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2
    
        FeaSpectroN3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Noise/'), 0.5, config) #1,999
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN3))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN3.shape[0])))
        del FeaSpectroN3
        
        FeaSpectroN4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_FP/'), 0.5, config) #6,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN4))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN4.shape[0])))
        del FeaSpectroN4
        
        FeaSpectroN5 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_FP/'), 0.5, config) #7,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN5))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN5.shape[0])))
        del FeaSpectroN5
        
        FeaSpectroN6 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_FP/'), 0.5, config) #7,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN6))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN6.shape[0])))
        del FeaSpectroN6
        
        FeaSpectroN7 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_FP/'), 0.5, config) #9,000
        feature = np.vstack((FeaSpectro0, FeaSpectroN7))
        label = np.hstack((Label0, np.zeros(FeaSpectroN7.shape[0])))
        del FeaSpectroN7

    elif config.DATASET == 'Data_2_vanilla':  # DCL13, no negative harvesting
        print('Extract features from DCL:')
        # Class right whale
        FeaSpectroP1 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_TP/'), 1.0,
            config)  # 800
        FeaSpectro0 = FeaSpectroP1
        Label0 = np.ones(FeaSpectroP1.shape[0])
        del FeaSpectroP1

        FeaSpectroP2 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_TP/'), 1.0,
            config)  # 2,300
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP2))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP2.shape[0])))
        del FeaSpectroP2

        FeaSpectroP3 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_TP/'), 1.0,
            config)  # 1,700
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP3))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP3.shape[0])))
        del FeaSpectroP3

        FeaSpectroP4 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_TP/'), 1.0,
            config)  # 2,200
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP4))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP4.shape[0])))
        del FeaSpectroP4

        # Class Non-right whale
        FeaSpectroN1 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Training/'), 1.0,
            config)  # 2,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1

        FeaSpectroN2 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Gunshot/'), 0.5,
            config)  # 1,500
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2

        FeaSpectroN3 = sound2fea_clips(
            os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Noise/'), 0.5,
            config)  # 1,999
        feature = np.vstack((FeaSpectro0, FeaSpectroN3))
        label = np.hstack((Label0, np.zeros(FeaSpectroN3.shape[0])))
        del FeaSpectroN3

    elif config.DATASET == 'Data_3': # BOEM
        print('Extract features from BOEM:')
        FeaSpectro0 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/PositiveClips/'), 1.0, config) # 3760
        Label0 = np.ones(FeaSpectro0.shape[0])
        
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/NegativeClips/'), 1.0, config) # 3,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/BOEM_FP'), 0.75, config) # 2,300
        feature = np.vstack((FeaSpectro0, FeaSpectroN2))
        label = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2

    elif config.DATASET == 'Data_4': # Kaggle and DCL13
        print('Extract features from Kaggle:')
        label_file = os.path.join(DATA_DIR, r'Kaggle2013/TrainList.csv')
        sound_path = os.path.join(DATA_DIR, r'Kaggle2013/train')
        
        Label0 = np.loadtxt(label_file, delimiter=',')            
        FeaSpectro0 = sound2fea(sound_path, Label0.shape[0], 
                                config)                              
        
        # DCL
        print('Extract features from DCL:')
        FeaSpectroP1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_TP/'), 1.0, config) # 800
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP1))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP1.shape[0])))
        del FeaSpectroP1
    
        FeaSpectroP2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_TP/'), 1.0, config) # 2,300
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP2))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP2.shape[0])))
        del FeaSpectroP2
        
        FeaSpectroP3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_TP/'), 1.0, config) # 1,700
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP3))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP3.shape[0])))
        del FeaSpectroP3
        
        FeaSpectroP4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_TP/'), 1.0, config) # 2,200
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP4))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP4.shape[0])))
        del FeaSpectroP4    
    
        # Class Non-right whale
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Training/'), 0.5, config) # 2,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Gunshot/'), 0.5, config) #1,500
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2
    
        FeaSpectroN3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Noise/'), 0.5, config) #1,999
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN3))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN3.shape[0])))
        del FeaSpectroN3
        
        FeaSpectroN4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_FP/'), 0.5, config) #6,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN4))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN4.shape[0])))
        del FeaSpectroN4
        
        FeaSpectroN5 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_FP/'), 0.5, config) #7,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN5))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN5.shape[0])))
        del FeaSpectroN5
        
        FeaSpectroN6 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_FP/'), 0.5, config) #7,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN6))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN6.shape[0])))
        del FeaSpectroN6
        
        FeaSpectroN7 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_FP/'), 0.5, config) #9,000
        feature = np.vstack((FeaSpectro0, FeaSpectroN7))
        label = np.hstack((Label0, np.zeros(FeaSpectroN7.shape[0])))
        del FeaSpectroN7
    
    elif config.DATASET == 'Data_5': # BOEM & DCL13
        print('Extract features from BOEM:')
        FeaSpectro0 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/PositiveClips/'), 1.0, config) # 3760
        Label0 = np.ones(FeaSpectro0.shape[0])
        
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/NegativeClips/'), 1.0, config) # 3,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/BOEM_FP'), 0.75, config) 
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2
        
        # DCL
        print('Extract features from DCL:')
        FeaSpectroP1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_TP/'), 1.0, config) # 800
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP1))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP1.shape[0])))
        del FeaSpectroP1
    
        FeaSpectroP2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_TP/'), 1.0, config) # 2,300
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP2))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP2.shape[0])))
        del FeaSpectroP2
        
        FeaSpectroP3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_TP/'), 1.0, config) # 1,700
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP3))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP3.shape[0])))
        del FeaSpectroP3
        
        FeaSpectroP4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_TP/'), 1.0, config) # 2,200
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP4))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP4.shape[0])))
        del FeaSpectroP4    
    
        # Class Non-right whale
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Training/'), 0.5, config) # 2,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Gunshot/'), 0.5, config) #1,500
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2
    
        FeaSpectroN3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Noise/'), 0.5, config) #1,999
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN3))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN3.shape[0])))
        del FeaSpectroN3
        
        FeaSpectroN4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_FP/'), 0.5, config) #6,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN4))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN4.shape[0])))
        del FeaSpectroN4
        
        FeaSpectroN5 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_FP/'), 0.5, config) #7,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN5))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN5.shape[0])))
        del FeaSpectroN5
        
        FeaSpectroN6 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_FP/'), 0.5, config) #7,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN6))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN6.shape[0])))
        del FeaSpectroN6
        
        FeaSpectroN7 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_FP/'), 0.5, config) #9,000
        feature = np.vstack((FeaSpectro0, FeaSpectroN7))
        label = np.hstack((Label0, np.zeros(FeaSpectroN7.shape[0])))
        del FeaSpectroN7

    elif config.DATASET == 'Data_6': # BOEM + DCL + Kaggle
        # Kaggle
        print('Extract features from Kaggle:')
        label_file = os.path.join(DATA_DIR, r'Kaggle2013/TrainList.csv')
        sound_path = os.path.join(DATA_DIR, r'Kaggle2013/train')
        Label0 = np.loadtxt(label_file, delimiter=',')    
        FeaSpectro0 = sound2fea(sound_path, Label0.shape[0], config)
        
        # BOEM
        print('Extract features from BOEM:')
        FeaSpectroP1 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/PositiveClips/'), 1.0, config) # 800
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP1))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP1.shape[0])))
        del FeaSpectroP1
        
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/NegativeClips/'), 1.0, config) # 3,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'VA_BOEM/BOEM_FP'), 0.75, config) 
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2
        
        # DCL
        print('Extract features from DCL:')
        FeaSpectroP1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_TP/'), 1.0, config) # 767
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP1))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP1.shape[0])))
        del FeaSpectroP1
    
        FeaSpectroP2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_TP/'), 1.0, config) # 2,280
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP2))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP2.shape[0])))
        del FeaSpectroP2
        
        FeaSpectroP3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_TP/'), 1.0, config) # 1,663
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP3))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP3.shape[0])))
        del FeaSpectroP3
        
        FeaSpectroP4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_TP/'), 1.0, config) # 2,206
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroP4))
        Label0 = np.hstack((Label0, np.ones(FeaSpectroP4.shape[0])))
        del FeaSpectroP4    
    
        # Class Non-right whale
        FeaSpectroN1 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Training/'), 0.5, config) # 2,000
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN1))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN1.shape[0])))
        del FeaSpectroN1
        
        FeaSpectroN2 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Gunshot/'), 0.5, config) #1,500
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN2))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN2.shape[0])))
        del FeaSpectroN2
    
        FeaSpectroN3 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/Neg_Noise/'), 0.5, config) #1,999
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN3))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN3.shape[0])))
        del FeaSpectroN3
        
        FeaSpectroN4 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090328_FP/'), 0.5, config) #6,333
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN4))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN4.shape[0])))
        del FeaSpectroN4
        
        FeaSpectroN5 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090329_FP/'), 0.5, config) #7,270
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN5))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN5.shape[0])))
        del FeaSpectroN5
        
        FeaSpectroN6 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090330_FP/'), 0.5, config) #7,175
        FeaSpectro0 = np.vstack((FeaSpectro0, FeaSpectroN6))
        Label0 = np.hstack((Label0, np.zeros(FeaSpectroN6.shape[0])))
        del FeaSpectroN6
        
        FeaSpectroN7 = sound2fea_clips(os.path.join(DATA_DIR, r'DCL_St_Andrew/20090331_FP/'), 0.5, config) #8,685
        feature = np.vstack((FeaSpectro0, FeaSpectroN7))
        label = np.hstack((Label0, np.zeros(FeaSpectroN7.shape[0])))
        del FeaSpectroN7
    else:
        print('No dataset found...')
        
    return label, feature