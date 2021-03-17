# -*- coding: utf-8 -*-
"""
Run detetection for DCLDE 2018
Run multiple models on multiple days and then generate performance numbers and figures

Created on Wed May  2 14:47:09 2018

@author: ys587

"""
import os, sys
ROOT_DIR = os.path.abspath("/home/ys587/tensorflow3/upcall-basic-net") 
# the path where upcall-basic-net is
sys.path.append(ROOT_DIR)

from upcall.accuracy_measure import map_build_day_file
import glob, os
from upcall.config import Config
from upcall.full_process import full_process_interface
#import matplotlib.pyplot as plt

# parameter
config = Config()

DATA_DIR = r'/home/ys587/__Data'
# truth data:
TruthFolder = r'/home/ys587/__Data/__TruthLabelVAPlusStAndrew'
day_file_map = map_build_day_file(TruthFolder)

# testing sound stream of St Andrew DCL
SoundPath = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days')
day_list = sorted(glob.glob(os.path.join(SoundPath+'/','*')))

# Test
expt_label = 'Test detection'
classifier_file = r'weights-improvement-38-0.8361.hdf5'
classifier_path = r'/home/ys587/__Data/DetectionTestCase/'
###seltab_detect_path = '/home/ys587/__Data/DetectionTestCase/DetectionResults'
seltab_detect_path = '/tmp/AllZeroSamples' #<<<===
if not os.path.exists(seltab_detect_path):
    os.makedirs(seltab_detect_path, exist_ok=True)

full_process_interface(expt_label, day_list, classifier_file, classifier_path, seltab_detect_path, day_file_map, config)    



if False: # test on Virginia/BOEM
    test_sound_path = r'/mnt/159NAS/users/yu_shiu_ys587/__DeepContext/__NARW_Data/VA_Historical/'
    day_list = sorted(glob.glob(os.path.join(test_sound_path+'/','*')))
    day_list = day_list[:3]
    
    # Test
    expt_label = 'Test detection'
    classifier_file = r'weights-improvement-38-0.8361.hdf5'
    classifier_path = r'/home/ys587/__Data/DetectionTestCase/'
    ###seltab_detect_path = '/home/ys587/__Data/DetectionTestCase/DetectionResults'
    seltab_detect_path = '/tmp/AllZeroSamples' #<<<===
    if not os.path.exists(seltab_detect_path):
        os.makedirs(seltab_detect_path, exist_ok=True)
    
    full_process_interface(expt_label, day_list, classifier_file, classifier_path, seltab_detect_path, day_file_map, config)    


if True:
    # testing sound stream of St Andrew DCL
    SoundPath = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days')
    day_list = sorted(glob.glob(os.path.join(SoundPath+'/','*')))

    # Test
    expt_label = 'Test detection'
    classifier_file = r'weights-improvement-38-0.8361.hdf5'
    classifier_path = r'/home/ys587/__Data/DetectionTestCase/'
    ###seltab_detect_path = '/home/ys587/__Data/DetectionTestCase/DetectionResults'
    seltab_detect_path = '/tmp/AllZeroSamples' #<<<===
    if not os.path.exists(seltab_detect_path):
        os.makedirs(seltab_detect_path, exist_ok=True)
    
    full_process_interface(expt_label, day_list, classifier_file, classifier_path, seltab_detect_path, day_file_map, config)    
