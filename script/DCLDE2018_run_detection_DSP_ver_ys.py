# -*- coding: utf-8 -*-
"""
Run detetection for DCLDE 2018
Run multiple models on multiple days and then generate performance numbers and figures

Created on Wed May  2 14:47:09 2018

@author: ys587

"""
import os, sys, time

ROOT_DIR = os.path.abspath("/home/ys587/tensorflow3/upcall-basic-net") 
# the path where upcall-basic-net is
sys.path.append(ROOT_DIR)

from upcall.accuracy_measure import map_build_day_file
import glob, os
from upcall.config import Config
from upcall.full_process import full_process_interfaceDSP, full_process_interface
from upcall.RunDetectorDSPdependency import MakeSoundStream
#import matplotlib.pyplot as plt

start_time = time.time()

# parameter
config = Config()

DATA_DIR = r'/home/ys587/__Data'

# truth data:
#TruthFolder = r'/home/ys587/__Data/__TruthLabelVAPlusStAndrew'
truth_folder = os.path.join(DATA_DIR, 'DCL_St_Andrew/Sound_3_days_seltab')
day_file_map = map_build_day_file(truth_folder)

# testing sound stream of St Andrew DCL
#SoundPath = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days_short_test')
SoundPath = os.path.join(DATA_DIR, r'DCL_St_Andrew/Sound_3_days')
#SoundPath = '/cache/kpalmer/quick_ssd/data/dclmmpa2013/Testing/Upcalls_NOPPset2'
SoundFileLoc = file_days = os.listdir(SoundPath)
file_days = [os.path.join(SoundPath , s) for s in file_days]
# Make soundstream of all folders
sample_stream = MakeSoundStream(file_days)


# Test
expt_label = 'evaluate'
#classifier_file = r'weights-improvement-38-0.8361.hdf5'
#classifier_path = r'/home/ys587/__Data/DetectionTestCase/'
classifier_path ='/home/ys587/tensorflow3/__ModelOutput/__DCLDE2018/ModelSearch201805_DCL/LeNet_Dropout_Layer_p2/weights-improvement-56-0.9133.hdf5'

###seltab_detect_path = '/home/ys587/__Data/DetectionTestCase/DetectionResults'
seltab_detect_path = '/home/ys587/tensorflow3/tmp/DSP_test' #<<<===

# If the selcetion table path does not exist create it
if not os.path.exists(seltab_detect_path):
    os.makedirs(seltab_detect_path, exist_ok=True)

full_process_interfaceDSP(expt_label, 
                          sample_stream, 
                          classifier_path, 
                          seltab_detect_path, 
                          day_file_map,
                          config)   

print('The running time is '+str(time.time()-start_time)+ 'Sec.')

#classifier_path_orig ='/home/kpalmer/Desktop/dsp test folder'
#classifier_file_orig ='Trained Model For DSP Test.hdf5'

#full_process_interface(expt_label, day_list, classifier_file_orig, 
#                           classifier_path_orig, seltab_detect_path, 
#                           day_file_map, config)





