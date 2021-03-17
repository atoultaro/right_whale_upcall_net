#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 7/9/19
@author: atoultaro
"""
from __future__ import print_function
import os
import sys
import time
import glob

# ROOT_DIR = os.path.abspath("../../upcall-basic-net")
# where the path where upcall-basic-net is
# sys.path.append(ROOT_DIR)

from upcall.config import Config
from upcall.train_classifier import find_best_model
from upcall.run_detector_dsp import make_sound_stream, run_detector_1day_parallel_fft

start_time = time.time()

proj_path = u'/mnt/drive_W/projects/2018_ORStateU_CCB_85941/Dep01'
sound_dir = os.path.join(proj_path, r'AIFFs_UTCz')
# sound_dir = r'/mnt/drive_W/projects/2018_ORStateU_CCB_85941/Dep01/detection_results/__sound_test'  # for test
model_path = os.path.join(proj_path, r'detection_results/__model/lenet_dropout_conv_train_Data_6_augment_Run0_epoch_100_F1_0.8733.hdf5')
# seltab_detect_output = os.path.join(proj_path, r'detection_results/__seltab')
seltab_detect_output = os.path.join(proj_path, r'detection_results/__seltab_test_drive')

best_model_path, best_accu = find_best_model(model_path)

config = Config(dataset=None, model=u"lenet_dropout_conv")
config.SCORE_THRE = 0.5

print('Detecting:')
day_list = os.listdir(sound_dir)
day_list2 = []
for dd in day_list:
    day_list2.append(sound_dir+'/'+dd)
day_list = day_list2
del day_list2
day_list.sort()
# day_list = day_list[1:]  # avoid the first one, 20190216 has problem
# day_list = day_list[0]
day_list_err = [day_list[ii] for ii in [0]]  # debugging 20190215 & 20190612
day_list = day_list_err

expt_label = "85941_CCB27"
for ff in day_list:
    print('\nDay: ' + str(ff))
    sample_stream = make_sound_stream(ff)
    run_detector_1day_parallel_fft(sample_stream, model_path,
                                   seltab_detect_output,
                                   SelectionTableName=expt_label,
                                   config=config)

print('\nThe run time of training the classifier is ' + str(
    time.time() - start_time) + ' Sec')

