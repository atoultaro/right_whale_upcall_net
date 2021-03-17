#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on 7/11/19
@author: atoultaro
"""
import os
import glob

from upcall.config import Config
from upcall.accuracy_measure import calc_TP_FP_FN, map_build_day_file, \
    get_date_str

result_path = os.path.join('/home/ys587/__ExptResult/cv_lenet_dropout_conv_train_Data_2_augment', '__full_data_large')


DATA_DIR = r'/home/ys587/__Data' # Data
truth_folder = os.path.join(DATA_DIR, '__large_test_dataset/__truth_seltab')

config = Config(dataset='Data_2', model='lenet_dropout_conv')
config.NUM_RUNS = 1
for rr in range(config.NUM_RUNS):
    day_list = sorted(glob.glob(os.path.join(result_path, 'Run'+str(rr), '__TP_FN_FP','*.txt')))  # find selection table for all days
    print (rr)
    for dd in day_list:
        basename = os.path.basename(dd)
        map_day_file = map_build_day_file(truth_folder)
        truth_file = map_day_file[get_date_str(basename)]
        accu_result, _ = calc_TP_FP_FN(dd, truth_file, result_path, config) # TP, FP, TP2, FN