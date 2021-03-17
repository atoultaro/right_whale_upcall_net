# -*- coding: utf-8 -*-
"""
Temporarily reproduce the TP, TN and FP

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
Created on Tue Oct  9 13:02:30 2018

@author: ys587
"""
from __future__ import print_function
import os, sys, glob
ROOT_DIR = os.path.abspath("../../upcall-basic-net") 
# the path where upcall-basic-net is
sys.path.append(ROOT_DIR)
import time

from upcall.config import Config
from upcall.accuracy_measure import map_build_day_file, accu_days, plot_precision_recall #, calc_ap_over_runs
#from upcall.run_detector_dsp import make_sound_stream
#from upcall.full_process import full_process_interfaceDSP, full_process_interface_dsp

start_time = time.time()

def calc_accu_over_runs(model_path, day_file_map, config):
    for rr in range(config.NUM_RUNS):
    #for rr in range(2):
        print('Run '+str(rr))
        expt_label = 'Run'+str(rr)
        seltab_detect_path = os.path.join(model_path, 'Run'+str(rr))
        
        accu_result_path = seltab_detect_path +'/__TP_FN_FP'
        if not os.path.exists(accu_result_path):
            os.makedirs(accu_result_path)
        accu_tab = accu_days(day_file_map, seltab_detect_path, accu_result_path, expt_label, config)
        # save accu_tab two levels up above accu_result_path
        accu_result_path_one_above = os.path.abspath(os.path.join(seltab_detect_path,'..'))
        accu_tab.to_csv(os.path.join(accu_result_path_one_above, expt_label+'_TP_FN_FP.txt'), float_format='%.3f',index=False, sep="\t")  
        
        plot_precision_recall(accu_tab.values, expt_label, seltab_detect_path)


config = Config()
config.NUM_RUNS = 10

DATA_DIR = r'/home/ys587/__Data' # Data
truth_folder = os.path.join(DATA_DIR, 'DCL_St_Andrew/Sound_3_days_seltab')
day_file_map = map_build_day_file(truth_folder)

truth_folder2 = os.path.join(DATA_DIR, '__large_test_dataset/__truth_seltab')
day_file_map2 = map_build_day_file(truth_folder2)



#result_only_folder = r'/home/ys587/__ExptResult/__V4_Paper/__result_only_play_small'
result_only_folder = r'/home/ys587/__ExptResult/__V4_Paper/__result_only_play'

model_folder_list = sorted(glob.glob(os.path.join(result_only_folder+'/','cv*')))
for mm in model_folder_list:
    print(os.path.basename(mm))
    task1 = '__full_data'
    mm1 = os.path.join(mm, task1)
    calc_accu_over_runs(mm1, day_file_map, config)
    
    task2 = '__split_data'
    mm2 = os.path.join(mm, task2)
    calc_accu_over_runs(mm2, day_file_map, config)


model_folder_list_large = sorted(glob.glob(os.path.join(result_only_folder,'__full_data_large','__full*')))
for mm in model_folder_list_large:
    print(os.path.basename(mm))
    calc_accu_over_runs(mm, day_file_map2, config)


print('\nThe run time of training the classifier is '+str(time.time() - start_time)+' Sec')


