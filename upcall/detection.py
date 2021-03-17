# -*- coding: utf-8 -*-
"""
given trained classifier, detect sound events

(1) A vanilla detector that can run=> done.
(2) Non-maximum suppresion => merging multi detection boxex

@author: ys587
"""
from __future__ import print_function

import numpy as np
import os
import glob
import re
import datetime
import soundfile as sf
import timeit

from keras.models import load_model
from upcall.sound_util import preprocess
from upcall.train_classifier import F1_Class
import matplotlib.pyplot as plt 

def get_time_stamp(TheString, TIME_FORMAT):
    m = re.search("(\d{8}_\d{6})", TheString) # YYYYMMDD_HHMMSS
    return datetime.datetime.strptime(m.groups()[0], TIME_FORMAT)

def non_max_suppress(boxes_in_batch, score_arr, config):
    """
    Keep the top-score box in a page and eliminate every box having 
    overlap over an threshold ratio, say, 0.5
    """
    overlap_size = int(config.FRAME_SIZE_SEC/config.FRAME_STEP_SEC
                    *config.OVERLAP_RATIO)
    box_sorted = boxes_in_batch[np.argsort(score_arr[boxes_in_batch])[::-1]] 
    # [::-1] reverse the sort order from ascending to descending
    # get the ordered values: score_arr[boxes_in_batch][box_sorted]
    boxes_separated = []
    for ii in range(box_sorted.shape[0]):
        if ii == 0:
            boxes_separated.append(box_sorted[ii])
        else:
            # compare with previous indices through a while loop
            # stupid for loop; need to modify to while loop to save computation
            IndTooClose = False
            for jj in boxes_separated:
                if( abs(jj-box_sorted[ii]) <= overlap_size ):
                    IndTooClose = True
            if IndTooClose == False:
                boxes_separated.append(box_sorted[ii])

    boxes_separated = np.sort(boxes_separated)
    
    return boxes_separated


def non_max_suppress_bypass(boxes_in_batch, score_arr, config):
    return boxes_in_batch


def run_detector_days(day_sound_path_list, seltab_out_path, 
                      classifier_model_path, config):
    """ run detector on multiple days of sound"""
    start_time = timeit.default_timer()
    for dd in day_sound_path_list:
        print('\n'+dd)
        seltab_detected = run_detector(dd, seltab_out_path, classifier_model_path, config)
    stop_time = timeit.default_timer()
    print('Runtime: '+str.format("{0:=.4f}", stop_time - start_time)+' Sec')
    
    return seltab_detected


def run_detector(day_sound_path, seltab_out_path, classifier_model_path, 
                 config):
    """ run detector on one day of sound"""
    if isinstance(classifier_model_path, str): # check if classifier_model_path 
    #is list or tuple
        MultiModel = False
    else:
        MultiModel = True
        
    if MultiModel == False:
        classifier_model = load_model(classifier_model_path, custom_objects={'F1_Class': F1_Class}) # Temporary for F1_Class metric
    else:
        # load multiple models into a list
        classifier_modelList = [load_model(cc, custom_objects={'F1_Class': F1_Class}) for cc in classifier_model_path]
        
    sound_list = sorted(glob.glob(os.path.join(day_sound_path+'/', '*.aif'))) # Virginia uses .aif # need to fix. #Use os.path to determine aif or wav
    if len(sound_list)==0:
        sound_list = sorted(glob.glob(os.path.join(day_sound_path+'/', '*.wav'))) # NOPP uses .wav
        if len(sound_list)==0:
            print('.wav files are not found, either. Leave.')
            return
    DayFile = os.path.join(seltab_out_path+'/',os.path.basename(day_sound_path)+'.txt')
    f = open(DayFile,'w')
    f.write('Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tScore\n')
    
    EventId = 0
    for ff in sound_list:
    ###for ff in [sound_list[0]]:
        ff2 = os.path.splitext(os.path.basename(ff))[0]
        print (ff2)                    
        # time stamp
        time_curr = get_time_stamp(ff2, config.TIME_FORMAT)
        samples0, sample_rate = sf.read(ff)
                
        if samples0.ndim==1: # make it a multi-dimensional array for single channel
            samples0 = np.reshape(samples0, (-1, 1))
        num_sample, num_chan = samples0.shape[0], samples0.shape[1]
        
        for cc in range(num_chan):
            ss_last = int(np.floor((num_sample-config.FRAME_SIZE)*1.0/config.FRAME_STEP))
            print('#',end='')
            
            # make prediction for each 15-min file or for each sliding window. The former is 3 times faster and the latter will be eliminated later.
            spectro_list = []
            for ss in range(ss_last):
                samples = samples0[ss*config.FRAME_STEP:ss*config.FRAME_STEP+config.FRAME_SIZE,cc]
                spectro = preprocess(samples, config)
                    
                spectro_list.append(spectro)
            
            fea_spectro = np.vstack(spectro_list)            
            fea_spectro = fea_spectro.reshape(fea_spectro.shape[0], config.IMG_X, config.IMG_Y, 1)

            if MultiModel == False:
                score_arr = classifier_model.predict(fea_spectro)[:,1]
            else:
                score_arr_list = [cc.predict(fea_spectro)[:,1] for cc in classifier_modelList]
                score_arr_sum = score_arr_list[0]
                for ss in range(1, len(score_arr_list)):
                    score_arr_sum += score_arr_list[ss]
                
                score_arr = score_arr_sum/float(len(score_arr_list))
            
            call_arr = np.where(score_arr > config.SCORE_THRE)[0]
            ###call_arr = np.where(score_arr > 0.0)[0]
            if call_arr.shape[0] != 0: # there's at least one window with score larger than the threshold
                # merging multi detection boxes / non-maximum suppresion
                # score & indices. 0.5 the ovelap
                call_arr_sepa = non_max_suppress(call_arr, score_arr, config)
                ###call_arr_sepa = non_max_suppress_bypass(call_arr, score_arr, config)
                print('==>> ',end='') #!!! move Non-max suppresion to the last stage using the score threshold set
                print(call_arr_sepa)

                for jj in call_arr_sepa:
                    EventId += 1
                    # Raven selection table format
                    Time1 = time_curr.hour*3600.0 + time_curr.minute*60.0 + time_curr.second + jj*config.FRAME_STEP_SEC
                    Time2 = Time1 + config.FRAME_SIZE_SEC
                    print('Found event: '+ str(EventId)+ ' Time1: '+str.format("{0:=.4f}",Time1)+' Score: '+str(score_arr[jj]))
                    f.write(str(EventId)+'\t'+'Spectrogram'+'\t'+str(cc+1)+'\t'+str.format("{0:=.4f}",Time1)+'\t'+str.format("{0:<.4f}",Time2)+'\t'+str.format("{0:=.1f}",config.BOX_OUT_F1)+'\t'+str.format("{0:=.1f}",config.BOX_OUT_F2)+'\t'+str.format("{0:<.5f}", score_arr[jj]) )
                    f.write('\n')
        print('')
    f.close()
    return True


#def ValidateDetection(SelTabTruth, SelTabDetected, ThresholdList):
#    
#    ''' from upcal_utils import detector_performance 
#    Results_TP, Results_FP, False_negative = detector_performance(results_file, 
#                            validation_file, overlap_thresh =0.4)
#    '''
#    
#    
#    return 0
    
#    
#
#if __name__ == "__main__":
#    ## Input ##    
#    # input: sound day folder: replace here with your sound path
#    day_sound_path = r'/mnt/159NAS/users/yu_shiu_ys587/__DeepContext/NARW_analyst_handbrowsed_truth_set/__Sound_BOEM_VA_Historical/BOEM_VA_Historical_20150108'
#    #day_sound_path = r'/mnt/159NAS/users/yu_shiu_ys587/__DeepContext/NARW_analyst_handbrowsed_truth_set/__Sound_BOEM_VA_Historical/BOEM_VA_Historical_20121216'
#    #day_sound_path = r'/mnt/159NAS/users/yu_shiu_ys587/__DeepContext/NARW_analyst_handbrowsed_truth_set/__Sound_BOEM_VA_Historical/BOEM_VA_Historical_20130307'
#    
#    # input: trained model of the classifier
#    classifier_model_path = r'./NARW_LeNet_V0.h5' # vanilla CNN using Kaggle training data only
#    #model_loaded = load_model('Cnn_placeholder.h5', custom_objects={'precision':precision,'recall':recall, 'Kb':Kb})
#    ModelLoaded = load_model(classifier_model_path)
#    
#    # output: the path to detected selection table
#    #seltab_out_path = os.getcwd()
#    seltab_out_path = r'/tmp'
#    
#    # TRUTH selection table: replace HERE with your selection table for validation purpose
#    SelTabTruth = r'/mnt/159NAS/users/yu_shiu_ys587/__DeepContext/NARW_analyst_handbrowsed_truth_set/VADep05_002K_M05_multi_20150108_NARW_HAND_cd.selections'
#    
#    # Score threshold
#    ScoreThre = 0.5
#    
#    # Run the detector
#    img_x, img_y = 40, 40 # For RNN, frequency, time # x: frequency; y: time
#    StartTime = timeit.default_timer()
#    SelTabDetected = RunDetector(day_sound_path, classifier_model_path, seltab_out_path, ModelLoaded, ScoreThre, img_x, img_y)
#    StopTime = timeit.default_timer()
#    print('Runtime: '+str.format("{0:=.4f}", StopTime - StartTime)+' Sec')
#    
#    # Validate detection result
#    #ThresholdList =  np.arange(0, 1.0, 0.05) # Genwerate threshold List
#    #PerfMetrics = ValidateDetection(SelTabTruth, SelTabDetected, ThresholdList)
    
