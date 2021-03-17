# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:52:01 2018

@author: ys587
"""
import os
from upcall.detection import run_detector_days
from upcall.accuracy_measure import accu_days, plot_precision_recall, calc_avg_prec
#from upcall.RunDetectorDSPdependency import RunDetector as RunDetctorDSP
# from upcall.run_detector_dsp import make_sound_stream, run_detector_1day, run_detector_1day_parallel_fft
from upcall.run_detector_dsp import make_sound_stream, run_detector_1day_parallel_fft

def full_process_interface(expt_label, day_list, classifier_file, 
                           classifier_path, seltab_detect_path, 
                           day_file_map, config):
    classifier_model = os.path.join(classifier_path, classifier_file)
    if not os.path.exists(seltab_detect_path):
        os.makedirs(seltab_detect_path)
    full_process(expt_label, day_list, classifier_model, seltab_detect_path, day_file_map, config)


def full_process(expt_label, 
                 day_list, 
                 classifier_model, 
                 seltab_detect_path, 
                 day_file_map, config):
    print(expt_label)
    if config.ACCU_ONLY is False: 
        run_detector_days(day_list, seltab_detect_path, classifier_model, config)
    
    accu_result_path = seltab_detect_path +'/__TP_FN_FP'
    if not os.path.exists(accu_result_path):
        os.makedirs(accu_result_path)
    accu_tab = accu_days(day_file_map, seltab_detect_path, accu_result_path, config)
    plot_precision_recall(accu_tab, expt_label, seltab_detect_path)    
    
    
def full_process_interfaceDSP(expt_label, 
                              sample_stream, 
                              model, 
                              seltab_detect_path, 
                              day_file_map,
                              config):
    '''
       expt_label - output label for the experiment
       file_days - 
       model - Either loaded keras model or path to model to run on the sample stream
       seltab_detect_path - output path for the completed selection tables
       day_file_map - dictionary of validation tables?
       config - configuration file
       
    '''
    # Check if selection table path exists, if not create it
    if not os.path.exists(seltab_detect_path):
        os.makedirs(seltab_detect_path)
        
    # Run the process to create selection tables
    full_processDSP(expt_label, 
                   sample_stream, 
                   model, 
                   seltab_detect_path, 
                   day_file_map,
                   config)
    

def full_processDSP(expt_label, 
                   sample_stream, 
                   model, 
                  seltab_detect_path, 
                  day_file_map,
                  config):
    
    print(expt_label)
    
    '''
       expt_label - output label for the experiment
       file_days - 
       model - Either loaded keras model or path to model to run on the sample stream
       seltab_detect_path - output path for the completed selection tables
       config - configuration file
       
    '''
    # Code example:                       
    # run detector using sound stream for multiple days
                      
    #sample_stream = make_sound_stream(file_days)
    run_detector(sample_stream, 
                      model,
                      seltab_detect_path,
                      SelectionTableName = expt_label,
                      config = config)
    
    accu_result_path = seltab_detect_path +'/__TP_FN_FP'
    if not os.path.exists(accu_result_path):
        os.makedirs(accu_result_path)
    accu_tab = accu_days(day_file_map, seltab_detect_path, accu_result_path, expt_label, config)
    # save accu_tab two levels up above accu_result_path
    accu_result_path_one_above = os.path.abspath(os.path.join(seltab_detect_path,'..'))
    accu_tab.to_csv(os.path.join(accu_result_path_one_above, expt_label+'_TP_FN_FP.txt'), index=False, sep="\t")  
    
    plot_precision_recall(accu_tab.values, expt_label, seltab_detect_path)
    
    
def full_process_interface_dsp(expt_label, 
                              day_list, 
                              model, 
                              seltab_detect_path, 
                              map_day_file,
                              config):
    '''
       expt_label - output label for the experiment
       file_days - 
       model - Either loaded keras model or path to model to run on the sample stream
       seltab_detect_path - output path for the completed selection tables
       day_file_map - dictionary of validation tables?
       config - configuration file
       
    '''
    # Check if selection table path exists, if not create it
    if not os.path.exists(seltab_detect_path):
        os.makedirs(seltab_detect_path)
        
    # Run the process to create selection tables
    full_process_dsp(expt_label, 
                   day_list, 
                   model, 
                   seltab_detect_path, 
                   map_day_file,
                   config)
    

def full_process_dsp(expt_label, 
                   day_list, 
                   model, 
                  seltab_detect_path, 
                  map_day_file,
                  config):
    
    print(expt_label)
    
    '''
       expt_label - output label for the experiment
       file_days - 
       model - Either loaded keras model or path to model to run on the sample stream
       seltab_detect_path - output path for the completed selection tables
       config - configuration file
       
    '''
    if config.ACCU_ONLY is False:      
        for ff in day_list:
            print('\nDay: '+str(ff))
            sample_stream = make_sound_stream(ff)
            #run_detector_1day(sample_stream, 
            #run_detector_1day(sample_stream, 
            run_detector_1day_parallel_fft(sample_stream, 
                      model,
                      seltab_detect_path,
                      SelectionTableName = expt_label,
                      config = config)
    
    accu_result_path = seltab_detect_path +'/__TP_FN_FP'
    if not os.path.exists(accu_result_path):
        os.makedirs(accu_result_path)
    accu_tab = accu_days(map_day_file, seltab_detect_path, accu_result_path, expt_label, config)
    # save accu_tab two levels up above accu_result_path
    accu_result_path_one_above = os.path.abspath(os.path.join(seltab_detect_path,'..'))
    accu_tab.to_csv(os.path.join(accu_result_path_one_above, expt_label+'_TP_FN_FP.txt'), index=False, sep="\t")  
    
    plot_precision_recall(accu_tab.values, expt_label, seltab_detect_path)

    # Calculate average precision (AP)    
    #ap_11pt, mpre, mrec = calc_avg_prec(map_day_file, seltab_detect_path, config)
    
    
    
    
    
    
    
    