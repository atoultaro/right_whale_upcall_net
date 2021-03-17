#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:30:48 2018

This is a modified version of 'RunDetector' by YS. It consists of a series
of functions that can be used to run a 


@author: kpalmer
"""
import os
import sys
import re
import datetime
import librosa
from soundfile import SoundFile
import numpy as np
from keras.models import load_model
import pandas as pd
import multiprocessing as mp
import time
from random import random
from itertools import repeat

# both dsp and upcall-basic-net are installed on the same level of directories
sys.path.append(os.path.join(os.path.dirname(sys.path[-1]), 'dsp'))
from dsp.SampleStream import SampleStream, Streams
from dsp.abstractstream import StreamGap, StreamEnd

from upcall.config import Config
from upcall.filter_list import bird_net_light_filter,bird_net_filter, \
     med2d_with_cropping, null_filter, normalize_median_filter, ys_Preprocess,\
     nearest_to
from upcall.train_classifier import F1_Class
from upcall.sound_util import preprocess
#disable GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 


# Helper function for subsetting keys 
def extract(full_dictionary, key_list):
    ''''return a subset of a dictionary of a larger dictionary using a keylist
    input- 
        full_dictionary - dictionary from which to subset 
        key_list - list of keys to extract
    returns -
        new_dict - subset of full_dictionary with only the keylist keys
    '''
    subdictionary = dict()
    for k in key_list:
        subdictionary[k] = full_dictionary[k]
        
    return subdictionary
    

def get_start_timestamp(f, format_str =  "%Y%m%d_%H%M%S"):
    ''' returns a datetime object (start time) given a soundfile/directory
    name in the standard Cornell format
    
    input: 
        f- filename from which to extract the timestamp
        format_str - format string of the timestamp default "%Y%m%d_%H%M%S"
            Cornell format
    '''
    fname = os.path.split(f)[1]
    match_date = re.search(r'\d{8}_\d{6}', fname)
    try:
        # Cornell Format
        start_time = datetime.datetime.strptime(match_date.group(), format_str)
    except AttributeError:
        #  not cornell format try scripps
        match_date = re.search(r'\d{6}_\d{6}', fname)

    try:
        start_time = datetime.datetime.strptime(match_date.group(), format_str)
    except AttributeError:
        # Also not scripts, try NOAA
        start_time = datetime.datetime.strptime(fname[-16:-4], format_str)

    return start_time
    
                
def non_max_suppress_orig(boxes_in_batch, score_arr, config):
    """
    Keep the top-score box in a page and eliminate every box having 
    overlap over an threshold ratio, say, 0.5
    
    Args:
        boxes_in_batch: indices of classification scores from sliding a window
        score_arr: the corrsponding score to boxes_in_batch
        config: parameter configuration
    
    Return:
        boxes_separated: distilled indices after applying non-max suppression
    """
    overlap_size = int(config.FRAME_SIZE_SEC/config.FRAME_STEP_SEC
                    *config.OVERLAP_RATIO)
    # boxes sorted by scores
    box_sorted_by_score = boxes_in_batch[np.argsort(score_arr[boxes_in_batch])[::-1]] 

    boxes_separated = separate_boxes_fasterer(box_sorted_by_score, overlap_size)

    return boxes_separated


def separate_boxes(box_sorted_by_score, overlap_size):
    boxes_separated = []
    for ii in range(box_sorted_by_score.shape[0]):
        if ii == 0:
            boxes_separated.append(box_sorted_by_score[ii])
        else:
            # compare with previous indices through a while loop
            # stupid for loop; need to modify to while loop to save computation
            ind_close = False
            for jj in boxes_separated:
                if( abs(jj-box_sorted_by_score[ii]) <= overlap_size ):
                    ind_close = True
            if ind_close == False:
                boxes_separated.append(box_sorted_by_score[ii])
    boxes_separated = np.sort(boxes_separated)

    return boxes_separated


def separate_boxes_faster(box_sorted_by_score, overlap_size):
    # use a while loop to replace the inside for loop.
    boxes_separated = []
    for ii in range(box_sorted_by_score.shape[0]):
        if ii == 0:
            boxes_separated.append(box_sorted_by_score[ii])
        else:
            # compare with previous indices through a while loop
            # stupid for loop; need to modify to while loop to save computation
            jj = 0  
            ind_close = False
            while(not ind_close and jj < len(boxes_separated)):
                if( abs(boxes_separated[jj]-box_sorted_by_score[ii]) <= overlap_size ):
                    ind_close = True
                jj += 1
            if ind_close == False:
                boxes_separated.append(box_sorted_by_score[ii])
    boxes_separated = np.sort(boxes_separated)
    return boxes_separated
    
def separate_boxes_fasterer(box_sorted_by_score, overlap_size):
    boxes_separated = []
    for ii in range(box_sorted_by_score.shape[0]):
        if ii == 0:
            boxes_separated.append(box_sorted_by_score[ii])
        else:
            # compare with previous indices through a while loop
            # stupid for loop; need to modify to while loop to save computation
            #ind_close = False
            IndCLoseArr = np.array((abs(boxes_separated - box_sorted_by_score[ii]) <= overlap_size))
            #for jj in boxes_separated:
            #    if( abs(jj-box_sorted_by_score[ii]) <= overlap_size ):
            #        ind_close = True
            if IndCLoseArr.sum() == 0:
                boxes_separated.append(box_sorted_by_score[ii])
    boxes_separated = np.sort(boxes_separated)

    return boxes_separated
    

def separate_boxes_time(boxes_sorted_by_time, score_arr, overlap_size):
    # do nothing yet. No idea how to do it in our case.
    # We use scores to groups time instants into a call.
    # it's good to apply a classifier that generates selective scores but it
    # could be a problem for a loose classifier.
    boxes_separated = []
    for ii in boxes_sorted_by_time:
        print()
        
    return boxes_separated
    
# =============================================================================
#  Load soundfiles into stream/streams
# =============================================================================


def make_sound_stream(day_sound_path, format_str = "%Y%m%d_%H%M%S"):
    ''' 
    Function for making a soundstream capable of iterating through lots of
    files.

    Input:
        day_sound_path - List of folder location(s) containing soundfiles
        format_str - format string of the date default is"%Y%m%d_%H%M%S" 
            for Cornell
    Returns
        Returns a soundstrem of all wav or aif files listed in the folder 
    '''
    
    # Declare the stream
    stream_elements = Streams()
    
    # if it's not a list make it a list
    if isinstance(day_sound_path, (list,)) is False:
        day_sound_path = [day_sound_path]
    
    for ii in range(len(day_sound_path)):
        # get the file director
        file_dir = day_sound_path[ii]

        # Iterate through the folders and extract associated
        for filename in os.listdir(file_dir):
            # if soundfile add it to the stream
            if filename.endswith(".wav") or filename.endswith(".aif") or \
                    filename.endswith("flac"):
                sound_fullfile = file_dir + '/' + filename
                start = get_start_timestamp(filename, format_str)
                aa = SoundFile(sound_fullfile)
                stream_elements.add_file(sound_fullfile, [len(aa)], 
                                         [start], aa.samplerate)
            else:
                continue
                
    # Combine streams into a sample stream
    stream = SampleStream(stream_elements)
        
    return stream


def make_predictions(model, 
                     spectro_list,
                     n_channels,
                     config,
                     streamevent = False
                     ):
    '''
    Make model predictions
    
    Inputs:
        model - trained and properly loaded keras model
        spectro_list - list of spectrogram elements
        n_channels - int number of channels in the data
        streamevent - bool whether the function was triggered by a stream event 
        
    Returns:
        predictions - array of predictions length of spectrolist by n_channels
    
    '''
    if streamevent is True:
        # Finish reading/prediciting the remaining features
        if len(spectro_list)>1:
            FeaSpectro = np.expand_dims(np.vstack(spectro_list[0:len(spectro_list)]),3)
            FeaSpectro = FeaSpectro.reshape(FeaSpectro.shape[0], config.IMG_T, config.IMG_F, 1)
            if config.RECURR is True:
                FeaSpectro = np.squeeze(FeaSpectro)
            predictions = model.predict(FeaSpectro)[:,1]
        else:
            predictions =[]
    
    else: 
        FeaSpectro = np.expand_dims(np.vstack(spectro_list[0:len(spectro_list)]),3)
        FeaSpectro = FeaSpectro.reshape(FeaSpectro.shape[0], config.IMG_T, config.IMG_F, 1)
        if config.RECURR is True:
            FeaSpectro = np.squeeze(FeaSpectro)
        predictions = model.predict(FeaSpectro)[:,1]
    
    return predictions


def make_raven_sndsel_table(
                            ScoreArr, 
                            OverlapRatioThre,
                            file_array,
                            timestamp_array,
                            ScoreThre,
                            config,
                            date_format = "%Y%m%d_%H%M%S"):
    """
    Function for creating a sound selection table
    inputs:
        ScoreArr - numpy array of scores
        file_array - array of file names
        OverlapRatioThres- float, between 0 and 1 throw out adjacent events
        closer than t
        timestamp_array
        ScoreThre
        date_format - datestring format of the aiff file
        config - optional configuration file
    """
    raven_table_out = pd.DataFrame()
    EventId = 0
    n_channels = ScoreArr.shape[1]

    # test channels
    for jj in range(n_channels):
        if ScoreArr[:, jj].tolist().count(ScoreArr[0, jj]) != len(ScoreArr):
            print('chan ' + str(jj) + ' ok')
        else:
            print( print('chan ' + str(jj) + ' bad'))
    
    for ii in range(n_channels):
        # Set the event number
        EventId = len(raven_table_out)+1
        
        # get the indicies of the events that are above the detection
        # threshold
        boxes_in_batch = np.where(ScoreArr[:, ii] > ScoreThre)

        # iterate through the channels and append to the dataframe that
        # will become the raven selection/ sound selection table
        
        # Also make sure that all predictions in the channel are not the same
        if (len(boxes_in_batch[0]) > 0) & \
            (ScoreArr[:,ii].tolist().count(ScoreArr[0,ii]) != len(ScoreArr)):

            # Return non- overlapping indices of the detections 
            CallArrSepa1 =  non_max_suppress_orig(boxes_in_batch[0],
                                                  ScoreArr[:,ii], config)
            
            # Pare down the file list to only files from non-max
            out_files1 = np.array(file_array)[CallArrSepa1].tolist()
            
            # Pare down the timestamp list to only timestamps from non-max
            begin_time_s1 = np.array(timestamp_array)[CallArrSepa1].tolist()
            
            # Set the event ID's
            EventIDs = EventId + np.arange(len(CallArrSepa1))
            
            # Calculate the begin time in s into day and file
            # for Sound Selection table
            begin_time_s1 = pd.to_datetime(begin_time_s1)
            #begin_time_s1 = pd.to_datetime(begin_time_s1) - \
            #        pd.Timedelta(seconds = 2.0) 
            #pd.Timedelta(seconds = 2) # YuShiu: why do this???
                
            file_offset_sec =[]
            file_start_name = []

#            for jj in range(len(out_files1)-1)
            for jj in range(len(out_files1)):
                file_start = get_start_timestamp(out_files1[jj],
                                             date_format)
                file_offset = begin_time_s1[jj] - file_start
                file_offset_sec.append(file_offset.total_seconds())
                file_start_name.append(os.path.basename(out_files1[jj]))

            # Start in seconds from which to draw start times
            sec_start = begin_time_s1.hour*60.*60+  \
                    begin_time_s1.minute*60.+  \
                    begin_time_s1.second + \
                    begin_time_s1.microsecond/(10.**6)
            
            # Data output for Raven Sound Selection table
            data = {'Selection': EventIDs,
                    'View':'Spectrogram 1',
                    'Channel': ii+1,
                    'Begin Time (s)': sec_start,
                    'End Time (s)':sec_start+2,
                    'Low Freq (Hz)':config.BOX_OUT_F1,
                    'High Freq (Hz)':config.BOX_OUT_F2,
                    'Begin Date':begin_time_s1.strftime("%Y/%m/%d"),
                    'Begin Hour':begin_time_s1.hour,
                    'Begin Path':out_files1,
                    'Begin File': file_start_name,   
                    'File Offset (s)':  file_offset_sec,  
                    'Score':ScoreArr[CallArrSepa1, ii],
                    'timestamp':begin_time_s1,
                    'Detection threshold': ScoreThre
                    }
            
            frames = [raven_table_out, pd.DataFrame(data)]
            raven_table_out = pd.concat(frames)
            EventId = max(EventIDs)

        # Change order to match what raven expects
        if len(raven_table_out)>0:
            raven_table_out = raven_table_out[['Selection',
                     'View',
                     'Channel',
                     'Begin Time (s)',
                     'End Time (s)',
                     'Low Freq (Hz)',
                     'High Freq (Hz)',
                     'Begin Date',
                     'Begin Hour', 
                     'Begin Path',
                     'Begin File',
                     'File Offset (s)',
                     'Score',
                     'timestamp',
                     'Detection threshold']]
    
    # Return the Raven table as a dataframe
    return raven_table_out


def write_raven_table_to_CSV(PandasDataFrame, 
                             seltab_path_detected,
                             SelectionTableName, 
                             SNDSel= False):
    '''
    Write the pandas dataframe to a txt file
    
    Inputs:
        PandasDataFrame - pandas dataframe with approperiate columns
        seltab_path_detected - output path
        SelectionTableName - Name of the output file
        SNDSel - boolean, whether to write the file as a Raven Sound selection
            table (default True) or raven selection table. Raven selection table
            will produce a table every time the stream hits a new day, change in
            sample rate or different number of channels
        
    Returns:
        Nothing- writes dataframe to specified location
    
    '''
    if len(PandasDataFrame)>0:
        # if not sound selection table then sound table, remove the sound selection
        # coulmns
        ssnd_sel_name = 'sndsel'
        
        if not SNDSel:
            PandasDataFrame = PandasDataFrame.drop(['Begin File', 
                                                   'File Offset (s)',
                                                   'Begin Path'], axis=1)
            ssnd_sel_name = 'seltab'
            
        
        # sort
        PandasDataFrame['Selection'] = np.arange(1, len(PandasDataFrame)+1)
        
        # make the file loc
        DayFile = os.path.join(seltab_path_detected,
                               SelectionTableName+ '_'+ssnd_sel_name + '_'+\
                               str(PandasDataFrame['timestamp'].iloc[0].strftime('%Y%m%d')) +'.txt')
        
        # export
        PandasDataFrame.to_csv(DayFile,
                              sep='\t', mode='a',
                              index = False)
    else:
        pass
            

# def run_detector(stream,
#                 model,
#                 seltab_path_detected,
#                 filter_args=None,
#                 N_read = 1000,
#                 N_previous= 1000,
#                 date_format = "%Y%m%d_%H%M%S",
#                 ScoreThre = 0.05,
#                 max_streams = 40000, # make it big to utilize GPU
#                 OverlapRatioThre = 0.4,
#                 SelectionTableName = 'temp',
#                 SNDSel= False,
#                 config = None):
#
#
#     """
#     Runs the model over the datastream and first produces an array of
#     score values for all times in the datset/Day of data. Then removes all
#     scores below a given threhsold and runs non-max suppression on the
#     remaining prediction values. Finally puts all predictions into
#     produces raven table(s) for
#     the detector. Read in N_read and N_previous samples from the sample stream
#     (e.g. context!). Step through the loaded samples making predictions from
#     the trained model. Run non-max supression to remove duplicate detections.
#
#     inputs -
#         stream - SampleStream of files (see DSP sampelstream of
#             MakeSampleStream)
#         model -  Either traied and loadede keras model or full path to
#             trained model
#
#         #####################################
#         # Filter args Dictionary
#         #####################################
#
#         FFTSize - FFTSize in samples
#         HopSize - HopSize in samples
#         filter_name - string, preprocessing filter name from filterlist.py
#         f_range - frequency range (Hz) to maintain (e.g. cropping limits)
#
#         #####################################
#         # Advance dictionary data. Could also be dictionary
#         #####################################
#
#         seltab_path_detected - output directory for the raven selection table
#         N_read, N_previous - number of sampels (combined) needed for the
#             prediction. For upcall basic net this is 2k
#         date_format - date string format of the sound files
#             (e.g. "%Y%m%d_%H%M%S")
#         ScoreThre- threshold above which the detector keeps the events
#             default 0.05
#         max_streams - number of stream advances to load into memory
#                         prior to making a prediction default 400
#         SelectionTableName - string for sound selection / sound table name
#         SNDSel - bool to indicate whether or not to make a sound table default
#                 true
#
#         config- optional configuration class containing input parameters
#
#         Conditions for wariting out the selection table
#         1) End of the stream
#         2) The stream has come across a new day AND sndsel is Fales
#         3) A file in the stream has different parameters (e.g. channels)
#             this is not fully populated yet. Currently everything runs on 2khz
#             so we are just going with that for the time being
#
#
#         Creates a Raven readable selection table of detections that include
#         prediction score
#     """
#     #########################################################################
#     # Check if there is a configureation file, if so load/override relevant
#     # filter parameters from the config file
#
#     if config is not None:
#         # preprocess arguments
#         filter_args = {'FFTSize': config.FFT_SIZE,
#                        'HopSize': config.HOP_SIZE,
#                        'fs': config.SAMPLE_RATE,
#                        'filter_fx' : 'ys_Preprocess'}
#         fs = filter_args['fs']
#         # Calculate number of samples to advance the streamer and the number
#         # of 'previous' samples to retain. Note this is not the same as
#         # stft and advance
#         FrameStepSec = config.FRAME_STEP_SEC
#         FrameStep = int(FrameStepSec*fs)
#         FrameSizeSec = config.FRAME_SIZE_SEC # each window is 2 sec long
#         FrameSize = int(FrameSizeSec*fs)
#
#         N_read = FrameStep
#         N_previous = FrameSize -N_read
#
#         date_format = config.TIME_FORMAT
#         OverlapRatioThre = config.OVERLAP_RATIO
#
#         # output frequencies of the upcall detector
#         low_f = config.BOX_OUT_F1,
#         high_f = config.BOX_OUT_F2
#
#         max_streams = config.MAX_STREAMS
#
#     if type(model) == str:
#         try:
#          model = load_model(model)
#         except ValueError:
#              try:
#                  model = load_model(model, custom_objects={'F1_Class': F1_Class})
#              except:
#                  print('Falure loading model')
#
#     #######################################################################
#     # chcek that low and high frequency are in the filter args for the
#     # output bounding box on the raven selection table
#     try:
#         filter_args['low_f']
#     except KeyError:
#         low_f = 50
#     try:
#         filter_args['high_f']
#     except KeyError:
#         high_f = 350
#
#     ########################################################################
#     # Initialize a table to export
#     raven_table_out = pd.DataFrame()
#
#     # Create empty timestamp array to get timestamp of detections
#     timestamp_array =[]
#
#     # Create empty ScoreArr
#     ScoreArr = np.array([])
#     counter = 0
#     spectro_list = []
#     file_array = []
#     previous_channels = stream.stream.channels
#     current_channels = stream.stream.channels
#
#     # Stream through the files, preprocess and apply the detector
#     previous_date = stream.get_current_timesamp().date()
#     current_date = stream.get_current_timesamp().date()
#
#     while True:
#         # Append the timestamp
#         timestamp = stream.get_current_timesamp()
#
#         # Set the current file
#         current_file = stream.stream.filename
#
#         try:
#             # load the samples
#             samps = stream.read(N = N_read, previousN=N_previous)[0]
#         except (StreamGap, StreamEnd) as S:
#             # If error was thrown after a read (as opposed to after writing
#             # a new table, then read the samples and predict)
#             if PredictAgain:
#                 ScoreArr = np.append(score_arr,
#                                     make_predictions(model,
#                                                      spectro_list,
#                                                      previous_channels,
#                                                      config,
#                                                      streamevent = True
#                                                      ))
#                 score_arr = score_arr.reshape(-1, previous_channels)
#                 score_arr = score_arr[:-1].reshape(-1, previous_channels)
#
#                 # Make the raven table
#                 raven_data_frame = make_raven_sndsel_table(
#                         score_arr,
#                         OverlapRatioThre,
#                         file_array,
#                         timestamp_array,
#                         ScoreThre,
#                         config,
#                         date_format = date_format)
#
#             # End of Stream write the dataframe
#             if type(S).__name__ == 'StreamEnd' :
#                 raven_table_out = raven_table_out.append(raven_data_frame)
#
#                 #print('writing selection table ' + current_file)
#                 if PredictAgain:
#                     write_raven_table_to_CSV(raven_table_out,
#                                              seltab_path_detected,
#                                              SelectionTableName,
#                                              SNDSel= SNDSel)
#                     PredictAgain = False
#                     continue
#                 return False
#
#             elif type(S).__name__ == 'StreamGap':
#                 # reset the list
#                 spectro_list = []
#                 file_array = []
#                 timestamp_array =[]
#                 counter = 0
#                 score_arr = []
#                 #print('I passed a stream gap!!')
#                 continue
#
#         #advance the counter
#         counter +=1
#         # number of channels in the new file
#         current_channels = stream.stream.channels
#
#         # current date
#         current_date = stream.get_current_timesamp().date()
#
#         # Append the timestamp
#         timestamp_array.append(timestamp)
#
#         # Set the current file
#         file_array.append(current_file)
#
#         #pre-process all the channels
#         for ii in range(samps.shape[1]):
#             #Spectro = preprocess(samps[:,ii], filter_args)
#             Spectro = preprocess(samps[:,ii], config)
#             spectro_list.append([Spectro])
#
#         if (current_channels != previous_channels) or \
#             (SNDSel is False and current_date != previous_date):
#
#             if PredictAgain:
#                 preds = make_predictions(model,
#                                          spectro_list,
#                                          previous_channels,
#                                          config,
#                                          streamevent = False
#                                          )
#                 score_arr = np.append(score_arr, preds)
#                 score_arr = score_arr.reshape(-1, previous_channels)
#
#                 raven_data_frame = make_raven_sndsel_table(
#                                         score_arr,
#                                         OverlapRatioThre,
#                                         file_array,
#                                         timestamp_array,
#                                         ScoreThre,
#                                         config,
#                                         date_format = date_format)
#
#                 raven_table_out = raven_table_out.append(raven_data_frame)
#
#                 #SelectionTableName_out = SelectionTableName + \
#                 #                    str(previous_channels) + \
#                 #                    '_'+ str(ScoreThre)
#
#
#                 # Export the raven table
#                 #print('writing selection table ' + current_file)
#                 write_raven_table_to_CSV(raven_table_out,
#                                          seltab_path_detected,
#                                          SelectionTableName,
#                                          SNDSel= SNDSel)
#
#                 # Reset everything
#                 raven_table_out  = pd.DataFrame()
#                 score_arr = []
#                 spectro_list = []
#                 timestamp_array =[]
#                 file_array = []
#                 previous_channels = current_channels
#                 previous_date = current_date
#                 counter = 0
#
#                 # tell the reader not to write a dataframe on next streamgap
#                 PredictAgain = False
#         else:
#             # If iterated the maximum number of times, make the predicitons and clear the list
#             if counter == max_streams:
#
#                 preds = make_predictions(model,
#                                          spectro_list,
#                                          current_channels,
#                                          config,
#                                          streamevent = False
#                                          )
#                 # make model predictions
#                 score_arr = np.append(score_arr, preds)
#                 #if bool(len(score_arr) % current_channels):
#                     #print('blarg')
#
#                 # reset the list
#                 spectro_list = []
#                 counter = 0
#
#             previous_channels = current_channels
#             previous_date = current_date
#             PredictAgain = True
#

# def run_detector_1day(stream,
#                 model,
#                 seltab_path_detected,
#                 filter_args=None,
#                 N_read = 1000,
#                 N_previous= 1000,
#                 date_format = "%Y%m%d_%H%M%S",
#                 ScoreThre = 0.05,
#                 max_streams = 40000, # make it big to utilize GPU
#                 OverlapRatioThre = 0.4,
#                 SelectionTableName = 'temp',
#                 SNDSel= False,
#                 config = None):
#     """ Runs the model over the datastream and first produces an array of
#     score values for all times in the datset/Day of data. Then removes all
#     scores below a given threhsold and runs non-max suppression on the
#     remaining prediction values. Finally puts all predictions into
#     produces raven table(s) for
#     the detector. Read in N_read and N_previous samples from the sample stream
#     (e.g. context!). Step through the loaded samples making predictions from
#     the trained model. Run non-max supression to remove duplicate detections.
#
#     inputs -
#         stream - SampleStream of files (see DSP sampelstream of
#             MakeSampleStream)
#         model -  Either traied and loadede keras model or full path to
#             trained model
#
#         #####################################
#         # Filter args Dictionary
#         #####################################
#
#         FFTSize - FFTSize in samples
#         HopSize - HopSize in samples
#         filter_name - string, preprocessing filter name from filterlist.py
#         f_range - frequency range (Hz) to maintain (e.g. cropping limits)
#
#         #####################################
#         # Advance dictionary data. Could also be dictionary
#         #####################################
#
#         seltab_path_detected - output directory for the raven selection table
#         N_read, N_previous - number of sampels (combined) needed for the
#             prediction. For upcall basic net this is 2k
#         date_format - date string format of the sound files
#             (e.g. "%Y%m%d_%H%M%S")
#         ScoreThre- threshold above which the detector keeps the events
#             default 0.05
#         max_streams - number of stream advances to load into memory
#                         prior to making a prediction default 400
#         SelectionTableName - string for sound selection / sound table name
#         SNDSel - bool to indicate whether or not to make a sound table default
#                 true
#
#         config- optional configuration class containing input parameters
#
#         Conditions for wariting out the selection table
#         1) End of the stream
#         2) The stream has come across a new day AND sndsel is Fales
#         3) A file in the stream has different parameters (e.g. channels)
#             this is not fully populated yet. Currently everything runs on 2khz
#             so we are just going with that for the time being
#
#
#         Creates a Raven readable selection table of detections that include
#         prediction score
#     """
#     #########################################################################
#     # Check if there is a configureation file, if so load/override relevant
#     # filter parameters from the config file
#
#     if config is not None:
#         # preprocess arguments
#         filter_args = {'FFTSize': config.FFT_SIZE,
#                        'HopSize': config.HOP_SIZE,
#                        'fs': config.SAMPLE_RATE,
#                        'filter_fx' : 'ys_Preprocess'}
#         fs = filter_args['fs']
#         # Calculate number of samples to advance the streamer and the number
#         # of 'previous' samples to retain. Note this is not the same as
#         # stft and advance
#         FrameStepSec = config.FRAME_STEP_SEC
#         FrameStep = int(FrameStepSec*fs)
#         FrameSizeSec = config.FRAME_SIZE_SEC # each window is 2 sec long
#         FrameSize = int(FrameSizeSec*fs)
#
#         N_read = FrameStep
#         N_previous = FrameSize -N_read
#
#         date_format = config.TIME_FORMAT
#         OverlapRatioThre = config.OVERLAP_RATIO
#
#         # output frequencies of the upcall detector
#         low_f = config.BOX_OUT_F1,
#         high_f = config.BOX_OUT_F2
#
#         max_streams = config.MAX_STREAMS
#
#     if type(model) == str:
#         try:
#          model = load_model(model)
#         except ValueError:
#              try:
#                  model = load_model(model, custom_objects={'F1_Class': F1_Class})
#              except:
#                  print('Falure loading model') # Need fix
#
#     ########################################################################
#     # Create empty timestamp array to get timestamp of detections
#     timestamp_array =[]
#
#     # Create empty score_arr
#     score_arr = np.array([])
#     counter = 0
#     spectro_list = []
#     file_array = []
#     previous_channels = stream.stream.channels
#     current_channels = stream.stream.channels
#
#     # Stream through the files, preprocess and apply the detector
#     #previous_date = stream.get_current_timesamp().date()
#     current_date = stream.get_current_timesamp().date()
#
#     # path to save and load spectrogram for reuse
#     seltab_path_detected = os.path.normpath(seltab_path_detected)
#     # two levels up from RunX: the model cv folder
#     #path_to_spectro_file = os.path.dirname(os.path.dirname(seltab_path_detected))
#
#     if config.USE_SAVED_FEATURE is True:
#         config.MAX_STREAMS = 864000 # 86400/.1
#
#     # __ExptResult
#     path_to_spectro_file = os.path.dirname(os.path.dirname(os.path.dirname(seltab_path_detected)))
#     if config.TEST_MODE is True:
#         spectro_file_curr_day = os.path.join(path_to_spectro_file, 'TEST_'+str(current_date))+'.npz'
#     else:
#         spectro_file_curr_day = os.path.join(path_to_spectro_file, str(current_date))+'.npz'
#
#     if config.USE_SAVED_FEATURE and os.path.exists(spectro_file_curr_day):
#         print('Load the saved feature files...')
#         stored_data = np.load(spectro_file_curr_day)
#         spectro_list = stored_data['arr_0']
#         file_array = stored_data['arr_1']
#         timestamp_array = stored_data['arr_2']
#         #spectro_list, file_array, timestamp_array
#         #np.savez(spectro_file_curr_day, spectro_list, file_array, timestamp_array)
#
#         print('Make prediction...')
#         score_arr = make_predictions(model,
#                                                      spectro_list,
#                                                      previous_channels,
#                                                      config,
#                                                      streamevent = True
#                                                      )
#         score_arr = score_arr.reshape(-1, previous_channels)
#         score_arr = score_arr[:-1].reshape(-1, previous_channels)
#
#         print('Make dataframe for selection tables')
#         raven_data_frame = make_raven_sndsel_table(
#                         score_arr,
#                         OverlapRatioThre,
#                         file_array,
#                         timestamp_array,
#                         ScoreThre,
#                         config,
#                         date_format = date_format)
#
#         print('Write to the selection table')
#         #print('writing selection table ' + current_file)
#         write_raven_table_to_CSV(raven_data_frame,
#                                              seltab_path_detected,
#                                              SelectionTableName,
#                                              SNDSel= SNDSel)
#
#     else:
#         #print('Extracting features from sound files...')
#         sec_advance = float((N_read+N_previous)/stream.get_all_stream_fs()[0])
#         while True:
#             try:
#                 # load the samples
#                 samps = stream.read(N = N_read, previousN=N_previous)[0]
#                 #advance the counter
#                 counter +=1
#
#                 # Append the timestamp
#                 #timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=2.0)
#                 timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=sec_advance)
#
#                 # Set the current file
#                 current_file = stream.stream.filename
#
#                 # current date
#                 #current_date = stream.get_current_timesamp().date()
#
#                 # Append the timestamp
#                 timestamp_array.append(timestamp)
#
#                 # Set the current file
#                 file_array.append(current_file)
#
#                 #pre-process all the channels
#                 for ii in range(samps.shape[1]):
#                     #Spectro = preprocess(samps[:,ii], filter_args)
#                     Spectro = preprocess(samps[:,ii], config)
#                     spectro_list.append([Spectro])
#                 #spectro_list.append([Spectro]) # this seems weird: only save the feature of the last channel
#
#                 if counter == config.MAX_STREAMS:
#                     preds = make_predictions(model,
#                                              spectro_list,
#                                              current_channels,
#                                              config,
#                                              streamevent = False
#                                              )
#                     # make model predictions
#                     score_arr = np.append(score_arr, preds)
#                     #if bool(len(score_arr) % current_channels):
#                         #print('blarg')
#
#                     # reset the list
#                     spectro_list = []
#                     counter = 0
#
#             except (StreamGap, StreamEnd) as S:
#
#                 # If error was thrown after a read (as opposed to after writing
#                 # a new table, then read the samples and predict)
#                 #score_arr = np.append(score_arr,
#                 print('Make prediction')
#                 preds = make_predictions(model,
#                                              spectro_list,
#                                              previous_channels,
#                                              config,
#                                              streamevent = True
#                                              )
#                 score_arr = np.append(score_arr, preds)
#                 score_arr = score_arr.reshape(-1, previous_channels)
#                 #score_arr = score_arr[:-1].reshape(-1, previous_channels)
#
#                 # save spectro_list
#                 if config.USE_SAVED_FEATURE is True:
#                     print('Save features to numpy binary files...')
#                     np.savez(spectro_file_curr_day, spectro_list, file_array, timestamp_array)
#
#                 # End of Stream write the dataframe
#                 if type(S).__name__ == 'StreamEnd' :
#                     # Make the raven table
#                     print('Make dataframe for selection tables')
#                     raven_data_frame = make_raven_sndsel_table(
#                         score_arr,
#                         OverlapRatioThre,
#                         file_array,
#                         timestamp_array,
#                         ScoreThre,
#                         config,
#                         date_format = date_format)
#
#                     #print('writing selection table ' + current_file)
#                     print('Write to the selection table')
#                     write_raven_table_to_CSV(raven_data_frame,
#                                              seltab_path_detected,
#                                              SelectionTableName,
#                                              SNDSel= SNDSel)
#                     return False
#
#                 elif type(S).__name__ == 'StreamGap':
#                     # reset the list
#                     spectro_list = []
#                     file_array = []
#                     timestamp_array =[]
#                     counter = 0
#                     score_arr = []
#                     #print('I passed a stream gap!!')
#                     continue
#
#
# def fft_sample_to_spectro_list_failed(sample_list, config):
#     spectro_list = []
#     for ss in sample_list:
#         for ii in range(ss.shape[1]): # over channels
#             spectro = preprocess(ss[:,ii], config)
#             spectro_list.append([spectro])
#     return spectro_list
#
#
def fft_sample_to_spectro(sample, config):
    spectro_list = []
    for ii in range(sample.shape[1]): # over channels
        spectro = preprocess(sample[:,ii], config)
        spectro_list.append([spectro])
    return spectro_list


def run_detector_1day_parallel_fft(stream,
                model,
                seltab_path_detected,
                filter_args=None,
                N_read = 1000,
                N_previous= 1000,
                date_format = "%Y%m%d_%H%M%S",
                ScoreThre = 0.05,
                max_streams = 40000, # make it big to utilize GPU
                OverlapRatioThre = 0.4,
                SelectionTableName = 'temp',
                SNDSel= False,
                config = None):
    """
    Derived from run_detector_1day_parallel but aim to parallel compute fft.
    """
    #########################################################################
    if config is not None:
        # preprocess arguments
        filter_args = {'FFTSize': config.FFT_SIZE,
                       'HopSize': config.HOP_SIZE,
                       'fs': config.SAMPLE_RATE,
                       'filter_fx' : 'ys_Preprocess'}
        fs = filter_args['fs']
        # Calculate number of samples to advance the streamer and the number
        # of 'previous' samples to retain. Note this is not the same as 
        # stft and advance
        FrameStepSec = config.FRAME_STEP_SEC
        FrameStep = int(FrameStepSec*fs)
        FrameSizeSec = config.FRAME_SIZE_SEC # each window is 2 sec long
        FrameSize = int(FrameSizeSec*fs)
        ScoreThre = config.SCORE_THRE
        N_read = FrameStep
        N_previous = FrameSize -N_read
        
        date_format = config.TIME_FORMAT
        OverlapRatioThre = config.OVERLAP_RATIO

    if type(model) == str:
        try:
            model = load_model(model)
        except ValueError:
             try:
                 model = load_model(model, custom_objects={'F1_Class': F1_Class})
             except:
                 print('Falure loading model') # Need fix

    # Create empty timestamp array to get timestamp of detections
    timestamp_array =[]
    
    # Create empty score_arr
    score_arr = np.array([])
    counter = 0
    spectro_list = []
    SampList = []
    file_array = []
    previous_channels = stream.stream.channels
    current_channels = stream.stream.channels
    
    # Stream through the files, preprocess and apply the detector
    #previous_date = stream.get_current_timesamp().date()
    current_date = stream.get_current_timesamp().date()

    # path to save and load spectrogram for reuse
    seltab_path_detected = os.path.normpath(seltab_path_detected)
    # two levels up from RunX: the model cv folder
    #path_to_spectro_file = os.path.dirname(os.path.dirname(seltab_path_detected))
    
    if config.USE_SAVED_FEATURE is True:
        config.MAX_STREAMS = 864000 # 86400/.1
    else:
        config.MAX_STREAMS = int(3.*60.*60./.1/current_channels) # 3 hours of sounds with 0.1 sec hop size
        #config.MAX_STREAMS = 1.*60.*60./.1/current_channels # 1 hours of sounds with 0.1 sec hop size
        
    # __ExptResult
    path_to_spectro_file = os.path.dirname(os.path.dirname(os.path.dirname(seltab_path_detected)))
    if config.TEST_MODE is True:
        spectro_file_curr_day = os.path.join(path_to_spectro_file, 'TEST_'+str(current_date))+'.npz'
    else:
        spectro_file_curr_day = os.path.join(path_to_spectro_file, str(current_date))+'.npz'


    #print('Extracting features from sound files...')
    sec_advance = float((N_read+N_previous)/stream.get_all_stream_fs()[0])
    N_tot = N_read + N_previous
    while True:
        try:
            # load the samples
            samps = stream.read(N = N_read, previousN=N_previous)[0]
            #advance the counter
            counter += 1

            # Append the timestamp
            #timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=2.0)
            timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=sec_advance)
            
            # Set the current file
            current_file = stream.stream.filename
                        
            # Append the timestamp
            timestamp_array.append(timestamp)
            
            # Set the current file
            file_array.append(current_file)
            if samps.shape[0] != N_tot:
                if samps.ndim == 1:
                    samps_temp = np.zeros((N_tot))
                    samps_temp[0:samps.shape[0]] = samps
                else:  # > 1
                    samps_temp = np.zeros((N_tot, samps.shape[1]))
                    samps_temp[0:samps.shape[0], :] = samps

                samps = samps_temp

            SampList.append(samps)
                 
            if counter == config.MAX_STREAMS:
                print(timestamp)
                print("Extracting features...")
                if config.PARALLEL_NUM == 1: # parallel computing using CPU
                    pool_fft = mp.Pool(processes=config.NUM_CORE)
                    results = pool_fft.starmap(fft_sample_to_spectro, zip(SampList, repeat(config)))
                    spectro_list = []
                    for result in results:
                        spectro_list += result
                        #spectro_list += [result]
                    pool_fft.close()
                    pool_fft.join()

                elif config.PARALLEL_NUM == 2: # GPU: pycuda, CuFFT, etc.
                    print("Under construction for GPU-based parallel computing")
                else: # 0 or others
                    print("No configuration on parallel computing. Using single core CPU")
                    spectro_list = []
                    for ss in SampList:
                        for ii in range(ss.shape[1]): # over channels
                            Spectro = preprocess(ss[:,ii], config)
                            spectro_list.append(Spectro)

                print("Making score prediction...\n")
                preds = make_predictions(model, 
                                         spectro_list,
                                         current_channels,
                                         config,
                                         streamevent = False
                                         )                
                # make model predictions
                score_arr = np.append(score_arr, preds)
                #if bool(len(score_arr) % current_channels):
                    #print('blarg')

                # reset the list
                spectro_list = []
                SampList = []
                counter = 0

        except (StreamGap, StreamEnd) as S:
            # If error was thrown after a read (as opposed to after writing
            # a new table, then read the samples and predict)          
            #score_arr = np.append(score_arr,
            #print('Make prediction')
            print("Extracting features and making prediction...\n")

            if config.PARALLEL_NUM == 1: # parallel computing using CPU
                pool_fft = mp.Pool(processes=config.NUM_CORE)
                results = pool_fft.starmap(fft_sample_to_spectro, zip(SampList, repeat(config)))
                spectro_list = []
                for result in results:
                    # if result.shape[0] is not 1600:
                    #     print('stop here')
                    spectro_list += result
                    #spectro_list += [result]
                pool_fft.close()
                pool_fft.join()
                
            elif config.PARALLEL_NUM == 2: # GPU: pycuda, CuFFT, etc.
                print("Under construction for GPU-based parallel computing")
            else:  # 0 or others
                print("No configuration on parallel computing. Using single core CPU")
                spectro_list = []
                for ss in SampList:
                    for ii in range(ss.shape[1]): # over channels
                        Spectro = preprocess(ss[:,ii], config)
                        spectro_list.append(Spectro)
                        
            preds = make_predictions(model,
                                         spectro_list,
                                         previous_channels,
                                         config,
                                         streamevent = True
                                         )
            score_arr = np.append(score_arr, preds)
            score_arr = score_arr.reshape(-1, previous_channels)
            #score_arr = score_arr[:-1].reshape(-1, previous_channels)
            
            # save spectro_list
            if config.USE_SAVED_FEATURE is True:
                print('Save features to numpy binary files...')
                np.savez(spectro_file_curr_day, spectro_list, file_array, timestamp_array)
                      
            # End of Stream write the dataframe
            if type(S).__name__ == 'StreamEnd' :
                # Make the raven table
                print('Make dataframe for selection tables')
                raven_data_frame = make_raven_sndsel_table(
                    score_arr,
                    OverlapRatioThre,
                    file_array,
                    timestamp_array,
                    ScoreThre,
                    config,
                    date_format = date_format)
                
                #print('writing selection table ' + current_file)
                print('Write to the selection table')
                write_raven_table_to_CSV(raven_data_frame,
                                         seltab_path_detected,
                                         SelectionTableName,
                                         SNDSel= SNDSel)                    
                return False
        
            elif type(S).__name__ == 'StreamGap':
                # reset the list
                spectro_list = []
                file_array = []
                timestamp_array =[]
                counter = 0
                score_arr = []
                #print('I passed a stream gap!!')
                continue

######################################################################
# The following attempts are all failed since my goal was mistakenly to 
# parallel compute the make prediction function, which is already parallel
# computed through GPU. That's the reason why it cannot be parallel 
# computed through multiprocessing package. 

#iolock = mp.Lock()
NCORE = 2
class queue_and_pool(object):
    def __init__(self):
        self.queue = mp.Queue()
        #self.pool = mp.Pool(processes=NCORE, initializer=self.worker_main,)
        self.pool = mp.Pool(processes=NCORE)
        self.preds_list = []
        self.result_list = []
        #self.temp_buffer = []
        
#    def add_to_queue(self, msg):
#        if self.q.full():
#            self.temp_buffer.append(msg)
#        else:
#            self.q.put(msg)
#            if len(self.temp_buffer) > 0:
#                self.add_to_queue(self.temp_buffer.pop())
    def retrieve_result(self):
        return self.result_list
    
    def pool_close_join(self):
        self.pool.close()
        self.pool.join()
        
    def write_to_queue(self, data):
        #print("Adding items...")
        self.queue.put(data)
        #print('Queue size: '+str(self.queue.qsize()))
        time.sleep(random()*2)

    def send_end_to_queue(self):
        for _ in range(NCORE):
            self.queue.put(None)
    
    def proess_worker_main(self):
        data = self.queue.get()
        print("{0} retrieved.".format(os.getpid()))
        print('Queue size: '+str(self.queue.qsize()))
        print("Processing the model!")
        
        if data is None:
            self.pool_close_join()
        
        print("Generate a new peocess...")
        self.pool.apply_async(self.worker_main, args=(data,), callback=self.collect_results)
        
    def collect_result(self, result):
        self.result_list.extend(result)
        return self.result_list
        
    def worker_main(self, data):
        """
        Waits indefinitely for an item to be written in the queue.
        Finishes when the parent process terminates.
        """
        print("Process {0} started".format(os.getpid()))
                
        model = data[0]
        #spectro_list = data[1]
        SampList = data[1]
        current_channels = data[2]
        config = data[3]

        spectro_list = []
        for ss in SampList:
            for ii in range(ss.shape[1]):
                Spectro = preprocess(ss[:,ii], config)
                spectro_list.append([Spectro])
        
        preds = make_predictions(model, 
                                 spectro_list,
                                 current_channels,
                                 config,
                                 streamevent = False
                                 )

        print("Finished the prediction!")                   
        # simulate some random length operations
        #time.sleep(random())

        return preds


def pred_process_lock(queue, iolock):
    from time import sleep
    while True:
        stuff = queue.get()
        if stuff is None:
            break
        
        #iolock.acquire()
        with iolock:
            print('Processing begins...')
            print('queue.qsize: '+str(queue.qsize()))
            print("Processing the model!")
        #iolock.release()
        
        model = stuff[0]
        #spectro_list = stuff[1]
        SampList = stuff[1]
        current_channels = stuff[2]
        config = stuff[3]
        
        spectro_list = []
        for ss in SampList:
            for ii in range(ss.shape[1]):
                Spectro = preprocess(ss[:,ii], config)
                spectro_list.append([Spectro])
        
        preds = make_predictions(model, 
                                     spectro_list,
                                     current_channels,
                                     config,
                                     streamevent = False
                                     )  
        sleep(0.5)


# def run_detector_1day_parallel_v1(stream,
#                 model,
#                 seltab_path_detected,
#                 filter_args=None,
#                 N_read = 1000,
#                 N_previous= 1000,
#                 date_format = "%Y%m%d_%H%M%S",
#                 ScoreThre = 0.05,
#                 max_streams = 40000, # make it big to utilize GPU
#                 OverlapRatioThre = 0.4,
#                 SelectionTableName = 'temp',
#                 SNDSel= False,
#                 config = None):
#
#     if config is not None:
#         # preprocess arguments
#         filter_args = {'FFTSize': config.FFT_SIZE,
#                        'HopSize': config.HOP_SIZE,
#                        'fs': config.SAMPLE_RATE,
#                        'filter_fx' : 'ys_Preprocess'}
#         fs = filter_args['fs']
#         # Calculate number of samples to advance the streamer and the number
#         # of 'previous' samples to retain. Note this is not the same as
#         # stft and advance
#         FrameStepSec = config.FRAME_STEP_SEC
#         FrameStep = int(FrameStepSec*fs)
#         FrameSizeSec = config.FRAME_SIZE_SEC # each window is 2 sec long
#         FrameSize = int(FrameSizeSec*fs)
#
#         N_read = FrameStep
#         N_previous = FrameSize -N_read
#
#         date_format = config.TIME_FORMAT
#         OverlapRatioThre = config.OVERLAP_RATIO
#
#         # output frequencies of the upcall detector
#         low_f = config.BOX_OUT_F1,
#         high_f = config.BOX_OUT_F2
#
#         max_streams = config.MAX_STREAMS
#
#     # Load the trained model
#     model = load_model(model, custom_objects={'F1_Class': F1_Class})
#
#     ########################################################################
#     # Create empty timestamp array to get timestamp of detections
#     timestamp_array =[]
#
#     # Create empty score_arr
#     score_arr = np.array([])
#     counter = 0
#     spectro_list = []
#     SampList = []
#     file_array = []
#     previous_channels = stream.stream.channels
#     current_channels = stream.stream.channels
#
#     # Stream through the files, preprocess and apply the detector
#     #previous_date = stream.get_current_timesamp().date()
#     current_date = stream.get_current_timesamp().date()
#
#     # path to save and load spectrogram for reuse
#     seltab_path_detected = os.path.normpath(seltab_path_detected)
#
#     queue_pool_run = queue_and_pool()
#
#     #print('Extracting features from sound files...')
#     sec_advance = float((N_read+N_previous)/stream.get_all_stream_fs()[0])
#
#     while True:
#         try:
#             # load the samples
#             samps = stream.read(N = N_read, previousN=N_previous)[0]
#             #advance the counter
#             counter +=1
#
#             # Append the timestamp
#             #timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=2.0)
#             timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=sec_advance)
#             # Set the current file
#             current_file = stream.stream.filename
#
#             # Append the timestamp
#             timestamp_array.append(timestamp)
#             # Set the current file
#             file_array.append(current_file)
#
#             SampList.append(samps)
#
#             if counter == config.MAX_STREAMS:
#                 #queue.put([model, spectro_list,current_channels,config])
#                 #queue.put([model, SampList, current_channels, config])
#                 queue_pool_run.write_to_queue([model, SampList, current_channels, config])
#                 queue_pool_run.proess_worker_main()
#
#                 # reset the list
#                 spectro_list = []
#                 SampList = []
#                 counter = 0
#
#         except (StreamGap, StreamEnd) as S:
#             queue_pool_run.send_end_to_queue()
#
#             #for _ in range(NCORE):  # tell workers we're done
#             #    queue.put(None)
#
#             # If error was thrown after a read (as opposed to after writing
#             # a new table, then read the samples and predict)
#             #score_arr = np.append(score_arr,
#             print('Make prediction')
#             queue_pool_run.write_to_queue([model, SampList, current_channels, config])
#
#             queue_pool_run.pool_close_join()
#             #score_arr = np.append(score_arr, preds)
#             #score_arr = queue_pool_run.retrieve_preds
#             score_arr = queue_pool_run.retrieve_result()
#
#             score_arr = score_arr.reshape(-1, previous_channels)
#             #score_arr = score_arr[:-1].reshape(-1, previous_channels)
#
#             # End of Stream write the dataframe
#             if type(S).__name__ == 'StreamEnd' :
#                 # Make the raven table
#                 print('Make dataframe for selection tables')
#                 raven_data_frame = make_raven_sndsel_table(
#                     score_arr,
#                     OverlapRatioThre,
#                     file_array,
#                     timestamp_array,
#                     ScoreThre,
#                     config,
#                     date_format = date_format)
#
#                 print('Write to the selection table')
#                 write_raven_table_to_CSV(raven_data_frame,
#                                          seltab_path_detected,
#                                          SelectionTableName,
#                                          SNDSel= SNDSel)
#                 return False
#
#             elif type(S).__name__ == 'StreamGap':
#                 # reset the list
#                 spectro_list = []
#                 file_array = []
#                 timestamp_array =[]
#                 counter = 0
#                 score_arr = []
#                 continue
#
#             pool.close()
#             pool.join()
#
#
# def run_detector_1day_parallel_v2(stream,
#                 model,
#                 seltab_path_detected,
#                 filter_args=None,
#                 N_read = 1000,
#                 N_previous= 1000,
#                 date_format = "%Y%m%d_%H%M%S",
#                 ScoreThre = 0.05,
#                 max_streams = 40000, # make it big to utilize GPU
#                 OverlapRatioThre = 0.4,
#                 SelectionTableName = 'temp',
#                 SNDSel= False,
#                 config = None):
#
#     if config is not None:
#         # preprocess arguments
#         filter_args = {'FFTSize': config.FFT_SIZE,
#                        'HopSize': config.HOP_SIZE,
#                        'fs': config.SAMPLE_RATE,
#                        'filter_fx' : 'ys_Preprocess'}
#         fs = filter_args['fs']
#         # Calculate number of samples to advance the streamer and the number
#         # of 'previous' samples to retain. Note this is not the same as
#         # stft and advance
#         FrameStepSec = config.FRAME_STEP_SEC
#         FrameStep = int(FrameStepSec*fs)
#         FrameSizeSec = config.FRAME_SIZE_SEC # each window is 2 sec long
#         FrameSize = int(FrameSizeSec*fs)
#
#         N_read = FrameStep
#         N_previous = FrameSize -N_read
#
#         date_format = config.TIME_FORMAT
#         OverlapRatioThre = config.OVERLAP_RATIO
#
#         # output frequencies of the upcall detector
#         low_f = config.BOX_OUT_F1,
#         high_f = config.BOX_OUT_F2
#
#         max_streams = config.MAX_STREAMS
#
#     # Load the trained model
#     model = load_model(model, custom_objects={'F1_Class': F1_Class})
#
#     # Create empty timestamp array to get timestamp of detections
#     timestamp_array =[]
#
#     # Create empty score_arr
#     score_arr = np.array([])
#     counter = 0
#     spectro_list = []
#     SampList = []
#     file_array = []
#     previous_channels = stream.stream.channels
#     current_channels = stream.stream.channels
#
#     # Stream through the files, preprocess and apply the detector
#     #previous_date = stream.get_current_timesamp().date()
#     current_date = stream.get_current_timesamp().date()
#
#     # path to save and load spectrogram for reuse
#     seltab_path_detected = os.path.normpath(seltab_path_detected)
#
#     # Parllel computing starts here:
#     NCORE = 2
#     pool = mp.Pool(processes=NCORE)
#
#     #print('Extracting features from sound files...')
#     sec_advance = float((N_read+N_previous)/stream.get_all_stream_fs()[0])
#
#     queue_pool_run = queue_and_pool()
#     while True:
#         try:
#             # load the samples
#             samps = stream.read(N = N_read, previousN=N_previous)[0]
#             #advance the counter
#             counter +=1
#
#             # Append the timestamp
#             #timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=2.0)
#             timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=sec_advance)
#             # Set the current file
#             current_file = stream.stream.filename
#
#             # Append the timestamp
#             timestamp_array.append(timestamp)
#             # Set the current file
#             file_array.append(current_file)
#
#             SampList.append(samps)
#
#             if counter == config.MAX_STREAMS:
#                 queue_pool_run.write_to_queue([model, SampList, current_channels, config])
#
#                 print("Queuing after reaching MAX_STREAMS")
#                 queue_pool_run.proess_worker_main()
#
#                 # make model predictions
#                 ###score_arr = np.append(score_arr, preds)
#
#                 # reset the list
#                 spectro_list = []
#                 SampList = []
#                 counter = 0
#         except (StreamGap, StreamEnd) as S:
#             queue_pool_run.send_end_to_queue()
#             # If error was thrown after a read (as opposed to after writing
#             # a new table, then read the samples and predict)
#             print('Make prediction')
#             queue_pool_run.write_to_queue([model, SampList, current_channels, config])
#
#             queue_pool_run.pool_close_join()
#             score_arr = queue_pool_run.retrieve_preds
#             score_arr = score_arr.reshape(-1, previous_channels)
#
#             # End of Stream write the dataframe
#             if type(S).__name__ == 'StreamEnd' :
#                 # Make the raven table
#                 print('Make dataframe for selection tables')
#                 raven_data_frame = make_raven_sndsel_table(
#                     score_arr,
#                     OverlapRatioThre,
#                     file_array,
#                     timestamp_array,
#                     ScoreThre,
#                     config,
#                     date_format=date_format)
#
#                 print('Write to the selection table')
#                 write_raven_table_to_CSV(raven_data_frame,
#                                          seltab_path_detected,
#                                          SelectionTableName,
#                                          SNDSel= SNDSel)
#                 return False
#             elif type(S).__name__ == 'StreamGap':
#                 # reset the list
#                 spectro_list = []
#                 file_array = []
#                 timestamp_array =[]
#                 counter = 0
#                 score_arr = []
#                 continue
