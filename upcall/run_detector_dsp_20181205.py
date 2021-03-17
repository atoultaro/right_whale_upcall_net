#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:30:48 2018

This is a modified version of 'RunDetector' by YS. It consists of a series
of functions that can be used to run a 


@author: kpalmer
"""


import os, sys, re, datetime, librosa
from soundfile import SoundFile
import numpy as np
from keras.models import load_model
import pandas as pd

# both dsp and upcall-basic-net are installed on the same level of directories
#sys.path.append("/home/kpalmer/AnacondaProjects/dsp")
#sys.path.append("/home/kpalmer/AnacondaProjects/upcall-basic-net/upcall")
sys.path.append(os.path.join(os.path.dirname(sys.path[-1]), 'dsp'))
#print(sys.path)

from dsp.SampleStream import SampleStream, Streams
from dsp.abstractstream import StreamGap, StreamEnd

from upcall.config import Config
from upcall.filter_list import bird_net_light_filter,bird_net_filter, \
     med2d_with_cropping, null_filter, normalize_median_filter, ys_Preprocess,\
     nearest_to
from upcall.train_classifier import F1_Class
from upcall.sound_util import preprocess

#import time

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
    
    fname= os.path.split(f)[1]
    
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
                 
  
    return(start_time)    
    
    

 
def preprocess_EMBARGO(SampleClip, filter_args = None): 
     
    ''' Pre-process the sample clip by creating an spcetrogram, clipping the 
    image and applying a image filter 
     
    inputs: 
        SampleClip - sound data (from sound read or similar) 
        filter_args - either Config file or dictionary containing the following keys 
            FFTSize - fft size in samples 
            HopSize - hop size in samples 
            f_idx_x - tuple with the frequency VALUES 
            f_idx_y - tuple with the time indexes to retain 
            filter_fx - which image prefilter (eg. birdnet, median2d) 
            kernel_size - kernel size for median2d filtering  
                            (median2d, normalize_Median2d) default 3 
            area_threshold - connected region threshold required to retain block 
                (bird net, birdnet_lite) 
         
    Returns: 
        flattend spectrogram image 
     
    ''' 
    
    
    # check if config file
    if filter_args == Config:
        filter_args = {'FFTSize': Config.FFT_SIZE, 
                       'HopSize': Config.HOP_SIZE, 
                       'fs': Config.SAMPLE_RATE, 
                       'filter_fx' : ys_Preprocess} 
    
    if filter_args is None: 
                # Define filter/preprocess arguments 
        filter_args = {'FFTSize': 256, 
                       'HopSize': 100, 
                       'fs': 2000, 
                       'filter_fx' : None, 
                       'f_range' : None, 
                       'kernel_size' : 3, 
                       'area_threshold': 5} 
         
     
     
     
    # Required for all values 
    FFT= filter_args['FFTSize'] 
    Hop= filter_args['HopSize'] 
    fs = filter_args['fs'] 
 
    # Check if YS preprocessing 
    if filter_args['filter_fx'] is not 'ys_Preprocess': 
        # spectrogram 
        Spectro = np.abs(librosa.stft(SampleClip, 
                                      n_fft=FFT, 
                                      hop_length=Hop)) 
         
        # get the indecies of the values to clip also set up dictionaries for 
        # filter functions 
        if filter_args['f_range'] is not None: 
            f_ind=[] 
             
 
            f_bins = np.linspace(0, fs, FFT) 
            f_bins = f_bins + f_bins[1]/2 
             
            # find the clipping indecies 
            f_ind.append(nearest_to(f_bins, filter_args['f_range'][0])[1]) 
            f_ind.append(nearest_to(f_bins, filter_args['f_range'][1])[1]) 
             
            # add it to the dictionary 
            filter_args['f_ind'] = f_ind 
             
        
         
        # else nothing there so we can set to 0, this could come out of the  
        # function but not sure it makes a huge difference compared to the 
        # issue above 
        else: 
            filter_args['f_ind'] = None 
         
        # median filter 2 dimensional 
        if filter_args['filter_fx'] == 'med2d_with_cropping': 
            filter_fn = med2d_with_cropping 
            processing_args = extract(filter_args, ['f_ind']) 
         
        # Normalizing filter 
        elif filter_args['filter_fx'] == 'normalize_median_filter': 
            filter_fn = med2d_with_cropping 
            processing_args = extract(filter_args, ['f_ind','kernel_size']) 
     
        # Birdnet FILTERS 
        elif filter_args['filter_fx'] == 'bird_net_filter': 
            filter_fn = bird_net_filter 
            processing_args = extract(filter_args, ['f_ind', 'area_threshold']) 
             
        # Birdnet FILTERS 
        elif filter_args['filter_fx'] == 'bird_net_light_filter': 
            filter_fn = bird_net_light_filter 
            processing_args = extract(filter_args, ['f_ind', 'area_threshold']) 
     
        else: 
            # no filter 
            filter_fn = null_filter 
            processing_args = extract(filter_args, ['f_ind', 'area_threshold']) 
 
        Spectro = filter_fn(Spectro, **processing_args) 
     
    # It is YS pre processing algorithm therefore just process samples     
    else: 
        filter_fn = ys_Preprocess 
        Spectro = filter_fn(SampleClip) 
 
    return Spectro 
    
                
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
#    FRAME_STEP_SEC = 0.1 # sec
#    FRAME_SIZE_SEC = 2.0 # each window is 2 sec long
#    
#    OVERLAP_RATIO = 1.0 # non-max suppression
    
    
    overlap_size = int(config.FRAME_SIZE_SEC/config.FRAME_STEP_SEC
                    *config.OVERLAP_RATIO)
    # boxes sorted by scores
    box_sorted_by_score = boxes_in_batch[np.argsort(score_arr[boxes_in_batch])[::-1]] 
    # [::-1] reverse the sort order from ascending to descending
    # get the ordered values: score_arr[boxes_in_batch][box_sorted_by_score]

    # original approach
#    time_start = time.time()
#    boxes_separated = separate_boxes(box_sorted_by_score, overlap_size)
#    print('Method 1: run time is: '+str(time.time() - time_start))
#    
#    time_start2 = time.time()
#    boxes_separated2 = separate_boxes_faster(box_sorted_by_score, overlap_size)
#    print('Method 2: run time is: '+str(time.time() - time_start2))

    #time_start3 = time.time()
    boxes_separated = separate_boxes_fasterer(box_sorted_by_score, overlap_size)
    #print('Method 3: run time is: '+str(time.time() - time_start3))

    
    # alternative approach
    #boxes_separated = separate_boxes_time(box_sorted_by_score.sort(), overlap_size)
        
    # computer vision approach: Malisiewicz et al.
    #boxes_separated = non_max_suppression_fast(boxes_in_batch, overlapThresh):
    #print(boxes_separated)
    #print(boxes_separated2)
    #print(boxes_separated3)
    
    return boxes_separated


def separate_boxes(box_sorted_by_score, overlap_size):
    boxes_separated = []
    for ii in range(box_sorted_by_score.shape[0]):
        if ii == 0:
            boxes_separated.append(box_sorted_by_score[ii])
        else:
            # compare with previous indices through a while loop
            # stupid for loop; need to modify to while loop to save computation
            IndClose = False
            for jj in boxes_separated:
                if( abs(jj-box_sorted_by_score[ii]) <= overlap_size ):
                    IndClose = True
            if IndClose == False:
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
            IndClose = False
            while(not IndClose and jj < len(boxes_separated)):
                if( abs(boxes_separated[jj]-box_sorted_by_score[ii]) <= overlap_size ):
                    IndClose = True
                jj += 1
            if IndClose == False:
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
            #IndClose = False
            IndCLoseArr = np.array((abs(boxes_separated - box_sorted_by_score[ii]) <= overlap_size))
            #for jj in boxes_separated:
            #    if( abs(jj-box_sorted_by_score[ii]) <= overlap_size ):
            #        IndClose = True
            if IndCLoseArr.sum() == 0:
                boxes_separated.append(box_sorted_by_score[ii])
    boxes_separated = np.sort(boxes_separated)

    return boxes_separated
    

def separate_boxes_time(boxes_sorted_by_time, score_arr, overlap_size):
    # do nothing yet. No idea how to do it in our case.
    # We use scores to groups time instants into a call.
    # it's good to apply a classifier that generates selective scores but it could be a problem for a loose classifier. 
    boxes_separated = []
    for ii in boxes_sorted_by_time:
        print()
        
    return boxes_separated
    
# =============================================================================
#  Load soundfiles into stream/streams
# =============================================================================

def make_sound_stream(DaySoundPath, format_str = "%Y%m%d_%H%M%S"):
    ''' 
    Function for making a soundstream capable of iterating through lots of
    files.

    Input:
        DaySoundPath - List of folder location(s) containing soundfiles
        format_str - format string of the date default is"%Y%m%d_%H%M%S" 
            for Cornell
    Returns
        Returns a soundstrem of all wav or aif files listed in the folder 
    '''
    
    # Declare the stream
    stream_elements = Streams()
    
    # if it's not a list make it a list
    if isinstance(DaySoundPath, (list,)) is False:
        DaySoundPath = [DaySoundPath]
        
    
    
    for ii in range(len(DaySoundPath)):
        
        # get the file director
        file_dir = DaySoundPath[ii]
        

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
                # print(os.path.join(directory, filename)) # debugging
            else:
                continue
                
    # Combine streams into a sample stream
    stream = SampleStream(stream_elements)
        
    return stream

def make_predictions(model, 
                     SpectroList, 
                     n_channels,
                     config,
                     streamevent = False
                     ):
    '''
    Make model predictions
    
    Inputs:
        model - trained and properly loaded keras model
        SpectroList - list of spectrogram elements
        n_channels - int number of channels in the data
        streamevent - bool whether the function was triggered by a stream event 
        
    Returns:
        predictions - array of predictions length of spectrolist by n_channels
    
    '''
    if streamevent is True:
        # Finish reading/prediciting the remaining features
        if len(SpectroList)>1:
            FeaSpectro = np.expand_dims(np.vstack(SpectroList[0:len(SpectroList)]),3)
            FeaSpectro = FeaSpectro.reshape(FeaSpectro.shape[0], config.IMG_T, config.IMG_F, 1)
            if config.RECURR is True:
                FeaSpectro = np.squeeze(FeaSpectro)
            predictions = model.predict(FeaSpectro)[:,1]
        else:
            predictions =[]
    
    else: 
        FeaSpectro = np.expand_dims(np.vstack(SpectroList[0:len(SpectroList)]),3)
        FeaSpectro = FeaSpectro.reshape(FeaSpectro.shape[0], config.IMG_T, config.IMG_F, 1)
        if config.RECURR is True:
            FeaSpectro = np.squeeze(FeaSpectro)
        predictions = model.predict(FeaSpectro)[:,1]
    
    return predictions



def make_raven_SndSel_table( 
                            ScoreArr, 
                            OverlapRatioThre,
                            file_array,
                            timestamp_array,
                            ScoreThre,
                            config,
                            date_format = "%Y%m%d_%H%M%S"):
    '''
    Function for creating a sound selection table
    
    inputs:

        ScoreArr - numpy array of scores 
        file_array - array of file names
      	OverlapRatioThres- float, between 0 and 1 throw out adjacent
            events closer than t
        timestamp_array
	    ScoreThre
	    date_format - datestring format of the aiff file
        config - optional configuration file

    '''
    
    RavenTable_out = pd.DataFrame()
    EventId = 0
    n_channels = ScoreArr.shape[1]

    # test channels
    for jj in range(n_channels):
         if  ScoreArr[:,jj].tolist().count(ScoreArr[0,jj]) != len(ScoreArr):
               print('chan ' + str(jj) + ' ok')
         else:
               print( print('chan ' + str(jj) + ' bad'))
    
    for ii in range(n_channels):
        # Set the event number
        EventId = len(RavenTable_out)+1
        
        # get the indicies of the events that are above the detection
        # threshold
        boxes_in_batch = np.where(ScoreArr[:,ii] > ScoreThre)
        
        
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
            
            frames = [RavenTable_out, pd.DataFrame(data)]
            RavenTable_out = pd.concat(frames)
            EventId = max(EventIDs)
            
                
            
                
        #Change order to match what raven expects          
        if len(RavenTable_out)>0:
            RavenTable_out = RavenTable_out[['Selection',
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
    return RavenTable_out


def write_raven_table_to_CSV(PandasDataFrame, 
                             SelTabPathDetected,
                             SelectionTableName, 
                             SNDSel= False):
    '''
    Write the pandas dataframe to a txt file
    
    Inputs:
        PandasDataFrame - pandas dataframe with approperiate columns
        SelTabPathDetected - output path
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
        SSND_sel_name = 'SndSel'
        
        if not SNDSel:
            PandasDataFrame = PandasDataFrame.drop(['Begin File', 
                                                   'File Offset (s)',
                                                   'Begin Path'], axis=1)
            SSND_sel_name = 'SelTab'
            
        
        # sort
        PandasDataFrame['Selection'] = np.arange(1, len(PandasDataFrame)+1)
        
        # make the file loc
        DayFile = os.path.join(SelTabPathDetected,  
                               SelectionTableName+ '_'+SSND_sel_name + '_'+\
                               str(PandasDataFrame['timestamp'].iloc[0].strftime('%Y%m%d')) +'.txt')
        
        # export
        PandasDataFrame.to_csv(DayFile,
                              sep='\t', mode='a',
                              index = False)
    else:
        pass
            




def run_detector(stream,
                model,
                SelTabPathDetected,
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

    
    ''' Runs the model over the datastream and first produces an array of 
    score values for all times in the datset/Day of data. Then removes all
    scores below a given threhsold and runs non-max suppression on the 
    remaining prediction values. Finally puts all predictions into 
    produces raven table(s) for
    the detector. Read in N_read and N_previous samples from the sample stream
    (e.g. context!). Step through the loaded samples making predictions from 
    the trained model. Run non-max supression to remove duplicate detections.
    
    inputs - 
        stream - SampleStream of files (see DSP sampelstream of
            MakeSampleStream)
        model -  Either traied and loadede keras model or full path to
            trained model
        
        #####################################
        # Filter args Dictionary
        #####################################
        
        FFTSize - FFTSize in samples
        HopSize - HopSize in samples
        filter_name - string, preprocessing filter name from filterlist.py
        f_range - frequency range (Hz) to maintain (e.g. cropping limits)
        
        #####################################
        # Advance dictionary data. Could also be dictionary 
        #####################################
        
        SelTabPathDetected - output directory for the raven selection table
        N_read, N_previous - number of sampels (combined) needed for the
            prediction. For upcall basic net this is 2k
        date_format - date string format of the sound files
            (e.g. "%Y%m%d_%H%M%S")
        ScoreThre- threshold above which the detector keeps the events
            default 0.05
        max_streams - number of stream advances to load into memory
                        prior to making a prediction default 400
        SelectionTableName - string for sound selection / sound table name
        SNDSel - bool to indicate whether or not to make a sound table default
                true
        
        config- optional configuration class containing input parameters
                
            

        
        
        Conditions for wariting out the selection table
        1) End of the stream
        2) The stream has come across a new day AND SndSel is Fales
        3) A file in the stream has different parameters (e.g. channels)
            this is not fully populated yet. Currently everything runs on 2khz
            so we are just going with that for the time being
        
        
        Creates a Raven readable selection table of detections that include
        prediction score
   
    
    '''
    
    
    #########################################################################
    # Check if there is a configureation file, if so load/override relevant
    # filter parameters from the config file
    
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
        
        N_read = FrameStep
        N_previous = FrameSize -N_read
        
        date_format = config.TIME_FORMAT
        OverlapRatioThre = config.OVERLAP_RATIO
        
        # output frequencies of the upcall detector
        low_f = config.BOX_OUT_F1,
        high_f = config.BOX_OUT_F2
        
        max_streams = config.MAX_STREAMS
        
     
    if type(model) == str:
        try:
         model = load_model(model)
        except ValueError:
             try:
                 model = load_model(model, custom_objects={'F1_Class': F1_Class})
             except:
                 print('Falure loading model')
                 
    #######################################################################
    # chcek that low and high frequency are in the filter args for the
    # output bounding box on the raven selection table 
    try:
        filter_args['low_f']
    except KeyError:
        low_f = 50
    try:
        filter_args['high_f']
    except KeyError:
        high_f = 350
                 
    ########################################################################
    # Initialize a table to export
    RavenTable_out = pd.DataFrame()
    
    # Create empty timestamp array to get timestamp of detections
    timestamp_array =[]
    
    # Create empty ScoreArr
    ScoreArr = np.array([])
    counter = 0
    SpectroList = []
    file_array = []
    previous_channels = stream.stream.channels
    current_channels = stream.stream.channels
    
    # Stream through the files, preprocess and apply the detector
    previous_date = stream.get_current_timesamp().date()
    current_date = stream.get_current_timesamp().date()


    while True:            
        # Append the timestamp 
        timestamp = stream.get_current_timesamp()
        
        # Set the current file
        current_file = stream.stream.filename
                
        try:     
            # load the samples
            samps = stream.read(N = N_read, previousN=N_previous)[0]
        except (StreamGap, StreamEnd) as S:
            # If error was thrown after a read (as opposed to after writing
            # a new table, then read the samples and predict)
            if PredictAgain:
                ScoreArr = np.append(ScoreArr, 
                                    make_predictions(model,
                                                     SpectroList, 
                                                     previous_channels,
                                                     config,
                                                     streamevent = True
                                                     ))
                ScoreArr = ScoreArr.reshape(-1, previous_channels)
                ScoreArr = ScoreArr[:-1].reshape(-1, previous_channels)
            
                # Make the raven table
                Raven_data_frame = make_raven_SndSel_table( 
                        ScoreArr, 
                        OverlapRatioThre,
                        file_array,
                        timestamp_array,
                        ScoreThre,
                        config,
                        date_format = date_format)
                
                      
            # End of Stream write the dataframe
            if type(S).__name__ == 'StreamEnd' :
                RavenTable_out = RavenTable_out.append(Raven_data_frame)
                
                #print('writing selection table ' + current_file)
                if PredictAgain:
                    write_raven_table_to_CSV(RavenTable_out,
                                             SelTabPathDetected,
                                             SelectionTableName,
                                             SNDSel= SNDSel)
                    PredictAgain = False
                    continue
                    
                return False
    
        
            elif type(S).__name__ == 'StreamGap':
                                        
                # reset the list
                SpectroList = []
                file_array = []
                timestamp_array =[]
                counter = 0
                ScoreArr = []               
                #print('I passed a stream gap!!')
                continue

        #advance the counter
        counter +=1   
        # number of channels in the new file
        current_channels = stream.stream.channels
        
        # current date
        current_date = stream.get_current_timesamp().date()

        # Append the timestamp 
        timestamp_array.append(timestamp)
        
        # Set the current file
        file_array.append(current_file)                    
            
        #pre-process all the channels
        for ii in range(samps.shape[1]):
            #Spectro = preprocess(samps[:,ii], filter_args)
            Spectro = preprocess(samps[:,ii], config)
            SpectroList.append([Spectro])

    
        if (current_channels != previous_channels) or \
            (SNDSel is False and current_date != previous_date):
            
            if PredictAgain:
                preds = make_predictions(model, 
                                         SpectroList,
                                         previous_channels,
                                         config,
                                         streamevent = False
                                         )
                ScoreArr = np.append(ScoreArr, preds)
                ScoreArr = ScoreArr.reshape(-1, previous_channels)
                
                Raven_data_frame = make_raven_SndSel_table( 
                                        ScoreArr, 
                                        OverlapRatioThre,
                                        file_array,
                                        timestamp_array,
                                        ScoreThre,
                                        config,
                                        date_format = date_format)
                
                RavenTable_out = RavenTable_out.append(Raven_data_frame)
                
                #SelectionTableName_out = SelectionTableName + \
                #                    str(previous_channels) + \
                #                    '_'+ str(ScoreThre)
    
                    
                # Export the raven table
                #print('writing selection table ' + current_file)
                write_raven_table_to_CSV(RavenTable_out,
                                         SelTabPathDetected,
                                         SelectionTableName, 
                                         SNDSel= SNDSel)
            
                # Reset everything
                RavenTable_out  = pd.DataFrame()                  
                ScoreArr = []
                SpectroList = [] 
                timestamp_array =[]
                file_array = []
                previous_channels = current_channels
                previous_date = current_date
                counter = 0
                
                # tell the reader not to write a dataframe on next streamgap
                PredictAgain = False
        else:
            # If iterated the maximum number of times, make the predicitons and clear the list
            if counter == max_streams:
                
                preds = make_predictions(model, 
                                         SpectroList,
                                         current_channels,
                                         config,
                                         streamevent = False
                                         )                
                # make model predictions
                ScoreArr = np.append(ScoreArr, preds)
                #if bool(len(ScoreArr) % current_channels):
                    #print('blarg')

                # reset the list
                SpectroList = []
                counter = 0
                
            previous_channels = current_channels
            previous_date = current_date
            PredictAgain = True
            

            
def run_detector_1day(stream,
                model,
                SelTabPathDetected,
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

    
    ''' Runs the model over the datastream and first produces an array of 
    score values for all times in the datset/Day of data. Then removes all
    scores below a given threhsold and runs non-max suppression on the 
    remaining prediction values. Finally puts all predictions into 
    produces raven table(s) for
    the detector. Read in N_read and N_previous samples from the sample stream
    (e.g. context!). Step through the loaded samples making predictions from 
    the trained model. Run non-max supression to remove duplicate detections.
    
    inputs - 
        stream - SampleStream of files (see DSP sampelstream of
            MakeSampleStream)
        model -  Either traied and loadede keras model or full path to
            trained model
        
        #####################################
        # Filter args Dictionary
        #####################################
        
        FFTSize - FFTSize in samples
        HopSize - HopSize in samples
        filter_name - string, preprocessing filter name from filterlist.py
        f_range - frequency range (Hz) to maintain (e.g. cropping limits)
        
        #####################################
        # Advance dictionary data. Could also be dictionary 
        #####################################
        
        SelTabPathDetected - output directory for the raven selection table
        N_read, N_previous - number of sampels (combined) needed for the
            prediction. For upcall basic net this is 2k
        date_format - date string format of the sound files
            (e.g. "%Y%m%d_%H%M%S")
        ScoreThre- threshold above which the detector keeps the events
            default 0.05
        max_streams - number of stream advances to load into memory
                        prior to making a prediction default 400
        SelectionTableName - string for sound selection / sound table name
        SNDSel - bool to indicate whether or not to make a sound table default
                true
        
        config- optional configuration class containing input parameters
        
        Conditions for wariting out the selection table
        1) End of the stream
        2) The stream has come across a new day AND SndSel is Fales
        3) A file in the stream has different parameters (e.g. channels)
            this is not fully populated yet. Currently everything runs on 2khz
            so we are just going with that for the time being
        
        
        Creates a Raven readable selection table of detections that include
        prediction score
   
    
    '''
    
    
    #########################################################################
    # Check if there is a configureation file, if so load/override relevant
    # filter parameters from the config file
    
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
        
        N_read = FrameStep
        N_previous = FrameSize -N_read
        
        date_format = config.TIME_FORMAT
        OverlapRatioThre = config.OVERLAP_RATIO
        
        # output frequencies of the upcall detector
        low_f = config.BOX_OUT_F1,
        high_f = config.BOX_OUT_F2
        
        max_streams = config.MAX_STREAMS
        
     
    if type(model) == str:
        try:
         model = load_model(model)
        except ValueError:
             try:
                 model = load_model(model, custom_objects={'F1_Class': F1_Class})
             except:
                 print('Falure loading model') # Need fix
                 
    #######################################################################
    # chcek that low and high frequency are in the filter args for the
    # output bounding box on the raven selection table 
    try: # need fixz
        filter_args['low_f']
    except KeyError:
        low_f = 50
    try:
        filter_args['high_f']
    except KeyError:
        high_f = 350
                 
    ########################################################################      

    # Initialize a table to export
    RavenTable_out = pd.DataFrame()
    
    # Create empty timestamp array to get timestamp of detections
    timestamp_array =[]
    
    # Create empty ScoreArr
    ScoreArr = np.array([])
    counter = 0
    SpectroList = []
    file_array = []
    previous_channels = stream.stream.channels
    current_channels = stream.stream.channels
    
    # Stream through the files, preprocess and apply the detector
    #previous_date = stream.get_current_timesamp().date()
    current_date = stream.get_current_timesamp().date()

    # path to save and load spectrogram for reuse
    SelTabPathDetected = os.path.normpath(SelTabPathDetected)
    # two levels up from RunX: the model cv folder
    #path_to_spectro_file = os.path.dirname(os.path.dirname(SelTabPathDetected))
    # __ExptResult
    path_to_spectro_file = os.path.dirname(os.path.dirname(os.path.dirname(SelTabPathDetected)))
    if config.TEST_MODE is True:
        spectro_file_curr_day = os.path.join(path_to_spectro_file, 'TEST_'+str(current_date))+'.npz'
    else:
        spectro_file_curr_day = os.path.join(path_to_spectro_file, str(current_date))+'.npz'

    if config.USE_SAVED_FEATURE and os.path.exists(spectro_file_curr_day):
        print('Load the saved feature files...')
        stored_data = np.load(spectro_file_curr_day)
        SpectroList = stored_data['arr_0']
        file_array = stored_data['arr_1']
        timestamp_array = stored_data['arr_2']
        #SpectroList, file_array, timestamp_array 
        #np.savez(spectro_file_curr_day, SpectroList, file_array, timestamp_array)
        
        print('Make prediction...')
        ScoreArr = make_predictions(model,
                                                     SpectroList, 
                                                     previous_channels,
                                                     config,
                                                     streamevent = True
                                                     )
        ScoreArr = ScoreArr.reshape(-1, previous_channels)
        ScoreArr = ScoreArr[:-1].reshape(-1, previous_channels)
        
        print('Make dataframe for selection tables')
        Raven_data_frame = make_raven_SndSel_table( 
                        ScoreArr, 
                        OverlapRatioThre,
                        file_array,
                        timestamp_array,
                        ScoreThre,
                        config,
                        date_format = date_format)
    
        #RavenTable_out = RavenTable_out.append(Raven_data_frame)
        
        print('Write to the selection table')
        #print('writing selection table ' + current_file)
        write_raven_table_to_CSV(Raven_data_frame,
                                             SelTabPathDetected,
                                             SelectionTableName,
                                             SNDSel= SNDSel)                    
        
    else:
        print('Extract features from sound files...')
        sec_advance = float((N_read+N_previous)/stream.get_all_stream_fs()[0])
        while True:                                
            try:
                # load the samples
                samps = stream.read(N = N_read, previousN=N_previous)[0]
                #advance the counter
                counter +=1

                # Append the timestamp 
                #timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=2.0)
                timestamp = stream.get_current_timesamp() - datetime.timedelta(seconds=sec_advance)
                
                # Set the current file
                current_file = stream.stream.filename
                            
                # current date
                #current_date = stream.get_current_timesamp().date()
        
                # Append the timestamp 
                timestamp_array.append(timestamp)
                
                # Set the current file
                file_array.append(current_file)                    
                    
                #pre-process all the channels
                for ii in range(samps.shape[1]):
                    #Spectro = preprocess(samps[:,ii], filter_args)
                    Spectro = preprocess(samps[:,ii], config)
                    SpectroList.append([Spectro])
                #SpectroList.append([Spectro]) # this seems weird: only save the feature of the last channel
    
                if counter == config.MAX_STREAMS:
                    preds = make_predictions(model, 
                                             SpectroList,
                                             current_channels,
                                             config,
                                             streamevent = False
                                             )                
                    # make model predictions
                    ScoreArr = np.append(ScoreArr, preds)
                    #if bool(len(ScoreArr) % current_channels):
                        #print('blarg')
    
                    # reset the list
                    SpectroList = []
                    counter = 0
    
            except (StreamGap, StreamEnd) as S:
                
                # If error was thrown after a read (as opposed to after writing
                # a new table, then read the samples and predict)          
                #ScoreArr = np.append(ScoreArr, 
                print('Make prediction')
                preds = make_predictions(model,
                                             SpectroList, 
                                             previous_channels,
                                             config,
                                             streamevent = True
                                             )
                ScoreArr = np.append(ScoreArr, preds)
                ScoreArr = ScoreArr.reshape(-1, previous_channels)
                #ScoreArr = ScoreArr[:-1].reshape(-1, previous_channels)
                
                # save SpectroList
                if config.USE_SAVED_FEATURE is True:
                    print('Save features to numpy binary files...')
                    np.savez(spectro_file_curr_day, SpectroList, file_array, timestamp_array)
                          
                # End of Stream write the dataframe
                if type(S).__name__ == 'StreamEnd' :
                    # Make the raven table
                    print('Make dataframe for selection tables')
                    Raven_data_frame = make_raven_SndSel_table( 
                        ScoreArr, 
                        OverlapRatioThre,
                        file_array,
                        timestamp_array,
                        ScoreThre,
                        config,
                        date_format = date_format)
    
                    #RavenTable_out = RavenTable_out.append(Raven_data_frame)
                    
                    #print('writing selection table ' + current_file)
                    print('Write to the selection table')
                    write_raven_table_to_CSV(Raven_data_frame,
                                             SelTabPathDetected,
                                             SelectionTableName,
                                             SNDSel= SNDSel)                    
                    return False
            
                elif type(S).__name__ == 'StreamGap':
                    # reset the list
                    SpectroList = []
                    file_array = []
                    timestamp_array =[]
                    counter = 0
                    ScoreArr = []               
                    #print('I passed a stream gap!!')
                    continue

if __name__ == "__main__":
        
    # =============================================================================
    # User Defined
    # =============================================================================
    from UpcallExptV2_new import  F1_Class
    
    ### Load the Model and list output location  ##########################
    # Load upcall model
    model_loc = '/home/kpalmer/AnacondaProjects/upcall-basic-net/DetectorOutputValidation/Models'
    model_loc = 'C:\\Users\\Kaitlin\\Desktop\\NOAA SW Data\\Model'
    model_loc = '/home/kpalmer/Desktop/dsp test folder'

    #model_name = 'NARW_LeNet_V0_dropout.h5'
    model_name = 'Trained Model For DSP Test.hdf5'
    full_mod_loc = model_loc + '/' + model_name
    #model= load_model(full_mod_loc)
    model = load_model(full_mod_loc, custom_objects={'F1_Class': F1_Class}) 

    ######################################################################
    
    
##    ##### Input/output locations #####################################
#    DaySoundPath = ['/cache/kpalmer/quick_ssd/data/Corrnell Data/'+\
#                    'NARW_analyst_handbrowsed_truth_set/'+\
#                    '__Sound_BOEM_VA_Historical']
#
#
#    DaySoundPath = ['D:\\data\\DebugFile']
#    DaySoundPath = ['D:\\data\\dclmmpa2013\\Testing\\Upcalls_NOPPset2']
#
#    file_days = os.listdir(DaySoundPath[0])
#    file_days = [DaySoundPath[0] + '/' + s for s in file_days]

#    
    file_days = ['/cache/kpalmer/quick_ssd/data/TempForTesting']
    
    
#    file_days = os.listdir(DaySoundPath[0])
#    file_days = [DaySoundPath[0] + '/' + s for s in file_days]

    # Create the stream of files
    stream = make_sound_stream(file_days)
#    stream = MakeSoundStream(DaySoundPath)
    
    # Define where the output txt files go
    SelectionTableDir = '/home/kpalmer/Desktop/dsp test folder/Scratch '+\
        'SelectionTables'
            
    Project_alias = 'temp/'
    SelTabPathDetected = SelectionTableDir      
    ####################################################################
    
    

    
    #### Define Preprocessing Features and spectrogram advance ###############
    # Pick fft lenght and window length
    FFTSize = 256
    HopSize = 100
    fs =2000
    
    # preprocess arguments
    filter_args = {'FFTSize': FFTSize,
                   'HopSize': HopSize,
                   'fs': fs,
                   'filter_fx' : 'ys_Preprocess'}
    
    # Advance the streamer 0.1 second at a time
    FrameStepSec = .1 
    FrameStep = int(FrameStepSec*fs)
    FrameSizeSec = 2.0 # each window is 2 sec long
    FrameSize = int(FrameSizeSec*fs)
    
    N_read = FrameStep
    N_previous = FrameSize -N_read
    

    ###########################################
    
    

    
    #Run the detector 
    RunDetector(stream,
                model,
                filter_args,
                SelTabPathDetected,
                N_read, N_previous,
                date_format = "%Y%m%d_%H%M%S",
                SelectionTableName = 'DCLDEtests',
                SNDSel = False,
                ScoreThre = 0.2,
                max_streams = 200,
                OverlapRatioThre = 0.5)
    

    
    
#
#
#
#
#
#
#
#
#
#
#
#
