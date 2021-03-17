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
import regex as re
import datetime
import soundfile as sf
import timeit
import pandas as pd

from keras.models import load_model
from upcall.sound_util import preprocess
from upcall.train_classifier import F1_Class

from dsp.SampleStream import SampleStream, Streams

# from upcall.FilterList import bird_net_light_filter,bird_net_filter, \
#     med2d_with_cropping, null_filter, ys_Preprocess


def get_time_stamp(TheString, TIME_FORMAT):
    """ Retrieve the time information from filename of sound files.
    
    Args:
        TheString: base part of filename (extension is not included)
        TIME_FORMAT: target time format to be retrieved.
    
    Return: 
        datetime format of time informtions
    """
    m = re.search("(\d{8}_\d{6})", TheString) # YYYYMMDD_HHMMSS
    return datetime.datetime.strptime(m.groups()[0], TIME_FORMAT)

def get_start_timestamp(f, format_str =  "%Y%m%d_%H%M%S"):
    ''' returns a datetime object (start time) given a soundfile/directory
    name in the standard Cornell format
    '''
    
    fname= os.path.split(f)[1]
    
    match_date = re.search(r'\d{8}_\d{6}', fname)
    try:
        # Cornell Format
        start_time = datetime.datetime.strptime(match_date.group(), format_str)
    except AttributeError:
        # shit, not cornell format try scripps
         match_date = re.search(r'\d{6}_\d{6}', fname)
         try:
             start_time = datetime.datetime.strptime(match_date.group(), format_str)
         except AttributeError:
             # And if he still doesn't answer? Say, NOAA SW format, 
             # my sweet NOAAA you're the one...
             start_time = datetime.datetime.strptime(fname[-16:-4], format_str)
                 
    return(start_time)
    
def non_max_suppress(boxes_in_batch, score_arr, config):
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
 
    DaySoundPath = [day_sound_path]
    # Declare the stream
    stream_elements = Streams()
    
    for ii in range(len(day_sound_path)):
        print('ii: '+str(ii))
        # get the file director
        file_dir = day_sound_path[ii]
        

        # Iterate through the folders and extract associated
        for filename in os.listdir(file_dir):
            
            # if soundfile add it to the stream
            if filename.endswith(".wav") or filename.endswith(".aif"): 
                #start_time = timeit.timeit()
                
                sound_fullfile = file_dir + '/' + filename
                start = get_start_timestamp(filename, format_str)
                try:
                    aa = sf.SoundFile(sound_fullfile)
                    stream_elements.add_file(sound_fullfile, [len(aa)], [start]
                        , aa.samplerate)
                except TypeError:
                    raise Exception('Incorrect path or file format.')
            else:
                continue

    # Combine streams into a sample stream
    stream = SampleStream(stream_elements)
        
    return stream


def MakeSoundStream(day_sound_path, format_str = "%Y%m%d_%H%M%S"):
    """
    Function for making a soundstream capable of iterating through lots of
    files.

    Input:
        day_sound_path - List of folder location(s) containing soundfiles
        format_str - format string of the date default is"%Y%m%d_%H%M%S" 
            for Cornell
    Returns
        Returns a soundstrem of all wav or aif files listed in the folder 
    """
    # Declare the stream
    stream_elements = Streams()
    
    for ii in range(len(day_sound_path)):
        print('ii: '+str(ii))
        # get the file director
        file_dir = day_sound_path[ii]
        

        # Iterate through the folders and extract associated
        for filename in os.listdir(file_dir):
            
            # if soundfile add it to the stream
            if filename.endswith(".wav") or filename.endswith(".aif"): 
                #start_time = timeit.timeit()
                
                sound_fullfile = file_dir + '/' + filename
                start = get_start_timestamp(filename, format_str)
                aa = SoundFile(sound_fullfile)
                stream_elements.add_file(sound_fullfile, [len(aa)], 
                                         [start], aa.samplerate)

                #end_time = timeit.timeit()
                #print(start)
                #print('Duration: '+str(end_time-start_time))
                # print(os.path.join(directory, filename)) # debugging
            else:
                continue

    # Combine streams into a sample stream
    stream = SampleStream(stream_elements)
        
    return stream


def Make_raven_SndSel_table(n_channels, 
                            score_arr, 
                            OverlapRatioThre,
                            file_array,
                            timestamp_array,
                            ScoreThre,
                            date_format = "%Y%m%d_%H%M%S"):
    """
    Function for creating a sound selection table
    
    inputs:
        n_channels - number of channels
        idx - I forget from YS code 
        score_arr - numpy array of scores 
        OverlapRatioThres- float, between 0 and 1 throw out adjacent
            events closer than t
        file_array - array of file names
    """
    RavenTable_out = pd.DataFrame()
    EventId = 0

    for ii in range(n_channels):
        # select those greater than score
        idx = np.nonzero(score_arr[:,ii] > ScoreThre)[0]
        timestamp_array_sub= np.array(timestamp_array)[idx]
        file_array_sub = np.array(file_array)[idx]
        EventId = len(RavenTable_out)+1
        
        if idx.shape[0] != 0:
            # Run YS non-max supression
            CallArrSepa, begin_time_s, out_files = \
                            NonMaxSuppressTopOne(idx, 
                            score_arr[:,ii], 
                            FrameSize, 
                            FrameStepSec,
                            OverlapRatioThre,
                            FrameSizeSec,
                            timestamp_array_sub,
                            file_array = file_array_sub)
            
            # Run non-max supression                           
#            CallArrSepa, begin_time_s, out_files = \
#                non_max_suppression_fast(idx, 
#                                         timestamp_array_sub,
#                                         OverlapRatioThre,
#                                         file_array = file_array_sub)
        
            EventIDs = EventId + np.arange(len(CallArrSepa))
            
            # Calculate the begin time in s into day and file
            begin_time_s = pd.to_datetime(begin_time_s) -pd.Timedelta(seconds = 2)
            file_offset_sec =[]
            file_start_name = []

            for jj in range(len(out_files)):
                file_start = get_start_timestamp(out_files[jj],
                                                 date_format)
                file_offset = begin_time_s[jj]- file_start
                file_offset_sec.append(file_offset.total_seconds())
                file_start_name.append(os.path.basename(out_files[jj]))

            sec_start = begin_time_s.hour*60*60+  \
                    begin_time_s.minute*60+  \
                    begin_time_s.second + \
                    begin_time_s.microsecond/10**6
            
            # Data output for Raven Sound Selection table
            data = {'Selection': EventIDs,
                    'View':'Spectrogram 1',
                    'Channel': ii+1,
                    'Begin Time (s)': sec_start,
                    'End Time (s)':sec_start+2,
                    'Low Freq (Hz)':50.0,
                    'High Freq (Hz)':350.0,
                    'Begin Date':begin_time_s.strftime("%Y/%m/%d"),
                    'Begin Hour':begin_time_s.hour,
                    'Begin Path':out_files,
                    'Begin File': file_start_name,   
                    'File Offset (s)':  file_offset_sec,  
                    'Score':score_arr[CallArrSepa, ii-1],
                    'timestamp':begin_time_s,
                    'Detection threshold': ScoreThre
                    }
            
            frames = [RavenTable_out, pd.DataFrame(data)]
            RavenTable_out = pd.concat(frames)
            EventId = max(EventIDs)

        # Change order
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
    # RavenTable_out= RavenTable_out.sort_values(by=['Begin Time (s)'])

    return RavenTable_out


def write_raven_table_to_CSV(PandasDataFrame, SelTabPathDetected,
                             SelectionTableName, SNDSel= True):
    """
    Write the datafram to a txt file
    
    Inputs:
        PandasDataFrame - pandas dataframe with approperiate columns
        SelTabPathDetected - output path
        SelectionTableName - output name
        
    Returns:
        Nothing- writes dataframe to specified location
    """
    # if not sound selection table then sound table, remove the sound selection
    # coulmns
    if not SNDSel:
        PandasDataFrame = PandasDataFrame.drop([ 'Begin File', 
                                               'File Offset (s)',
                                               'Begin Path'], axis=1)

    # sort by start time
    PandasDataFrame = PandasDataFrame.sort_values(by=['Begin Time (s)'])
    
    # sort
    PandasDataFrame['Selection'] = np.arange(1, len(PandasDataFrame)+1)
    PandasDataFrame = PandasDataFrame.reset_index(drop=True) # reset index
    
    # make the file loc
    DayFile = os.path.join(SelTabPathDetected + '/'+ 
                           SelectionTableName + '_' +
                           str(PandasDataFrame['timestamp'].iloc[0].strftime('%Y%m%d')) +'.txt')
    
    # export
    PandasDataFrame.to_csv(DayFile,
                          sep='\t', mode='a',
                          index = False)    


def run_detector_days_dsp(day_sound_path_list, seltab_out_path, 
                          classifier_model_path, config):
    """ run detector on one day of sound; the main detection running engine
    
    Args:
        day_sound_path: path of a deployment sound folder
        seltab_out_path: directory where the selection tables will be output
        classifier_model_path: the path to the trained classifier model. Could 
        be a list instead for ensemble classifiers
        config: parameter configuration
    """
    classifier_model = load_model(classifier_model_path, custom_objects=\
        {'F1_Class': F1_Class}) # Temporary for F1_Class metric

    stream = MakeSoundStream(day_sound_path_list)

    IS_MAX_STREAMS = False
    previous_channels = stream.stream.channels
    previous_date = stream.get_current_timesamp().date()
    score_arr = np.array([])
    fea_spectro = []
    count_stream = 0

    while True:
        try:
            # Append the timestamp 
            timestamp = stream.get_current_timesamp()            
            #print(timestamp)
            
            # Set the current file
            current_file = stream.stream.filename
            # number of channels in the new file
            current_channels = stream.stream.channels
            # current date
            current_date = stream.get_current_timesamp().date()
            
            # load the samples
            samps = stream.read(N=config.N_READ, previousN=config.N_PREV)[0]        
            
            if (current_channels != previous_channels) or ( current_date != 
            previous_date): # find a new channel of new date
                # Write the previous raven table
               # Yu: for each new channel, generate sound selectio tables as well?
               # should remove the condition for channels part
                print('New Channel Format or sound selection for day writing table')
                
                preds = classifier_model.predict(fea_spectro)[:,1]
                #preds = make_predictions(model, fea_spectro, current_channels,streamevent = False)
                score_arr = np.append(score_arr, preds)
                score_arr = score_arr.reshape(-1, previous_channels)

                # Get the quantile threshold values
                if config.ScoreThre is None:
                    aa = pd.Series(score_arr.flatten())
                    thresholds = aa.quantile(np.linspace(min(aa), max(aa), 20))
                    del(aa)
                else:
                    thresholds = [config.ScoreThre]
                
                # Create selection tables for every threshold Yu: why do we need it?
                for threshold in thresholds:
                    print(threshold)                    

                    Raven_data_frame = Make_raven_SndSel_table(current_channels, 
                            score_arr, 
                            OverlapRatioThre,
                            file_array,
                            timestamp_array,
                            threshold,
                            date_format = date_format)
                    
                    RavenTable_out = RavenTable_out.append(Raven_data_frame)
                    
                    #SelectionTableName_out = SelectionTableName + \
                    #                    str(previous_channels) + \
                    #                    ''+ str(counter)
                    SelectionTableName_out = SelectionTableName
                    
                    # Export the raven table
                    write_raven_table_to_CSV(RavenTable_out,
                                             SelTabPathDetected,
                                             SelectionTableName_out, 
                                             SNDSel= SNDSel)
                
                    # Reset everything
                    RavenTable_out  = pd.DataFrame()                  
                    score_arr = []
                    SpectroList = [] 
                    timestamp_array =[]
                    file_array = []
                    previous_channels = current_channels
                    previous_date = current_date
                    
                    # Bool telling streamer not to write on the stream gap if
                    # it's thrown later on. There has got to be a nicer way to
                    # do this
                    IS_MAX_STREAMS = True           
            else:
                #advance the counter
                count_stream +=1
                # Append the timestamp 
                timestamp_array.append(timestamp)
                
                # Set the current file
                file_array.append(current_file)                    
                    
                #pre-process all the channels
                for cc in range(samps.shape[1]):
                    #Spectro = Preprocess(samps[:,cc], config)
                    Spectro = preprocess(samps[:,cc], config)
                    fea_spectro.append([Spectro])
                        
                # If iterated the maximum number of times, make the predicitons and clear the list
                if count_stream == config.MAX_STREAMS: # does it mean we will not predictions on those at the end of the file when count_stream hasnt achieved max_streams?
                    preds = classifier_model.predict(fea_spectro)[:,1]
                    #preds = make_predictions(model, fea_spectro, current_channels, streamevent = False)
                    
                    # make model predictions
                    score_arr = np.append(score_arr, preds)

                    # reset the list
                    fea_spectro = []
                    count_stream = 0
                    
                previous_channels = current_channels
                previous_date = current_date
                
                # On stream gap, do the normal stuff
                IS_MAX_STREAMS = False
        except Exception as S:
            
            if IS_MAX_STREAMS is not True:
                preds = classifier_model.predict(fea_spectro)[:,1]

                score_arr = np.append(score_arr, pred)
                score_arr = score_arr.reshape(-1, current_channels)
                
                # Make the raven table
                Raven_data_frame = Make_raven_SndSel_table(current_channels, 
                        score_arr, 
                        OverlapRatioThre,
                        file_array,
                        timestamp_array,
                        config.ScoreThre,
                        date_format = date_format)
                
                RavenTable_out = RavenTable_out.append(Raven_data_frame)
                
            # End of Stream write the dataframe
            if type(S).__name__ == 'StreamEnd' :
                if IS_MAX_STREAMS is not True:
            
                    score_arr = np.append(score_arr, 
                                    make_predictions(model, SpectroList,
                                                     current_channels,
                                                     streamevent = True))
                    # score_arr is not used in write_raven_table_to_CSV?

                    
                    write_raven_table_to_CSV(RavenTable_out,
                                             SelTabPathDetected,
                         SelectionTableName+'', SNDSel= SNDSel)
                    # RavenTable_out at this point only includes detection threshold .2
                
                return False
            
            elif type(S).__name__ == 'StreamGap':
                                        
                # reset the list
                SpectroList = []
                file_array = []
                timestamp_array =[]
                counter = 0
                iter_counter +=1
                score_arr = []               
                IS_MAX_STREAMS = False
                print('I passed a stream gap!!')
            else:
                continue # Yu: or break?               
                    
                    
                    #write_raven_table_to_CSV(RavenTable_out,
                    #                         SelTabPathDetected,
                    #     SelectionTableName+'', SNDSel= SNDSel)
                    # RavenTable_out at this point only includes detection threshold .2
                
                # return False


def Preprocess(SampleClip, filter_args = None):
    
    ''' Pre-process the sample clip by creating an spcetrogram, clipping the
    image and applying a image filter
    
    inputs:
        SampleClip - sound data (from sound read or similar)
        filter_args - dictionary containing the following keys
            FFTSize - fft size in samples (sorry Marie!)
            HopSize - hop size in samples (using YS code)
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
    
    if filter_args is None:
                # Define filter/preprocess arguments
        filter_args = {'FFTSize': 256,
                       'HopSize': 100,
                       'fs': 2000,
                       'center_fft' : False,
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
    

def run_detector_days(day_sound_path_list, seltab_out_path, 
                      classifier_model_path, config):
    """ interface of runing detector on multiple days of sound
    Args:
        day_sound_path_list: list of path of day sound folders
        seltab_out_path: directory where the selection tables will be output
        classifier_model_path: the path to the trained classifier model
        config: parameter configuration
    """
    start_time = timeit.default_timer()
    for dd in day_sound_path_list:
        print('\n'+dd)
        #run_detector(dd, seltab_out_path, classifier_model_path, config)
        run_detector(dd, seltab_out_path, classifier_model_path, config)
        
    stop_time = timeit.default_timer()
    print('Runtime: '+str.format("{0:=.4f}", stop_time - start_time)+' Sec')
                     
def run_detector(day_sound_path, seltab_out_path, classifier_model_path, 
                 config):
    """ run detector on one day of sound; the main detection running engine
    
    Args:
        day_sound_path: path of a day sound folder
        seltab_out_path: directory where the selection tables will be output
        classifier_model_path: the path to the trained classifier model. Could 
        be a list instead for ensemble classifiers
        config: parameter configuration
    """
    if isinstance(classifier_model_path, str): # check if classifier_model_path 
    #is list or tuple
        MultiModel = False
    else:
        MultiModel = True
        
    if MultiModel == False:
        classifier_model = load_model(classifier_model_path, custom_objects={'F1_Class': F1_Class}) # Temporary for F1_Class metric
    else:# load multiple models into a list
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
        ff2 = os.path.splitext(os.path.basename(ff))[0]
        print (ff2)                    
        # time stamp
        time_curr = get_time_stamp(ff2, config.TIME_FORMAT)
        samples0, sample_rate = sf.read(ff)
                
        if samples0.ndim==1: # make it a multi-dimensional array for single channel
            samples0 = np.reshape(samples0, (-1, 1))
        num_sample, num_chan = samples0.shape[0], samples0.shape[1]
        
        for cc in range(num_chan):
            ss_last = int(np.floor((num_sample-config.FRAME_SIZE)*1.0/
                config.FRAME_STEP))
            print('#',end='')
            
            # make prediction for each 15-min file or for each sliding window. The former is 3 times faster and the latter will be eliminated later.
            spectro_list = []
            for ss in range(ss_last):
                samples = samples0[ss*config.FRAME_STEP:ss*config.FRAME_STEP+
                    config.FRAME_SIZE,cc]
                spectro = preprocess(samples, config)
                    
                spectro_list.append(spectro)
            
            fea_spectro = np.vstack(spectro_list)            
            fea_spectro = fea_spectro.reshape(fea_spectro.shape[0], 
                                              config.IMG_X, config.IMG_Y, 1)
            #fea_spectro = fea_spectro.reshape(fea_spectro.shape[0], img_y, img_x) # RNN: x: frequency; y: time

            if MultiModel == False:
                score_arr = classifier_model.predict(fea_spectro)[:,1]
            else:
                score_arr_list = [cc.predict(fea_spectro)[:,1] for cc in 
                    classifier_modelList]
                score_arr_sum = score_arr_list[0]
                for ss in range(1, len(score_arr_list)):
                    score_arr_sum += score_arr_list[ss]
                
                score_arr = score_arr_sum/float(len(score_arr_list))
            
            call_arr = np.where(score_arr > config.SCORE_THRE)[0]
            if call_arr.shape[0] != 0: # there's at least one window with score larger than the threshold
                # merging multi detection boxes / non-maximum suppresion
                #call_arr_sepa = non_max_suppress(call_arr, score_arr, FrameSize, FrameStepSec, OverlapRatio) # score & indices. 0.5 the ovelap
                call_arr_sepa = non_max_suppress(call_arr, score_arr, config)
                print('==>> ',end='') #!!! move Non-max suppresion to the last stage using the score threshold set
                print(call_arr_sepa)

                for jj in call_arr_sepa:
                    EventId += 1
                    # Raven selection table format
                    Time1 = time_curr.hour*3600.0 + time_curr.minute*60.0 + \
                        time_curr.second + jj*config.FRAME_STEP_SEC
                    Time2 = Time1 + config.FRAME_SIZE_SEC
                    print('Found event: '+ str(EventId)+ ' Time1: '+
                        str.format("{0:=.4f}",Time1)+' Score: '+
                        str(score_arr[jj]))
                    f.write(str(EventId)+'\t'+'Spectrogram'+'\t'+str(cc+1)+
                        '\t'+str.format("{0:=.4f}",Time1)+'\t'+
                        str.format("{0:<.4f}",Time2)+'\t'+
                        str.format("{0:=.1f}",config.BOX_OUT_F1)+'\t'+
                        str.format("{0:=.1f}",config.BOX_OUT_F2)+'\t'
                        +str.format("{0:<.5f}", score_arr[jj]) )
                    f.write('\n')
        print('')
    f.close()


def run_detector_days_dsp_speedy(day_sound_path_list, seltab_out_path, 
                          classifier_model_path, config):
    """ run detector on one day of sound; the main detection running engine
    
    Args:
        day_sound_path: path of a deployment sound folder
        seltab_out_path: directory where the selection tables will be output
        classifier_model_path: the path to the trained classifier model. Could 
        be a list instead for ensemble classifiers
        config: parameter configuration
    """
    classifier_model = load_model(classifier_model_path, custom_objects=\
        {'F1_Class': F1_Class}) # Temporary for F1_Class metric
#
#    DayFile = os.path.join(seltab_out_path+'/',os.path.basename(day_sound_path)+'.txt')
#    f = open(DayFile,'w')
#    f.write('Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tScore\n')
#
    stream = MakeSoundStream(day_sound_path_list)
############################################
    IS_MAX_STREAMS = False
    previous_channels = stream.stream.channels
    previous_date = stream.get_current_timesamp().date()
    score_arr = np.array([])
    fea_spectro = []
    count_stream = 0
    #EventId = 1
    while True:
        try:
            # Append the timestamp 
            timestamp = stream.get_current_timesamp()            
            #print(timestamp)
            
            # Set the current file
            current_file = stream.stream.filename
            # number of channels in the new file
            current_channels = stream.stream.channels
            # current date
            current_date = stream.get_current_timesamp().date()
            
            # load the samples
            samps = stream.read(N=config.N_READ, previousN=config.N_PREV)[0]        
            
            if (current_channels != previous_channels) or ( current_date != 
            previous_date): # find a new channel of new date
                # Write the previous raven table
               # Yu: for each new channel, generate sound selectio tables as well?
               # should remove the condition for channels part
                print('New Channel Format or sound selection for day writing table')
                
                preds = classifier_model.predict(fea_spectro)[:,1]
                #preds = make_predictions(model, fea_spectro, current_channels,streamevent = False)
                score_arr = np.append(score_arr, preds)
                score_arr = score_arr.reshape(-1, previous_channels)

                # Get the quantile threshold values
                if config.ScoreThre is None:
                    aa = pd.Series(score_arr.flatten())
                    thresholds = aa.quantile(np.linspace(min(aa), max(aa), 20))
                    del(aa)
                else:
                    thresholds = [config.ScoreThre]
                
                # Create selection tables for every threshold Yu: why do we need it?
                for threshold in thresholds:
                    print(threshold)                    

                    Raven_data_frame = Make_raven_SndSel_table(current_channels, 
                            score_arr, 
                            OverlapRatioThre,
                            file_array,
                            timestamp_array,
                            threshold,
                            date_format = date_format)
                    
                    RavenTable_out = RavenTable_out.append(Raven_data_frame)
                    
                    #SelectionTableName_out = SelectionTableName + \
                    #                    str(previous_channels) + \
                    #                    ''+ str(counter)
                    SelectionTableName_out = SelectionTableName
                    
                    # Export the raven table
                    write_raven_table_to_CSV(RavenTable_out,
                                             SelTabPathDetected,
                                             SelectionTableName_out, 
                                             SNDSel= SNDSel)
                
                    # Reset everything
                    RavenTable_out  = pd.DataFrame()                  
                    score_arr = []
                    SpectroList = [] 
                    timestamp_array =[]
                    file_array = []
                    previous_channels = current_channels
                    previous_date = current_date
                    
                    # Bool telling streamer not to write on the stream gap if
                    # it's thrown later on. There has got to be a nicer way to
                    # do this
                    IS_MAX_STREAMS = True           
            else:
                #advance the counter
                count_stream +=1
                # Append the timestamp 
                timestamp_array.append(timestamp)
                
                # Set the current file
                file_array.append(current_file)                    
                    
                #pre-process all the channels
                for cc in range(samps.shape[1]):
                    #Spectro = Preprocess(samps[:,cc], config)
                    Spectro = preprocess(samps[:,cc], config)
                    fea_spectro.append([Spectro])
                        
                # If iterated the maximum number of times, make the predicitons and clear the list
                if count_stream == config.MAX_STREAMS: # does it mean we will not predictions on those at the end of the file when count_stream hasnt achieved max_streams?
                    preds = classifier_model.predict(fea_spectro)[:,1]
                    #preds = make_predictions(model, fea_spectro, current_channels, streamevent = False)
                    
                    # make model predictions
                    score_arr = np.append(score_arr, preds)

                    # reset the list
                    fea_spectro = []
                    count_stream = 0
                    
                previous_channels = current_channels
                previous_date = current_date
                
                # On stream gap, do the normal stuff
                IS_MAX_STREAMS = False
        except Exception as S:
            
            if IS_MAX_STREAMS is not True:
                preds = classifier_model.predict(fea_spectro)[:,1]

                score_arr = np.append(score_arr, pred)
                score_arr = score_arr.reshape(-1, current_channels)
                
                # Make the raven table
                Raven_data_frame = Make_raven_SndSel_table(current_channels, 
                        score_arr, 
                        OverlapRatioThre,
                        file_array,
                        timestamp_array,
                        config.ScoreThre,
                        date_format = date_format)
                
                RavenTable_out = RavenTable_out.append(Raven_data_frame)
                
            # End of Stream write the dataframe
            if type(S).__name__ == 'StreamEnd' :
                if IS_MAX_STREAMS is not True:
                    score_arr = np.append(score_arr, 
                                    make_predictions(model, SpectroList,
                                                     current_channels,
                                                     streamevent = True))
                    # score_arr is not used in write_raven_table_to_CSV?
                    
                    write_raven_table_to_CSV(RavenTable_out,
                                             SelTabPathDetected,
                         SelectionTableName+'', SNDSel= SNDSel)
                    # RavenTable_out at this point only includes detection threshold .2
                
                return False
            
            elif type(S).__name__ == 'StreamGap':
                                        
                # reset the list
                SpectroList = []
                file_array = []
                timestamp_array =[]
                counter = 0
                iter_counter +=1
                score_arr = []               
                IS_MAX_STREAMS = False
                print('I passed a stream gap!!')
            else:
                continue # Yu: or break?

                return False

