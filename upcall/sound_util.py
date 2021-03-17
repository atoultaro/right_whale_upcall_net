# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:49:45 2018

@author: ys587
"""
import librosa
import numpy as np
import os
import soundfile as sf
from shutil import copyfile
import random


def augment_data(spectro_list, label, config):
    '''
    Augment the data using roll functions
    spectro_list - list of spectrograms to roll
    label - list of lable values (typically 0/1)
    feature_shape - shape of the spectrogram
    augmentation_args - dictionary containing augmentation arguments
    
    '''
    # Number of images to add
    N_augment = int(config.AUGMENT_ARGS['proportion']*len(spectro_list))    
    
    # index of images to use
    idx = np.random.choice(range(len(spectro_list)), N_augment, replace = False)
    
    # Spectros to augment
    augment_spectros = np.array(spectro_list)
    augment_spectros = augment_spectros[idx]
    
    for jj in range(len(idx)):
        spectro = augment_spectros[jj,:,:,:].squeeze()   
        # roll along each specificed axis
        #for ii in range(len(config.AUGMENT_ARGS['dims'])):
        for ii in config.AUGMENT_ARGS['dims']:
            # number of spaces to roll
            N_roll_max = int(spectro.shape[ii] *
                             config.AUGMENT_ARGS['prop_of_image'][ii])
            # Roll the spectrogram
            #spectro  = np.roll(spectro,  np.random.randint(1, N_roll_max,
            #                    config.AUGMENT_ARGS['dims'][ii]))
            spectro  = np.roll(spectro,  np.random.randint(1, N_roll_max),
                                config.AUGMENT_ARGS['dims'][ii])
        augment_spectros[jj,:,:,0] = spectro

    try:
        augment_labels = np.array(label)[idx]
    except:
        print()
        
    return augment_spectros, augment_labels


#def preprocess(sample_clip, config):
#    """ 
#    Convert sound waveform into spectrogram & normalize spectrum into a 
#    unit-length vector
#    
#    Args:
#        sample_clip: the input sound in samples
#        config: configuration class object
#    Returns:
#        spectrogram, where each spectrum is unit-length vector
#    """
#    # Short-time Fourier transform
#    sftf_mat = np.abs(librosa.stft(sample_clip, n_fft=config.FFT_SIZE, 
#                                  hop_length=config.HOP_SIZE))
#    spectro = sftf_mat[config.IMG_X_START:(config.IMG_X_START+config.IMG_X), 
#                      config.IMG_Y_START:(config.IMG_Y_START+config.IMG_Y)].flatten()
#
#    # normalize the spectrum into unit vector
#    vec_len = np.sqrt(np.sum(spectro**2.)) # length of vector "spectro"
#    flatten_spectro = spectro/vec_len if vec_len else np.zeros(spectro.shape)
#    # set to zero if Veclen is zero
#
#    return flatten_spectro

def preprocess(sample_clip, config):
    """ 
    Convert sound waveform into spectrogram & normalize spectrum into a 
    unit-length vector
    
    Args:
        sample_clip: the input sound in samples
        config: configuration class object
    Returns:
        spectrogram, where each spectrum is unit-length vector
    """
    spectro = wav_to_spectrogram(sample_clip, config)
    flatten_spectro = mag_normalized(spectro, config)
    
    return flatten_spectro    
    
def wav_to_spectrogram(sample_clip, config):
    """ Convert sound waveform into spectrogram
    Args:
        sample_clip: the input sound in samples
        config: configuration class object
    Returns:
        spectrogram, of the original magnitude without normalization
    """
    sftf_mat = np.abs(librosa.stft(sample_clip, n_fft=config.FFT_SIZE, 
                                  hop_length=config.HOP_SIZE))
    spectro = sftf_mat[config.IMG_F_START:(config.IMG_F_START+config.IMG_F), 
                      config.IMG_T_START:(config.IMG_T_START+config.IMG_T)].T.flatten()
    # flatten default's is over row-major, i.e., 
    return spectro

def mag_normalized(spectro_flat, config):
    """ normalize each spectrum of a spectrogram into unit-length
    Args:
        spectro_flat: flatten spectrogram
    Return:
        flatten_spectro: spectrogram where each spectrum is normalized into 
        unit length
    """
    # normalize the spectrum into unit vector
    vec_len = np.sqrt(np.sum(spectro_flat**2.)) # length of vector "spectro"
    flatten_spectro = spectro_flat/vec_len if vec_len else np.zeros(spectro_flat.shape)
    # set to zero if Veclen is zero

    return flatten_spectro
    
def ind2sound(ind_set, sound_path, sound_path_out):
    """
    Copy sound files of training sound clips to a new folder by given 
    indices
    
    Args:
        IndSet: index set of selected sound clips
        sound_path: sound folder to read from
        sound_path_out: sound folder to write to
    Returns:
        True for completing the function
    """
    if not os.path.exists(sound_path_out):
        os.mkdir(sound_path_out)
    for ii in ind_set:
        sound_file = sound_path+'/train'+str(ii+1)+'.aiff'
        sound_file_out = sound_path_out+'/train'+str(ii+1)+'.aiff'
        copyfile(sound_file, sound_file_out)

def sound2fea(sound_path, num_data, config):
    """
    Apply preprocess/feature extraction for Kaggle training data
    
    Args:
        sound_path: path the sound folder
        num_data: number of sound clips used    
        config: configuration parameters
    Returns:
        FeaSpectro (numpy array): extracted features
    
    """
    spectro_list = []
    for ii in range(num_data):
        sound_file = os.path.join(sound_path,'train'+str(ii+1)+'.aiff')
        if os.path.isfile(sound_file) == False:
            print(sound_file)
    
        # read sound
        samples0, SampleRate = sf.read(sound_file)
        spectro = preprocess(samples0, config)
        spectro_list.append(spectro)

    fea_spectro = np.vstack(spectro_list) 
    # DataDimen x DataNum: 1,600 x 30,000? Or 30,000 x 1,600?

    return fea_spectro
    
def sound2fea_clips(sound_path, sampling_ratio, config):
    """
    Apply preprocess/feature extraction 
    
    Args:
        sound_path: path the sound folder
        sampling_ratio: the ratio of sampling sound clips to be used in 
                       the training process
        config: configuration parameters
    Returns:
        FeaSpectro (numpy array): extracted features 
    """
    spectro_list = []
    sound_list = os.listdir(sound_path)
    sound_list = sorted(sound_list)
    if sampling_ratio != 1.0:
        random.seed(config.SOUND_DATA_SEED)
        sound_list = random.sample(sound_list, int(len(sound_list)*sampling_ratio)) 
    for sound_file0 in sound_list:
        sound_file = os.path.join(sound_path, sound_file0)
        if os.path.isfile(sound_file) == False:
            print(sound_file)
    
        samples0, SampleRate = sf.read(sound_file)    
        spectro = preprocess(samples0, config)
        spectro_list.append(spectro)

    fea_spectro = np.vstack(spectro_list)

    return fea_spectro