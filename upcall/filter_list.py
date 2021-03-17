#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:54:52 2018

@author: kpalmer/Y.Shiu
"""
from scipy.ndimage.morphology import binary_closing, binary_dilation

from scipy.signal import medfilt2d, butter, lfilter
import numpy as np
#from simage.measure import label, regionprops
from skimage.measure import label, regionprops
from sklearn.preprocessing import normalize

import librosa


# Function for finding nearest neighbour
def nearest_to(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def normalize_filter(spectra):
    ''' Apply a normalisation filter to the spectra'''
    return normalize(spectra)
    
def normalize_median_filter(spectra, kernel_size = 3):
    ''' Apply a normalisation filter to the spectra then median2d filtering'''
    
    spectra = normalize(spectra)
    spectra = medfilt2d(spectra)
    
    return spectra

def null_filter(spectra, f_ind = None):
    ''' Null filter with clropping, returns self if no arguments supplied
    and self within the frequency indexes if f_inds are supplied where f_inds
    are the indes of the low and high frequencies in the spectra to retain'''
    if f_ind is not None:
        spectra = spectra[f_ind[0]:f_ind[1],:]
    return(spectra)
    

def med2d_with_cropping(spectra, f_ind = None, kernel_size = 3):
    ''' Median 2d filter with cropping, returns 2dmedian if no arguments
    supplied. If arguments are supplied then it cropps THEN applies the median2d
    filter.'''
    spectra = np.real(spectra)
    if f_ind is not None:
        spectra = spectra[f_ind[0]:f_ind[1],:]
        spectra = medfilt2d(spectra)
        
    else:
        spectra = medfilt2d(spectra)
            
    return(spectra)
      
    

def bird_net_light_filter(spectra, f_ind = None, area_threshold = 5):
    ''' Series of pre-proecessing filters  based on 
    http://www.animalsoundarchive.org/RefSys/Nips4b2013NotesAndSourceCode/ 
    WorkingNotes_Mario.pdf
    
    Input: 
        spectra - magnitude squared spectra
        f_idx1 - index of the lowest frequency to retain
        f_idx2 - index of the highest frequency to retain
    
    1) Normalise
    2) Brute band pass filter 
    3) Set pixles to 1 if >3x median rows AND columns
    4) closing
    5) Dialation
    5) Median filtering
    6) Remove non-connnected pixels at a given threshold

    '''
    
    flag = False
    if len(spectra.shape)==2:
        flag = True
        spectra = np.reshape(spectra, (1, spectra.shape[0], spectra.shape[1]))
        
    
    # 1) normalize 
    spectra = np.real(spectra)
    sums = np.sum(np.sum(spectra, axis = 1), axis=1)
    dvision = spectra/ np.reshape(sums, [spectra.shape[0],1,1])
    spectra = np.sqrt(dvision)
    
    
    
    #2) Clip lowest and highest bins
    if f_ind is not None:
        spectra = spectra[:,:, f_ind[0]:f_ind[1]]
    
    for ii in range(spectra.shape[0]):

        spec = spectra[ii,:,:]
        
        #3) set values greter than 3x the median to 1 and others to 0
        med_threshold = max(max(np.median(spec, axis=0)),
                        max(np.median(spec, axis=1)))
        
        spec[spec > med_threshold] = 1
        spec[spec <= med_threshold] = 0

        #7) Threshold remove pixels
        # 7a) Label regions
        spec = label(spec, connectivity=1)
        # 7b) get metrics for each region
        metrics = regionprops(spec)
        
        # 7c) remove the values less than image
        for m in metrics:
            if m.area < area_threshold:
                spec[spec == m.label] = 0

        
        #8) Set all labeled values back to 1
        spec[spec>1] = 1
        
        spectra[ii,:,:] = spec
    
    np.sum(np.sum(spectra,axis=1), axis = 1)    
    
    if flag is True:
        spectra = np.reshape(spectra, (spectra.shape[1], spectra.shape[2]))
    
    return  spectra
    
        


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bird_net_filter(spectra, f_ind = None, area_threshold = 5):
    ''' Series of pre-proecessing filters  based on 
    http://www.animalsoundarchive.org/RefSys/Nips4b2013NotesAndSourceCode/ 
    WorkingNotes_Mario.pdf
    
    Input: 
        spectra - magnitude squared spectra
        f_idx1 - index of the lowest frequency to retain
        f_idx2 - index of the highest frequency to retain
    
    1) Normalise
    2) Brute band pass filter 
    3) Set pixles to 1 if >3x median rows AND columns
    4) closing
    5) Dialation
    5) Median filtering
    6) Remove non-connnected pixels at a given threshold

    '''
    
    flag = False
    if len(spectra.shape)==2:
        flag = True
        spectra = np.reshape(spectra, (1, spectra.shape[0], spectra.shape[1]))
        
    
    # 1) normalize 
    spectra = np.real(spectra)
    sums = np.sum(np.sum(spectra, axis = 1), axis=1)
    dvision = spectra/ np.reshape(sums, [spectra.shape[0],1,1])
    spectra = np.sqrt(dvision)
    
    
    
    #2) Clip lowest and highest bins
    if f_ind is not None:
        spectra = spectra[:,:, f_ind[0]:f_ind[1]]
    
    for ii in range(spectra.shape[0]):

        spec = spectra[ii,:,:]
        
        #3) set values greter than 3x the median to 1 and others to 0
        med_threshold = max(max(np.median(spec, axis=0)),
                        max(np.median(spec, axis=1)))
        
        spec[spec > med_threshold] = 1
        spec[spec <= med_threshold] = 0
            
        #4) Closing
        spec = binary_closing(spec).astype(np.int)

        #5) Dialation
        spec = binary_dilation(spec).astype('float32')
        
        #6) Median filter
        spec = medfilt2d(spec).astype('float32')  
        
        #7) Threshold remove pixels
        # 7a) Label regions
        spec = label(spec, connectivity=1)
        # 7b) get metrics for each region
        metrics = regionprops(spec)
        
        # 7c) remove the values less than image
        for m in metrics:
            if m.area < area_threshold:
                spec[spec == m.label] = 0
                
            
        
        #8) Set all labeled values back to 1
        spec[spec>1] = 1
        
        spectra[ii,:,:] = spec
    
    np.sum(np.sum(spectra,axis=1), axis = 1)    
    
    if flag is True:
        spectra = np.reshape(spectra, (spectra.shape[1], spectra.shape[2]))

    return  spectra

# Yu Shiu Preprosess function
def ys_Preprocess(SampleClip, FMin=5, FMax=45, TMin=0, TMax=40):
    ''' preprocessing system used by basic upcall detector '''
    SftfMat = np.abs(librosa.stft(SampleClip, 
                                  n_fft=256,
                                  hop_length=100))
    
    Spectro = SftfMat[FMin:FMax,TMin:TMax].flatten() # 40x40, LeNet! 
    #plt.pcolormesh(SftfMat[FMin:FMax,TMin:TMax]); plt.show()
    VecLen = np.sqrt(np.sum(Spectro**2.))

    if VecLen:
        spectro_processed = Spectro/VecLen
        spectro_processed = spectro_processed.reshape([40,40])
    else:
        spectro_processed = np.zeros([40, 40])

    return spectro_processed 





def PowerLawMatCal(SpectroMat, Nu1, Nu2, Gamma):
    DimF, DimT = SpectroMat.shape
    Mu_k = [PoweLawFindMu(SpectroMat[ff,:]) for ff in range(SpectroMat.shape[0])]
    
    Mat0 = SpectroMat**Gamma - np.array(Mu_k).reshape(DimF,1)*np.ones((1, DimT))
    MatADenom = [(np.sum(Mat0[:,tt]**2.))**.5 for tt in range(DimT)]
    MatA = Mat0 / (np.ones((DimF,1)) * np.array(MatADenom).reshape(1, DimT) )
    MatBDenom = [ (np.sum(Mat0[ff,:]**2.))**.5 for ff in range(DimF)]
    MatB = Mat0 / (np.array(MatBDenom).reshape(DimF,1) * np.ones((1, DimT)))
    PowerLawMat = (MatA**Nu1)*(MatB**Nu2)
    return PowerLawMat

def PoweLawFindMu(SpecTarget):
    SpecSorted = np.sort(SpecTarget)
    SpecHalfLen = int(np.floor(SpecSorted.shape[0]*.5))
    IndJ = np.argmin(SpecSorted[SpecHalfLen:SpecHalfLen*2] - SpecSorted[0:SpecHalfLen])
    Mu = np.mean(SpecSorted[IndJ:IndJ+SpecHalfLen])
    return Mu

def FindIndexNonZero(DiffIndexSeq):
    IndSeq = []
    ItemSeq = []
    for index, item in enumerate(DiffIndexSeq):
        if(item != 0):
            IndSeq.append(index)
            ItemSeq.append(item)
    return IndSeq, ItemSeq
    
  

    
