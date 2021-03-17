'''
Created on Sep 4, 2017

@author: mroch
'''

import pandas as pd

def time_delta_s(s):
    "Return a TimeDelta object"
    return pd.Timedelta(s*1e9, 'ns')

class TimeIndex:
    '''
    TimeIndex class - Convert index to time
    
    Constructor timeindex(starttime, Fs)
    timeindex[N] yields time associate with sample N
    
    '''


    def __init__(self, starttime, Fs):
        '''
        TimeIndex(starttime, Fs)
        '''
        
        self.start = pd.Timestamp(starttime)
        # Best precision we can get is ns with current pandas
        self.Fs = Fs
        self.increment = pd.Timedelta(1e9/Fs, 'ns')
        
    def __getitem__(self, idx):
        return self.start + idx * self.increment
    
    def __repr__(self):
        return "{} starting at {} with Fs={}".format(
            self.__class__, self.start, self.Fs)
        
        
        
        