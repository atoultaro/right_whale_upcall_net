'''
Created on Aug 21, 2017

@author: mroch
'''

import math
import sys




from dsp.abstractstream import Streamer
from dsp.soundfilestream import SoundFileStream
import numpy as np
from rope.base.exceptions import ModuleNotFoundError


# underlying reader
try:
    from soundfile import SoundFile, _SoundFileInfo, SEEK_SET
    from dsp.soundfilestream import SoundFileStream
    soundfile_avail = True
    _default_reader = "SoundFile"
except ImportError:
    # somewhat limited wave reader, but okay for now
    print("""soundfile module not available, using scipy.io.wavfile (2 ch wave only, no 24 bit)
    Available at https://pysoundfile.readthedocs.io""")
    soundfile_avail = False
    _default_reader = "wavfile"
    
# less capable audio reader (16 bit wave files)
try:
    import scipy.io.wavfile
    scipywav_avail = True
except ImportError:
    scipywav_avail = False
    if not soundfile_avail:
        raise ValueError("No audio interface available")
    

class AudioFrames(Streamer):
    """AudioFrames
    A class for iterating over frames of audio data
    """
    
    def __init__(self, filename, adv_ms, len_ms, incomplete=False, streamstart=None,
                 interface=_default_reader):
        """"AudioSamples(adv_ms, len_ms, incomplete)
        Create a stream of audio frames where each is in len_ms milliseconds long
        and frames are advanced by adv_ms. 
        Optional parameters:
            streamstart - numpy.datetime64 timestamp for first sample in contiguous stream
            soundfile - True (default) - use libsoundfile (soundfile module)
                        False - use scipy.io.wavfie (more limited)
            incomplete - False (default) - incomplete frames discarded
                        True - incomplete frames retained
            interface - Class to use for sound I/O interface
                "SoundFile" - soundfile.SoundFile
                "SoundFileStream" - derived from SoundFile but returns multiple partial frames
                    that can be used to build frames across files
                "wavfile" - scipy wave reader (only reads wavefiles, does not support 24 bit
                defaults to SoundFile if available, otherwise wavfile                
        """
        #fileinfo =info()

        self.len_ms = len_ms
        self.adv_ms = adv_ms
        self.offset = 0     # offset start of frames to sample N (currently unused
        
        # Set up audio file
        self.filename = filename

        # Determine which interface to use
        self.interface = interface            
        self.use_soundfile = soundfile_avail and self.interface in ["SoundFile", "SoundFileStream"]
        if not self.use_soundfile and self.interface not in ["wavfile"]:
            raise ValueError("Bad I/O interface specification") 
        
        if self.use_soundfile:
            self.init_soundfile()
        else:
            self.init_scipywavfile()
            if incomplete:
                raise NotImplementedError("scipy.io.wavfile does not support incomplete frames")
        
        # Compute framing parameters in samples
        self.adv_N = int(self.Fs * (self.adv_ms / 1000.0))
        self.len_N = int(self.Fs * (self.len_ms / 1000.0))
        
        self.start_time = streamstart
        if streamstart:
            if not isinstance(streamstart, np.datetime64):
                raise ValueError("streamstart not of type numpy.datetime64")
            
        self.framedelta = np.timedelta64(adv_ms, 'ms')
        self.cumoffset = np.timedelta64(0, 's')
        
        # Compute parameters related to frame overlap and skip
        # number new samples each frame
        self.nonoverlap_N = min([self.len_N - self.adv_N, self.len_N])
        # number of samples from previous frame
        self.overlap_N = max([self.len_N - self.adv_N, 0])
        # number of samples needed to advance past the end of frame
        # to the beginning of the next
        self.next_frame_adv = max([0, self.adv_N - self.len_N])  
        
        self.current_sample = 0  # initial position
        
        self.incomplete = incomplete
        
        self.repositioned = True  # Let iterator know that this will be first frame
        
        
        
        
    def init_scipywavfile(self):
        "init_scipywavfile() - initialize scientific python wavfile interface"            
                
        # get data as memory mapped file
        [self.Fs, self.data] = scipy.io.wavfile.read(self.filename, mmap=True)

        self.samplesN = self.data.shape[0]
        self.channels = 1 if len(self.data.shape) == 1 else self.data.shape[1] 
        self.format = "WAV"
        
    def init_soundfile(self):
        "init_soundfile() - initialize flexible SoundFile audio interface"
        self.soundfileinfo =  _SoundFileInfo(self.filename, verbose = True)
        self.data = None
        if self.interface == "SoundFile":
            self.fileobj = SoundFile(self.filename)
        else:
            self.fileobj = SoundFileStream(self.filename)
        
        self.Fs = self.fileobj.samplerate
        self.samplesN = self.soundfileinfo.frames
        self.channels = self.soundfileinfo.channels
        self.format = self.soundfileinfo.format
        self.subtype = self.soundfileinfo.subtype
    
    def get_framelen_samples(self):
        "get_framelen_ms - Return frame length in samples"
        return self.len_N
    
    def get_framelen_ms(self):
        "get_framelen_ms - Return frame length in ms"
        return self.len_ms
    
    def get_frameadv_samples(self):
        "get_frameadv_ms - Return frame advance in samples"
        return self.adv_N  

    def get_frameadv_ms(self):
        "get_frameadv_ms - Return frame advance in ms"
        return self.adv_ms
    
    def get_Fs(self):
        "get_Fs() - Return sample rate"
        return self.Fs
    
    def __len__(self):
        "len() - number of frames"
        
        # Number of frames computation
        remainingN = self.samplesN - self.offset  # account for possible non-zero start
        #
        if self.incomplete:
            if self.interface == "SoundFileStream":
                # Returns all possible frames
                return math.ceil((remainingN-1)/ self.adv_N)
            else:
                return math.floor(remainingN / self.adv_N)
        else:
            # complete frames
            # length of frame - advance can be subtracted from sample count and divided by
            # advance
            return math.floor((remainingN - (self.len_N - self.adv_N)) / self.adv_N);

    def get_Nyquist(self):
        return self.Fs/2.0
    
    def get_params(self):
        "Return dict with file parameters"
        
        params = {
            "filename" : self.filename, 
            "Fs" : self.Fs,
            "samples" : self.samplesN,
            "framing" : {"adv_ms" : self.adv_ms, "len_ms" : self.len_ms, 
                         "adv_N":self.adv_N, "len_N" : self.len_N},
            "format" : self.format
            }
        if self.use_soundfile:
            params["subtype"] = self.subtype
            
        return params
    
    def shape(self):
        "shape() - shape of tensor generated by iterator"
        return np.asarray([self.len_N, 1])
    
    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return np.asarray(np.product(self.shape()))
        
    def __iter__(self):
        """"iter() - Return a frame iterator
        WARNING:  Multiple iterators on same soundfile are not guaranteed to work as expected"""
        if self.use_soundfile:
            it = soundfile_it(self)
        else:
            it = scipywavefile_it(self)
        return it
    
    def seek_sample(self, N):
        "seek_sample(N) - Next iterator will start with sample N"

        if N > self.samplesN:
            raise ValueError("File %s seek to sample {}:  past end of file {}"%(
                self.filename, N, self.samplesN))
        else:
            if self.use_soundfile:
                self.current_sample = self.fileobj.seek(N, whence = SEEK_SET)
            else:
                # memory mapped, just set counter
                self.current_sample = N
    
    def get_data(self, startidx, N):
        """get_data(startidx, N) - Retrieve N samples starting at startidx.
        This has no side effects, the file position of iterators is unchanged
        """
        if startidx > len(self)*self.adv_N:
            raise ValueError("Read past end of data")
        
        if self.use_soundfile:
            current = self.fileobj.tell()
            self.fileobj.seek(startidx)
            data = self.fileobj.read(N)
            self.fileobj.seek(current)  # put back
        else:
            stopidx = min(startidx+N, len(self))
            if self.channels > 1:
                data = self.data[startidx:stopidx,:]
            else:
                data = self.data[startidx:stopidx]
            
        return data
    
    def seek_frame(self, N):
        "seek_frame(N) - Next read will start at frame N"
        raise NotImplemented()
            
class scipywavefile_it(object):
    
    def __init__(self, frameobj):
        self.frameobj = frameobj
        self.start_sample = frameobj.current_sample
        self.current_sample = self.start_sample
        
    def __iter__(self):
        return self
    
    def __next__(self):
        "next() - Return next frame, offset (s), and absolute time (if available, otherwise None"
        
        # setup frame indices
        start = self.current_sample
        stop = start + self.frameobj.len_N
        
        # check for valid frame indices
        
        # start past end check
        if start >= self.frameobj.samplesN:
            raise StopIteration
        
        # Compute offsets
        offset_s = self.current_sample / self.frameobj.Fs  #MOVE UP
        if self.frameobj.start_time:
            time = self.frameobj.start_time + np.timedelta64(offset_s, 's')
        else:
            time = None

        # past end check
        if stop > self.frameobj.samplesN:
            # frame runs off end
            if self.frameobj.incomplete:
                # User wants the incomplete frame.  Set stop to last
                # sample and set current_sample to trigger StopIteration
                # on next call
                stop = self.frameobj.samplesN
                self.current_sample = self.frameobj.samplesN
            else:
                raise StopIteration    
        else:               
            self.current_sample = start + self.frameobj.adv_N
        
            
        if self.frameobj.channels > 1:
            frames = self.frameobj.data[start:stop,:]
        else:
            frames = self.frameobj.data[start:stop]

        # frame data, frame offset (s) from start, frame start time 
        return (frames, offset_s, time)  
        
class soundfile_it(object):

    def __init__(self, frameobj):
        "soundfile_it(frameobj) - Generate frame iterator"
        
        self.frameobj = frameobj
        self.sample_offset = frameobj.current_sample

        # Store underlying libraries frame iterator
        if self.frameobj.interface == "SoundFile":
            self.framegen = frameobj.fileobj.blocks(
                blocksize = frameobj.len_N,
                overlap = frameobj.overlap_N)
        else:
            self.framegen = frameobj.fileobj.blocks(
                blocksize = frameobj.len_N,
                overlap = frameobj.overlap_N,
                incomplete = frameobj.incomplete)
            
    def __iter__(self):
        return self
    
    def __next__(self):
        "next() - Return next frame, offset (s), and absolute time (if available, otherwise None"

        frames = next(self.framegen)
        if frames.shape[0] != self.frameobj.len_N and not self.frameobj.incomplete:
            # User did not want incomplete frames
            raise StopIteration

        offset_s = self.sample_offset / self.frameobj.Fs
        if self.frameobj.start_time:
            time = self.frameobj.start_time + np.timedelta64(offset_s, 's')
        else:
            time = None

        self.sample_offset = self.sample_offset + self.frameobj.adv_N
        
        return (frames, offset_s, time)
    
    
        
            
            
            
        
    
        
        
        
