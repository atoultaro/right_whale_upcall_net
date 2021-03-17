'''
Created on Aug 21, 2017

@author: mroch
'''
from abc import abstractmethod, ABCMeta
import math

class Streamer(object):
    '''
    streamer - An abstract class for streaming signals
    '''
    __metaclass = ABCMeta

    @abstractmethod
    def __init__(self):
        '''
        streamer - Abstract class for streaming.  Constructors indicate framing and other parameters
        '''
        ...
    
    @abstractmethod
    def __iter__(self):
        "Return stream iterator object"
        ...  
    
    @abstractmethod
    def shape(self):
        "shape() Return shape of tensors contained in stream"
        ...
        
    @abstractmethod
    def size(self):
        "size() - Return total number of elements in tensors from stream"
        ...
        
    @abstractmethod
    def __len__(self):
        "len() - Return number of tensors to be generated, Inf means unknown or unbounded"
        return math.inf
    
    @abstractmethod
    def set_position(self, N):
        """set_position(N)
        If possible, position the streamer such that the next iterator will start at position N
        The interpretation of N is based upon the type of stream object.
        For example, frame-based streams will set to the N'th frame
        """
        ...
        
    @abstractmethod
    def set_sample_offset(self, N):
        "set_sample_offset(N)"
        ...
      
class StreamGap(Exception):
    
    def __init__(self, samples=None, timestamp=None, contig=None, filename=None):
        """StreamGap(samples, timetstamp, contig, filename)
        The stream has a gap in it.  Optional parameters
        samples - Sample number within file that will be read next
        timestamp - timestamp associated with next read
        contig - Some files have multiple sections of contiguous data 
            e.g. duty cycled, event driven samplin
            The number sotred here indicates that the next read will be from
            the Nth contiguous group of data
        filename - source file
        """
        
        self.samples = samples
        self.timestamp = timestamp
        self.contig = contig
        self.filename = filename
        
    def __str__(self):
        info = []
        info.append("{}: Next read after gap".format(self.__class__.__name__))
        for n in ["samples", "timestamp", "contig", "filename"]:
            try:
                info.append("{}={}".format(n, getattr(info,n)))
            except AttributeError:
                pass # attribute not populated
                
        return ", ".join(info)

        
class StreamEnd(EOFError):
    def __init__(self):
        pass
    
    def __str__(self):
        return "StreamEnd - Read past the last sample in the stream"
    
        