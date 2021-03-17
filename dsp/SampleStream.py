'''
Created on Aug 27, 2017

@author: mroch
'''

# sortedcontainers module
# http://www.grantjenks.com/docs/sortedcontainers/
# pip install sortedcontainers
import sortedcontainers

from dsp.abstractstream import Streamer, StreamGap, StreamEnd
from dsp.timeindex import TimeIndex

import math
import sys
import numpy as np
import pandas as pd


import soundfile
import scipy.io.wavfile
from pandas.core.algorithms import isin
#from builtins import None

class Streams:
    def __init__(self):
        self.streams = sortedcontainers.SortedList()
        self.__total_stream_samps = 0
        self.__samps_in_streams = []
        
    def add_file(self, file, samples, timestamps, Fs):
        """add_file(file, samples, timestamps)
        Given a file, provide two lists describing the contiguous
        data segments within the file:
        samples - List of number of samples in each contiguous segment
        timestamps - Start time of each contigous segment as a pandas timestamp
        Fs - sample rate
        """
        if len(samples) != len(timestamps):
            raise ValueError(
                "sample and timestamp lists must be of same size: {} != {}".format(
                len(samples), len(timestamps)))
        self.streams.add((timestamps[0], file, timestamps, samples, Fs))
        
        # documnet the number of samples in the entire stream
        self.set_total_stream_samps((self.get_total_stream_samps()+ samples[0]))
        self.set_samps_in_streams(samples[0])
        
    def get_samps_in_steams(self):
        '''return list of the number of samples in each stream'''
        return self.__samps_in_streams
    
    def set_samps_in_streams(self, x):
        '''returns a list indicating the end sample of each stream '''
        if len(self.get_samps_in_steams()) == 0:
            new_x = x
        else:
            new_x = x + self.get_samps_in_steams()[-1]
        self._Streams__samps_in_streams.append(new_x)
        
        
    def get_total_stream_samps(self):
        ''' Returns the total number of samples in the entire samplestream'''
        return self._Streams__total_stream_samps
    
    def set_total_stream_samps(self, x):
        self.__total_stream_samps = x

    
    
    def __len__(self):
        return len(self.streams)
    
    def __iter__(self):
        return iter(self.streams)
    
    def __getitem__(self, key):
        return self.streams[key]
    
    def get_stream(self, idx):
        
        return SampleFile(self.streams[idx])
    
   
    def timestamp_to_file_offset(self, timestamp):
        """timestamp_to_file_offset(timestamp)
        Convert timestamp to a file, contiguous data section and sample offset 
        relative to the start of file
        """
        
        # Each stream is a tuple:  (
        #    -start_time,
        #    [list of start times within the file]  start times of each
        #        contiguous section (e.g. duty cycle)
        #    [list of number of samples]    # samples associated with each
        #        contiguous section
        # )
        
        notfound = (None, None, None)
        
        # Search for file in which this time might fall
        streamsN = len(self.streams)
        streamidx = 0
        infile = None   # stream file # where we found this 
        
        while (self.streams[streamidx][0] < timestamp and streamidx < streamsN):
            previous = streamidx
            streamidx = streamidx + 1
            if (streamidx >= streamsN or self.streams[streamidx][0] >= timestamp):
                # This segment is after the timestamp.  Check if the previous
                # one starts before it.  If so, we might have it.
                if timestamp >= self.streams[previous][0]:
                    infile = previous
                break
            
        if infile is None:
            # Nothing
            result = notfound
        else:
            fileidx = previous
            contigidx = 0
            previous = None
            starts = self.streams[fileidx][2]
            contigsN = len(starts)
            incontig = None
            while (starts[contigidx] < timestamp and contigidx < contigsN):
                previous = contigidx
                contigidx = contigidx + 1
                if (contigidx >= contigsN or starts[previous] > timestamp):
                    # This contig is after the timestamp.  Check previous
                    # contig.
                    if timestamp >= starts[previous]:
                        incontig = previous
                    break
            
            if incontig is None:
                result = notfound
            else:
                # Check if timestamp is within contig
                duration = timestamp - starts[incontig]
                samples_per_contig = self.streams[infile][2]
                Fs = self.streams[infile][4]
                sample_into_contig= int(duration.total_seconds() * Fs + 0.5)
                sample_into_file = sample_into_contig
                if incontig > 0:
                    sample_into_file += sum(samples_per_contig[0:incontig]) 

                result = (infile, incontig, sample_into_file)
                      
        return result
        
        
        
class SampleFile:
    "An instance of a stream"
    
    def __init__(self, streamtuple):
        """Create a stream object for reading data
        streamtuple must have:
            
            [0] - Starting timestamp for file
            [1] - Filename
            Lists describing contiguous sections of data within file
 
            These lists are of the same size
            [2] - Start time of contiguous data block
            [3] - Number of samples in contiguous block
 
        """
        # streamtuple
        self.filename = streamtuple[1]
        self.audio = soundfile.SoundFile(self.filename)
        
        self.Fs = self.audio.samplerate
        #self.samplesN = np.uint64(self.audio.frames)
        self.samplesN = len(self.audio)
        self.channels = self.audio.channels
        self.format = self.audio.format
        self.subtype = self.audio.subtype
        
        self.times = [TimeIndex(d, self.Fs) for d in streamtuple[2]]

        # User may have provide NaN if the last block length was unknown
        # Set count now that we know the number of samples
        self.samples = np.zeros(len(streamtuple[3]), dtype="uint64")
        if np.isnan(streamtuple[3][-1]):
            if len(streamtuple[3]) == 1:
                self.samples[0] = self.samplesN   # only one block
            else:
                # Infer unknown last block size
                self.samples[:-1] = streamtuple[3][:-1]
                self.samples[-1] = self.samplesN - np.sum(self.samples)
        else:
            self.samples[:] = streamtuple[3]
            
        self.contig_idx = 0  # nth contiguous block
        
        # When a transition event is about to occur such as the end
        # of a contiguous block or the end of file, the event is populated
        # with the exception that will be raised. 
        self.event = None
        
        # Store the index into the current contiguous block, aka "contig"
        # as well the number of samples in the contig
        self.index = np.uint64(0)
        self.contig_samples = np.uint64(self.index + self.samples[self.contig_idx])
        # Store the starting sample into the stream for each contig
        cumulative = np.cumsum(self.samples)
        if len(self.samples) > 1:
            self.contigs_first_sample = np.stack(
                [[np.uint64(0)], cumulative[0:-1]])
        else:
            self.contigs_first_sample = np.zeros([1], dtype="uint64")        
    
    def read(self, N):
        """"read(N) - Read N samples

        Returns tuple:
            0 : sample data
            1 : starting index within the current contiguous block
            2 : timestamp of the first sample
            3 : index of contiguous block
            
        When approaching the end of a contiguous block, fewer than N samples
        will be returned.  
        
        Whenever a read results in reaching the end of the set of files
        or the end of a contiguous data block, a subsequent read will
        produce a StreamGap (more data can be obtained by subsequent
        calls to read) or EndOfStream in which case there are no
        more data to be had.
        """

        
        if self.event:
            # We have reached a StreamGap or the end of file
            event = self.event
            if isinstance(event, StreamGap):
                # There's more to read, just let caller know that their next
                # read is not contiguous
                self.event = None
                self.index = np.uint64(0)  # Starting a new block
                print("File {} Contig {} index set to {}".format(
                    self.filename, self.contig_idx, self.index))

            raise event
        
        # Store index of current contiguous block
        # (will be updated if we read to the end)
        contig_idx = self.contig_idx
        
        if self.index + N >= self.contig_samples:
            # We will reach or go past the end of block
            
            # Find out how much data we can read (might be everything the
            # caller wanted if we read up to the boundary

            new_N = int(self.contig_samples - self.index)

            data = self.audio.read(new_N)
            
            # Next read should produce some type of event
            self.event = self._next_contig()
                
        else:  
            data = self.audio.read(N)
            
        # Single channel data needs to be reshaped to be consistent
        if len(data.shape) == 1:
            data = np.reshape(data, [data.size, 1])

        data_N = np.uint64(data.shape[0])  # number samples read
        
        # Compute start time of sample
        timestamp = self.times[contig_idx][self.index]
        startidx = self.index
        self.index += data_N
        return (data, startidx, timestamp, contig_idx)
    

    def _next_contig(self):
        """_next_contig() - Prepare next contiguous block.  Returns one of the
           following:
            EOF - Next read will go past the end of file
            StreamGap - Next read will start a new contiguous section
        """
        
        self.contig_idx = self.contig_idx + 1
        if self.contig_idx < len(self.samples):
            # More to read
            self.contig_samples = np.uint64(self.samples[self.contig_idx])
            value = StreamGap(filename=self.filename, 
                            samples=self.contigs_first_sample[self.contig_idx],
                            contig=self.contig_idx, 
                            timestamp=self.times[self.contig_idx][0])
        else:
            value = EOFError(self.filename)
            print("SampleFile.read - EOF {}".format(value))
            
        return value
    
    def next_sample_idx(self):
        "next_sample_idx() - Return sample that will be read next"
        return self.contig_samples
    
    def next_timestamp(self):
        """next_timestamp() - Return the timestamp of the next sample to be read
        When the end of file is reached, the time returned is one sample
        past the last one read.
        """
        if self.event != None and isinstance(self.event, EOFError):
            # next read is past this file
            # Return time of next sample past end
            t = self.times[-1][self.samples[-1] + 1]
        else:
            t = self.times[self.contig_idx][self.index]
        return t
    
    def start_timestamp_contigN(self, idx):
        "start_timestamp_contigN(N) - Return starting time Nth contig"
        return self.times[idx][0]
    
    def samplefile_contiguous(self, start_time):
        """"samplefile_contiguous(start_time) 
        Input: start time- the start time of the NEXT file
        Given the start_time of another file, does it continue this one
        without a StreamGap?
        Returns True or False
        """
        expected = self.times[-1][self.samples[-1]]        
        return expected == start_time
    
    def at_end_of_file(self):
        "at_end_of_file() - Will the enxt read result in end of file?"
        return isinstance(self.event, EOFError)
    
    def at_stream_gap(self):
        "at_stream_gap() - Will the next read result in a stream gap?"
        return isinstance(self.event, StreamGap)
        
            
        
    
class SampleStream(Streamer):
    """
    SampleStream
    
    Class that merges multiple files with the same characteristics (sample rate,
    quantization, etc.) into a single stream. The stream may have gaps both
    across and within files. See constructor for gap specification.

    """
    def __init__(self, streams, maxhistory=40000):
        """"AudioSamples(filename, optional arguments)
        Create a stream of audio samples. 
        Parameters:
        streams - Streams object describing files in stream
                
        maxhistory - Maximum number of history samples to retain            
        
        CAVEATS:  It is assumed that all files are ordered by time (currently no checking)
        """

        self.streams = streams
        
        # offsets into stream
        self.stream_idx = 0   # current stream
             
        # Set up first audio file
        self.stream = self.streams.get_stream(self.stream_idx)
        
        # initialize history buffer
        self._hist_size = maxhistory
        self._reset_hist()
        
        self.channel_select = self.stream.channels   # channel select:  All channels by default
        self.allchan = True
        
        self._readtype = "int16"  # change to float64 after debugged

    def _reset_hist(self):
        "_reset_hist - reset history buffer"
        self._hist = np.empty([self._hist_size, self.stream.channels]) * np.NaN
        self._hist_avail = 0
        # Our next read must be inside a file, so reset any
        # pending StreamGap or StreamEnd event
        self.event = None
             
    def _next_stream(self):
        """_next_stream() - Prepare to read from next stream
        Returns:
            None - Data are contiguous, can continue reading
            StreamGap() - Return any data read so far, but the next
                read should produce a StreamGap exception
            StreamEnd() - Return any data read so far, but the next
                read should produce an StreamEnd excpetion <<== YuShiu: ?? Why not raise error now?
        """
        if self.stream_idx + 1 < len(self.streams):
            # Is there a gap?
            contig_continues = self._contig_continues_next_file()
#            # debug
#            if not contig_continues:
#                xyzzy = self._contig_continues_next_file()
#                
            
            # Move to next stream 
            self.stream = self.streams.get_stream(self.stream_idx+1)
            
            if contig_continues:
                action = None
            else:
                # Currently don't track exact position in samples
                # Update this if we start doing so.  For now, first
                # sample in next file
                action = StreamGap(
                    samples = self.stream.next_sample_idx(),
                    timestamp = self.stream.next_timestamp(),
                    contig = self.stream.contig_idx,
                    filename = self.stream.filename)
            self.stream_idx = self.stream_idx + 1

        else:
            # Reached end of stream, no more files to process
            action = StreamEnd()
        return action
            
    def get_stream_len(self):
        return len(self.streams)
    

    def get_all_stream_filenames(self):
        '''
        Returns a list of the filename of each stream in the streamlist
        '''
        
        filenamelist = []
        
        for ii in range(len(self.streams)):
            filenamelist.append(self.streams[ii][1])
            
        return filenamelist


    def get_all_stream_fs(self):
        '''
        Returns a list of the sample rate in each file in the stream
        '''
        
        file_fs = []
        
        for ii in range(len(self.streams)):
            file_fs.append(self.streams[ii][4])
            
        return file_fs
    
    def get_all_stream_samps(self):
        '''
        Returns a numpy array indcating the number of samples (total) in each
        file
        '''
        
        samp_durs = np.zeros(self.get_stream_len())
        
        for ii in range(len(self.streams)):
            samp_durs[ii] =   self.streams[ii][3][0]
            
        return samp_durs
        
            
    def _contig_continues_next_file(self):
        """"_contig_continues_next_file() - Return True if the current
        SampleFile's contiguous section continues in the next file
        """
        next_stream = self.stream_idx + 1
        if next_stream  < len(self.streams):
            # See if start time of next stream is consistent
            # with the end of this stream's time
            next_start = self.streams[next_stream][0]
            result = self.stream.samplefile_contiguous(next_start)
        else:
            result = False  # no more, doesn't continue
        return result
         
        
    def read(self, N, previousN=None, _previous=None):
        """read(N, previousN=None, _previous=None)
        Read N samples from the stream.  
        
        If previousN is specified, one of two things will occur:
        
            If this is not the first read of the stream or after a StreamGap, 
            at most previousN samples will be prepended to the returned data.
            (Number prepended depends on how many previous samples are
             available)
            
            If it is the first read, N+previousN samples will be read
            
        Argument _previous is for internal use only.
        
        May raise the following exceptions:
            StreamGap - End of a contiguous block of data was reached. Next read
                will start a new non-contiguous block of data.  Examples of 
                events that can cause data gaps are recorder servicing, 
                duty cycles, event triggers, etc.
                StreamGap events resent the history and previous bytes 
                will no longer be available.
                
            StreamEnd - No more samples are available            
        """

        # About to read across a StreamGap or past StreamEnd?
        # Raise the exception
        # Note that StreamGap can be raised within a SampleFile read,
        # this handles StreamGaps across files
        if self.event != None:
            event = self.event
            # No history is available across a gap
            self._hist_avail = 0
            if isinstance(self.event, StreamGap):
                # Reset the history
                self._reset_hist()

            print("Read triggered {}".format(event))            
            raise event
        
        # If user wanted history and there is not enough history
        # available, increase samples to read
        if previousN != None:
            missing = previousN - self._hist_avail
            if missing > 0: 
                N = N + missing  # increase samples to read
                previousN = self._hist_avail  # Only use available history
            
        (data, idx, tstamp, contig_idx)= self.stream.read(N) #dtype=self._readtype)
        readN = data.shape[0]

        # How many of the read samples can be stored to the history?        
        storeN = min(readN, self._hist_size)  
        
        if previousN:
            if previousN + storeN > self._hist_size:
                raise RuntimeError(
                    ("Attempted to read {} samples with {} samples " +
                     "of history when history only permits {} samples"
                     ).format(readN, previousN, self._hist_size))
                
            # Take a slice of the history.  Note that when we update the history
            # with np.roll, the pointers in previus_data still point to the same
            # data, but if we physically modify the indices these values go away.
            previous_data = self._hist[-previousN:,:]

            
        # add read data to history
        # We move the oldest entries into the end of the array with
        # numpy.roll and then populate the last entries
        self._hist = np.roll(self._hist, -storeN, axis=0)  # make room
        self._hist[-storeN:,:] = data[-storeN:, :]
        self._hist_avail = min(self._hist_avail + storeN, self._hist_size)
    
        # Update what was read if needed
        if previousN:
            data = np.vstack([previous_data, data])
        
        if self.stream.at_end_of_file():
            # We have reached the end of the current stream.
            
            # Move to the next stream
            self.event = self._next_stream()
            # The next stream might be a continuation 
            
            if self.event == None:
                # Next stream is a continuation.
                if N > readN:
                    # Caller requested more than we read, get the rest
                    # We adjust the number of samples used from the history
                    # in a new call
                    if previousN == None:
                        # Use the samples we just read
                        previousN = readN
                    else:
                        # Use the samples we just read and the history the user
                        # already requested
                        previousN = previousN + readN
                    new_read = N - readN
                    return self.read(int(new_read), previousN=previousN)
        # Need to compute timestamps and samples to return
        return (data, idx, tstamp, contig_idx, self.stream_idx)
    
    def set_time(self, timestamp):
        """set_time(timestamp) - Given a Panda timestamp, 
        seek to the appropriate time.  
        """
        
        # Find time within series of times
        infile, incontig, sample_into_file = \
            self.streams.timestamp_to_file_offset(timestamp)
        
        # Throw an error if time isn't in the sample stream otherwise
        # update the index
        if infile is None:
            print('NE! Time not in specified files. Go away or we shall taunt'+
                  'you a second time!')
        else:
            # Else update which stream and the sample location 
            self.stream = self.streams.get_stream(infile)
            self.stream.index = int(sample_into_file)
            self.stream_idx = infile
            # Reset the history
            self._reset_hist()
            
            #check seek time and move the curser using audio.seek
            if self.stream.audio.tell() != self.stream.index:
                self.stream.audio.seek(sample_into_file)
        
        # set the (*&)^ stream index
        self.stream_idx = infile
                
                
    def set_sample(self, sample):
        """set_sample(int) - Given sample ID, 
        seek to the appropriate time.  
        
        """
        temp = np.asarray(self.streams.get_samps_in_steams())
        temp_stream_idx = np.argmax(temp>sample)
        
        if temp_stream_idx == 0:

           self.stream = self.streams.get_stream(temp_stream_idx)
           self.stream.index = int(sample_into_file)           
           
           
           
        else:
           sample_into_file = sample- temp[temp_stream_idx-1]
        
           # Else update which stream and the sample location 
           self.stream = self.streams.get_stream(temp_stream_idx)
           self.stream.index = int(sample_into_file)
           self.stream_idx = temp_stream_idx
        
        # Reset the history
        self._reset_hist()
        
        # set the (*&)^ stream index
        self.stream_idx = temp_stream_idx
        
        #check seek time and move the curser using audio.seek
        if self.stream.audio.tell() != self.stream.index:
            self.stream.audio.seek(sample_into_file)    
            
  
        # Seek to appropriate sample        
    def set_channel(self, channels):
        """set_channel(channels) - Iterators return only specified channel(s)
        By default, all channels are returned.
        
        Select a set of channels to be returned by:
            list, e.g. [0, 1, 5]
            scalar, 0    Use total number of channels to specify all channels
        
        CAVEAT:  Setting channel will change the the size and shape reported
        """
        
        raise NotImplementedError("Channel selection code is not functional YET. Needs implementation.")
        
        if channels == None:
            raise ValueError("channels must be specified")
        if isinstance(channels, int):
            if channels == self.stream.channels:
                # special case, all channels
                self.channel_select = self.stream.channels
                channels = list([x for x in range(self.stream.channels)])
            else:            
                channels = list(channels)
                
        for c in channels:
            if c < 0 or channels >= self.stream.channels:
                raise ValueError(
                    "Bad channel specification list, must be in [0,{}] ".format(
                        self.stream.channels-1))
        self.channel_select = channels
        if len(set.intersection(set(channels),
                                set([x for x in range(self.stream.channels)]))
               ) == self.stream.channelsN:
            self.allchan = True
        else:
            self.allchan = False
    
    def get_stream_event(self):
        """get_steam_event() - Indicate if the last read prepared us for a 
        stream event.
        Event codes:
            None - Can likely continue to read
            StreamGap() - End of contiguous data was reached  during
                last read.  Next read will trigger a StreamGap exception
            StreamEnd() - End of data stream has been reached.  Next
                read should produce an StreamEnd exception
                
        CAVEAT:  If a read ended on a stream boundary, i.e. the user was able to
        read all the data they requested, the stream event will not be set, so
        this is not a good way to check if the next read will trigger an
        exception.  It is useful for checking why the last read did not return
        as much data as was expected.
        """
        
        event = None
        if self.event:
            event = self.event
        else:
            # The current stream may have triggered the event
            gap = self.stream.at_stream_gap()
            # At stream end - Need to double check how this works, we should really set up
            # for an EndOfStream when we hit this condition.
            lasteof = self.stream.at_end_of_file() and self.stream_idx >= len(self.streams)
            if lasteof:
                print("Last EOF, should be StreamEnd instead... investigate")
            if gap or lasteof:
                event = self.stream.event
                 
        return event
    
    def get_current_timesamp(self):
        ''' returns the datetimestamp at the current sample'''
    
        # timestamp at start of current file
        file_timestamp_start = self.stream.times[0][0]
        
        # Samples into file
        samp_number = self.stream.audio.tell()
        
        # fs of the sample stream
        fs = self.stream.Fs
        
        #seconds into file
        seconds_into_file = samp_number/fs
        
        # add the seconds to the timestamp and return it        
        timestamp = file_timestamp_start +\
            pd.Timedelta(seconds=seconds_into_file)
        
    
        return timestamp
    

    def get_current_sample(self):
            ''' returns the number of samples into the streamer'''
                    
            sample_into_stream = np.sum(self.get_all_stream_samps() \
                                        [0:int(self.stream_idx)]) +\
                                        self.stream.audio.tell()
            return(sample_into_stream)



























