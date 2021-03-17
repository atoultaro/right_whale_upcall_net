'''
Created on Aug 29, 2017

@author: mroch
'''

from soundfile import *

class SoundFileStream(SoundFile):
    '''
    SoundFileStream - subclass of soundfile.SoundFile that supports returning multiple
    incomplete blocks
    '''
           
    def blocks(self, blocksize=None, overlap=0, frames=-1, dtype='float64',
               always_2d=False, fill_value=None, out=None, incomplete=False):
        """Return a generator for block-wise reading.

        By default, the generator yields blocks of the given
        `blocksize` (using a given `overlap`) until the end of the file
        is reached; `frames` can be used to stop earlier.

        Parameters
        ----------
        blocksize : int
            The number of frames to read per block. Either this or `out`
            must be given.
        overlap : int, optional
            The number of frames to rewind between each block.
        frames : int, optional
            The number of frames to read.
            If ``frames < 0``, the file is read until the end.
        dtype : {'float64', 'float32', 'int32', 'int16'}, optional
            See :meth:`.read`.
        incomplete : {False (default), True}
        

        Yields
        ------
        numpy.ndarray or type(out)
            Blocks of audio data.
            If `out` was given, and the requested frames are not an
            integer multiple of the length of `out`, and no
            `fill_value` was given, the last block will be a smaller
            view into `out`.


        Other Parameters
        ----------------
        always_2d, fill_value, out
            See :meth:`.read`.

        Examples
        --------
        >>> from soundfile import SoundFile
        >>> with SoundFile('stereo_file.wav') as f:
        >>>     for block in f.blocks(blocksize=1024):
        >>>         pass  # do something with 'block'

        """
        if 'r' not in self.mode and '+' not in self.mode:
            raise RuntimeError("blocks() is not allowed in write-only mode")

        if overlap != 0 and not self.seekable():
            raise ValueError("overlap is only allowed for seekable files")

        if out is None:
            if blocksize is None:
                raise TypeError("One of {blocksize, out} must be specified")
        else:
            if blocksize is not None:
                raise TypeError(
                    "Only one of {blocksize, out} may be specified")
            blocksize = len(out)

        advance = blocksize - overlap
        # We stop when there are no more frames left to read
        # Interpretations:
        #
        #    incomplete False
        #    There are less than blocksize samples left after backing up
        #    by overlap.
        #
        #    incomplete True
        #    There are samples left after backing up by overlap
        if incomplete:
            stopcond = 1
        else:
            stopcond = blocksize
        
        # determine how many frames of data remain
        # frame is defined as all channels of a single sample
        frames = self._check_frames(frames, fill_value)
        
        while frames > 0:
            if frames < blocksize:
                # Not a complete frame
                if fill_value is not None and out is None:
                    out = self._create_empty_array(blocksize, always_2d, dtype)
                if not incomplete:
                    blocksize = frames
                    
            # Read partial or full block
            block = self.read(blocksize, dtype, always_2d, fill_value, out)
            readN = block.shape[0]  # sample count read
          
            # If we got the whole thing, seeking for the next frame will be easy.
            # otherwise we need to figure out how far back to move

            # samples remaining must take into account the overlap
            frames -= readN  # reduce by amount read
            # find amount eligible to read again        
            overlap_fulfilled = max([0, readN - advance]) # How many samples past advance
            #print("read: frames left {} (posn={}) len {}, overlap {}".format(frames, self.tell(), readN, overlap))

            frames += overlap_fulfilled
            #print("after adjust frames {}".format(frames))
            
            if frames >= stopcond:
                if self.seekable():
                    # If we read past the advance, position for next frame
                    if block.shape[0] > advance:
                        self.seek(-overlap_fulfilled, SEEK_CUR)
                        posn = self.tell()
                        #print('position {}'.format(posn))
            else:
                frames = 0  # stop
            yield block
        