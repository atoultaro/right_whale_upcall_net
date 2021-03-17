# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:03 2017

@author: mroch
"""

import numpy

class AEfeatures(object):
    "Class for generating autoencoder features"
    
    def __init__(self, htklist, dummy=None, batch_size=100):
        """Given a list of HTKFeat_read objects, 
        create an object that behaves like Keras Sequence.
        The dummy argument would normally be the target set,
        but autoencoders use inputs as targets."""
        
        self.htklist = htklist
        self.batch_size = batch_size
        self.current = 0
        self.epoch = 0
        # number of features in each HTK set
        self.featsperfile = [feats.getlen() for feats in self.htklist]
        # array showing cumulative index of features across files in set
        self.cumfeats = numpy.cumsum(self.featsperfile)
        # total number of features
        self.numfeatures = sum(self.featsperfile)
        self.dim = htklist[0].getdim()
        
    def __len__(self):
        "len(AEfeatures) - Return number of features"
        return self.numfeatures
    
    def __iter__(self):
        return self
    
    def get_dim(self):
        "get_dim() - Return dimensionality of feature vector"
        return self.dim
    
    def get_epoch(self):
        return self.epoch
    
    def __next__(self):
        """"Return next training and test batch
        Data is looped in a circular manner and will never raise
        a StopIteration exception.  Use .get_epoch() to see how many
        times the data has been looped over completely.
        """

        count = 0
        features = []
        while count < self.batch_size:
            try:
                features.append(next(self.htklist[self.current]))
                count = count + 1
            except StopIteration:
                # End of current list, move to next or wrap around
                self.current = self.current + 1
                # Have we made it all the way through the data?
                if self.current >= len(self.htklist):
                    # Reset, note epoch change
                    self.current = 0
                    self.epoch = self.epoch + 1
                # Reset list, not our first time visiting it
                if self.epoch > 0:
                    self.htklist[self.current].seek(0)

        featmatrix = numpy.stack(features, axis=0)
        return (featmatrix, featmatrix)
            
        