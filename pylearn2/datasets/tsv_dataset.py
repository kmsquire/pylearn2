# -*- coding: utf-8 -*-
"""
A simple general tsv dataset wrapper for pylearn2.
Can do automatic one-hot encoding based on labels present in a file.
"""
__authors__ = "Zygmunt Zając"
__copyright__ = "Copyright 2013, Zygmunt Zając"
__credits__ = ["Zygmunt Zając"]
__license__ = "3-clause BSD"
__maintainer__ = "?"
__email__ = "zygmunt@fastml.com"

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess


class TSVDataset(DenseDesignMatrix):
    """
    A generic class for accessing TSV files
    labels, if present, should be in the first column
    if there's no labels, set expect_labels to False
    if there's no header line in your file, set expect_headers to False
    """
    def __init__(self, 
                 path = 'train.tsv',
                 one_hot = False,
                 start = None,
                 stop = None,
                 expect_labels = True,
                 expect_headers = True,
                 delimiter = '\t'):
        """
        .. todo::

            WRITEME
        """

        self.path = path
        self.one_hot = one_hot
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        
        self.view_converter = None

        # and go

        self.path = preprocess(self.path)
        X, y = self._load_data()
        
        if start is not None:
            assert start >= 0
            if stop > X.shape[0]:
                raise ValueError('stop='+str(stop)+'>'+'m='+str(X.shape[0]))
            assert stop > start
            X = X[start:stop,:]
            if X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d" % (X.shape[0], start, stop))
            if len(y.shape) > 1:
                y = y[start:stop,:]
            else:
                y = y[start:stop]
            assert y.shape[0] == stop - start

        super(TSVDataset, self).__init__(X=X, y=y)

    def _load_data(self):
        """
        .. todo::

            WRITEME
        """
    
        if self.expect_headers:
            data = np.loadtxt(self.path, delimiter = self.delimiter, skiprows = 1)
        else:
            data = np.loadtxt(self.path, delimiter = self.delimiter)
        
        if self.expect_labels:
            y = data[:,0]
            X = data[:,1:]
            
            # get unique labels and map them to one-hot positions
            labels = np.unique(y)
            #labels = { x: i for i, x in enumerate(labels) }    # doesn't work in python 2.6
            labels = dict((x, i) for (i, x) in enumerate(labels))

            if self.one_hot:
                one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
                for i in xrange(y.shape[0]):
                    label = y[i]
                    label_position = labels[label]
                    one_hot[i,label_position] = 1.
                y = one_hot

        else:
            X = data
            y = None

        return X, y
