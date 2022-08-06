# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang
"""

import numpy as np

def find_idx(array, values):
    '''
    Find the indexes of values in array.
    
    If not found, will return -l-1, which is out of
    the range of array.

    Parameters
    ----------
    array : Iterable
        .
    values : Iterable
        .

    Returns
    -------
    idx : np.ndarray (int)
    
    found : np.ndarray (bool)

    '''
    l = len(array)
    sorter = np.argsort(array)
    ss = np.searchsorted(array, values, sorter=sorter)
    isin = np.isin(values, array)
    not_found = (ss==l) | (~isin)
    found = ~not_found
    ss[not_found] = -1
    idx = sorter[ss]
    idx[not_found] = -l-1
    return idx, found

# Modified dictionary. If each value has the same length, it is similar to pandas.DataFrame, but simpler.
class objdict(dict):
    # from https://goodcode.io/articles/python-dict-object/?msclkid=daff3822c47111eca4f572e5716ccae3
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

