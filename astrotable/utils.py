# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang
"""

import numpy as np
import warnings
import pickle
import os
from functools import wraps

#%% array/Iterable operation

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

def grid(x, y, flat=False):
    if flat:
        xx = [xi for yi in y for xi in x]
        yy = [yi for yi in y for xi in x]
    else:
        xx = [[xi for xi in x] for yi in y]
        yy = [[yi for xi in x] for yi in y]
    return xx, yy

#%% basic types

# Modified dictionary. If each value has the same length, it is similar to pandas.DataFrame, but simpler.
class objdict(dict):
    # author: Senko Rašić
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

#%% interactive functions

def pause_and_warn(message=' ', choose='Proceed?', default = 'n', yes_message='', no_message='raise', timeout=None):
    '''
    calling this function will do something like this:
            [print]  <message>
            [print]  <choose> y/n >>> 
    default choice is <default>
    if yes:
            [print] <yes_message>
    if no:
            [print] <no_message>
        if no_message is 'raise':
            [raise] Error: <message>
    [return] the choise, True for yes, False for no.
    '''
    print('{:-^40}'.format('[WARNING]'))
    
    if isinstance(message, Exception):
        message = str(type(message)).replace('<class \'','').replace('\'>', '')+': '+'. '.join(message.args)
    warnings.warn(message)
    print(message)
    
    question = '{} {} >>> '.format(choose, '[y]/n' if default == 'y' else 'y/[n]')
    if timeout is None:
        cont = input(question)
    else:
        raise NotImplementedError
    if not cont in ['y', 'n']:
        cont = default
    if cont == 'y':
        print(yes_message)
        return True
    elif cont == 'n':
        if no_message == 'raise':
            raise RuntimeError(message)
        else:
            print(no_message)
            return False

#%% file IO

def save_pickle(fname, yes=False, *data):
    '''
    save data to fname

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    yes : bool
        if ``True``, file will be overwritten without asking.
    *data : TYPE
        DESCRIPTION.

    '''
    if not '.pkl' in fname:
        fname+='.pkl'
    if os.path.exists(fname):
        if os.path.isdir(fname):
            raise ValueError('fname should be the file name, not the directory!')
        if yes:
            print(f'OVERWRITTEN: {fname}')
        else:
            pause_and_warn('File "{}" already exists!'.format(fname), choose='overwrite existing files?',
                           default='n', yes_message='overwritten', no_message='raise')
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(fname):
    '''
    load pkl and return. 
    If there is only one object in the pkl, will return it.
    Otherwise, return a tuple of the objects.

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if fname[-4:] != '.pkl':
        fname+='.pkl'
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        if len(data) == 1:
            return data[0]
        else:
            return data

# def save_zip(fname, ext='zip', overwrite=False)

#%% wrappers
def deprecated_keyword_alias(**aliases):
    '''
    Returns wrapper for deprecated alias of keyword arguments.

    Parameters
    ----------
    **aliases : old = new
        old (deprecated) name and new name
    '''
    def wrapper(f):
        @wraps(f)
        def fnew(*args, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"Both {old} and {new} found in arguments; use {new} only.")
                    kwargs[new] = kwargs.pop(old)
                    warnings.warn(f"argument '{old}' is deprecated; use '{new}' instead",
                                  category=FutureWarning, # this is a warning for end-users rather than programmers
                                  stacklevel=2)
            return f(*args, **kwargs)
        return fnew
    return wrapper

