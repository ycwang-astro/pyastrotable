# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang

Main tools to store, operate and visualize data tables.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Column, Table, hstack
from astrotable.utils import objdict, save_pickle, load_pickle, keyword_alias, bitwise_all, pause_and_warn
import astrotable.plot as plot
import astrotable
import warnings
import multiprocessing as mp
from collections.abc import Iterable
from collections import OrderedDict, Counter
import inspect
from itertools import repeat, chain
import os
import zipfile
import pickle
from copy import deepcopy
import json
import re
from keyword import iskeyword
# import time

try:
    import pandas as pd # used to handle pd.DataFrame input
except ImportError:
    has_pd = False
else:
    has_pd = True

subplot_arrange = {
    1: [1, 1],
    2: [1, 2],
    3: [1, 3],
    4: [2, 2]
    }

plot_funcs = {
    'plot': plot.plot,
    'scatter': plot.scatter,
    'hist': plot.hist,
    'hist2d': plot.hist2d,
    'errorbar': plot.errorbar,
    }

plot_array_funcs = plot_funcs

# plot_array_funcs = {
#     'plot': lambda ax: ax.plot,
#     'scatter': scatter_with_colorbar, # lambda ax: ax.scatter,
#     'hist': lambda ax: ax.hist,
#     'hist2d': lambda ax: ax.hist2d,
#     'errorbar': lambda ax: ax.errorbar,
#     }

class FailedToLoadError(Exception):
    pass

class FailedToEvaluateError(Exception):
    pass

class SubsetMergeError(Exception):
    pass

class SubsetNotFoundError(LookupError):
    def __init__(self, name, kind='path'):
        if kind == 'path':
            info = f"'{name}'. Maybe missing/incorrect group name or incorrect subset name?"
        elif kind in ['subset', 'name']:
            info = f"'{name}'"
        else:
            raise ValueError(f"unknown kind '{kind}'")
        super().__init__(info)
        
class GroupNotFoundError(KeyError):
    pass

class Subset():
    '''
    A class to specify a row subset of an ``astrotable.table.Data`` object.
    Although this class is independent to the ``Data`` class, it should be only used together with a ``Data`` object.
    
    To specify the selection criteria, name, etc. of a subset, the general way is::

        Subset(<selection>, name=<name>, <...>)

    Convenient methods for specifying a subset are::

        Subset.by_range(<column name>=<value range>, <...>)
        Subset.by_value(<column name>, <value>)

    See ``help(Subset.__init__)``, ``help(Subset.by_range)`` and ``help(Subset.by_value)`` for more information.
    
    In practice, a subset of ``data`` is usually defined as:
        
        >>> subset = data.add_subsets(Subset(<...>))
    
    You may also define multiple subsets at a time:
        
        >>> subset1, subset2 = data.add_subsets(
        ...     Subset(<...>),
        ...     Subset(<...>))
    
    ``Subset`` objects can be used as if they are arrays (for most cases).
    For example, you can get the intersection set ``subset1 & subset2``,
    the union set ``subset1 | subset2``, and the complementary set ``~subset1``.
    
    Notes for developers
    --------------------
    Currently, the '&', '|', '~' operations can only be performed if selection is an boolean array,
    or ``Subset.eval_`` is called. (This is always called when a subset is defined in ``data.add_subsets()``.)
    Operation before evalutaion may be supported in the future.
    '''
    def __init__(self, selection, name=None, expression=None, label=None):
        '''
        Specify a subset of ``Data``.

        Parameters
        ----------
        selection : callable (e.g. function), iterable (e.g. array-like) or string
            If it is iterable, it should be a boolean array indicating whether each row is included in this subset.
            It should have a shape of ``(len(data),)`` where ``data`` is an ``astrotable.table.Data`` instance.  
        
            If it is callable, should be defined like 
            
                >>> def selection(table): # input: astropy.table.Table object
                ...     <...>
                ...     return arr # boolean array 
                ...                # whether each row is included in subset 
            
            If it is a string, should be expressions using the column names of the data, e.g.
            ``'(column1 > 0) & (column2 < 1)'``.
        name : str, optional
            The name of the subset. The default is None.
        expression : str, optional
            The expression [e.g. '(col1 > 0) & (col2 == "A")'] used to recognize the conditions. 
            The default is None.
        label : str, optional
            The label used in figures. 
            The default is None
            
        Notes
        -----
        The inputs of ``__init__()`` will be attributes of the object.
        By executing the ``eval_`` method, an ``astrotable.table.Data`` object ``data`` is inputted, and:
            - The attribute ``selection`` will be converted to a boolean array;
            - The attribute ``name`` will be set to the default name if it is None;
            - The attribute ``expression`` will be automatically set if it is None;
            - The attribute ``label`` will be set to ``name`` if it is None; strings will be replaced 
              according to the mapping of dict ``data.col_labels``.
              
        If the input/evaluation of ``selection`` is/results in a masked (boolean) array, the masked elements 
        are always filled with False (which means that they do NOT belong to this subset).
        This often happens when ``selection`` is calculated from a masked column of the table.
        The final ``selection`` after executing the ``eval_`` method is never a masked array.

        '''
        if name is not None and '/' in name:
            raise ValueError('"/" not supported in Subset names')
        
        self.selection = selection
        self.name = name 
        self.expression = expression
        self.label = label
        self.data_name = None
        
    @classmethod
    def by_range(cls, **ranges):
        '''
        Initializes a subset by specifying ranges for the data.
        
        For example, ``Subset.by_range(col1=[0, 1], col2=[0, np.inf])`` 
        defines a subset with `(0 < col1 < 1) & (col2 > 0)`.

        Parameters
        ----------
        **ranges : key - value pairs:
            key : str
                Name of the column in the data. 
            value : list or tuple (or other similar objects) with length=2
                List of 2 numbers, e.g. ``[0, 1]``, specifying a range of the column.

        Returns
        -------
        ``astrotable.table.Subset``

        '''
        # get Subset from range
        def selection(t):
            selected = True
            for col, range_ in ranges.items():
                selected &= (t[col] > range_[0]) & (t[col] < range_[1])
            return selected # the boolean array
        name = '&'.join([f'{col}({range_[0]}-{range_[1]})' for col, range_ in ranges.items()])
        expression = ' & '.join([f'({col} > {range_[0]}) & ({col} < {range_[1]})' for col, range_ in ranges.items()])
        label = ', '.join([f'{col}$\\in$({range_[0]}, {range_[1]})' for col, range_ in ranges.items()])
       
        name = Subset._remove_slash(name)
        return cls(selection, name=name, expression=expression, label=label)
      
    @classmethod
    def by_value(cls, column, value):
        '''
        Initializes a subset by specifying the exact value of column.

        Parameters
        ----------
        column : str
            Name of the data column.
        value : 
            Value of the column.

        Returns
        -------
        ``astrotable.table.Subset``

        '''
        def selection(t):
            return t[column] == value # the boolean array
        name = f'{column}={value}'
        expression = name
        label = value if type(value) in [str, np.str_] else f'{column}$=$' + '$\\mathrm{' + f'{value}' + '}$'

        name = Subset._remove_slash(name)
        return cls(selection, name=name, expression=expression, label=label)
    
    @staticmethod
    def _remove_slash(name, rep='_'):
        if '/' in name:
            oldname = name
            name = oldname.replace('/', rep)
            warnings.warn(f"'/' is not supported as a subset name. '{oldname}' renamed to '{name}'.")
        return name
    
    def eval_(self, data, existing_keys=()):
        '''
        Evaluate the selection array, expression, name and label, given data.
        This method should be executed if, self.selection is not a boolean array
        OR either self.name or self.expression is None.

        Parameters
        ----------
        data : ``astrotable.table.Data``
            
        existing_keys : Iterable, optional
            Names of subsets that already exists. 
            This is used to generate automatic subset names.
            The default is ().
        '''
        self.data_name = data.name
        
        # get selection array and expression
        if callable(self.selection):
            if self.expression is None: 
                self.expression = inspect.getsource(self.selection)
            self.selection = self.selection(data.t)
        elif type(self.selection) is str:
            if self.expression is None: 
                self.expression = self.selection
            
            if self.selection in ['all', 'All']:
                self.selection = np.full(len(data), True)
            
            else:
                self.selection = data.eval(self.selection)
                
                ### old implementation below:
                # # check string: avoid error if the string contains something like "self", "data" but are not real column names
                # names = list(chain(locals(), globals()))
                # for name in names:
                #     if name in ['np', 'os']:
                #         continue
                #     if name in self.selection and name not in data.colnames:
                #         raise KeyError(name)
                
                # for colname in data.colnames: # replace colnames to expression
                #     self.selection = self.selection.replace(colname, f"data.t['{colname}']")
                
                # try:
                #     print('[subset] evaluating', self.selection)
                #     self.selection = eval(self.selection)
                # # except NameError as e:
                # #     raise KeyError(e.name)
                # except Exception as e:
                #     raise FailedToEvaluateError(f"Auto-generated expression cannot be evaluated: ({self.selection}). Check your input (see above for error) or try other methods to specify a subset.") from e
                # except:
                #     raise
                ### end
                
        elif isinstance(self.selection, Iterable):
            if len(self.selection) != len(data):
                raise ValueError(f'length of array should be {len(data)}, got {len(self.selection)}.')
            if self.expression is None: 
                self.expression = '<array>'
            if not isinstance(self.selection, np.ndarray):
                self.selection = np.array(self.selection)
                # else: if it is masked, keep it masked (otherwise np.array(self.selection) will get its data only)
            if self.selection.dtype != bool:
                self.selection = self.selection.astype(bool)
            
            self.selection = self.selection.copy()

        else:
            raise TypeError(f"keyword argument should be function, array-like object or string, got '{type(self.selection)}'.")

        # directly fill masked to False (this makes a difference when using e.g. __or__, __invert__)
        if np.ma.is_masked(self.selection):
            self.selection = self.selection.filled(False) # IMPORTANT: Masked elements do NOT belong to this subset!

        # get name
        if self.name is None:
            i = 0
            while f'subset{i}' in existing_keys:
                i += 1
            self.name = f'subset{i}'
        
        # get label
        if self.label is None:
            self.label = self.name
            
        # replace colname with label
        # TODO: this is not robust. This should be used to modify the labels for subset
        # initialized by Subset.by_range and Subset.by_value, because they do not know the labels during
        # initialization. If a subset is initialized by Subset() and label is given by the user, 
        # it should not be modified.
        for colname, labelstr in data.col_labels.items():
            self.label = self.label.replace(colname, labelstr)
    
    def _cut(self, index, data_name):
        # return a cutted Subset (cutted with ``index``)
        cutted_subset = Subset(
            self.selection[index], # Note: this is not a copy of self.selection. This is not a problem, since the array it refers to is never modified in the code.
            self.name,
            self.expression,
            self.label,
            )
        cutted_subset.data_name = data_name
        return cutted_subset
    
    @property
    def size(self): # the size of the subset
        return np.sum(np.array(self))
    
    def __and__(self, subset): # the & (bitwise AND)
        selection = self.selection & subset.selection
        name = f'{self.name} & {subset.name}'
        expression = f'({self.expression}) & ({subset.expression})'
        if self.label != 'All' and subset.label != 'All':
            label = f'{self.label}, {subset.label}'
        elif self.label == 'All':
            label = f'{subset.label}'
        else: # subset.label == 'All'
            label = f'{self.label}'
        
        new_subset = Subset(selection, name, expression, label)
        if self.data_name == subset.data_name:
            new_subset.data_name = self.data_name
        return new_subset
    
    def __or__(self, subset): # the | (bitwise OR)
        selection = self.selection | subset.selection
        name = f'{self.name} | {subset.name}'
        expression = f'({self.expression}) | ({subset.expression})'
        label = f'[{self.label}] or [{subset.label}]'
        
        new_subset = Subset(selection, name, expression, label)
        if self.data_name == subset.data_name:
            new_subset.data_name = self.data_name
        return new_subset
    
    def __invert__(self): # the ~ (bitwise NOT)
        selection = ~self.selection
        name = f'~({self.name})'
        expression = f'~({self.expression})'
        label = f'not [{self.label}]'
        
        new_subset = Subset(selection, name, expression, label)
        new_subset.data_name = self.data_name
        return new_subset
    
    def __array__(self):
        if not hasattr(self.selection, 'dtype') or self.selection.dtype != bool: #not isinstance(self.selection, Iterable):
            raise RuntimeError('Selection should be a boolean array. Maybe forgot to run eval_()?')
        if np.ma.is_masked(self.selection): # this never happens after I directly fill masked to False. This is kept to handle instances of old versions.
            return self.selection.filled(False) # IMPORTANT: Masked elements do NOT belong to this subset!
        else:
            return self.selection
    
    def __len__(self):
        # this is actually (and should be) the same as len(data).
        return len(np.array(self))
    
    def __repr__(self):
        # return f"Subset('{self.selection}')"
        # return f"Subset(name='{self.name}', selection={self.selection.__repr__()})"
        datastr = f" of Data '{self.data_name}'" if self.data_name is not None else ''
        return f"<Subset '{self.name}'{datastr} ({self.size}/{len(self)})>"
    
    def __setstate__(self, state):
        # Call __init__() to initialize some attributes in case not provided by state.
        # this can be useful when restoring Subset from a pkl file of an older version (that may lack ``self.data_name``).
        self.__init__(None)
        # Restore instance attributes
        self.__dict__.update(state)
        

class Data():
    '''
    A class to store, manipulate and visualize data tables.
    
    Notes
    -----
    - The data table of a ``Data`` instance (i.e. ``data.t``) is not expected to be changed since creation. 
      If ``data.t`` is changed, the matching and subset information may be inconsistent with the table.
      Create a new ``Data`` instance instead.
    '''
    def __init__(self, data, name=None, **kwargs):
        '''
        

        Parameters
        ----------
        data : str, `astropy.table.Table`, etc.
            path to the data file, an `astropy.table.Table` object, or anything that can be initialized as an `astropy.table.Table` object by `astropy.table.Table.read`.
        name : str, optional
            The name of this Data object. This name will be used in many cases to distinguish datasets. The default is None.
        **kwargs : 
            Keyword arguments passed to `astropy.table.Table.read`, 
            if a str is passed to argument `data`.

        '''
        if type(data) is str and 'format' in kwargs and kwargs['format'] in ['data', 'pkl']: # should use Data.load
            raise ValueError(f"to load data file saved with Data.save, use Data.load('{data}', format='{kwargs['format']}')")
        
        if name is None:
            warnings.warn('It is recommended to input a name.',
                          stacklevel=2)

        # TODO: save a pkl (or other formats) file while reading an ascii file,
        # so that the next time this ascii file is read (if not modified), use the pkl
        # file to accelerate data loading process.
        
        # get data
        if type(data) is str: # got a path
            self.t = Table.read(data, **kwargs)
            self.path = data
        elif isinstance(data, Table): # got astropy table
            self.t = data
            self.path = '(initialized from Table)'
        # for large ascii files, loading with pd abd converting it to astropy.table.Table seems to be faster
        elif has_pd and type(data) == pd.DataFrame:
            self.t = Table.from_pandas(data)
            self.path = '(initialized from DataFrame)'
        else: # try to convert to data
            self.t = Table(data)
            self.path = f'(initalized from a {type(data)} object)'
        self.meta['path'] = self.path
         
        # basic properties
        self.name = name
        if self.name is None and self.path is not None:
            self.name = self.path.split('/')[-1].split('\\')[-1]
        # self.id = time.time() #time.strftime('%y%m%d%H%M%S')
        
        # set metadata for columns (TODO: experimental feature)
        if type(data) is str: # got a path
            for colname in self.colnames:
                col = self.t[colname]
                if 'src' not in col.meta.keys():
                    col.meta['src'] = self.name
                    col.meta['src_detail'] = f'Loaded "{self.path}"'
                    col.meta['set_by_user'] = False # whether the value of this column is modified by the user. 
                    # TODO: CAUTION: modification can only be detected when using data[] instead of data.t[]
        else:
            pass # TODO: not safe to set metadata for other input format, as they may have their own metadata
        
        # matching
        self.matchinfo = []
        self.matchnames = []
        self.matchlog = []
        
        # subset
        self.subset_all = Subset(np.ones(len(self)).astype(bool), name='all', expression='all', label='All') # subset named "all"
        self.subset_all.data_name = self.name
        self.subset_groups = {
            'default': {'all': self.subset_all}
            }
        
        # plot
        self.col_labels = {} # {column_name: label_in_figures}
        self.plot_axes = None # the axes for the last plot
        self.plot_fig = None # the fig for the last plot
        self.plot_returns = [] # the returns of the last plot
    
    #### properties
    @property
    def colnames(self):
        return self.t.colnames
    
    @property
    def meta(self):
        return self.t.meta
    
    #### matching & merging 
    def match(self, data1, matcher, verbose=True, replace=False):
        '''
        Match this data object with another `astrotable.table.Data` object `data1`.

        Parameters
        ----------
        data1 : `astrotable.table.Data`
            Data to be matched to this Data.
        matcher : any recognized matcher object
            A matcher object used to match the two data objects.
            Built-in matchers includes, e.g., `astrotable.matcher.ExactMatcher` and `astrotable.matcher.SkyMatcher`.
            See e.g. `help(astrotable.matcher.SkyMatcher)` for more information.
            
            A matcher object should be defined like below:
                
                >>> class MyMatcher():
                ...     def __init__(self, args): # 'args' means any number of arguments that you need
                ...         # initialize it with args you need
                ...         pass
                ...     
                ...     def get_values(self, data, data1, verbose=True): # data1 is matched to data
                ...         # prepare the data that is needed to do the matching (if necessary)
                ...         pass
                ...     
                ...     def match(self):
                ...         # do the matching process and calculate:
                ...         # idx : array of shape (len(data), ). 
                ...         #     the index of a record in data1 that best matches the records in data
                ...         # matched : boolean array of shape (len(data), ).
                ...         #     whether the records in data can be matched to those in data1.
                ...         return idx, matched
        verbose : bool, optional
            Whether to output matching information. The default is True.
        replace : bool, optional
            When ``data1`` (Data to be matched) has the same name as a Data object that has already been matched to this Data,
            whether to replace the old matching.
            If False, a ValueError is raised.
            The default is False.

        Raises
        ------
        ValueError
            - Data with the same name to be matched to this Data twise.
        
        Returns
        -------
        
        
        '''
        if not (isinstance(data1, Data) or type(data1) == type(self)):
            raise TypeError(f"only supports matching 'astrotable.table.Data' type; got {type(data1)}")
        
        if data1.name in self.matchnames:
            if replace:
                self.unmatch(data1)
            else:
                raise ValueError(f"Data with name '{data1.name}' has already been matched. This may result from name duplicates or re-matching the same catalog. \
Set 'replace=True' to replace the existing match with '{data1.name}'.")
        
        matcher.get_values(self, data1, verbose=verbose)
        idx, matched = matcher.match()
        info = objdict(
            matcher = matcher,
            data1 = data1,
            idx = idx,
            matched = matched,
            )
        self.matchinfo.append(info)
        self.matchnames.append(data1.name)
        
        matchstr = f'"{data1.name}" matched to "{self.name}": {np.sum(matched)}/{len(matched)} matched.'
        self.matchlog.append(matchstr)
        if verbose: print('[match] ' + matchstr)
        
        return self
    
    def unmatch(self, data1, verbose=True):
        '''
        Remove match to ``data1``.

        Parameters
        ----------
        data1 : ``astrotable.table.Data`` or str
            The Data or the name of the Data.
        verbose : bool, optional
            Whether to output information. The default is True.

        Returns
        -------
        None.

        '''
        # warnings.warn('Data.unmatch method not tested')
        if (isinstance(data1, Data) or type(data1) == type(self)):
            name1 = data1.name
        elif type(data1) is str:
            name1 = data1
        else:
            raise TypeError(f"only supports 'astrotable.table.Data' or str; got {type(data1)}")
        
        if name1 not in self.matchnames:
            warnings.warn(f"Data with name '{data1.name}' has never been matched. Nothing is done.")
            return
        
        self.matchinfo = [info for info in self.matchinfo if info.data1.name != name1]
        self.matchnames.remove(name1)
        
        unmatchstr = f'"{name1}" unmatched to "{self.name}".'
        if verbose: print('[match] ' + unmatchstr)
        self.matchlog.append(unmatchstr)
    
    def reset_match(self):
        '''
        Remove all match information.

        '''
        self.matchinfo = []
        self.matchlog = []
        self.matchnames = []
    
    def _match_propagate(self, idx=None, matched=None, depth=-1, ignore_id=[]):
        '''
        Propagate all of the match to self's "child" data to self's "parent" data.

        Parameters
        ----------
        idx : Iterable, optional
            'idx' information for self matched to parent data. The default is None.
        matched : Iterable, optional
            'matched' information for self matched to parent data. The default is None.
        depth : int, optional
            Depth. The default is -1.
        ignore_id : Iterable, optional
            Data id to be ignored. The default is [].

        Returns
        -------
        idxs, matcheds : list
            The 'idx', 'matched' information for (all of self's child data) matched to (self's parent data).

        '''
        ignore_id = ignore_id.copy()
        
        if id(self) not in ignore_id:
            ignore_id.append(id(self))
        
        if idx is None:
            idx = np.arange(len(self))
        if matched is None:
            matched = np.full((len(self),), True)
        
        data1s, idxs, matcheds = [], [], []
        if depth != 0:
            for info in self.matchinfo: # analyze the child data of self
                data1 = info.data1
                if id(data1) in ignore_id:
                    continue
                
                # get match info for data1  (self's parent to self's child "data1")
                idx_s = info.idx # _s: self - child match
                matched_s = info.matched
                
                idx_temp = idx.copy()
                # l_p = len(idx) # length of parent data
                idx_temp[~matched] = 0
                idx_ps = idx_s[idx_temp] # _ps: parent - child match
                matched_ps = matched_s[idx_temp]
                matched_ps &= matched
                idx_ps[~matched_ps] = -len(data1) - 1
                
                data1s.append(data1)
                ignore_id.append(id(data1))
                idxs.append(idx_ps)
                matcheds.append(matched_ps)
                
                # ask data1 to give me all of its child data
                data1_data1s, data1_idxs, data1_matcheds, ignore_id = data1._match_propagate(idx=idx_ps, matched=matched_ps, depth=depth-1, ignore_id=ignore_id)
                data1s += data1_data1s
                idxs += data1_idxs
                matcheds += data1_matcheds
        
        return data1s, idxs, matcheds, ignore_id
                
    def merge_matchinfo(self, depth=-1):
        '''
        Merge the matchinfo for all of the children data of this data,
        so that each info is the match with repect to **this** data.
        If there are duplicates in the child data, only the first found is used.

        Parameters
        ----------
        depth : int, optional
            The depth of merging.
            For example, if depth == 1, only the direct children (without grandchildren) of 
            this data are merged.
            if depth == -1, all children (including all grandchildren) are merged.
            The default is -1.

        Returns
        -------
        outinfo : list of objdicts
            .

        '''
        outinfo = []
        data1s, idxs, matcheds, _ = self._match_propagate(depth=depth)
        for data1, idx, matched in zip(data1s, idxs, matcheds):
            outinfo.append(objdict(
                matcher = None,
                data1 = data1,
                idx = idx,
                matched = matched,
                ))
        return outinfo
    
    @staticmethod
    def _cut_subset_groups(subset_groups, index, new_name):
        # cut each subset with [index]
        subset_groups = deepcopy(subset_groups)
        for group in subset_groups:
            for subset in subset_groups[group]:
                subset_groups[group][subset] = subset_groups[group][subset]._cut(index, new_name)
        return subset_groups
    
    @staticmethod
    def _mask_subset_groups(subset_groups, index, in_place=True, val=False):
        # change all subsets' [index] values to val (False).
        if not in_place:
            subset_groups = deepcopy(subset_groups)
        for group in subset_groups:
            for subset in subset_groups[group]:
                subset_groups[group][subset].selection[index] = val
        return subset_groups
       
    
    @staticmethod
    def _merge_subset_groups(data_subset_groups, data_names, verbose=True):
        # get all
        all0 = data_subset_groups[0]['default']['all']
        l = len(all0)
        subset_all = Subset(np.ones(l).astype(bool), name='all', expression='all', label='All') # subset named "all"
        subset_all.data_name = all0.data_name
        
        merged_subset_groups = {
            'default': {'all': subset_all},
            }
        # data_groupnames = 
        
        ## merge 'default' group ##
        for i in range(len(data_names)):
            datai_subset_groups, data_name = data_subset_groups[i], data_names[i]
            datai_default = datai_subset_groups['default']
            other_names = [list(g['default']) for g in data_subset_groups]
            other_names.pop(i)
            for name, subset in datai_default.items():
                outname = name
                if name == 'all':
                    # if 'all' not in merged_subset_groups['default']:
                    #     merged_subset_groups['default']['all'] = subset
                    continue
                if any(name in other_namesi for other_namesi in other_names):
                    outname = '_'.join([name, data_name])
                    if verbose:
                        print(f"[merge] subset renamed: {name} -> {outname}")
                if outname in merged_subset_groups['default']:
                    raise SubsetMergeError(f"Subset merging results in name duplicates: '{outname}'. "
                                           f"You may change the name of your data to avoid duplicate Data names ('{data_name}').")
                merged_subset_groups['default'][outname] = subset
                    
        ## merge other groups ##
        for i in range(len(data_names)):
            datai_subset_groups, data_name = data_subset_groups[i], data_names[i]
            other_names = [list(g) for g in data_subset_groups]
            other_names.pop(i)
            for name, group in datai_subset_groups.items():
                outname = name
                if name == 'default':
                    continue
                if any(name in other_namesi for other_namesi in other_names):
                    outname = '/'.join([data_name, name])
                    if verbose:
                        print(f"[merge] group renamed: {name} -> {outname}")
                if outname in merged_subset_groups:
                    raise SubsetMergeError(f"Subset merging results in name duplicates: '{outname}'. "
                                           f"You may change the name of your data to avoid duplicate Data names ('{data_name}').")
                merged_subset_groups[outname] = group
        
        return merged_subset_groups
    
    @staticmethod
    def _decide_missing_value(col):
        # decide the value representin missing values
        # refer to https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
        kind = col.dtype.kind
        if np.ma.is_masked(col):
            col = col[~col.mask]
            
        miss = None
        # TODO: below
        if kind in 'b': # boolean
            pass
        elif kind in 'i': # signed int
            miss = -99
            while np.min(col) < miss/9:
                miss = (miss - 1)*10 + 1
        elif kind in 'u': # unsigned int
            miss = 99
            while np.max(col) > miss/9:
                miss = (miss + 1) * 10 - 1
        elif kind in 'f': # float
            miss = np.nan
        elif kind in 'c': # complex float
            pass
        elif kind in 'm': # timedelta
            pass
        elif kind in 'M': # datetime
            pass
        elif kind in 'O': # object
            miss = 'N/A'
        elif kind in 'S': # (byte-)string
            pass
        elif kind in 'U': # Unicode
            n = col.dtype.itemsize / 4
            if n == 1:
                return '?'
            elif n == 2:
                return 'NA'
            else:
                return 'N/A'
        elif kind in 'V': # void
            pass
        return miss
    
    def merge(self, depth=-1, keep_unmatched=[], merge_columns={}, ignore_columns={}, outname=None, keep_subsets=False, matchinfo_subset=False, verbose=True):
        '''
        Merge all data objects that are matched to this data.
        
        The data that are directly matched to this data are called "chilren" of this data, and are on "depth 1".
        The data directly matched to data on "depth 1" are on "depth 2", etc.

        Parameters
        ----------
        depth : int, optional
            The depth of merging.
            For example, if depth == 1, only the direct children (without grandchildren) of 
            this data are merged.
            if depth == -1, all children (including all grandchildren) are merged.
            The default is -1.
        keep_unmatched : Iterable, optional
            A list of names of `astrotable.table.Data` objects (you can check the names with e.g. `data.name`).
            A record (row) of THIS data is kept even if a dataset in the above list cannot be matched to this data.
            The default is [] (which means that only those that can be matched to each child data of this data are kept).
        merge_columns : dict, optional
            A dict that specifies fields (columns) to be merged.
            For example, if ``data1`` with name 'Data_1' is matched to this object, and you want to merge only 
            'column1', 'column2' in ``data1`` into the merged catalog, use:
                {'Data_1': ['column1', 'column2']}
            If, e.g, ``merge_columns`` for ``data2`` (with name 'Data_2') is not specified, every fields (columns) of ``data2`` will be merged.
            The default is {}.
        ignore_columns : dict, optional
            A dict that specifies fields (columns) not to be merged.
            Similar to argument ``merge_columns``.
            If both ``merge_columns`` and ``ignore_columns`` are specified for a field, 
            the columns IN ``merge_columns`` AND NOT IN ``ignore_columns`` are merged.
            The default is {}.
        outname : str, optional
            The name of the merged data. 
            If not given, will be automatically generated from the names of data that are merged.
            The default is None.
        keep_subsets : bool, optional
            Whether the subsets of the data are kept and merged. The default is False.
        matchinfo_subset : bool, optional
            If ``keep_unmatched != []``, whether to add a subset 'matched/<this_data_name>/<name_of_data_matched_to_this_data>',
            indicating whether each row can be matched to that data.
            The default is False.
        verbose : bool, optional
            Whether to show more information on merging. The default is True.

        Returns
        -------
        matched_data : ``astrotable.table.Data``
            An ``astrotable.table.Data`` object containing the merged catalog.

        Notes
        -----
            If the ``keep_unmatched`` is not empty (``[]``), say ``keep_unmatched=['data1']``. 
            Then, the rows in THIS data that has no match with the dataset called 'data1' are kept,
            and the columns from 'data1' for this row are missing values.
            
            'data1' may also have its subsets. When ``keep_subsets`` is set to True, the subsets of 'data1'
            are also merged. The rows with no match with 'data1' always do NOT belong to the subsets merged 
            from 'data1'.
        '''
        if type(keep_unmatched) is str:
            keep_unmatched = [keep_unmatched]
        elif keep_unmatched is True:
            keep_unmatched, _, _ = self._match_tree(depth=depth, silent=True)
            keep_unmatched = keep_unmatched[1:]
            if verbose: print(f'[merge] `keep_unmatched` set to all subsets matched to {self}: {keep_unmatched}')
        
        matched = np.full((len(self),), True) # whether a record in self is matched to ALL the child data
        data1_matched_tables = []
        unnamed_count = 0
        data_names = [self.name]
        data_metas = {self.name: self.meta}
        data_subset_groups = [] # list of cutted subset groups for each data
        
        merged_matchinfo = self.merge_matchinfo(depth=depth)
        
        ## get matched indices
        for matchinfo in merged_matchinfo:
            data1 = matchinfo.data1
            if data1.name not in keep_unmatched:
                data1_matched = matchinfo.matched
                matched &= data1_matched # boolean array indicating whether a row of the base data is matched to ALL the child data
                
            if data1.name is None:
                unnamed_count += 1
                data_names.append(str(unnamed_count))
                data_metas[str(unnamed_count)] = data1.meta
            else:
                data_names.append(data1.name)
                data_metas[data1.name] = data1.meta
                
        if outname is None:
            outname = 'match_' + '_'.join(data_names)
            
        if unnamed_count > 0 and verbose:
            print(f'found no names for {unnamed_count} sets of data, automatically named with numbers.')
        
        ## cut data and subsets (if needed) ##
        # cut myself
        data = self.t[matched]
        if self.name in merge_columns:
            data.keep_columns(merge_columns[self.name])
        if self.name in ignore_columns:
            data.remove_columns(ignore_columns[self.name])
        
        if keep_subsets:
            subset_groups = Data._cut_subset_groups(self.subset_groups, matched, outname)
            data_subset_groups.append(subset_groups)
        
        subsets_to_be_added = [] # will be used if matchinfo_subset
        
        # cut data matched to me
        for matchinfo in merged_matchinfo:
            data1 = matchinfo.data1
            idx = matchinfo.idx
            data1_matched = matchinfo.matched
            
            data1_table = data1.t.copy()
            
            if data1.name in merge_columns:
                data1_table.keep_columns(merge_columns[data1.name])
            if data1.name in ignore_columns:
                data1_table.remove_columns(ignore_columns[data1.name])
            
            if data1.name in keep_unmatched: # keep unmatched
                if verbose: print(f'[merge] entries with no match for {data1.name} is kept.')
                idx[~data1_matched] = 0 # TODO: DANGER: The masked "data" will be valid values, i.e. the value on the first row! These may emerge when using e.g. np.array, plt.hist2d (which uses np.histogram2d).
                data1_table = Table(data1_table, masked=True)
                data1_matched_table = data1_table[idx]
                for c in data1_matched_table.columns:
                    miss_val = Data._decide_missing_value(data1_matched_table[c])
                    if miss_val is not None:
                        data1_matched_table[c][~data1_matched] = miss_val
                    data1_matched_table[c].mask[~data1_matched]=True
                data1_matched_table = data1_matched_table[matched]
                
                if keep_subsets:
                    subset_groups = Data._cut_subset_groups(data1.subset_groups, idx, outname)
                    subset_groups = Data._mask_subset_groups(subset_groups, ~data1_matched)
                    subset_groups = Data._cut_subset_groups(subset_groups, matched, outname)
                    
                if matchinfo_subset:
                    subsets_to_be_added.append(Subset(
                        data1_matched, 
                        name=data1.name, 
                        expression=f"<'{data1.name}' matched to '{self.name}'>",
                        label=f"'{data1.name}' matched to '{self.name}'"))

            else: # do not keep unmatched
                data1_matched_table = data1_table[idx[matched]]
                
                if keep_subsets:
                    subset_groups = Data._cut_subset_groups(data1.subset_groups, idx[matched], outname)
            
            data1_matched_tables.append(data1_matched_table)
            
            if keep_subsets:
                data_subset_groups.append(subset_groups)
               
        # merge table and get data
        matched_table = hstack([data] + data1_matched_tables, table_names=data_names)
        matched_data = Data(matched_table, name=outname)
        matched_data.meta['path'] = '(merged data)'
        
        # merge subsets
        if keep_subsets:
            merged_subset_groups = Data._merge_subset_groups(data_subset_groups, data_names)
            matched_data.subset_groups = merged_subset_groups
        
        # add subsets if matchinfo_subset
        if matchinfo_subset:
            group = f'matched/{self.name}'
            if group in matched_data.subset_groups:
                raise RuntimeError(f"A subset group named '{group}' already exists.")
            matched_data.add_subsets(*subsets_to_be_added, group=group)
        
        if verbose: print('[merge] merged: ' + ', '.join(data_names))
        
        ## generate data meta
        assert list(data_metas.keys()) == data_names
        matched_names, _, tree_str = self._match_tree(depth=depth, silent=True)
        assert data_names == matched_names
        merging = OrderedDict({ # detailed information for merging
            'notes': 'This is a table merged from several tables. The merging information is recorded below.'\
                     'The metadata for merged datasets are recorded in "metas".',
            'options': dict(
                depth=depth,
                keep_unmatched=keep_unmatched,
                keep_subsets=keep_subsets,
                matchinfo_subset=matchinfo_subset,
                ),
            'tree': tree_str, # the match tree of the base data
            'merged': data_names, # names of the data merged
            'metas': data_metas, # metas for the data merged
            })
        matched_data.meta['merging'] = merging
        
        return matched_data
    
    def from_which(self, colname=None, detail=True):
        '''
        When reading a dataset from a file using ``Data(<path>, name=<name>)``, 
        the name of the data is associated with each columns. 
        After matching and merging it with other datasets, you may want to 
        check the name of the data from which ``colname`` is matched.
        See examples below.
        
        **WARNING**: The information for user-added columns may be invalid.

        Parameters
        ----------
        colname : str, optional
            Column name.
            If this argument is not given, a dict with the information for all columns 
            will be returned.
        detail : bool, optional
            Whether the detail of the data is returned. The default is True.
        
        Returns
        -------
        str or dict
            The name (str) of the data from which ``colname`` is matched,
            or a dict containing the information for all columns.

        Examples
        --------
        Say you have two catalog files, ``cat1.csv`` and ``cat2.csv``.
        
            >>> cat1 = Data('cat1.csv', name=cat1) # with columns 'col1', etc.
            >>> cat2 = Data('cat2.csv', name=cat2) # with columns 'col2', etc.
            >>> cat_merged = cat1.match(cat2, SkyMatcher()).merge()
            ... # cat_merged has columns 'col1', 'col2', etc.
            >>> cat_merged.from_which('col1')
            cat1 (loaded from "cat1.csv")
            >>> cat_merged.from_which('col2')
            cat2 (loaded from "cat2.csv")
            
        '''
        warnings.warn('WARNING: The information for user-added columns may be invalid.')
        if colname is None:
            return OrderedDict((name, self.from_which(name, detail=detail)) for name in self.colnames)
        elif colname not in self.colnames:
            raise KeyError(colname)
        else:
            meta = self.t[colname].meta
            if 'src' in meta.keys():
                src, src_detail = meta['src'], meta['src_detail']
                info = src
                if detail:
                    info += f' ({src_detail})'
                return info
            else:
                return ''
    
    def match_merge(self, data1, matcher, keep_unmatched=[], merge_columns={}, ignore_columns={}, outname=None, verbose=True):
        '''
        Match this data with ``data1`` and immediately merge everything that can be matched to this data.
        See ``help(astrotable.table.Data.match)`` and ``help(astrotable.table.Data.merge)`` for more information.
        '''
        self.match(data1=data1, matcher=matcher, verbose=verbose)
        return self.merge(keep_unmatched=keep_unmatched, merge_columns=merge_columns, ignore_columns=ignore_columns, outname=outname, verbose=verbose)
    
    def _match_tree(self, depth=-1, detail=True, matched_names=[], matched_ids=[], indent='', matcher='base', tree_str='', silent=False):
        # copy lists to avoid modifying it in-place (which will cause the method to "remember" them!)
        matched_names = matched_names.copy()
        matched_ids = matched_ids.copy()
        
        # print this name
        matcher = '' if not detail else f' [{matcher}]'
        name = 'Unnamed' if self.name is None else self.name
        if id(self) in matched_ids: # this data is already matched 
            print_str = f'{indent}({name}){matcher}'
            if not silent: print(print_str)
            tree_str += print_str + '\n'
            return matched_names, matched_ids, tree_str # do not expand this anymore
        else:
            matched_names.append(name)
            matched_ids.append(id(self))
            print_str = f'{indent}{name}{matcher}'
            if not silent: print(print_str)
            tree_str += print_str + '\n'
       
        # print data matched to this
        if depth != 0:
            for info in self.matchinfo:
                data = info.data1
                matcher = info.matcher
                matched_names, matched_ids, tree_str = data._match_tree(depth=depth-1, detail=detail, matched_names=matched_names, matched_ids=matched_ids, indent=indent+':   ', matcher=matcher, tree_str=tree_str, silent=silent)
        return matched_names, matched_ids, tree_str
            
    def match_tree(self, depth=-1, detail=True):
        '''
        Generate a "match tree", showing all data that can be matched and merged to this data.
        
        The data that are directly matched to this data are called "chilren" of this data, and are on "depth 1".
        The data directly matched to data on "depth 1" are on "depth 2", etc.

        Parameters
        ----------
        depth : int, optional
            The depth.
            For example, if depth == 1, only the direct children (without grandchildren) of 
            this data are shown.
            if depth == -1, all children (including all grandchildren) are shown.
            The default is -1.
        detail : bool, optional
            Whether to show detail (including how the data are matched). The default is True.

        '''
        print('Names with parentheses are already matched, thus they are not expanded and will be ignored when merging.')
        print('---------------')
        self._match_tree(depth=depth, detail=detail)
        print('---------------')
    
    #### operation
    def apply(self, func, processes=None, args=(), **kwargs):
        '''
        Apply function ``func`` to each row of the Table (``data.t``) to get a new column.
        This operation is not vectorized.

        Parameters
        ----------
        func : function
            A function to be applied to each row. Example::
            
                >>> def func(row): # row is a row of the Table.
                ...     return row['a'] + row['b']
            
            Note that if processes is not None, func should be a global function 
            and should not be a lambda function and only accepts one single argument ``row``.
        processes : None or int
            if int (>0) is given, this specifies the number of processes used to get the results.
            if -1 is given, will automatically use all available cpu cores.
            if None, multiprocess will not be used.
        args : Iterable, optional
            Additional arguments to be passed to func. The default is ().
        **kwargs :
            Additional keyword arguments to be passed to func.

        Returns
        -------
        ``astropy.table.Column``
            Result of applying ``func`` to each row.

        '''
        if processes is None:
            result = []
            for row in self.t:
                result.append(func(row, *args, **kwargs))
        elif type(processes) is int:
            if processes == -1:
                processes = None
            pool = mp.Pool(processes)
            result = pool.map(func, self.t)
        else:
            raise TypeError('"processes" should be None or int.')
        return Column(result)
    
    def _get_colnames_variable(self):
        '''
        Get the colnames that can be regarded as names, 
        and do not have duplicates (no other column has the same name).
        '''
        colname_counts = Counter(self.colnames)
        self.colnames_as_variables = []
        for colname, count in colname_counts.items():
            if count == 1 and colname.isidentifier() and not iskeyword(colname):
                self.colnames_as_variables.append(colname)
        return self.colnames_as_variables
    
    def eval(self, expression, to_col=None, **kwargs):
        '''
        Evaluate the value with an expression.
        
        In the expression, the columns of the table can be referred to with:
            - The name of the column, if the name can be regarded as a Python variable name, 
              and they do not coincidence with names in the local/global namespace.
            - ``$(<column name>)``.
            - ``self['<column name>']``.
        
        The Data object itself can be referred to as ``self``.

        Parameters
        ----------
        expression : str
            The expression to be evaluated.
        to_col : str, optional
            Sets ``data[to_col]`` to the evaluated values of the expression.
            This is preferred to using ``data['name'] = data.eval(...)``, 
            because the information of the expression is added to the metadata with ``data.eval(..., to_col='name')``.
            The default is None.
        **kwargs :
            If the expression uses some name that is not recognized (e.g. using a user-defined name will result in NameError),
            you can pass the values of the names here.
            
            For example, if you use an expression 'my_function(col) + my_value' (where 'col' is a column name in the data),
            you can pass ``my_function`` and ``my_value`` by::
                Data.eval('my_function(col) + my_value', my_function=my_function, my_value=my_value)

        Returns
        -------
        result : 
            The result of the evaluation.
        '''
        locals().update(**kwargs)
        self._get_colnames_variable()
        for _colname in self.colnames_as_variables:
            _existing_names = []
            if _colname not in locals() and _colname not in globals():
                locals()[_colname] = self[_colname]
            elif _colname in expression and f"$({_colname})" not in expression:
                _existing_names.append(_colname)
            if _existing_names:
                warnings.warn(f'Column names {_existing_names} coincidence with existing names in the local/global namespace, '\
                              'thus are not interpretated as column names. '\
                              "Consider refering column names with $(column name).")
        _eval_expression = re.sub(
            r"\$\((.*?)\)",  # replace $(...)
            lambda match: f"self['{match.group(1)}']", # to self['...']
            expression,
            )
        try:
            result = eval(_eval_expression)
        except SyntaxError as e:
            raise SyntaxError('invalid syntax (are you trying to directly refer to unsupported column names?)') from e
        except NameError as e:
            raise NameError(f"Unrecognized name '{e.name}'. Check if you have misspelled a column name. If you are using a name defined in your script, consider passing '{e.name}={e.name}' when calling eval().") from e
            
        if to_col is not None:
            self[to_col] = result
            self[to_col].meta['src_detail'] += f' with expr "{expression}"'
        return result
    
    def mask_missing(self, cols=None, missval=None, verbose=True):
        '''
        Mask missing values represented by ``missval`` (e.g. -999)
        for columns ``cols``.
        
        For example, ``data.mask_missing(cols='col', missval=-999)``
        masks all -999 values in column "col", indicating that they are missing.
        
        If verbose, the information for the process will be printed. 
        Note that the printed information indicates the number of elements masked 
        in this process, rather than the total number of masked elements in the columns.
        To get the number of unmasked elements in a column, try::
            print(data.get_subsets('$unmasked/<column_name>'))

        Parameters
        ----------
        cols : str or list of str, optional
            Name(s) of the columns to be masked. The default is all columns.
        missval : optional
            The value regared as missing value. The default is NaN.
        verbose : bool, optional
            If verbose, the information for missing values will be printed.
            The default is True.
        '''
        if cols is None:
            cols = self.colnames
        if isinstance(cols, str):
            cols = (cols,)
        if missval is None:
            missval = np.nan
        
        if not self.t.masked:
            self.t = Table(self.t, masked=True, copy=False)  # convert to masked table
            
        for col in cols:
            if np.isnan(missval):
                mask = np.isnan(self.t[col])
                n_miss = np.sum(mask)
            else:
                mask = self.t[col] == missval
                n_miss = np.sum(mask)
            self.t[col].mask[mask] = True
            if verbose:
                n = len(self)
                print(f"[mask missing] col '{col}': {n_miss}/{n} ({n_miss/n*100:.2f}%) masked (value: {missval}).")
    
    #### subsets
    
    def add_subsets(self, *subsets, group=None):
        '''
        Add subsets to a subset group.
        
        A subset refers to a subset (selection) of rows;
        a subset group is a group of subsets.
        
        Beware that a subset does not "watch" the changes in the data:
        once added to the data, it never changes, even if the data changes.
        If you would like to update your subset, you may add it again to overwrite the old one.

        Parameters
        ----------
        group : str, optional
            The name of the subset group. If not specified, the default subset group will be used.
        *subsets : ``astrotable.table.Subset``
            See ``help(astrotable.table.Subset)`` for more information.
        
        Return
        ------
        subsets : tuple
            The arguments, i.e. a tuple of subset objects.
            
        '''
        # TODO: add arg "overwrite=False". if not overwrite, pass existing_names to Subset.eval_ to rename; otherwise, overwrite the subset
        if group is None:
            group = 'default'
        if group == '$unmasked':
            raise ValueError("Subset group '$unmasked' is a special group that cannot be modified.")
            
        elif group not in self.subset_groups.keys():
            self.subset_groups[group] = {} # create the new group
        # subset_objects = []
        for subset in subsets:
            subset.eval_(self, self.subset_groups[group].keys())
            name = subset.name
            if group == 'default' and name == 'all':
                raise ValueError("Subset name 'all' in the 'default' group is reserved and cannot be re-written.")
            self.subset_groups[group][name] = subset
            # subset_objects.append(subset)
        # return subset_objects
        if len(subsets) == 1:
            return subsets[0]
        else:
            return subsets
    
    def subset_group_from_values(self, column, group_name=None, overwrite=False):
        '''
        Create a subset group by the unique values of a column.
        
        For example, if a column named "class" has 3 possible values, "A", "B" and "C",
        a subset group will be defined with 3 subsets for class=A, B, C, respectively.

        Parameters
        ----------
        column : str
            The name of the column.
        group_name : str, optional
            The name of the created subset group. 
            The default is the name of the column.
        overwrite : bool, optional
            When a group with ``group_name`` already exists, whether to overwrite the group. 
            The default is False.

        Raises
        ------
        ValueError
            A group with ``group_name`` already exists, and ``overwrite`` set to ``False``.
        '''
        if group_name is None:
            group_name = column
            
        if group_name == 'default':
            raise ValueError("name 'default' is not allowed for new subset groups")
        
        if group_name == '$unmasked':
            raise ValueError("Subset group '$unmasked' is a special group that cannot be modified.")
        
        if group_name in self.subset_groups and not overwrite:
            raise ValueError(f'A subset group with name "{group_name}" already exists.')
        
        values = np.unique(self.t[column])
        if len(values) > 10:
            warnings.warn(f'A total of {len(values)} unique values found in column "{column}". This will result in a subset group with a lot of subsets.')
        
        # TODO: use self.add_subsets (with overwrite=True) [make sure to test the changes before using it!]
        self.subset_groups[group_name] = {}
        for value in values:
            subset = Subset.by_value(column, value)
            subset.eval_(self)
            name = subset.name
            self.subset_groups[group_name][name] = subset
    
    def subset_group_from_ranges(self, column, ranges, group_name=None, overwrite=False):
        '''
        Create a subset group by setting several ranges of values of a column.
        
        For example, ``data.subset_group_from_ranges(column='col1', ranges=[[0, 1], [1, 2]])``
        defines a subset group named 'col1', which consists of 2 subsets, `0<col1<1` and `1<col1<2`.

        Parameters
        ----------
        column : str
            The name of the column.
        ranges : list of lists (or similar objects)
            List of ranges.
        group_name : str, optional
            The name of the created subset group. 
            The default is the name of the column.
        overwrite : bool, optional
            When a group with ``group_name`` already exists, whether to overwrite the group. 
            The default is False.

        Raises
        ------
        ValueError
            A group with ``group_name`` already exists, and ``overwrite`` set to ``False``.
        '''
        if group_name is None:
            group_name = column
        
        if group_name == 'default':
            raise ValueError("name 'default' is not allowed for new subset groups")

        if group_name == '$unmasked':
            raise ValueError("Subset group '$unmasked' is a special group that cannot be modified.")
              
        if group_name in self.subset_groups and not overwrite:
            raise ValueError(f'A subset group with name "{group_name}" already exists.')
        
        # TODO: use self.add_subsets (with overwrite=True) [make sure to test the changes before using it!]
        self.subset_groups[group_name] = {}
        for range_ in ranges:
            subset = Subset.by_range(**{column: range_})
            subset.eval_(self)
            name = subset.name
            self.subset_groups[group_name][name] = subset
    
    def clear_subsets(self, group=None):
        '''
        Clear user-defined subsets.

        Parameters
        ----------
        group : str, optional
            Name of the subset group to be cleared. 
            If not specified, all user-defined subsets are deleted.
        '''
        if group in (None, 'all'): # clear all groups
            # print('INFO: subsets reset to default.')
            self.subset_groups = {
                'default': {'all': self.subset_all}
                }
        elif group in ('default',): # clear default group
            self.subset_groups['default'] = {'all': self.subset_all}
        elif group in self.subset_groups.keys(): # clear a certain group 
            del self.subset_groups[group]
        elif group in ['$unmasked']: # trying to clear a special group
            raise ValueError(f"'{group}' is a special group that cannot be cleared")
        else:
            warnings.warn(f"group name '{group}' does not exist, no need to clear")
    
    def get_subsets(self, path=None, name=None, group=None, listalways=False):
        '''
        Get a subset (or several subsets) given the subset groups, subset names 
        or paths (i.e. ``'<group_name>/<subset_name>'``)
        Different from the ``subset_data`` method, which returns the ``Subset`` objects.
        
        Note that a special group (and the subsets in it) is virtual (it does not really "exist"). 
        Subsets from special groups (including '$unmasked') can only be retrieved with its path
        (e.g. ``'$unmasked/<column name>'``). Otherwise, a GroupNotFoundError will be raised.

        Parameters
        ----------
        path : str or list of str, optional
            The path or a list of paths.
            If this is given, arguments ``name`` and ``group`` are ignored.
            The default is None.
        name : str or list of str, optional
            The names of subsets, or a list of names. 
            The default is all subsets in the specified group.
        group : str, optional
            The name of the group. The default is the default group.
        listalways : bool, optional
            If True, always returns list of subsets (even if len(list) == 1).
            The default is False.

        Returns
        -------
        ``astrotable.table.Subset`` or list of ``astrotable.table.Subset``
            The subset or list of subsets specified.
            
        Special subsets (groups)
        ------------------------
        A special subset group is a virtual group that do not actually "exists".
        This is used to create a (new) subset as if retrieving an existing subset of
        the data. That means that these (virtual) subsets are 
        only created when ``get_subsets()`` is called to "retrieve" it, and it is not actually
        added to the data. If you would like to make it a "normal" subset stored in
        the ``astrotable.table.Data`` instance, you can use::
            data.add_subsets(
                data.get_subsets('<path to the special subset>'),
                )
        
        Recognized special subsets are listed below:
        
        - ``$unmasked``. This subset group contains virtual subsets indicating whether the values in
          a certain column is not masked (i.e., a subset in this group contains rows where the value 
          for the specified column is not masked). 
          To get such a subset, simply use::
              data.get_subsets('$unmasked/<column name>')
            
          Note that a new subset is created whenever ``get_subsets()`` is called; 
          `the old subsets never change even when the mask of the column is changed`. For example::
              subset0 = data.get_subsets('$unmasked/col1')
              # changing the mask of column 'col1'
              subset1 = data.get_subsets('$unmasked/col1')
              subset0 is not subset1 # True 
              # NOTE: subset0 is the old subset that does not represent the masking of 'col1' now.
        
        Examples
        --------
        

        '''
        # (i.e. a special subset group and the virtual subsets therein 
        # is never remembered by a ``astrotable.table.Data`` instance)
        
        autosearch = False # do not search in other groups unless the user inputs ONLY subsets
        
        if path is not None:
            if group is not None or name is not None:
                warnings.warn('Since the argument "path" is given, arguments "name"/"group" are ignored.',
                              stacklevel=2)
            if isinstance(path, Subset): # it is itself a Subset!
                return path
            if type(path) is str:
                if listalways:
                    path = [path]
                else:
                    return self._get_subset_from_path(path, autosearch=autosearch)
            if isinstance(path, Iterable):
                subsets = []
                for p in path:
                    subsets.append(self._get_subset_from_path(p, autosearch=autosearch))
                return subsets
            else:
                raise TypeError('path should be str or Iterable')
        else:
            if group is None: # user inputs ONLY subsets
                autosearch = True
                group = 'default'
            if type(group) is not str:
                raise TypeError('group should be a string')
            
            if group not in self.subset_groups.keys():
                raise GroupNotFoundError(group)
            
            if name is None:
                name = self.subset_groups[group].keys()
            if type(name) is str:
                if listalways:
                    name = [name]
                else:
                    return self._get_subset_from_path(f'{group}/{name}', autosearch=autosearch)
            if isinstance(name, Iterable):
                subsets = []
                for n in name:
                    if isinstance(n, Subset): # it is itself a Subset!
                        subsets.append(n)
                    else:
                        subsets.append(self._get_subset_from_path(f'{group}/{n}', autosearch=autosearch))
                return subsets
            elif isinstance(name, Subset): # it is itself a Subset
                return name
            else:
                raise TypeError(f'name should be str or Iterable, not {type(name)}')
                
    def _get_subset_from_path(self, path, autosearch=False):
        # get the subset from path
        # autosearch: search this subset name in other groups if does not found this name in this group
        if isinstance(path, Subset): # it is itself a Subset
            return path
        
        if '/' not in path:
            group = 'default'
            name = path
        else:
            names = path.split('/')
            group, name = '/'.join(names[:-1]), names[-1] # allows '/' in groupname
            
        if group in ['$unmasked']: # this is a special group
            return self._get_special_subset(group, name)
        
        if group not in self.subset_groups.keys():
            raise GroupNotFoundError(group)
        
        if name not in self.subset_groups[group].keys():
            if autosearch:
                subset = self._get_subset_from_name(name) # search for this name in all groups
            else:
                raise SubsetNotFoundError(f"{group}/{name}", kind='path')
        else:
            subset = self.subset_groups[group][name]
        return subset
    
    def _get_subset_from_name(self, name, verbose=True):
        # get subset for name without knowing the name of the group
        group = None
        subset = None
        for group_name, subsets in self.subset_groups.items():
            if name in subsets.keys():
                if subset is not None:
                    raise ValueError(f"subset name is ambiguous: '{name}' found in multiple groups")
                subset = subsets[name]
                group = group_name
        if subset is None:
            raise SubsetNotFoundError(name, kind='name')
            
        if verbose: print(f"[subset] Found subset '{name}' in group '{group}'.")
        return subset
    
    def _get_special_subset(self, group, name):
        if group == '$unmasked': # '$unmasked/<column_name>'
            if np.ma.is_masked(self[name]):
                unmasked = ~self[name].mask # here name is a column name
                # Notes: the '~' operation creates a new array. Consider this: a = np.array((True, True)); b = ~a; a[0] = not a[0]; print(np.all(b == ~a)) # False
            else:
                unmasked = np.full(len(self), True)
            subset = Subset(unmasked, 
                            name=f'$unmasked({name})', 
                            expression=f"~self['{name}'].mask",
                            label=self.get_labels(name)+' unmasked')
            subset.eval_(self)
            return subset
        else:
            raise ValueError(f"unrecognized special subset group: '{group}'")
    
    def _data_from_subset(self, subsets):
        '''
        Returns the sub-dataset from ``Subset`` objects

        Parameters
        ----------
        subsets : ``Subset`` or list of ``Subset``

        Returns
        -------
        ``Data``

        '''
        if not isinstance(subsets, Iterable):
            subset = subsets
            table_subset = self.t[np.array(subset)]
            return Data(table_subset, name=f'{self.name}_{subset.name}')
        else:
            subset_data = []
            for subset in subsets:
                table_subset = self.t[np.array(subset)]
                subset_data.append(Data(table_subset, name=f'{self.name}_{subset.name}'))
            return subset_data
        

    def subset_data(self, path=None, name=None, group=None):
        '''
        Get a subset (or several subsets) of data given the subset groups, subset names 
        or paths (i.e. ``'<group_name>/<subset_name>'``).
        This is different from the ``get_subsets`` method, which returns the ``Subset`` objects.
        
        You may also pass a ``Subset`` object or a list of ``Subset`` objects to the ``path`` parameter,
        to directly get the data.

        Parameters
        ----------
        path : Subset OR list of Subset OR str OR list of str, optional
            A Subset object or a list of Subset objects,
            or the path or a list of paths.
            If this is given, arguments ``name`` and ``group`` are ignored.
            The default is None.
        name : str or list of str, optional
            The names of subsets, or a list of names. 
            The default is all subsets in the specified group.
        group : str, optional
            The name of the group. The default is the default group.


        Returns
        -------
        ``astrotable.table.Data`` or list of ``astrotable.table.Data``
            The subset of data or list of subsets of data specified.
            
        Examples
        --------
        

        '''
        subsets = self.get_subsets(path=path, name=name, group=group)
        return self._data_from_subset(subsets)
    
    def subset_summary(self, group=None):
        '''
        Get a summary table for the subsets and subset groups.
        
        The table consists of the following columns:
            - `group`: name of the subset group
            - `name`: name of the subset
            - `size`: size of the subset
            - `fraction`: fracion of the size to the total number
            - `expression`: expression/source code that specifies the selection of the subset
            - `label`: label of the subset used for plotting

        Parameters
        ----------
        group : str or list of str, optional
            The name (or list of names) of the subset group(s) to be shown in the table. 
            If not given, all groups will be shown by default.

        Returns
        -------
        summary : ``astropy.table.Table``
        '''
        summary = Table(names=['group', 'name', 'size', 'fraction', 'expression', 'label'], 
                        dtype=['str', 'str', 'int', 'float', 'str', 'str']
                        )
        if group is None: # no group specified
            # add information for special groups
            summary.add_row(dict(
                group='$unmasked', name='-',
                size='-1', fraction=np.nan,
                expression='<special group of subsets for unmasked elements>',
                label='-',
                ))
            # show all groups
            group = self.subset_groups.keys()
        elif type(group) is str:
            group = [group]
        shown_groups = {k: self.subset_groups[k] for k in group}
        for groupname, subsets in shown_groups.items():
            for subsetname, subset in subsets.items():
                n_selected = subset.size # same as np.sum(subset.selection)
                summary.add_row(dict(
                    group=groupname, name=subsetname,
                    size=n_selected, fraction=n_selected/len(subset.selection),
                    expression=subset.expression,
                    label=subset.label,
                    ))
        return summary
    
    #### plot
    
    def set_labels(self, **kwargs):
        '''
        label(<column_name>=<label>)
        
        Add/update the labels used for, e.g., the labels on the axes of the plots.
        
        Example: if ``col1='$x_1$'``, the data in ``data.t['col1']`` will be labeled as '$x_1$' on the plots.

        Parameters
        ----------
        **kwargs : <column_name:str>=<label:str>
        '''
        self.col_labels.update(**kwargs)
    
    def get_labels(self, *cols, listalways=False):
        '''
        Get the labels of columns (if not set by ``set_labels``, the column name will be used).
        
        Parameters
        ----------
        *cols : str
            names of the columns
        listalways : bool, optional
            If True, always returns list of labels (even if len(list) == 1).
            The default is False.
        
        Returns
        -------
        str or list of str
        '''
        labels = [self.col_labels[col] if col in self.col_labels else col for col in cols]
        if len(labels) == 1 and not listalways:
            return labels[0]
        else:
            return labels
    
    # @property
    # def labels(self):
    #     return self.get_labels()
    
    # TODO. argument col_input: plan: make it possible to input something like 'col1', 'col2', c='col3',
    # and translate it to make it the same as columns=('col1', 'col2'), kwarg_columns={'c': 'col3'}.
    @keyword_alias('deprecated', columns='cols', kwarg_columns='kwcols')
    @keyword_alias('accepted', group='groups')
    def plot(self, func, *args, col_input=None, cols=None, kwcols={}, paths=None, subsets=None, groups=None, autolabel=True, ax=None, global_selection=None, title=None, iter_kwargs={}, **kwargs):
        '''
        Make a plot given a plotting function.
        
        Arguments ``paths``, ``subsets``, ``groups`` are used to specify the subsets of data 
        that are plotted in the same subplot.

        Parameters
        ----------
        func : str or Callable
            Function to make plots, e.g. ``plt.plot``,
            or name of the function, e.g. ``'plot'``.
        *args : 
            Arguments to be passed to func.
        cols : str or list of str, optional
            The name of the columns to be passed to ``func``. 
            For example, if ``cols = ['col1', 'col2']``, ``func`` will be called by:
                ``func(data.t['col1'], data.t['col2'], *args)``
            `Note`: When ``autolabel`` is True, the len of this argument is used to guess the dimension of the plot (e.g. 2D/3D).
            The default is None.
        kwcols : dict, optional
            Names of data columns that are passed to ``func`` as keyword arguments.
            For example, if ``kwcols={'x': 'col1', 'y':'col2'}``, ``func`` will be called by:
                ``func(x=data.t['col1'], y=data.t['col2'])``
        paths : str or list of str, optional
            The full path of a subset (e.g. ``'<group_name>/<subset_name>'``) or a list of paths.
            If this is given, arguments ``subsets`` and ``group`` are ignored.
            The default is None.
        subsets : str or list of str, optional
            The names of subsets, or a list of names. 
            The default is all subsets in the specified group.
        groups : str, optional
            The name of the group. The default is the default group.
        autolabel : bool, optional
            If True, will try to automatically add labels to the plot (made by ``func``) as well as the axes,
            using the labels stored in Data and Subset objects.
            
            NOTE: The labels for axes are auto-set according to the argument ``columns``, 
            and may not get the results you expects.
            Label for axes and legends are only possible for axes if argument ``ax`` is given.
            
            The default is True.
        ax : axes, optional
            The axis of the plot. 
            Note that this is ONLY used for adding axis labels and legends; 
            if you would like to plot on a specific axis, consider passing e.g. ``ax.plot`` to argument ``func``.
            The default is None.
        global_selection : ``astrodata.table.Subset`` or str or list of str, optional
            The global selection [or the path(s) of the selection(s)] for this plot. 
            If not None, only data selected by this argument is plotted.
            Accepted input:
                - An ``astrotable.table.Subset`` object. Note that logical operations of subsets are supported, e.g. ``subset1 & subset2 | subset3``.
                - The path to the subset, i.e. ``'groupname/subsetname'``. If group name is 'default', you can directly use 'subsetname'.
                - A list/tuple/set of paths to the subsets. The global selection will be the logical AND (i.e. the intersection set) of the subsets.
            The default is None.
        title : str
            Manually setting the title of the plot. This will overwrite the title automatically generated.
            The default is None (automatically generated if autolabel is True).
        iter_kwargs : dict, optional
            Lists of keywoard arguments that are different for each subset specified. 
            Suppose 3 subsets are specified using the ``subsets`` argument, an example value for 
            ``iter_kwargs`` is :
                ``{'color': ['b', 'r', 'k'], 'linestyle': ['-', '--', '-.']}``
            The default is {}.
        **kwargs : 
            Additional keyword arguments to be passed to ``func``.

        Raises
        ------
        ValueError
            len of one item of iter_kwargs is not equal to 
            the len of paths/subsets

        Examples
        --------
        One example:
            
        >>> from astrotable.table import Data
        >>> data = Data({'col1': [1, 2, 3], 'col2': [1, 4, 9]})
        >>> fig, ax = plt.subplots()
        >>> data.plot(ax.plot, columns=('col1', 'col2'), color='k')
        '''
        iter_kwargs = iter_kwargs.copy()
        kwarg_columns = kwcols.copy()
        columns = cols
        
        if type(columns) is str:
            columns = [columns]
        if type(func) is str:
            if func not in plot_funcs:
                raise ValueError(f'Supported func names are: {",".join(plot_funcs.keys())}')
            func = plot_funcs[func]
        else:
            func = plot.plotFuncAuto(func)
            
        if ax is None:
            ax = plt.gca()
        
        if type(global_selection) in (str, tuple, list, set):
            global_selection = bitwise_all(self.get_subsets(path=global_selection, listalways=True))       

        subset_names = subsets
        subsets = self.get_subsets(path=paths, name=subset_names, group=groups)
        if type(subsets) is Subset:
            subsets = [subsets]
        
        local_subsets = subsets
        if global_selection is not None:
            subsets = [(subset & global_selection) for subset in subsets]
            
        subset_data_list = self._data_from_subset(subsets)
        # subset_data_list = self.subset_data(path=paths, name=subset_names, group=groups)
        # if type(subset_data_list) is Data:
        #     subset_data_list = [subset_data_list]
        
        # try to automatically set label
        if autolabel:
            if 'label' not in iter_kwargs and 'label' not in kwargs: # label for single plot element
                # check if func supports label as input
                if isinstance(func, plot.PlotFunction):
                    func_params = func.func_sig.parameters
                    func_name = func.func_defname
                else:
                    func_params = inspect.signature(func).parameters
                    func_name = func.__name__
                if 'label' in func_params.keys() or any([param.kind == param.VAR_KEYWORD for param in func_params.values()]): # check if func supports label as argument
                    iter_kwargs['label'] = [subset.label for subset in local_subsets]
                else:
                    warnings.warn(f'Failed to automatically set labels: user-defined function "{func_name}" does not support "label" as argument.')
            
            if 'label' in kwargs and len(subset_data_list) > 1:
                warnings.warn('You are setting the same label for plots of multiple subsets.')
            
            # set axis label
            if ax is not None and columns is not None:
                ax.set(**dict(zip(
                    ['xlabel', 'ylabel', 'zlabel'], 
                    self.get_labels(*columns, listalways=True),
                    )))
            
            # special case for my scatter()
            if type(func) == plot.PlotFunction and type(func.func) == plot.Scatter and 'c' in kwarg_columns and 'barlabel' not in kwargs:
                kwargs['barlabel'] = self.get_labels(kwarg_columns['c'])
        
        if iter_kwargs != {}:
            # check values
            for key, values in iter_kwargs.items():
                if not isinstance(values, Iterable):
                    values = [values]
                if len(values) != len(subset_data_list):
                    raise ValueError(f"len of iter_kwargs '{key}' should be {len(subset_data_list)}, got {len(values)}")
            # get kwargs for each subset
            iter_kwargs_list = [dict(zip(iter_kwargs.keys(), value)) for value in zip(*(iter_kwargs[i] for i in iter_kwargs))]
        else:
            iter_kwargs_list = repeat({})
        
        if isinstance(func, plot.PlotFunction):
            plot_func = func.in_plot # callback of func should not be recursively called.
        else:
            plot_func = func
        
        for subset_data, iter_kwargs in zip(subset_data_list, iter_kwargs_list):
            if columns is None:
                input_data = () # input data for the func (as *args)
            else:
                input_data = []
                for column in columns:
                    input_data.append(subset_data.t[column])
            this_kwarg_columns = {}
            for argname in kwarg_columns:
                argval = kwarg_columns[argname]
                if isinstance(argval, str):
                    this_kwarg_columns[argname] = subset_data.t[argval]
                elif isinstance(argval, (list, tuple)) and all(isinstance(v, str) for v in argval):
                    this_kwarg_columns[argname] = [subset_data.t[v] for v in argval]
                else:
                    raise TypeError(f'expected str or list/tuple of str for values of kwcols, got "{type(argval)}"')
            
            ret = plot_func(*input_data, *args, **this_kwarg_columns, **iter_kwargs, **kwargs)
        
        if hasattr(func, 'ax_callback'): 
            func.ax_callback(ax) # call ax_callback attached to func only once
            
        if autolabel and ax is not None:
            if len(subset_data_list) > 1:
                legend = ax.legend()
                if len(legend.get_texts()) == 0: # no legend generated?
                    legend.remove()
                
                if global_selection is not None:
                    ax.set_title(global_selection.label)
            else: # only one data plot, just use title instead of legend
                ax.set_title(subsets[0].label)
        
        if title is not None:
            if ax is None:
                warnings.warn('To set the title, please input the axis, e.g. data.plot(<...>, ax=your_axis)')
            else:
                ax.set_title(title)
        
        return ret
    
    @keyword_alias('deprecated', columns='cols', kwarg_columns='kwcols') # deprecated old names
    @keyword_alias('accepted', group='plotgroups', groups='plotgroups', paths='plotpaths', subsets='plotsubsets', ax='axes') # make plot() arguments acceptable here 
    def plots(self, func, *args, cols=None, kwcols={}, plotpaths=None, plotsubsets=None, plotgroups=None, arraygroups=None, global_selection=None, share_ax=False, autobreak=False, autolabel=True, ax_callback=None, returns='fig', axes=None, fig=None, iter_kwargs={}, **kwargs):
        '''
        Make a plot given the function ``func`` used for plotting.
        
        If ``arraygroups`` is not ``None``, plot an "array" of subplots (panels; subplots with several rows and columns) for different selections given in ``arraygroups``;
        Each of the panels consists of several plots for different selections given in ``plotgroups``.
        This is useful if one wishes to compare a plot for different subsets of the data. 
        For example, say ``plotgroups='group1'``, ``arraygroups=['group2', 'group3']``.
        Then each panel compares different subsets in ``'group1'``; different panels compares the results 
        between subsets in ``'group2'`` and ``'group3'``.
        Note that the dataset for each plot in each panel is the INTERSECTION of the corresponding subsets in 
        ``'group1'``, ``'group2'`` and ``'group3'``.

        Parameters
        ----------
        func : str or Callable or ``astrotable.plot.PlotFunction``
            Name of the ``matplotlib.pyplot`` function used to make plots, e.g. ``'plot'``, ``'scatter'``.
            
            Also accepts custum functions that receives an axis as the only argument, 
            and returns a function (called "plotting function" hereafter) to make plots.
            Example:``lambda ax: ax.plot``.
            
            You can also input your custom plot function ``func`` defined by::
                
                from astrotable.plot import plotFunc
                @plotFunc
                def func(<your inputs>):
                    <make the plot>
                    return # you can return somthing here
                    
            Or::
                
                from astrotable.plot import plotFuncAx
                @plotFuncAx
                def func(ax): # input ax axis
                    def plot(<your inputs>):
                        <make the plot>
                        return # you can return somthing here
                    return plot
            
        *args : 
            Arguments to be passed to the plotting function.
        cols : str or list of str, optional
            The name of the columns to be passed to the plotting function. 
            For example, if ``cols = ['col1', 'col2']``, the plotting function will be called by:
                ``plotting_function(data.t['col1'], data.t['col2'], *args)``
            `Note`: When ``autolabel`` is True, the len of this argument is used to guess the dimension of the plot (e.g. 2D/3D).
            The default is None.
        kwcols : dict, optional
            Names of data columns that are passed to the plotting function as keyword arguments.
            For example, if ``kwcols={'x': 'col1', 'y':'col2'}``, the plotting function will be called by:
                ``plotting_function(x=data.t['col1'], y=data.t['col2'])``
        paths, subsets, groups :
            aliases of "plotpaths", "plotsubsets" and "plotgroups".
        plotpaths : str or list of str, optional
            The full path of a subset (e.g. ``'<group_name>/<subset_name>'``) or a list of paths, for plots in each subplot.
            If this is given, arguments ``plotsubsets`` and ``plotgroups`` are ignored.
            The default is None.
        plotsubsets : str or list of str, optional
            The names of subsets, or a list of names, for plots in each subplot. 
            The default is all subsets in the specified group.
        plotgroups : str, optional
            The name of the subset group used to make different plots in each one of the panels. 
            For example, when the plotting function plots curves and ``plotgroups`` consists of 
            3 subsets, 3 curves for the 3 subsets are plotted in each of the panels.
            The default is None.
        arraygroups : str or iterable of len <= 2, optional
            The name of subset groups used to make different panels. 
            Examples:
                - ``arraygroups = ['group1']``, where `'group1'` consists of 3 subsets. 
                  Then subplots with ``nrow=1, ncol=3`` (1x3) are generated.
                - ``arraygroups = ['group1', 'group2']``, where `'group1', 'group2'` consists of 3, 4 subsets respectively. 
                  Then subplots with ``nrow=3, ncol=4`` (3x4) are generated.
            The default is None.
        global_selection : ``astrotable.table.Subset`` or str or list of str, optional
            Only consider data in subset ``global_selection``.
            Accepted input:
                - An ``astrotable.table.Subset`` object. Note that logical operations of subsets are supported, e.g. ``subset1 & subset2 | subset3``.
                - The path to the subset, i.e. ``'groupname/subsetname'``. If group name is 'default', you can directly use 'subsetname'.
                - A list/tuple/set of paths to the subsets. The global selection will be the logical AND (i.e. the intersection set) of the subsets.
            The default is None (the whole dataset is considered).
        share_ax : bool, optional
            Whether the x, y axes are shared. The default is False.
        autobreak : bool, optional
            When ``arraygroups`` consists of only one group, whether to automatically break the row
            into several rows (since the default result is a group of subplots with only one row). 
            The default is False.
        autolabel : bool, optional
            If True, will try to automatically add labels to the plot (made by ``func``) as well as the axes,
            using the labels stored in Data and Subset objects.
            
            NOTE: The labels for axes are auto-set according to the argument ``columns``, 
            and may not get the results you expects.
            Label for axes and legends are only possible for axes if argument ``ax`` is given.
            
            The default is True.
        ax_callback : function, optional
            The function to be called as ``ax_callback(ax)`` after plotting in each panel,
            where ``ax`` is the axis object of this panel.
        returns : str, optional
            Decide what to return.
            
            - ``'fig'`` or ``'fig, axes'``:
                return figure and axes.
            - ``'plot'`` or ``'return'``:
                return a list of the returned values of the plot function.
            
            Whatever this argument is, you can always retrive the figure, axes and the returned values (of the 
            plot function) of the last call of ``data.plot()`` with ``self.plot_fig, self.plot_axes, self.plot_returns``.
        
        ax :
            alias of "axes".
        axes : list of axes, optional
            The axes of the subplots. 
            The default is None.
        fig : ``matplotlib.figure.Figure``, optional
            The figure on which the subplots are made. 
            The default is None.
        iter_kwargs : dict, optional
            Lists of keywoard arguments that are different for each subset in ``plotgroups``. 
            Suppose ``plotgroups='group1'`` consists of 3 subsets, an example value for 
            ``iter_kwargs`` is :
                ``{'color': ['b', 'r', 'k'], 'linestyle': ['-', '--', '-.']}``
            The default is {}.
        **kwargs : 
            Additional keyword arguments to be passed to the plotting function.

        Raises
        ------
        ValueError
            - len(arraygroups) >=3: plot array of dim >= 3 not supported.
            - inferred ``nrow*ncol`` != ``len(axes)`` given

        Returns
        -------
        fig : ``matplotlib.figure.Figure``
            
        axes : list if axes

        '''
        # TODO (not implemented)
        if share_ax: raise NotImplementedError('This feature is not implemented, and whether it will be added is undetermined.')
        
        self.plot_returns = []
        
        iter_kwargs = iter_kwargs.copy()
        kwarg_columns = kwcols.copy()
        columns = cols
        
        if type(func) is str:
            if func not in plot_array_funcs:
                raise ValueError(f'Supported func names are: {",".join(plot_array_funcs.keys())}')
            func = plot_array_funcs[func]
        else:
            func = plot.plotFuncAuto(func)

        if type(global_selection) in (str, tuple, list, set):
            global_selection = bitwise_all(self.get_subsets(path=global_selection, listalways=True))

        # special case for my scatter()
        if type(func) == plot.PlotFunction and type(func.func) == plot.Scatter and 'c' in kwarg_columns and 'barlabel' not in kwargs:
            kwargs['barlabel'] = self.get_labels(kwarg_columns['c'])

        if arraygroups is None:
            if axes is None:
                axes = plt.gca()
            if isinstance(axes, Iterable):
                axes = axes[0]
            if fig is None:
                fig = axes.figure
            ret = self.plot(func(axes), *args, cols=columns, kwcols=kwarg_columns, paths=plotpaths, subsets=plotsubsets, groups=plotgroups, autolabel=autolabel, global_selection=global_selection, ax=axes, iter_kwargs=iter_kwargs, **kwargs)
            self.plot_returns.append(ret)
        
        else:
    
            # get subsets for each panel
            if type(arraygroups) is str:
                arraygroups = [arraygroups]
            subsets = [self.get_subsets(group=group, listalways=True) for group in arraygroups]
            # if global_selection is not None:
            #     subsets = [[subset & global_selection for subset in subseti] for subseti in subsets]
            if len(subsets) >= 3:
                raise ValueError('len(arraygroups) >=3: plot array of dim >= 3 not supported. ')
            elif len(subsets) == 2:
                subset_array = [[(xi & yi) for xi in subsets[0]] for yi in subsets[1]]
            else: # len(subsets) == 1
                subset_array = subsets
                
            # decide nrow and ncol and check inputs
            if len(subsets) == 1 and autobreak: # autobreak is not a comprehensive function
                if len(subsets[0]) in subplot_arrange:
                    nrow, ncol = subplot_arrange[len(subsets[0])]
                else:
                    pass # TODO: not implemented
            else:
                nrow, ncol = len(subset_array), len(subset_array[0])
            figsize = [6.4*(1+.7*(ncol-1)), 4.8*(1+.7*(nrow-1))]
            
            # prepare and check consistency with axes
            if axes is None:
                if fig is None:
                    fig = plt.figure(figsize=figsize)
                axes = fig.subplots(nrow, ncol)
            else:
                if fig is None:
                    if isinstance(axes, Iterable):
                        fig = axes.flatten()[0]
                    else:
                        fig = axes.figure
                if not isinstance(axes, Iterable):
                    axes = [axes]
            axes = np.array(axes)
            axes_flat = axes.flatten()
            if len(axes_flat) != nrow*ncol:
                raise ValueError(f'Expected {nrow}*{ncol}={nrow*ncol} axes; got {len(axes)}.')
            
            # plot subplots
            for ax, subset in zip(axes_flat, chain(*subset_array)):
                if global_selection is not None:
                    subset_with_global = subset & global_selection
                else:
                    subset_with_global = subset
                ret = self.plot(func(ax), *args, cols=columns, kwcols=kwarg_columns, paths=plotpaths, subsets=plotsubsets, groups=plotgroups, autolabel=autolabel, ax=ax, global_selection=subset_with_global, title=subset.label, iter_kwargs=iter_kwargs, **kwargs)
                self.plot_returns.append(ret)
                
                if ax_callback is not None:
                    ax_callback(ax)
            if autolabel and global_selection is not None:
                fig.suptitle(global_selection.label)
        
        self.plot_fig = fig
        self.plot_axes = axes
        
        if returns in ['fig', 'fig, axes']:
            return fig, axes
        elif returns in ['plot', 'return']:
            return self.plot_returns
        else: 
            raise ValueError(f'Unrecognized input for returns: "{returns}"')
    
    #### IO
    
    # data when saving and loading "data" (zip) files.
    data_to_save = {
        # attribute name: save method
        'col_labels': 'json',
        'subset_groups': 'pkl',
        't': 'astropy.table',
        'name': 'txt',
        'matchlog': 'json',
        'meta': 'json',
        }
    table_format = 'fits' # 'asdf' # 'ascii.ecsv'
    table_ext = '.fits' # '.asdf' # '.ecsv'
    save_meta = dict( # all information for saving
        data_to_save=data_to_save,
        table_format=table_format,
        table_ext=table_ext,
        package_version=astrotable.__version__,
        )
    
    # old values before save_meta is saved
    _old_data_to_save = {
        # attribute name: save method
        'col_labels': 'pkl',
        'subset_groups': 'pkl',
        't': 'astropy.table',
        'name': 'pkl',
        'matchlog': 'pkl',
        }
    _old_table_format = 'fits'
    _old_table_ext = '.fits'
    
    def save(self, path, format='data', overwrite=False):
        '''
        Save data to file.

        Parameters
        ----------
        path : str
            Path to the file.
        format : str, optional
            The format of the file. 
            The default is 'data'.
            Supported formats include:
                'pkl': 
                    Saving the full data object to a "*.pkl" file. 
                'data' (default):
                    Saving key data (including the data table, the subsets, etc.) to a "*.data" file.
                    Note that the matching data is not saved.
                Other formats: Any format supported by ``astropy.table.Table.write``. 
                    Only saving the data table (``astropy.table.Table``). 
                    This is equivalent to ``data.t.write(<...>)``.
        overwrite : bool, optional
            Whether to overwrite the file if it exists. 
            If set to ``False``, a ``FileExistsError`` will be raised.
            The default is False.

        Raises
        ------
        FileExistsError
            The file already exists.
            
        Notes for developers
        --------------------
        When setting ``format='pkl'``, a Data object will be saved with the standard ``pickle`` module.
        This means that all data for the object is converted and saved as a byte stream. When setting ``format='data'``,
        only a selected subset of attributes will be saved `separately`, and are not necessarily saved 
        with the Python's standard pickling protocols. This makes it possible to retrieve some data from the '*.data' file 
        even without e.g. Python's ``pickle`` module. 
        '''
        if format == 'pkl':
            save_pickle(path, overwrite, self)
            
        elif format == 'data': # save important data in a zip file
            if path[-5:] != '.data':
                path += '.data'
            if not overwrite and os.path.exists(path):
                raise FileExistsError(f'File "{path}" already exists. To overwrite, use the argument "overwrite=True".')
            with zipfile.ZipFile(path, mode='w', compression=zipfile.ZIP_DEFLATED) as datazip:
                # save data_to_save
                for attr, method in Data.data_to_save.items():
                    if method == 'astropy.table':
                        fname = attr + Data.table_ext
                        table = getattr(self, attr)
                        assert type(table) == Table
                        with datazip.open(fname, mode='w') as f:
                            table.write(f, format=Data.table_format) # ascii.ecsv
                    elif method == 'pkl':
                        fname = attr + '.pkl'
                        with datazip.open(fname, mode='w') as f:
                            pickle.dump(getattr(self, attr), f)
                    elif method == 'json':
                        fname = attr + '.json'
                        with datazip.open(fname, mode='w') as f:
                            json_str = json.dumps(getattr(self, attr), indent=4)
                            json_str = bytes(json_str, 'ascii')
                            f.write(json_str)
                    elif method == 'txt':
                        fname = attr + '.txt'
                        with datazip.open(fname, mode='w') as f:
                            s = getattr(self, attr)
                            assert isinstance(s, str), "'txt' mode only for str"
                            s = s.encode()
                            f.write(s)
                    else:
                        raise ValueError(f'unrecognized saving method: {method}')
                
                # save save_meta
                with datazip.open('.save_meta.json', mode='w') as f:
                    meta = json.dumps(Data.save_meta, indent=4)
                    meta = bytes(meta, 'ascii')
                    f.write(meta)
                        
        else:
            self.t.write(path, format=format, overwrite=overwrite)
    
    @classmethod
    def load(cls, path, format='data', **kwargs):
        '''
        Load a data file saved with ``Data.save()`` (usually with ".data" or ".pkl" format).
        
        **Note**: You may also read a raw table file like "*.csv", but it 
        is suggested to use ``Data('your_catalog.csv')`` instead of 
        ``Data.load('your_catalog.csv', format='ascii.csv')``.

        Parameters
        ----------
        path : str
            Path to the file.
        format : str, optional
            The format of the file (see ``help(Data.save)``). 
            The default is 'data'.
        **kwargs :
            other arguments passed when initializing ``Data`` 
            [Only used when format is neither 'data' nor 'pkl'.]

        Returns
        -------
        data : ``astrotable.table.Data``
        '''
        if format == 'data':
            attrs = {}
            try:
                with zipfile.ZipFile(path) as datazip:
                    if '.save_meta.json' in datazip.namelist():
                        with datazip.open('.save_meta.json') as f:
                            save_meta = json.load(f)
                        # locals().update(save_meta)
                        data_to_save, table_ext, table_format = save_meta['data_to_save'], save_meta['table_ext'], save_meta['table_format']
                    elif '.meta.json' in datazip.namelist(): # the old name
                        with datazip.open('.meta.json') as f:
                            save_meta = json.load(f)
                        data_to_save, table_ext, table_format = save_meta['data_to_save'], save_meta['table_ext'], save_meta['table_format']
                    else:
                        save_meta = None
                        data_to_save, table_ext, table_format = Data._old_data_to_save, Data._old_table_ext, Data._old_table_format
                        
                    for attr, method in data_to_save.items():
                        if method == 'astropy.table':
                            fname = attr + table_ext
                            with datazip.open(fname) as f:
                                attrs[attr] = Table.read(f, format=table_format, # ascii.ecsv
                                                         # unit_parse_strict='silent',
                                                         )
                        elif method == 'pkl':
                            fname = attr + '.pkl'
                            with datazip.open(fname) as f:
                                attrs[attr] = pickle.load(f)
                        elif method == 'json':
                            fname = attr + '.json'
                            with datazip.open(fname) as f:
                                attrs[attr] = json.load(f)
                        elif method == 'txt':
                            fname = attr + '.txt'
                            with datazip.open(fname) as f:
                                attrs[attr] = f.read().decode()
                        else:
                            raise ValueError(f'unrecognized saving method: {method}')
            except zipfile.BadZipFile as e:
                raise ValueError(f'The file is not a ".data" file. Did you mean "Data(\'{path}\', <...>)"?') from e
            except KeyError as e:
                ver = f" ({save_meta['package_version']})" if save_meta and 'package_version' in save_meta else ''
                raise FailedToLoadError(f"Failed to load '{path}': is not a '.data' file or is saved with an older version{ver} of astrotable.") from e
            except:
                raise
            dataname = attrs['name'] if 'name' in attrs else None
            data = cls(attrs['t'], name=dataname)
            update_names = [i for i in attrs if i not in ['name', 't']] # attrs that need to be updated
            for name in update_names:
                if name == 'meta': # a special case: can't set attribute 'meta'
                    getattr(data, name).clear()    
                    getattr(data, name).update(attrs[name])
                else: # the normal cases
                    setattr(data, name, attrs[name])
            return data
        elif format == 'pkl':
            return load_pickle(path)
        else:
            return cls(path, format=format, **kwargs)
    
    #### basic methods
    
    def metaJson(self, save_path=None, yes=False):
        '''
        Generate a json string for the metadata of this Data.
    
        The metadata of an ``astrotable.table.Data`` object typically saves the information
        on where the data we loaded, how was it merged (if it is a merged catalog), etc.
        It can be retrieved with ``data.meta``.
        This is saved as the metadata of ``data.t``, i.e. ``data.meta is data.t.meta``.

        Parameters
        ----------
        save_path : str, optional
            A path to save the json as a file. The default is None (do not save).
        yes : bool, optional
            If set to True, existing files will be overwritten without prompts. The default is False.

        Returns
        -------
        meta : str
            A json string.

        '''
        def break_lines(odict, key):
            odict[key] = odict[key].split('\n')
            
            # lines = odict[key] 
            # assert isinstance(lines, str)
            # keys = list(odict.keys())
            # idx = keys.index(key)
            # after_keys = keys[idx+1:]
            # lines = lines.split('\n')
            # odict.pop(key)
            # for i, line in enumerate(lines):
            #     odict[f'{key}_l{i+1}'] =  line
            # for key in after_keys:
            #     odict.move_to_end(key)
                
        def prepare_meta(odict):
            if 'merging' in odict.keys():
                break_lines(odict['merging'], 'tree')
                for data in odict['merging']['metas']:
                    prepare_meta(odict['merging']['metas'][data])
            
        meta = deepcopy(self.meta)
        prepare_meta(meta)
        
        meta = json.dumps(meta, ensure_ascii=False, indent=4)
        if save_path:
            if os.path.exists(save_path) and not yes:
                pause_and_warn(f"file '{save_path}' already exists!",
                               choose='Proceed to overwrite this file?',
                               yes_message=f"file '{save_path}' overwritten.")
            with open(save_path, 'w', encoding='utf8') as f:
                f.write(meta)
        return meta
    
    ## below are magic methods
    
    def __repr__(self):
        name = f"'{self.name}'" if self.name is not None else 'without name'
        return f"<Data {name}>"
    
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, item):
        # warnings.warn('Although supported, it is not suggested to access table by directly subscripting Data objects. Use e.g. data.t[index] instead of data[index].')
        return (self.t[item])
        # return Data(self.t[item])
    
    def __setitem__(self, item, value):
        # only changes metadata when adding one new column; 
        # see data.t's __setitem__ (astropy.table.table.Table.__setitem__)
        # TODO: better handle metadata (but maybe no need to change metadata for 
        # changing existing columns; adding several new column seems to be unsupported)
        
        # warnings.warn('Although supported, it is not suggested to set items of the table by directly subscripting Data objects. Use e.g. data.t[index] instead of data[index].')
        if not isinstance(item, str):
            raise NotImplementedError('Currently, we only accept a str as the index. You may consider directly setting the values by `data.t[...] = ...` instead of `data[...] = ...` (WITH CAUTION).')
            
        new = isinstance(item, str) and item not in self.colnames
        if new:
            self.t[item] = value
            self.t[item].meta['src'] = 'user-added'
            self.t[item].meta['src_detail'] = 'set by user'
            self.t[item].meta['set_by_user'] = True
            self.t[item].description = ''
            self.t[item].unit = ''
        else: # modifying existing column?
            # meta not modified; description and unit cleared
            old_meta = self.t[item].meta
            self.t[item] = value 
            self.t[item].meta = old_meta
            self.t[item].description = ''
            self.t[item].unit = ''
            self.t[item].meta['src_detail'] += '; modified by user'
            self.t[item].meta['set_by_user'] = True
        
    #### alternative names
    @property
    def labels(self): # alternative name for col_labels
        return self.col_labels
    
    def df(self, index=None, use_nullable_int=True): # convenient method to get the pandas DataFrame from data
        return self.t.to_pandas(index=index, use_nullable_int=use_nullable_int)
    
    subsets = subset_data # another name for subset_data
    subplot_array = plots
