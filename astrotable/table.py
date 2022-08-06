# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang

Main tools to store, operate and visualize data tables.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, hstack
from astrotable.utils import objdict
import warnings
# import time

class Data():
    '''
    A class to store, manipulate and visualize data tables.
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
        if name is None:
            warnings.warn('It is strongly suggested to input a name.')

        if type(data) is str: # got a path
            self.t = Table.read(data, **kwargs)
            self.path = data
        elif isinstance(data, Table): # got astropy table
            self.t = data
            self.path = None
        else: # try to convert to data
            self.t = Table(data)
            self.path = None
            
        self.colnames = self.t.colnames
        self.name = name
        if self.name is None and self.path is not None:
            self.name = self.path.split('/')[-1].split('\\')[-1]
        self.matchinfo = []
        self.matchnames = []
        self.matchlog = []
        # self.id = time.time() #time.strftime('%y%m%d%H%M%S')
        
    def match(self, data1, matcher, verbose=True):
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
                ```
                class MyMatcher():
                    def __init__(self, args): # 'args' means any number of arguments that you need
                        # initialize it with args you need
                        pass
                    
                    def get_values(self, data, data1, verbose=True): # data1 is matched to data
                        # prepare the data that is needed to do the matching (if necessary)
                        pass
                    
                    def match(self):
                        # do the matching process and calculate:
                        # idx : array of shape (len(data), ). 
                        #     the index of a record in data1 that best matches the records in data
                        # matched : boolean array of shape (len(data), ).
                        #     whether the records in data can be matched to those in data1.
                        return idx, matched
                ```
        verbose : bool, optional
            Whether to output matching information. The default is True.

        Raises
        ------
        ValueError
            - Data with the same name is not allowed to be matched to this Data twise.

        '''
        if data1.name in self.matchnames:
            raise ValueError(f"Data with name '{data1.name}' has already been matched. This may result from name duplicates or re-matching the same catalog.")
        
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
        if verbose: print(matchstr)
        
        return self
    
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
    
    def merge(self, depth=-1, keep_unmatched=[], merge_columns={}, outname=None, verbose=True):
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
            A record (row) of THIS data is kept even if it cannot be matched to data in the above list.
            The default is [] (which means that only those that can be matched to each child data of this data are kept).
        merge_columns : dict, optional
            A dict that specifies fields (columns) to be matched.
            For example, if `data1` with name 'Data_1' is matched to this object, and you want to merge only 
            'column1', 'column2' in `data1` into the merged catalog, use:
                {'Data_1': ['column1', 'column2']}
            If, e.g, `merge_columns` for `data2` (with name 'Data_2') is not specified, every fields (columns) of `data2` will be merged.
            The default is {}.
        outname : str, optional
            The name of the merged data. 
            If not given, will be automatically generated from the names of data that are merged.
            The default is None.
        verbose : bool, optional
            Whether to show more information on merging. The default is True.

        Returns
        -------
        matched_data : `astrotable.table.Data`
            An `astrotable.table.Data` object containing the merged catalog.

        '''
        if type(keep_unmatched) is str:
            keep_unmatched = [keep_unmatched]
        
        matched = np.full((len(self),), True)
        data1_matched_tables = []
        unnamed_count = 0
        data_names = [self.name]
        
        merged_matchinfo = self.merge_matchinfo(depth=depth)
        
        # get matched indices
        for matchinfo in merged_matchinfo:
            data1 = matchinfo.data1
            if data1.name not in keep_unmatched:
                data1_matched = matchinfo.matched
                matched &= data1_matched
                
            if data1.name is None:
                unnamed_count += 1
                data_names.append(str(unnamed_count))
            else:
                data_names.append(data1.name)
                
        if unnamed_count > 0 and verbose:
            print(f'found no names for {unnamed_count} sets of data, automatically named with numbers.')
        
        # cut data
        for matchinfo in merged_matchinfo:
            data1 = matchinfo.data1
            idx = matchinfo.idx
            data1_matched = matchinfo.matched
            
            data1_table = data1.t
            
            if data1.name in merge_columns:
                data1_table = data1_table[merge_columns[data1.name]]
            
            if data1.name in keep_unmatched: # keep unmatched
                if verbose: print(f'entries with no match for {data1.name} is kept.')
                idx[~data1_matched] = 0
                data1_table = Table(data1_table, masked=True)
                data1_matched_table = data1_table[idx]
                for c in data1_matched_table.columns:
                    data1_matched_table[c].mask[~data1_matched]=True
                data1_matched_table = data1_matched_table[matched]

            else: # do not keep unmatched
                data1_matched_table = data1_table[idx[matched]]
            
            data1_matched_tables.append(data1_matched_table)
        
        data = self.t[matched]
        
        if self.name in merge_columns:
            data = data[merge_columns[self.name]]
        matched_table = hstack([data] + data1_matched_tables, table_names=data_names)
        
        if outname is None:
            outname = 'match_' + '_'.join(data_names)
            
        matched_data = Data(matched_table, name=outname)
        
        if verbose: print('merged: ' + ', '.join(data_names))
        
        return matched_data
    
    def match_merge(self, data1, matcher, keep_unmatched=[], merge_columns={}, outname=None, verbose=True):
        '''
        Match this data with `data1` and immediately merge everything that can be matched to this data.
        See `help(astrotable.table.Data.match)` and `help(astrotable.table.Data.merge)` for more information.
        '''
        self.match(data1=data1, matcher=matcher, verbose=verbose)
        return self.merge(keep_unmatched=keep_unmatched, merge_columns=merge_columns, outname=outname, verbose=verbose)
    
    def _match_tree(self, depth=-1, detail=True, matched_names=[], matched_ids=[], indent='', matcher='base'):
        # copy lists to avoid modifying it in-place (which will cause the method to "remember" them!)
        matched_names = matched_names.copy()
        matched_ids = matched_ids.copy()
        
        # print this name
        matcher = '' if not detail else f' [{matcher}]'
        name = 'Unnamed' if self.name is None else self.name
        if id(self) in matched_ids:
            print(f'{indent}({name}){matcher}')
            return matched_names, matched_ids
        else:
            matched_names.append(name)
            matched_ids.append(id(self))
            print(f'{indent}{name}{matcher}')
        
        # print data matched to this
        if depth != 0:
            for info in self.matchinfo:
                data = info.data1
                matcher = info.matcher
                matched_names, matched_ids = data._match_tree(depth=depth-1, detail=detail, matched_names=matched_names, matched_ids=matched_ids, indent=indent+':   ', matcher=matcher)
        return matched_names, matched_ids
            
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
    
    def save(self, path):
        pass
    
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, item):
        # return Data(self.t[item])
        warnings.warn('Although supported, it is not suggested to access table by directly subscripting Data objects. Use e.g. data.t[index] instead of data[index].')
        return (self.t[item])

