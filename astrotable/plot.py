# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:19:10 2022

@author: Yuchen Wang
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from inspect import signature
from astrotable.utils import objdict

Axes = matplotlib.axes.Axes

#%% Class
class PlotFunction():
    def __init__(self, func, input_ax=True):
        self.func = func
        self.input_ax = input_ax
        if hasattr(func, 'ax_callback'):
            self.ax_callback = func.ax_callback
        else:
            self.ax_callback = lambda ax: None
        
        if input_ax:
            plot_func = func(Axes)
            self.func_doc = plot_func.__doc__
            self.func_name = func.__name__ # the appearent name when using this function
            self.func_defname = plot_func.__name__ # the real name in the definition of plot function
            self.func_sig = (signature(plot_func))
        else:
            self.func_doc = func.__doc__
            self.func_name = func.__name__ # the appearent name when using this function
            self.func_defname = func.__name__ # the real name in the definition of plot function
            self.func_sig = (signature(self.func))
        self.func_def = self.func_name + str(self.func_sig) + '\n\n' + self.func_name + '(axis)' + str(self.func_sig)
        
        # TODO: below may cause bugs
        self.func_def = self.func_def.replace('(self, ', '(')
        if self.func_doc is None: self.func_doc = ''
        if self.func_doc and self.func_doc[0] == '\n': 
            self.func_doc = self.func_doc[1:]
        
        # self.__call__.__func__.__doc__ = self.func_doc
    
    def _call_with_ax(self, ax):
        if self.input_ax:
            @wraps(self.func(ax))
            def plot(*args, **kwargs):
                return self.func(ax)(*args, **kwargs)
        else:
            @wraps(self.func)
            def plot(*args, **kwargs):
                ca = plt.gca()
                plt.sca(ax)
                out = self.func(*args, **kwargs)
                plt.sca(ca)
                return out
        plot.ax_callback = self.ax_callback
        return plot
    
    def __call__(self, *args, **kwargs): # direct call
        '''
        Calling the plot function modified by plotFuncAx or plotFunc. 
        ''' # For documentation, execute ``<function name>.help()``.
        
        # if f is called as f(ax), f(ax=ax):
        if len(args) == 0 and list(kwargs.keys()) == ['ax']: # f called as f(ax=ax)
            ax = kwargs['ax']
            if isinstance(ax, Axes):
                return self._call_with_ax(ax)
        elif len(kwargs) == 0 and len(args) == 1: # f called as f(ax) or f(x)
            ax = args[0]
            if isinstance(ax, Axes):
                return self._call_with_ax(ax)
        
        # seem that f not called with only one axis as input:
        ax = plt.gca()
        out = self._call_with_ax(ax)(*args, **kwargs)
        self.ax_callback(ax)
        return out
    
    def in_plot(self, *args, **kwargs): # use in Data.plot
        # do not call ax_callback.
        # plot function may be called several times in one subplot,
        # but ax_callback should be called ONLY ONCE.
        ax = plt.gca()
        return self._call_with_ax(ax)(*args, **kwargs)
        # return self.ax_callback
    
    def in_subplot_array(self, ax): # use in Data.subplot_array
        return self._call_with_ax(ax)
    
    # def help(self):
    #     print(self.func_doc)
    
    def __getattr__(self, attr):
        return getattr(self.func, attr)
        
    @property
    def __doc__(self): # manually generate doc
        return self.func_def + '\n\nFunction modified to accomodate astrotable.table.Data. Original documentaion shown below:\n\n' + self.func_doc + '\n\n'
    
    @property
    def __name__(self):
        return self.func_name

class DelayedPlot():
    def __init__(self):
        raise NotImplementedError()
        pass
    
    def __call__(self):
        pass
    
#%% stand-alone functions
            
#%% wrapper for plot functions
def plotFuncAx(f):
    '''
    Makes a function compatible to astrotable.table.Data. 

    Usage::
        
        @plotFuncAx
        def f(ax): # inputs axis object `ax`
            def plot_func(<your inputs ...>):
                <make the plot>
            return plot_func
    '''
    return PlotFunction(f, input_ax=True)

def plotFunc(f):
    '''
    Makes a function compatible to astrotable.table.Data. 

    Usage::
        
        @plotFunc
        def plot_func(<your inputs ...>):
            <make the plot>
    '''
    return PlotFunction(f, input_ax=False)

#%% axis callbacks
def colorbar(ax):
    # TODO: automatically detect and add a colorbar
    raise NotImplementedError()
    pass

#%% plot functions

# to generate a universal colorbar for several scatter plots in the same panel,
# we need to play a trick: do not actually plot scatter in the main part;
# save it to ax_callback.
# TODO: Scatter is not elegant. Improve it.
class Scatter():
    def __init__(self):
        self.__name__ = 'scatter'
        self.params = []
        self.autobar = None
        # self.ax = None
    
    @staticmethod
    def _decide_autobar(c, x, autobar):
        # parse c input and decide autobar or not
        if not autobar or c is None:
            return False
        else:
            try:
                carr = np.asanyarray(c, dtype=float)
            except ValueError:
                return False
            else:
                if not (c.shape == (1, 4) or c.shape == (1, 3)) and carr.size == x.size:
                    return True
                else:
                    return False
    
    def __call__(self, ax):
        # if self.ax is not None and self.ax != ax:
        #     self.params = []
        # self.ax = ax
        def scatter(x, y, s=None, c=None, *, cmap=None, vmin=None, vmax=None, autobar=True, barlabel=None, **kwargs):
            self.autobar = self._decide_autobar(c, x, autobar)
            # self.autobar = autobar and (c is not None and len(c)==len(x))
            param = {key: value for key, value in locals().items() if key not in ('self', 'kwargs')}
            param.update(kwargs)
            self.params.append(param)
        return scatter
    
    def ax_callback(self, ax):
        if self.autobar: # decide colorbar information
            # the general parameters for the whole plot
            cs = []
            barinfo = objdict(
                vmin = None,
                vmax = None,
                barlabel = None,
                cmap = None)
            
            for param in self.params:
                for name in ['vmin', 'vmax', 'barlabel', 'cmap']: # check consistency for different calls
                    if barinfo[name] is None:
                        barinfo[name] = param[name]
                    elif barinfo[name] != param[name]:
                        raise ValueError(f'colorbar cannot be generated due to inconsistency of "{name}": {barinfo[name]} != {param[name]}')
                    
                cs.append(param['c'])
            
            # decide vmin, vmax
            if barinfo.vmin is None:
                barinfo.vmin = min([np.min(c) for c in cs])
            if barinfo.vmax is None:
                barinfo.vmax = max([np.max(c) for c in cs])
            
            param_exclude = ['cmap', 'vmin', 'vmax', 'autobar', 'barlabel']
            color_param_keys = ['vmin', 'vmax', 'cmap']
            for param in self.params:
                param = {key: value for key, value in param.items() if key not in param_exclude}
                colorparams = {key: value for key, value in barinfo.items() if key in color_param_keys}
                s = ax.scatter(**param, **colorparams)
            
            # make colorbar
            cax = plt.colorbar(s, ax=ax)
            cax.set_label(barinfo.barlabel)
            
        else:
            param_exclude = ['autobar', 'barlabel']
            for param in self.params:
                param = {key: value for key, value in param.items() if key not in param_exclude}
                ax.scatter(**param)
                
        self.params = []

scatter = plotFuncAx(Scatter())    

@plotFuncAx
def plot(ax):
    return ax.plot

@plotFuncAx
def hist(ax):
    return ax.hist

@plotFuncAx
def hist2d(ax):
    return ax.hist2d

@plotFuncAx
def errorbar(ax):
    return ax.errorbar
