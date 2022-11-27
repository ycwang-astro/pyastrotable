# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 2022

@author: Yuchen Wang

Built-in matchers.
"""

import numpy as np
from astrotable.utils import find_idx
from astropy.coordinates import SkyCoord
import astropy.units as u

class ExactMatcher():
    def __init__(self, value, value1):
        '''
        Used to match `astrotable.table.Data` objects `data1` to `data`.
        Match records with exact values.
        This should be passed to method `data.match()`.
        See `help(data.match)`.

        Parameters
        ----------
        value : str or Iterable
            Specify values for `data` used to match catalogs. Possible inputs are:
            - str, name of the field used for matching.
            - Iterable, values for `data`. `len(value)` should be equal to `len(data)`.
        value1 : str or Iterable
            Specify values for `data1` used to match catalogs. Possible inputs are:
            - str, name of the field used for matching.
            - Iterable, values for `data1`. `len(value1)` should be equal to `len(data1)`.
        '''
        self.value = value
        self.value1 = value1
    
    def get_values(self, data, data1, verbose=True):
        if type(self.value) is str:
            self.value = data.t[self.value]
        if type(self.value1) is str:
            self.value1 = data1.t[self.value1]
    
    def match(self):
        idx, matched = find_idx(self.value1, self.value)
        return idx, matched
    
    def __repr__(self):
        return f"ExactMatcher('{self.value.name}', '{self.value1.name}')"


class SkyMatcher():
    def __init__(self, thres=1, coord=None, coord1=None, unit=u.deg, unit1=u.deg):
        '''
        Used to match `astrotable.table.Data` objects `data1` to `data`.
        Match records with nearest coordinates.
        This should be passed to method `data.match()`.
        See `help(data.match)`.

        Parameters
        ----------
        thres : number, optional
            Threshold in arcsec. The default is 1.
        coord : str or astropy.coordinates.SkyCoord, optional
            Specify coordinate for the base data. Possible inputs are:
            - astropy.coordinates.SkyCoord (recommended), the coordinate object.
            - str, should be like 'RA-DEC', which specifies the column name for RA and Dec.
            - None (default), will try ['ra', 'RA'] and ['DEC', 'Dec', 'dec'].
            The default is None.
        coord1 : str or astropy.coordinates.SkyCoord, optional
            Specify coordinate for the matched data. Possible inputs are:
            - astropy.coordinates.SkyCoord (recommended), the coordinate object.
            - str, should be like 'RA-DEC', which specifies the column name for RA and Dec.
            - None (default), will try ['ra', 'RA'] and ['DEC', 'Dec', 'dec'].
            The default is None.
        unit : astropy.units.core.Unit or list/tuple/array of it
            If astropy.coordinates.SkyCoord object is not given for coord, 
            this is used to specify the unit of auto-found coord.
            The default is astropy.units.deg.
        unit1 : astropy.units.core.Unit or list/tuple/array of it
            If astropy.coordinates.SkyCoord object is not given for coord1, 
            this is used to specify the unit of auto-found coord1.
            The default is astropy.units.deg.
           
        '''
        self.thres = thres
        self.coord = coord
        self.coord1 = coord1
        self.unit = unit
        self.unit1 = unit1
    
    def get_values(self, data, data1, verbose=True):
        # TODO: this method has not been debugged!
        # USE WITH CAUTION!
        ra_names = np.array(['ra', 'RA'])
        dec_names = np.array(['DEC', 'Dec', 'dec'])
        coords = []
        for coordi, datai, uniti in [[self.coord, data, self.unit], [self.coord1, data1, self.unit1]]:
            if coordi is None:
                found_ra = np.isin(ra_names, datai.colnames)
                if not np.any(found_ra):
                    raise KeyError(f'RA for {datai.name} not found.')
                self.ra_name = ra_names[np.where(found_ra)][0]
                ra = datai.t[self.ra_name]

                found_dec = np.isin(dec_names, datai.colnames)
                if not np.any(found_dec):
                    raise KeyError(f'Dec for {datai.name} not found.')
                self.dec_name = dec_names[np.where(found_dec)][0]
                dec = datai.t[self.dec_name]
                coords.append(SkyCoord(ra=ra, dec=dec, unit=uniti))
                if verbose: print(f"Data {datai.name}: found RA name '{self.ra_name}' and Dec name '{self.dec_name}'.")
            
            elif type(coordi) is str:
                self.ra_name, self.dec_name = coordi.split('-')
                ra = datai.t[self.ra_name]
                dec = datai.t[self.dec_name]
                coords.append(SkyCoord(ra=ra, dec=dec, unit=uniti))
            
            elif type(coordi) is SkyCoord:
                self.ra_name, self.dec_name = None, None
                coords.append(coordi)
            
            else:
                raise TypeError(f"Unsupported type for coord/coord1: '{type(coordi)}'")
        
        self.coord, self.coord1 = coords
        
    def match(self):
        idx, d2d, d3d = self.coord.match_to_catalog_sky(self.coord1)
        matched = d2d.arcsec < self.thres
        return idx, matched
    
    def explore(self, data, data1):
        '''
        Plot as simple histogram to 
        check the distribution of the minimum (2-d) sky separation.

        Parameters
        ----------
        data : ``astrotable.table.Data``
            The base data of the match.
        data1 : ``astrotable.table.Data``
            The data to be matched to ``data1``.

        Returns
        -------
        None.

        '''
        self.get_values(data, data1)
        idx, d2d, d3d = self.coord.match_to_catalog_sky(self.coord1)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(np.log10(d2d.arcsec), bins=min((200, len(data)//20)), histtype='step', linewidth=1.5, log=True)
        plt.axvline(np.log10(self.thres), color='r', linestyle='--')
        plt.xlabel('lg (d / arcsec)')
        plt.title(f"Min. distance to '{data1.name}' objects for each '{data.name}' object\nthreshold={self.thres}\"")
        
        
    def __repr__(self):
        return f'SkyMatcher with thres={self.thres}'
