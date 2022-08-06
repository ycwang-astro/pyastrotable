# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 18:08:42 2022

@author: Yuchen Wang
"""

from setuptools import setup, find_packages
import astrotable

setup(
    name='astrotable',
    version=astrotable.__version__,
    packages=find_packages(include=['astrotable', 'astrotable.*']),
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
    ]
)
