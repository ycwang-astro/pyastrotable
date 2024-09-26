from setuptools import setup, find_packages

import astrotable

setup(
    name='astrotable',
    version=astrotable.__version__,
    packages=find_packages(include=['astrotable', 'astrotable.*']),
    install_requires=[
        'pyttop',
    ],
)
