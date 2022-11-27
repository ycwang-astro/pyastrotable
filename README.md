# A Python package with tools for table operation

`astrotable` is a Python package that provides tools for operations on catalogs and tables, especially those common in astronomy. The aim is to make table operations as simple and intuitive as possible. Mainly based on `astropy`, this package helps you match, analyze and visualize catalogs. 

Of course, you can do anything you want with `astropy`, `numpy`, `matplotlib`, etc., but this package is designed for lazy people who would like to do some common operations in a simple and intuitive way.

Please note that this project is still in its early stage, and the names of the APIs, modules and even the package itself might be changed in a future update. Currently, the main features of this package include:
- Matching and merging catalogs
- Row subsets or groups of row subsets of catalogs
- Easily making plots comparing different subsets

## Installation

To install this package, run:
```
pip install git+https://github.com/ycwang-astro/pyastrotable.git
```

## Documentation
For tutorials, see [tutorial on matching](tutorials/tutorial1_matching.ipynb) and [tutorial on subsets and plots](tutorials/tutorial2_subset_and_plot.ipynb).

## Future updates
Planned future updates:
- More documentation and demostrations
- New feature: convenient plotting functions

## Change log
### 0.2.0
New features:
- Row subsets or groups of row subsets of catalogs
- Easily making plots comparing different subsets
- Applying functions to each row of the catalog
