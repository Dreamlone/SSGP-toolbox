# SSGP-toolbox

SimpleSpatialGapfiller - python class for filling gaps in matrices based on machine learing techniques. Main purpose is to provide convenient and simple instruments for modeling geophysical parameters, derived with Earth Remote Sensing, under clouds. But it also could be used for any matrices.


## Requirements
    'python>=3.7',
    'gdal>=2.4',
    'numpy',
    'scikit-learn==0.21.3',
    'pandas',
    'scipy',
    'netCDF4',
    'pyproj' 

###### If errors occur when installing "gdal", you should install the gdal library before running the command to install this module

## Install module

```python
pip install git+https://github.com/Dreamlone/SSGP-toolbox
```

## Modules

For now SSGT-toolbox is presented with:
 - Gapfiller class
 - Discretizator class
 - Several preparators: for Sentinel 3 LST data; for MODIS LST products; for MODIS NDVI based on reflectance product.
 
 By the way, you can prepare any data by yourself, it must be in binary numpy matrices format (.npy) and organized in several directories, as shown in docs.

## Documentation and examples

All documentation and examples for now are described in Jupyter Notebooks:
 - [Theoretical basis](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Theoretical_basis.ipynb)
 - [Sentinel 3 LST preparation](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Prepare_S3LST.ipynb)
 - [MODIS LST preparation](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Prepare_MODIS_LST.ipynb)
 - ##### [Gapfiller class, how to organize data and how to use](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Gapfilling.ipynb)
 - ##### [Discretizator class, how to organize data and how to use](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Discretizator.ipynb)

## Contacts

Feel free to contact us:

Mikhail Sarafanov (maintainer) | mik_sar@mail.ru

Eduard Kazakov | ee.kazakov@gmail.com





