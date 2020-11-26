# ![SSGP_label.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/label.png)


SimpleSpatialGapfiller - python class for filling gaps in matrices based on machine learing techniques. Main purpose is to provide convenient and simple instruments for modeling geophysical parameters, derived with Earth Remote Sensing, under clouds. But it also could be used for any matrices.


## Citation
Sarafanov, M.; Kazakov, E.; Nikitin, N.O.; Kalyuzhnaya, A.V. 
[A Machine Learning Approach for Remote Sensing Data Gap-Filling 
with Open-Source Implementation: An Example Regarding Land Surface 
Temperature, Surface Albedo and NDVI](https://www.mdpi.com/2072-4292/12/23/3865). Remote Sens. 2020, 12, 3865.


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

For now SSGP-toolbox is presented with:
 - Gapfiller class
 - Discretizator class
 - Several preparators: for Sentinel 3 LST data; for MODIS LST products; for MODIS NDVI based on reflectance product.
 - Algorithm for identifying cloud-shaded pixels in temperature field
 
 By the way, you can prepare any data by yourself, it must be in binary numpy matrices format (.npy) and organized in several directories, as shown in docs.

## Documentation and examples

All documentation and examples for now are described in Markdown files and Jupyter Notebooks:
 - [Theoretical basis](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Theoretical_basis.md)
 - [Sentinel 3 LST preparation](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Prepare_S3LST.ipynb)
 - [MODIS LST preparation](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Prepare_MODIS_LST.ipynb)
 - [Identifying cloud-shaded pixels](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Cellular_automaton.ipynb)
 - ##### [Gapfiller class, how to organize data and how to use](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Gapfilling.ipynb)
 - ##### [Discretizator class, how to organize data and how to use](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Notebooks/Discretizator.ipynb)

## Comparison 
If you want to compare the accuracy of your algorithm with ours, you can use the dataset we have prepared. You can find it in the "Comparison" folder. The dataset already contains the layers filled in by our model, as well as the ["CRAN gapfill"](https://cran.r-project.org/web/packages/gapfill/index.html) and ["gapfilling rasters"](https://github.com/HughSt/gapfilling_rasters) layers.
- [Detailed description of the data](https://github.com/Dreamlone/SSGP-toolbox/tree/master/Comparison/Description.md)

## Contacts

Feel free to contact us:

Mikhail Sarafanov (maintainer) | mik_sar@mail.ru

Eduard Kazakov | ee.kazakov@gmail.com


