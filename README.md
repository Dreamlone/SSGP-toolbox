<img src="https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/redesign/docs/media/images/label.png" width="800"/>

**Simple Spatial Gapfilling Processor - toolbox** - python library for filling 
gaps in matrices (can be applied on any matrices) based on machine learning techniques and 
spatial relations in the data. 

**Main purpose** is to provide convenient and simple tools for modeling geophysical parameters, 
derived with Earth Remote Sensing, under clouds. This module includes the tools 
needed to download data via the API of the most common 
Remote Sensing data sources, preprocessing algorithms, as well 
as the core of the library - the gap-filling algorithm.

## Brief description 

In this section a brief overview of the functionality of SSGP-toolbox has been prepared.


## Installation

Use the following commands to install this module

Using pip:

```Bash
pip install ssgp
```

Using poetry:

```Bash
poetry add ssgp
```

## Usage examples

In progress - documentation, jupyter notebooks and simple python code in example folder

## Documentation

In progress - see build on readthedocs

## Issues related to backward compatibility

**SSGP-toolbox** was developed as a minor MVP (A minimum viable product) product 
to test some hypotheses about how 
to recover gaps in remote sensing images during **`2019-2020`** (how long ago it was!). 

Since then, new technologies have appeared, and the programming skills of 
the maintainers of this repository have grown too. Therefore, in **`July 2023`**, 
significant refactoring and redesign of the module and documentation was performed.
The functionality of the module has been greatly expanded. Architecture 
refactoring has made the code more scalable, and a new user-friendly 
interface allows running pipelines for processing remote sensing data in just 
a few lines of code. 

However, we have tried to make sure that the module interface 
from the old version is also available in the new version (backwards compatibility).
Thus there are two SSGP-toolbox interfaces:

- Old via `SimpleSpatialGapfiller` and other related classes;
- New via `Extractor`, `Preprocessor`, `Gapfiller`, `Converter` and other more 
  convenient wrappers

It is recommended to use the new interface version. 
More information about each version can be found in the official documentation.

## Comparison 

If you want to compare the efficiency of your algorithm with ours, you 
can use the dataset we have prepared. You can find it in the "Comparison" folder. 
The dataset already contains the layers filled in by our model, as well as the 
["CRAN gapfill"](https://cran.r-project.org/web/packages/gapfill/index.html) 
and ["gapfilling rasters"](https://github.com/HughSt/gapfilling_rasters) layers.

Section with [detailed description of the data used for comparison](./Comparison)

## Citation

Sarafanov, M.; Kazakov, E.; Nikitin, N.O.; Kalyuzhnaya, A.V. 
[A Machine Learning Approach for Remote Sensing Data Gap-Filling 
with Open-Source Implementation: An Example Regarding Land Surface 
Temperature, Surface Albedo and NDVI](https://www.mdpi.com/2072-4292/12/23/3865). Remote Sens. 2020, 12, 3865.

## Contacts

If you have any questions or suggestions feel free to contact us:

| Person                         |        e-mail           |
|--------------------------------|-------------------------|
| Mikhail Sarafanov (maintainer) | mik.sarafanov@gmail.com |
| Eduard Kazakov                 | ee.kazakov@gmail.com    |


