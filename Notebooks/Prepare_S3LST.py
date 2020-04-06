
# Import class
from SSGPToolbox.Preparators.Sentinel3.S3_L2_LST import S3_L2_LST

# Additional inputs
import os
import zipfile
import shutil

import numpy as np
import gdal, osr
import json
from netCDF4 import Dataset
from pyproj import Proj, transform

# Пример использования класса
# file_path  --- путь до архива
# extent     --- словарь формата {'minX': ..., 'minY': ...,'maxX': ..., 'maxY': ...}, где указаны координаты в WGS
# resolution --- словарь формата {'xRes': 1000, 'yRes': 1000}, пространственное разрешение, единицы измерения, разрешение по X, по Y
# key_values --- словарь формата {'gap': -100.0, 'skip': -200.0,'noData': -32768.0}, обозначающий пропущенные пиксели
Preparator_S3LST = S3_L2_LST(file_path = '/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/source/S3A_SL_2_LST____20190831T194258_20190831T194558_20190831T215631_0179_048_356_0900_LN2_O_NR_003.zip',
                       extent = {'minX': 30, 'minY': 58,'maxX': 31, 'maxY': 59},
                       resolution = {'xRes': 1000, 'yRes': 1000},
                       key_values = {'gap': -100.0, 'skip': -200.0,'NoData': -32768.0})

# save_path --- место, в которое нужно поместить файл с результатом
Preparator_S3LST.archive_to_geotiff(save_path = '/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/prepared')

# save_path --- место, в которое нужно поместить файл с результатом
Preparator_S3LST.archive_to_npy(save_path = '/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/prepared')

# Сохранение метаданных в файл JSON
Preparator_S3LST.save_metadata(output_path = '/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/prepared/20190831T194258_metadata.json')


# Reconstruct geotiff from NPY. import function:
from SSGPToolbox.Preparators.common_functions import reconstruct_geotiff

# Run reconstruction. Just set path to NPY and saved metadata, and to output GeoTiff
reconstruct_geotiff(npy_path = os.path.join('/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/prepared', '20190831T194258.npy'),
                    metadata_path = os.path.join('/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/prepared', '20190831T194258_metadata.json'),
                    output_path = os.path.join('/media/mikhail/Data/SSGP-toolbox/Samples/S3LST_preparation_example/prepared', '20190831T194258_reconstructed.tif'))

