from abc import abstractmethod
from pathlib import Path
from typing import Union

import os, json

from pyproj import Proj, transform

from ssgp.preprocessing.common_functions import convert_path_into_string
from ssgp.preprocessing.default_params import DEFAULT_PIXELS_KEYS


class DefaultPreprocessor:
    """
    Base class for preprocessing operations with archives and files

    :param file_path: full path to target HDF file
    :param extent: dictionary like
    {'minX': ..., 'minY': ...,'maxX': ..., 'maxY': ...}, coordinated are WGS84
    :param resolution: dictionary like {'xRes': 1000, 'yRes': 1000}, with
    spatial resolution of target array in meters
    :param key_values: dictionary like
    {'gap': -100.0, 'skip': -200.0,'noData': -32768.0}, with values for gaps,
    pixels to be skipped and noData pixels
    """

    def __init__(self, file_path: Union[str, Path],
                 extent: dict, resolution: dict,
                 key_values: Union[dict, None] = None):

        # Process paths to archive
        if isinstance(file_path, Path):
            file_path = str(file_path)
        self.file_path = os.path.abspath(file_path)

        self.extent = extent
        self.resolution = resolution
        if key_values is None:
            # Use default values
            key_values = DEFAULT_PIXELS_KEYS
        self.key_values = key_values

        # Empty metadata for base class
        self.metadata: dict = {}

    def save_metadata(self, output_path: Union[str, Path]):
        """ Save metadata into json file

        :param output_path: path to JSON file where need to save metadata
        """
        output_path = convert_path_into_string(output_path)

        if '.json' not in output_path:
            raise ValueError(f'Metadata must be saved into json file, not {output_path}')

        with open(output_path, 'w') as f:
            f.write(json.dumps(self.metadata))

    @abstractmethod
    def archive_to_geotiff(self, save_path: Union[str, Path], **kwargs):
        """ Convert archive into geotiff file """
        raise NotImplementedError()

    @abstractmethod
    def archive_to_npy(self, save_path: Union[str, Path], **kwargs):
        """ Convert archive into npy file """
        raise NotImplementedError()

    def __get_utm_code_from_extent(self):
        """
        The private method for the selection of the most appropriate metric
        projection is called when initializing the class

        :return utm_code: UTM projection code
        :return utm_extent: dictionary {'minX': ..., 'minY': ...,'maxX': ...,
        'maxY': ...}, where are the coordinates in UTM
        """
        min_x = self.extent.get('minX')
        min_y = self.extent.get('minY')
        max_x = self.extent.get('maxX')
        max_y = self.extent.get('maxY')

        y_centroid = (min_y + max_y) / 2
        # 326NN or 327NN - where NN is the zone number
        if y_centroid < 0:
            base_code = 32700
        else:
            base_code = 32600

        x_centroid = (min_x + max_x) / 2
        zone = int(((x_centroid + 180) / 6.0) % 60) + 1
        utm_code = base_code + zone

        wgs = Proj(init="epsg:4326")
        utm = Proj(init="epsg:" + str(utm_code))
        min_corner = transform(wgs, utm, *[min_x, min_y])
        max_corner = transform(wgs, utm, *[max_x, max_y])
        utm_extent = {'minX': min_corner[0], 'minY': min_corner[1],
                      'maxX': max_corner[0], 'maxY': max_corner[1]}
        return utm_code, utm_extent
