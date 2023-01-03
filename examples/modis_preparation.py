from pathlib import Path

from ssgp.paths import get_samples_path
from ssgp.preprocessing.modis.mod_myd import ModisModMydPreprocessor


def modis_convert_example():
    """
    Demonstration how to convert archive of MODIS LST data (MOD11A1, MOD11A2,
    MOD11_L2, MYD11A1, MYD11A2, MYD11_L2) into npy files or geotiff
    """
    input_mod11_l2 = Path(get_samples_path(), 'MODIS_preparation_example',
                          'source', 'MOD11_L2.A2020048.1915.006.2020050045344.hdf')
    # Extent for final file
    extent = {'minX': 35, 'minY': 60, 'maxX': 36, 'maxY': 61}

    # Spatial resolution for final file
    resolution = {'xRes': 1000, 'yRes': 1000}

    # Key values for cloud and no-data pixels.
    # Which values should they have in final file
    key_values = {'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}

    # What to do with quality flags?
    # 0 - Do not use non-confident data
    # 1 - Use everything
    qa_policy = 0

    # Initialize class
    modis_preprocessor = ModisModMydPreprocessor(input_mod11_l2,
                                                 key_values=key_values,
                                                 extent=extent,
                                                 resolution=resolution,
                                                 qa_policy=qa_policy)
    print('Demonstration of Preprocessor metadata')
    print(modis_preprocessor.metadata)

    # You can save metadata - it will be useful for restoring spatial dataset
    # from .NPY produced by Gapfiller
    modis_preprocessor.save_metadata(Path(get_samples_path(),
                                          'MODIS_preparation_example',
                                          'prepared', 'L2', 'l2_metadata.json'))

    modis_preprocessor.archive_to_geotiff(Path(get_samples_path(),
                                               'MODIS_preparation_example',
                                               'prepared', 'L2'))

    modis_preprocessor.archive_to_npy(Path(get_samples_path(),
                                           'MODIS_preparation_example',
                                           'prepared', 'L2'))


if __name__ == '__main__':
    modis_convert_example()
