import gdal, osr
import os, json
from pyproj import Proj, transform
from datetime import datetime
import numpy as np

class MODIS_NDVI_MOD09GA():

    bits_map = np.array(['00', '01', '10', '11'])

    # file_path  --- full path to target HDF file
    # extent     --- dictionary like {'minX': ..., 'minY': ...,'maxX': ..., 'maxY': ...}, coordinated are WGS84
    # resolution --- dictionary like {'xRes': 1000, 'yRes': 1000}, with spatial resolution of target array in meters
    # key_values --- dictionary like {'gap': -100.0, 'skip': -200.0,'noData': -32768.0}, with values for gaps, pixels to be skipped and noData pixels.
    # layer      --- Day, Night
    # qa_policy  --- mode qa flags sensitivity. 0 - Do not use non-confident data 1 - Use everything

    def __init__(self, file_path, extent, resolution, key_values={'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}, qa_policy=0):
        self.file_path = file_path
        self.extent = extent
        self.resolution = resolution
        self.key_values = key_values

        self.file_name = os.path.basename(self.file_path)
        self.utm_code, self.utm_extent = self.__get_utm_code_from_extent()
        self.qa_policy = qa_policy

        self.datetime = datetime.strptime(self.file_name.split('.')[1], 'A%Y%j')
        self.datetime = self.datetime.strftime('%Y%m%dT%H%M%S')

        self.metadata = {}
        self.metadata.update({'file_name': self.file_name,
                              'datetime': self.datetime,
                              'extent': self.extent,
                              'utm_code': self.utm_code,
                              'utm_extent': self.utm_extent,
                              'resolution': self.resolution,
                              'qa_policy': self.qa_policy})

    # Choosing optimal UTM projection
    # return utm_code   --- UTM code
    # return utm_extent --- dictionary like {'minX': ..., 'minY': ...,'maxX': ..., 'maxY': ...}, with UTM coordinates
    def __get_utm_code_from_extent(self):
        minX = self.extent.get('minX')
        minY = self.extent.get('minY')
        maxX = self.extent.get('maxX')
        maxY = self.extent.get('maxY')

        y_centroid = (minY + maxY) / 2
        # 326NN или 327NN- где NN это номер зоны
        if y_centroid < 0:
            base_code = 32700
        else:
            base_code = 32600

        x_centroid = (minX + maxX) / 2
        zone = int(((x_centroid + 180) / 6.0) % 60) + 1
        utm_code = base_code + zone

        wgs = Proj(init="epsg:4326")
        utm = Proj(init="epsg:" + str(utm_code))
        min_corner = transform(wgs, utm, *[minX, minY])
        max_corner = transform(wgs, utm, *[maxX, maxY])
        utm_extent = {'minX': min_corner[0], 'minY': min_corner[1], 'maxX': max_corner[0], 'maxY': max_corner[1]}
        return (utm_code, utm_extent)

    def save_metadata(self, output_path):
        with open(output_path, 'w') as f:
            f.write(json.dumps(self.metadata))

    def file_path_to_product_name(self, file_path, product):
        if product in ['sur_refl_b01_1','sur_refl_b02_1']:
            new_path = 'HDF4_EOS:EOS_GRID:\"%s\":MODIS_Grid_500m_2D:%s' % (file_path, product)
        elif product in ['state_1km_1']:
            new_path = 'HDF4_EOS:EOS_GRID:\"%s\":MODIS_Grid_1km_2D:%s' % (file_path, product)
        return new_path

    def last_two_bits(self, arr_in):
        return self.bits_map[arr_in & 3]

    def create_quality_array(self, qc_array):
        # 0 - OK
        # 1 - Gap
        # 2 - Skip
        # 3 - NoData
        quality = np.zeros_like(qc_array)

        if self.qa_policy == 0:
            last_two_bits_array = self.last_two_bits(qc_array)
            quality[last_two_bits_array == '00'] = 0
            quality[last_two_bits_array == '01'] = 1
            quality[last_two_bits_array == '10'] = 1
            quality[last_two_bits_array == '11'] = 0
            quality[qc_array == -32768] = 3

        if self.qa_policy == 1:
            last_two_bits_array = self.last_two_bits(qc_array)
            quality[last_two_bits_array == '00'] = 0
            quality[last_two_bits_array == '01'] = 1
            quality[last_two_bits_array == '10'] = 0
            quality[last_two_bits_array == '11'] = 0
            quality[qc_array == -32768] = 3

        return quality

    def __preparation(self, mode='gtiff'):
        projection = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'

        # Get QC
        qc_full_file_path = self.file_path_to_product_name(self.file_path, 'state_1km_1')
        qc = gdal.Warp('', qc_full_file_path, format='MEM', srcNodata=-1, dstNodata=-32768, outputType=gdal.GDT_Int16)
        qc_array = qc.GetRasterBand(1).ReadAsArray()
        quality = self.create_quality_array(qc_array)
        qc.GetRasterBand(1).WriteArray(quality)

        # QC extent
        qc_geotransform = qc.GetGeoTransform()
        xMin = qc_geotransform[0]
        yMax = qc_geotransform[3]
        xMax = xMin + qc_geotransform[1] * qc.RasterXSize
        yMin = yMax + qc_geotransform[5] * qc.RasterYSize
        xRes = qc_geotransform[1]
        yRes = qc_geotransform[5]

        # Get RED band
        red_full_file_path = self.file_path_to_product_name(self.file_path, 'sur_refl_b01_1')
        red = gdal.Warp('', red_full_file_path, format='MEM', dstSRS=projection, dstNodata=-32768,
                        outputType=gdal.GDT_Float32, xRes=xRes,yRes=yRes,
                        outputBounds = [xMin,yMin,xMax,yMax], copyMetadata=False)

        red_array = red.GetRasterBand(1).ReadAsArray()
        red_array = red_array / 10000.0
        red.GetRasterBand(1).WriteArray(red_array)

        # Get NIR band
        nir_full_file_path = self.file_path_to_product_name(self.file_path, 'sur_refl_b02_1')
        nir = gdal.Warp('', nir_full_file_path, format='MEM', dstSRS=projection, dstNodata=-32768,
                        outputType=gdal.GDT_Float32, xRes=xRes,yRes=yRes,
                        outputBounds = [xMin,yMin,xMax,yMax], copyMetadata=False)

        nir_array = nir.GetRasterBand(1).ReadAsArray()
        nir_array = nir_array / 10000.0
        nir.GetRasterBand(1).WriteArray(nir_array)

        drv = gdal.GetDriverByName('MEM')
        ndvi = drv.Create('', nir.RasterXSize, nir.RasterYSize, 1, gdal.GDT_Float32)
        ndvi.SetGeoTransform(nir.GetGeoTransform())
        ndvi.SetProjection(nir.GetProjection())
        ndvi_array = (nir_array - red_array) / (nir_array + red_array)
        ndvi_array[ndvi_array > 1] = self.key_values['gap']
        ndvi_array[ndvi_array < -1] = self.key_values['gap']
        ndvi_array[quality == 1] = self.key_values['gap']
        ndvi_array[quality == 2] = self.key_values['skip']
        ndvi_array[quality == 3] = self.key_values['NoData']
        ndvi.GetRasterBand(1).WriteArray(ndvi_array)

        output_rs = osr.SpatialReference()
        output_rs.ImportFromEPSG(self.utm_code)

        if mode == 'gtiff':
            warpOptions = gdal.WarpOptions(dstNodata=self.key_values['NoData'], format='GTiff',
                                           srcSRS = projection, dstSRS = output_rs,
                                           outputBounds=[self.utm_extent.get('minX'), self.utm_extent.get('minY'),
                                                         self.utm_extent.get('maxX'), self.utm_extent.get('maxY')],
                                                         xRes=self.resolution.get('xRes'), yRes=self.resolution.get('yRes'),
                                                         creationOptions=['COMPRESS=LZW'])
        if mode == 'npy':
            warpOptions = gdal.WarpOptions(dstNodata=self.key_values['NoData'], format='MEM',
                                           srcSRS = projection, dstSRS = output_rs,
                                           outputBounds=[self.utm_extent.get('minX'), self.utm_extent.get('minY'),
                                                         self.utm_extent.get('maxX'), self.utm_extent.get('maxY')],
                                                         xRes=self.resolution.get('xRes'), yRes=self.resolution.get('yRes'))

        return ndvi, warpOptions

    def archive_to_geotiff(self, save_path):
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

        lst, warpOptions = self.__preparation(mode='gtiff')
        geotiff_name = self.datetime + '.tif'
        geotiff_path = os.path.join(save_path, geotiff_name)
        gdal.Warp(geotiff_path, lst, dstNodata=self.key_values.get('NoData'), options=warpOptions)

    def archive_to_npy(self, save_path):
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

        lst, warpOptions = self.__preparation(mode='npy')
        raster = gdal.Warp('', lst, dstNodata=self.key_values.get('NoData'), options=warpOptions)

        npy_name = self.datetime + '.npy'
        npy_path = os.path.join(save_path, npy_name)
        matrix = raster.ReadAsArray()
        matrix = np.array(matrix)
        np.save(npy_path, matrix)

a = MODIS_NDVI_MOD09GA('/home/ekazakov/MOD09GA.A2019197.h20v03.006.2019199030333.hdf',
                       extent={'minX': 47, 'minY': 56,'maxX': 48, 'maxY': 57},
                       resolution={'xRes': 1000, 'yRes': 1000},
                       qa_policy=0)

a.archive_to_geotiff('/home/ekazakov/modis5')
