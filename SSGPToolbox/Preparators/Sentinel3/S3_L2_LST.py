'''

class S3_LST_transformer - Класс, предназначенный для перевода файлов из архивов с данными LST в нужный формат
Mission: Sentinel-3
Satellite Platform: S3A_*, S3B_*
Product Type: SL_2_LST___
Timeliness: "Near Real Time", "Short Time Critical", "Short Time Critical"

Приватные методы:
get_utm_code_from_extent  --- метод, позволяющий подбирать подходящую метрическую проекцию для нужной территории
preparation               --- метод, выполняющий извлечение матриц из архива, их привязку и осуществляющий подготовку опций для обрезки растра

Публичные методы:
archive_to_geotiff        --- метод, выполняющий обрезку матриц по выбранным опциям, сохраняет файл .geotiff в нужной директории
archive_to_npy            --- метод, выполняющий обрезку матриц по выбранным опциям, сохраняет файл .npy в нужной директории

'''

import os
import zipfile
import shutil

import numpy as np
import gdal, osr
from netCDF4 import Dataset
from pyproj import Proj, transform

class S3_L2_LST():

    # file_path  --- путь до архива
    # extent     --- словарь формата {'minX': ..., 'minY': ...,'maxX': ..., 'maxY': ...}, где указаны координаты в WGS
    # resolution --- словарь формата {'xRes': 1000, 'yRes': 1000}, пространственное разрешение, единицы измерения, разрешение по X, по Y
    # key_values --- словарь формата {'gap': -100.0, 'skip': -200.0,'noData': -32768.0}, обозначающий пропущенные пиксели
    # При инициализации формируется словарь с метаданными self.metadata
    def __init__(self, file_path, extent, resolution, key_values = {'gap': -100.0, 'skip': -200.0,'NoData': -32768.0}):
        self.file_path = file_path
        self.extent = extent
        self.resolution = resolution
        self.key_values = key_values

        # Разбираемся с директориями, что и где будем размещать
        main_path = os.path.split(self.file_path)[0]
        self.temporary_path = os.path.join(main_path, 'temporary') # Временная директория, в которую будут складываться все файлы

        # Подбор наиболее подходящей метрической проекции
        self.utm_code, self.utm_extent = self.__get_utm_code_from_extent()
        # Записываем в переменные сведения о спутнике и дате съемки
        archive_name = os.path.basename(self.file_path)
        self.datetime = archive_name[16:31]
        self.satellite = archive_name[0:3]

        # Формируем словарь с метаданными
        self.metadata = {}
        self.metadata.update({'file_name': archive_name,
                              'satellite': self.satellite,
                              'datetime': self.datetime,
                              'extent': self.extent,
                              'utm_code': self.utm_code,
                              'utm_extent': self.utm_extent,
                              'resolution': self.resolution,})

    # Приватный метод для подбора наиболее подходящей метрической проекции, вызывается при инициализации класса
    # return utm_code   --- код UTM проекции
    # return utm_extent --- словарь формата {'minX': ..., 'minY': ...,'maxX': ..., 'maxY': ...}, где указаны координаты в UTM
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
        utm_extent = {'minX': min_corner[0], 'minY': min_corner[1],'maxX': max_corner[0], 'maxY': max_corner[1]}
        return(utm_code, utm_extent)

    # Приватный метод для формирования необходимых файлов для пространственной привязки NetCDF матриц
    # return warpOptions  --- список опций для формирования привязанного растра
    # return imageVRTPath --- путь до сгенерированного растра
    def __preparation(self):
        # Определение временной директории - если её не существует (что скорее всего так), то создаем её
        if os.path.isdir(self.temporary_path) == False:
            os.mkdir(self.temporary_path)

        archive = zipfile.ZipFile(self.file_path, 'r')  # Открываем архив
        arch_files = archive.namelist()  # Какие есть папки/файлы в архиве

        # Обращаемся к файлам NetCDF в архиве, предварительно извлекая из архивов файлы в temporary_path
        for file in arch_files:
            if file.endswith("geodetic_in.nc"):
                geodetic_in_nc = file
                geodetic_in = archive.extract(geodetic_in_nc, path = self.temporary_path)
            elif file.endswith("LST_in.nc"):
                LST_in_nc = file
                LST_in = archive.extract(LST_in_nc, path = self.temporary_path)
            elif file.endswith("flags_in.nc"):
                flags_in_nc = file
                flags_in = archive.extract(flags_in_nc, path = self.temporary_path)
            elif file.endswith("LST_ancillary_ds.nc"):
                LST_ancillary_ds_nc = file
                LST_ancillary_ds = archive.extract(LST_ancillary_ds_nc, path = self.temporary_path)

        flags_in = Dataset(flags_in)
        clouds = np.array(flags_in.variables['cloud_in'])                                         # Матрица с облаками

        geodetic_in = Dataset(geodetic_in)
        el = np.array(geodetic_in.variables['elevation_in'])                                      # Матрица высот
        lat = np.array(geodetic_in.variables['latitude_in'])                                      # Матрица широт
        long = np.array(geodetic_in.variables['longitude_in'])                                    # Матрица долгот

        LST_in = Dataset(LST_in)
        LST_matrix = np.array(LST_in.variables['LST'])                                            # Матрица LST

        LST_ancillary_ds = Dataset(LST_ancillary_ds)
        biome = np.array(LST_ancillary_ds.variables['biome'])                                     # Матрица биомов

        # ВНИМАНИЕ! Важен порядок присвоения флагов, сначала облакам - потом, все остальное
        # Иначе мы будем заполнять пиксель от облаков, в котором значение -inf потому что это море
        # Помечаем все пиксели с облаками на нашей матрице значениями - "gap"
        LST_matrix[clouds > 0] = self.key_values.get('gap')
        # Помечаем все пиксели занятые морской водой в нашей матрице значениями - "skip"
        LST_matrix[biome == 0] = self.key_values.get('skip')

        # дружно обратим последовательность строк во всех массивах
        div = np.ma.array(LST_matrix)
        div = np.flip(div, axis = 0)
        lats = np.flip(lat, axis = 0)
        lons = np.flip(long, axis = 0)

        # Список из строк, которые больше стольки-то градусов и меньше стольки-то градусов по широте, берем с запасом
        Higher_border = self.extent.get('maxY') + 10
        Lower_border = self.extent.get('minY') - 10

        wrong_raws_1 = np.unique(np.argwhere(lats > Higher_border)[:, 0])
        wrong_raws_2 = np.unique(np.argwhere(lats < Lower_border)[:, 0])
        # Объединяем списки индексов строк, которые необходимо убрать
        wrong_raws = np.hstack((wrong_raws_1, wrong_raws_2))

        div = np.delete(div, (wrong_raws), axis = 0)
        lats = np.delete(lats, (wrong_raws), axis = 0)
        lons = np.delete(lons, (wrong_raws), axis = 0)

        # выставим настройки типа данных и типа используемого драйвера, а также всех путей:
        dataType = gdal.GDT_Float64
        driver = gdal.GetDriverByName("GTiff")
        latPath = os.path.join(self.temporary_path, 'lat.tif')
        lonPath = os.path.join(self.temporary_path, 'lon.tif')
        imagePath = os.path.join(self.temporary_path, 'image.tif')
        imageVRTPath = os.path.join(self.temporary_path, 'image.vrt')

        # Создаем растр для широт (..\TEMP\lat.tif):
        dataset = driver.Create(latPath, div.shape[1], div.shape[0], 1, dataType)
        dataset.GetRasterBand(1).WriteArray(lats)

        # Создаем растр для долгот (..\TEMP\lon.tif):
        dataset = driver.Create(lonPath, div.shape[1], div.shape[0], 1, dataType)
        dataset.GetRasterBand(1).WriteArray(lons)

        # Создаем растр для данных (..\TEMP\image.tif)
        dataset = driver.Create(imagePath, div.shape[1], div.shape[0], 1, dataType)
        dataset.GetRasterBand(1).WriteArray(div)

        # Установим СК WGS84
        gcp_srs = osr.SpatialReference()
        gcp_srs.ImportFromEPSG(4326)
        proj4 = gcp_srs.ExportToProj4()

        # На основе tif-а создадим vrt (..\TEMP\image.vrt)
        vrt = gdal.BuildVRT(imageVRTPath, dataset, separate = True, resampleAlg = 'cubic', outputSRS = proj4)
        band = vrt.GetRasterBand(1)

        # Привяжем координаты к виртуальному растру...
        metadataGeoloc = {
            'X_DATASET': lonPath,
            'X_BAND': '1',
            'Y_DATASET': latPath,
            'Y_BAND': '1',
            'PIXEL_OFFSET': '0',
            'LINE_OFFSET': '0',
            'PIXEL_STEP': '1',
            'LINE_STEP': '1'
        }

        # ...записав это все в <Metadata domain='Geolocation'>:
        vrt.SetMetadata(metadataGeoloc, "GEOLOCATION")

        dataset = None
        vrt = None

        output_rs = osr.SpatialReference()
        output_rs.ImportFromEPSG(self.utm_code)

        warpOptions = gdal.WarpOptions(geoloc = True, format = 'GTiff', dstNodata = self.key_values.get('NoData'), srcSRS = proj4, dstSRS = output_rs,
                                       outputBounds = [self.utm_extent.get('minX'), self.utm_extent.get('minY'), self.utm_extent.get('maxX'), self.utm_extent.get('maxY')],
                                       xRes = self.resolution.get('xRes'), yRes = self.resolution.get('yRes'), creationOptions = ['COMPRESS=LZW'])

        # Закрываем разаархивированные NetCDF файлы
        archive.close()
        geodetic_in.close()
        LST_in.close()
        flags_in.close()
        LST_ancillary_ds.close()
        return(warpOptions, imageVRTPath)
    pass

    # Метод для формирования файла .geotiff в нужной директории
    # save_path --- место, в которое нужно поместить файл с результатом
    def archive_to_geotiff(self, save_path):
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        warpOptions, imageVRTPath = self.__preparation()

        geotiff_name = self.datetime + '.tif'
        geotiff_path = os.path.join(save_path, geotiff_name)
        raster = gdal.Warp(geotiff_path, imageVRTPath, dstNodata = self.key_values.get('NoData'), options = warpOptions)

        # Удаляем временную директорию
        shutil.rmtree(self.temporary_path, ignore_errors = True)

    # Метод для формирования файла .npy в нужной директории
    # save_path --- место, в которое нужно поместить файл с результатом
    def archive_to_npy(self, save_path):
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        warpOptions, imageVRTPath = self.__preparation()

        geotiff_name = self.datetime + '.tif'
        geotiff_path = os.path.join(self.temporary_path, geotiff_name)
        raster = gdal.Warp(geotiff_path, imageVRTPath, dstNodata = self.key_values.get('NoData'), options = warpOptions)

        # Сохраняем матрицу в формате .npy
        npy_name = self.datetime + '.npy'
        npy_path = os.path.join(save_path, npy_name)
        matrix = raster.ReadAsArray()
        matrix = np.array(matrix)
        np.save(npy_path, matrix)

        raster = None
        # Удаляем временную директорию
        shutil.rmtree(self.temporary_path, ignore_errors = True)