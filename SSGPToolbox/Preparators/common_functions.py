import gdal, json, os, osr
import numpy as np
import scipy.spatial
import random

def reconstruct_geotiff (npy_path, metadata_path, output_path):
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        
    npy_data = np.load(npy_path)
    extent = metadata['utm_extent']
    resolution = metadata['resolution']
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create( output_path, npy_data.shape[1], npy_data.shape[0], 1, gdal.GDT_Float32 )
        
    geotransform = [extent['minX'],resolution['xRes'],0,extent['maxY'],0,-1*resolution['yRes']]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(metadata['utm_code']))
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform(geotransform)
        
    ds.GetRasterBand(1).WriteArray(npy_data)
    del ds

# "Расширение" территории облака с помощью вероятностного клеточного автомата
# Задача алгоритма - идентификация затененных облаком пикселей на снимке
def cellular_expand (matrix, biome_matrix, gap = -100.0, iter = 10):

    #############################
    #     Requires revision     #
    #############################

    # Функция, которая определяет все преобразования, которые должны произойти с матрицей
    def step(matrix, biome_matrix, gap = gap):
        # Матрице с биомами мы присваиваем значение пропуска в тех местах, где есть облака в данный момент
        biome_matrix[matrix == gap] = gap

        # Копия матрицы, в которую будут добавляться пиксели
        next_matrix = np.copy(matrix)

        # Рассчет порога
        masked_array = np.ma.masked_where(matrix == gap, matrix)
        minimum = np.min(masked_array)
        maximum = np.max(masked_array)
        # Амплитуда температур на снимке - это потребуется вдальнейшем для нормализации
        amplitide = maximum - minimum

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Если пиксель расположен в верхней строке
                if i == 0:
                    # Крайняя левая ячейка
                    if j == 0:
                        # Урезанная окрестность Мура
                        arr = matrix[i: i + 2, j: j + 2]
                    # Крайняя правая ячейка
                    elif j == (matrix.shape[1] - 1):
                        arr = matrix[i: i + 2, j - 1: j + 1]
                    # Весь остальной ряд
                    else:
                        arr = matrix[i: i + 2, j - 1: j + 2]

                # Если пиксель расположен в левом столбце
                elif j == 0:
                    # Крайняя нижняя ячейка
                    if i == (matrix.shape[0] - 1):
                        arr = matrix[i - 1: i + 1, j: j + 2]
                    # Весь остальной ряд
                    else:
                        arr = matrix[i - 1: i + 2, j: j + 2]

                # Если пиксель расположен в нижней строке
                elif i == (matrix.shape[0] - 1):
                    # Крайняя правая ячейка
                    if j == (matrix.shape[1] - 1):
                        arr = matrix[i - 1: i + 1, j - 1: j + 1]
                    # Весь остальной ряд
                    else:
                        arr = matrix[i - 1: i + 1, j - 1: j + 2]

                # Если пиксель расположен в правом столбце
                elif j == (matrix.shape[1] - 1):
                    arr = matrix[i - 1: i + 2, j - 1: j + 1]

                # Если пиксель закрыт облаком
                elif matrix[i, j] == gap:
                    arr = np.zeros((2, 2))
                else:
                    # Окрестность Мура для точки
                    arr = matrix[i - 1: i + 2, j - 1: j + 2]

                # Проверка, есть ли в окрестности облако
                id_cloud = np.argwhere(arr == gap)
                # Если есть и при этом сам пиксель не закрыт облаком
                # то сравниваем температуру пикселя со средней температурой окрестности
                if len(id_cloud) != 0 and matrix[i, j] != gap:

                    #####################################
                    # Используется вероятностный подход #
                    #####################################

                    # Генерируем случайное число в диапазоне от 0 до 1
                    prob_number = random.random()

                    # Выбираем число, такое, что если prob_number превысит его, то пиксель станет облачным на следующем шаге
                    if len(id_cloud) >= 8:
                        # Чем ближе значение fact_number к 0, тем больше вероятность того, что пиксель станет "облаком"
                        fact_number = 0.8
                    elif len(id_cloud) == 7:
                        fact_number = 0.85
                    elif len(id_cloud) == 6:
                        fact_number = 0.9
                    elif len(id_cloud) == 5:
                        fact_number = 0.95
                    else:
                        fact_number = 0.99

                    # К изначально определенному значению вероятности добавляем некоторую величину
                    # в зависимости от температуры рассматриваемого пикселя

                    # Код биома (группы пикселей) для ячейки
                    biome_code = biome_matrix[i, j]
                    # Индексы точек, которые попадают в данный биом и при этом в данный момент не являются пропусками
                    coords = np.argwhere(biome_matrix == biome_code)
                    # Если точек в биоме недостаточно, то берем все известные точки
                    if len(coords) < 41:
                        coords = np.argwhere(matrix != gap)
                    else:
                        pass

                    # Координаты пикселя, от которого мы рассчитываем расстояния
                    target_pixel = np.array([[i, j]])
                    # Вектор из рассчитаных расстояний от целевого пикселя до всех остальных
                    distances = scipy.spatial.distance.cdist(target_pixel, coords)[0]

                    # Выбираем ближайшие 40 пикселей
                    new_coords = []
                    for iter in range(0, 40):
                        # Какой индекс в массиве coords имеет элемент с наименьшим расстоянием от целевого пикселя
                        index_min_dist = np.argmin(distances)

                        # Координата данного пикселя в матрице
                        new_coord = coords[index_min_dist]

                        # Добавляем координаты
                        new_coords.append(new_coord)

                        # Заменяем минимальный элемент из массива на очень большое число
                        distances[index_min_dist] = np.inf

                    # Записываем температуру этих ближайших 15 пикселей
                    list_tmp = []
                    for coord in new_coords:
                        list_tmp.append(matrix[coord[0], coord[1]])
                    list_tmp = np.array(list_tmp)

                    # Берем медианное значение
                    median_local_tmp = np.median(list_tmp)

                    # Нормализованная разница температур (к амплитуде)
                    value = ((matrix[i, j] - median_local_tmp) / amplitide)
                    # Если температура выше, чем медианная по биому, то ничего не делаем
                    if value >= 0:
                        pass
                    # Если количество соседних облачных пикселей строго меньше 3, то тоже не рассматриваем данный пиксель
                    elif len(id_cloud) < 3:
                        pass
                    # Чем меньше температура, тем выше вероятность того, что данный пиксель станет "облаком" на следующем шаге
                    else:
                        fact_number = fact_number + value

                        # Если сгенерированное число больше fact_number, то пиксель станет облачным
                        if prob_number >= fact_number:
                            next_matrix[i, j] = gap

        # Возвращаем матрицу, которая будет получена на следующем шаге
        return (next_matrix)

    # Заданное количество итераций применяется процедура "расширения облака"
    for iteration in range(0, iter):
        matrix = step(matrix, biome_matrix)
    return(matrix)