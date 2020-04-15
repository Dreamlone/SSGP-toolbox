'''

Discretizator --- a class intended for creating time series and filling in gaps in it

Private methods:
__sampling    --- this method allows you to bring data to the specified time step
__gap_process --- method for filling in missing values in a time series

Public methods:
make_time_series --- a method for placing matrices on a regular time grid and filling in gaps in time series
save_npy         --- a method for storing the results obtained as .npy matrices
save_netcdf      --- method for saving the results as a netCDF file

'''

import os
import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset

class Discretizator():

    # При инициализации класса необходмо указать
    # directory - директория, в которой расположены слои, которые требуется свести в временной ряд
    # key_values - словарь с обозначениями пропусков и нерелевантных значений
    # averaging - нужно ли осреднять слои, которые попадают в один временной интервал ('None', 'weighted', 'simple')
    def __init__(self, directory, key_values = {'gap': -100.0, 'skip': -200.0}, averaging = 'None'):
        self.directory = directory
        self.averaging = averaging

        # Все skip значения в матрицах будут восприниматься как те, которые заполнять не требуется
        self.skip = key_values.get('skip')
        # Если пропуски во временном ряду не будут заполняться, то пропуски будут заполнены значением gap
        self.gap = key_values.get('gap')

        # Составим отсортированный список всех файлов, которые находятся в папке
        layers = os.listdir(self.directory)
        layers.sort()

        # Формируем словарь с матрицами и список из ключей
        matrices_dictionary = {}
        keys = []
        for layer in layers:
            key = datetime.datetime.strptime(layer[:-4], '%Y%m%dT%H%M%S')
            keys.append(key)
            matrix_path = os.path.join(self.directory, layer)

            matrix = np.load(matrix_path)
            matrices_dictionary.update({key: matrix})

        # Упорядоченный список из названий файлов
        self.keys = keys
        # Словарь с матрицами
        self.matrices_dictionary = matrices_dictionary

    # Приватный метод позволяет привести данные к указанному временному шагу
    # timestep --- временной интервал, через который слои будут размещаться на сетке временного ряда
    # return tensor --- многомерная матрица, в которой каждый слой занимает свое место на временном ряду
    def __sampling(self, timestep):
        example_matrix = self.matrices_dictionary.get(self.keys[0])
        rows = example_matrix.shape[0]
        cols = example_matrix.shape[1]
        print('The time series will be composed with frequency -', timestep)

        start_date = str(self.keys[0])
        start_date = start_date[:10]
        print('Start date -', start_date)
        end_date = str(self.keys[-1] + datetime.timedelta(days = 1))
        end_date = end_date[:10]
        print('Final date -', end_date)

        # Синтез временного ряда с регулярными отступами
        times = pd.date_range(start = start_date, end = end_date, freq = timestep)
        # Вообще формируем мы не тензор, а многомерную матрицу, но для названия переменной вполне себе
        tensor = []
        tensor_timesteps = []
        for i in range(0, len(times) - 1):
            # Берем середину из данного временного промежутка
            time_interval = (times[i + 1] - times[i])/2
            centroid = times[i] + time_interval

            # Рассматриваем какие слои могут подходить для данного интервала
            suitable_keys = []
            for key in self.keys:
                if key >= times[i] and key < times[i + 1]:
                    suitable_keys.append(key)

            if len(suitable_keys) == 0:
                # Сначала проверка: если мы не смогли для последнего отрезка найти слой - значит и генерировать его не следует
                if i == len(times) - 2:
                    break
                else:
                    # В данном случае мы сами будем генерировать слои, пока заполняем "болванку" значением gap
                    matrix = np.full((rows, cols), self.gap)
            elif len(suitable_keys) == 1:
                # Если слой всего один, то добавляется именно он
                main_key = suitable_keys[0]
                matrix = self.matrices_dictionary.get(main_key)
            else:
                # Если осреднение не требуется, то выбирается тот слой, который ближе всего расположен к временному интервалу
                if self.averaging == 'None':
                    # Выбор одного слоя, расположенного ближе всего к инересующему нас интервалу
                    distances = []
                    for element in suitable_keys:
                        if element < centroid:
                            distances.append(centroid - element)
                        elif element > centroid:
                            distances.append(element - centroid)
                        else:
                            distances.append(0)
                    distances = np.array(distances)
                    # Ищем индекс наименьшего элемента
                    min_index = np.argmin(distances)
                    # По индексу получаем подходящий слой
                    main_key = suitable_keys[min_index]
                    matrix = self.matrices_dictionary.get(main_key)

                # Если выбрана процедура осреднения с параметром 'simple', то будет произведено простое усреднение
                elif self.averaging == 'simple':
                    # Создаем небольшую матрицу для данного временного интервала
                    matrix_batch = []
                    for element in suitable_keys:
                        step_matrix = self.matrices_dictionary.get(element)
                        matrix_batch.append(step_matrix)
                    matrix_batch = np.array(matrix_batch)

                    # Создаем "болванку" - матрицу заполненную нулями
                    matrix = np.zeros((rows, cols))
                    # Для кадого пикселя производим процедуру осреднения
                    for row_index in range(0, matrix_batch[0].shape[0]):
                        for col_index in range(0, matrix_batch[0].shape[1]):
                            mean_value = np.mean(matrix_batch[:, row_index, col_index])
                            # Записываем значение
                            matrix[row_index, col_index] = mean_value

                # Если выбрана процедура усреднения с параметром "weighted", то все слои, которые попали во временной интервал усредняются с весами
                elif self.averaging == 'weighted':

                    # Создаем небольшую матрицу для данного временного интервала
                    matrix_batch = []
                    for element in suitable_keys:
                        step_matrix = self.matrices_dictionary.get(element)
                        matrix_batch.append(step_matrix)
                    matrix_batch = np.array(matrix_batch)

                    # Необходимо определить насколько слои "близки" к временной метке
                    distances = []
                    for element in suitable_keys:
                        if element < centroid:
                            distances.append(centroid - element)
                        elif element > centroid:
                            distances.append(element - centroid)
                        else:
                            distances.append(0)
                    distances = np.array(distances)

                    # Воспользуемся функцией, которая вернет массив из индексов элементов, если отсортировать массив distances
                    distances_id_sorted = np.argsort(distances)
                    # Теперь, когда мы знаем какое место занимает по величине в массиве каждое расстояние, зададим веса
                    weights = np.copy(distances)
                    weight = len(distances)
                    # Присваиваются веса каждому элементу в зависимости от расстояния (чем ближе расположен элемент, тем больше вес)
                    for index in distances_id_sorted:
                        weights[index] = weight
                        weight -= 1

                    # Создаем "болванку" - матрицу заполненную нулями
                    matrix = np.zeros((rows, cols))
                    # Для кадого пикселя производим процедуру осреднения
                    for row_index in range(0, matrix_batch[0].shape[0]):
                        for col_index in range(0, matrix_batch[0].shape[1]):
                            mean_value = np.average(matrix_batch[:, row_index, col_index], weights = weights)
                            # Записываем значение
                            matrix[row_index, col_index] = mean_value

            # Добавляем матрицу
            tensor.append(matrix)
            tensor_timesteps.append(centroid)
        tensor = np.array(tensor)
        return(tensor, tensor_timesteps)

    # Приватный метод для заполнения пропущенных значений во временном ряду
    def __gap_process(self, timeseries, filling_method, n_neighbors = 5):
        # Индексы точек на временном ряду, которые необходимо заполнить
        i_gaps = np.argwhere(timeseries == self.gap)
        i_gaps = np.ravel(i_gaps)

        # В зависимости от выбранной стратегии заполнения пропусков во временных рядах
        # Пропуски во временном ряду не заполняются
        if filling_method == 'None':
            pass
        elif filling_method == None:
            pass
        # Пропуски заполняются локальными медианами
        elif filling_method == 'median':
            # Для каждого пропуска во временном ряду находим n_neighbors "известных соседей"
            for gap_index in i_gaps:
                # Индексы известных элементов (обновляются на каждой итерации)
                i_known = np.argwhere(timeseries != self.gap)
                i_known = np.ravel(i_known)

                # На основе индексов рассчитываем насколько далеко от пропуска расположены известные значения
                id_distances = np.abs(i_known - gap_index)

                # Теперь узнаем индексы наименьших значений в массиве, для этого сортируем для индексов
                sorted_idx = np.argsort(id_distances)
                # n_neighbors ближайших известных значений к пропуску
                nearest_values = []
                for i in sorted_idx[:n_neighbors]:
                    # Получаем значение индекса для ряда - timeseries
                    time_index = i_known[i]
                    # По этому индексу получаем значение каждого из "соседей"
                    nearest_values.append(timeseries[time_index])
                nearest_values = np.array(nearest_values)

                est_value = np.nanmedian(nearest_values)
                timeseries[gap_index] = est_value
        elif filling_method == 'poly':
            # Для каждого пропуска строим свой полином невысокой степени
            for gap_index in i_gaps:
                # Индексы известных элементов (обновляются на каждой итерации)
                i_known = np.argwhere(timeseries != self.gap)
                i_known = np.ravel(i_known)

                # На основе индексов рассчитываем насколько далеко от пропуска расположены известные значения
                id_distances = np.abs(i_known - gap_index)

                # Теперь узнаем индексы наименьших значений в массиве, для этого сортируем для индексов
                sorted_idx = np.argsort(id_distances)
                # Ближайшие известные значения к пропуску
                nearest_values = []
                # И их индексы
                nearest_indices = []
                for i in sorted_idx[:n_neighbors]:
                    # Получаем значение индекса для ряда - timeseries
                    time_index = i_known[i]
                    # По этому индексу получаем значение каждого из "соседей"
                    nearest_values.append(timeseries[time_index])
                    nearest_indices.append(time_index)
                nearest_values = np.array(nearest_values)
                nearest_indices = np.array(nearest_indices)

                # Локальная аппроксимация полиномом n-й степени
                local_coefs = np.polyfit(nearest_indices, nearest_values, 2)

                # В соответствии с подобранными коэффициентами оцениваем нашу точку
                est_value = np.polyval(local_coefs, gap_index)
                timeseries[gap_index] = est_value

        return(timeseries)

    # timestep --- временной интервал, через который слои будут размещаться на сетке временного ряда
    def make_time_series(self, timestep = '12H', filling_method = 'None'):
        # Из указанной директории и с помощью выбранного временного шага формируем многомерную матрицу и временные шаги
        tensor, tensor_timesteps = self.__sampling(timestep = timestep)

        # Для кадого пикселя в ряду строим свою модель
        for row_index in range(0, tensor[0].shape[0]):
            for col_index in range(0, tensor[0].shape[1]):
                # Получаем временной ряд для конкретного пикселя
                pixel_timeseries = tensor[:, row_index, col_index]

                # Если во временном ряду есть значение skip, то оно будет записано во все ячейки
                if any(value == self.skip for value in pixel_timeseries):
                    pixel_timeseries = np.full(len(pixel_timeseries), self.skip)

                # Если есть хотя бы один пропуск во временном ряду, то его необходимо заполнить
                elif any(value == self.gap for value in pixel_timeseries):
                    # С помощью приватного метода __gap_process заполняем пропуск во временном ряду
                    pixel_timeseries = self.__gap_process(timeseries = pixel_timeseries, filling_method = filling_method)

                # Если пропусков в ряду нет, то и заполнять ничего не нужно
                else:
                    pass

                # Временной ряд заполнен, поэтому записываем заполненный временной ряд в многомерную матрицу
                tensor[:, row_index, col_index] = pixel_timeseries

        return(tensor, tensor_timesteps)


    # Метод, позволяющий сохранять полученные результаты в виде матриц npy
    # save_path --- папка, в которую требуется сохранить результат
    def save_npy(self, tensor, tensor_timesteps, save_path):
        # Создаем папку 'Discretisation_output'; если есть, то используем существующую
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

        # Теперь в данную папку будут сохраняться заполненные матрицы
        for index in range(0, len(tensor)):
            matrix = tensor[index]
            time = tensor_timesteps[index]
            # Переводим формат datetime в строку
            time = time.strftime('%Y%m%dT%H%M%S')

            npy_name = os.path.join(save_path, time)
            np.save(npy_name, matrix)


    # Метод, позволяющий сохранять полученные результаты в виде netCDF файла
    # save_path --- папка, в которую требуется сохранить результат
    def save_netcdf(self, tensor, tensor_timesteps, save_path):

        # Создаем папку 'Discretisation_output'; если есть, то используем существующую
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

        netCDF_name = os.path.join(save_path, 'Result.nc')

        # Переводим временные метки в строковый тип данных
        str_tensor_timesteps = []
        for time in tensor_timesteps:
            # Переводим формат datetime в строку
            str_tensor_timesteps.append(time.strftime('%Y%m%dT%H%M%S'))
        str_tensor_timesteps = np.array(str_tensor_timesteps)

        # Формирует netCDF файл
        root_grp = Dataset(netCDF_name, 'w', format='NETCDF4')
        root_grp.description = 'Discretized matrices'

        # Размерности для данных, которые будем записывать в файл
        dim_tensor = tensor.shape
        root_grp.createDimension('time', len(str_tensor_timesteps))
        root_grp.createDimension('row', dim_tensor[1])
        root_grp.createDimension('col', dim_tensor[2])

        # Записываем данные в файл
        time = root_grp.createVariable('time', 'S2', ('time',))
        data = root_grp.createVariable('matrices', 'f4', ('time', 'row', 'col'))

        data[:] = tensor
        time[:] = str_tensor_timesteps

        root_grp.close()