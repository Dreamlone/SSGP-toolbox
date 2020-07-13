'''

class SimpleSpatialGapfiller --- a class that allows you to fill in gaps in matrices based on machine learning method

Private methods:
__make_training_sample --- creating a training sample from matrices in the "History" folder
__learning_and_fill    --- filling in gaps for the matrix, writing the result to the "Outputs" folder

Public methods:
fill_gaps        --- using the __learning_and_fill method for each of the matrices in the "Inputs" folder, creating a file with metadata about the quality of the algorithm
nn_interpolation --- using the nearest neighbor interpolation for each of the matrices in the "Inputs" folder
'''

import os
import random
import timeit
import json

import numpy as np
import pandas as pd
import scipy.spatial

# scikit-learn version is 0.21.3.
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing
from scipy import interpolate

class SimpleSpatialGapfiller():

    # При инициализации класса необходмо указать
    # directory - расположение папок проекта: "History", "Inputs", и "Extra"
    def __init__(self, directory):
        # Пороговое значение для невключения слоев в обучающую выборку при превышении (изменяется от 0.0 до 1.0)
        self.main_threshold = 0.05
        self.directory = directory

        # Создаем папку 'Outputs'; если есть, то используем существующую
        self.Outputs_path = os.path.join(self.directory, 'Outputs')
        if os.path.isdir(self.Outputs_path) == False:
            os.makedirs(self.Outputs_path)

        # Доступ ко всем оставшимся папкам в проекте
        self.Inputs_path = os.path.join(self.directory, 'Inputs')
        self.Extra_path = os.path.join(self.directory, 'Extra')
        self.History_path = os.path.join(self.directory, 'History')

        # Создаем словарь с данными, который будет заполняться по мере работы алгоритма
        self.metadata = {}

    # Приватный метод формирует обучающую выборку из матриц в папке "History"
    # return dictionary --- словарь, в котором ключу (название файла) соответсвует матрица
    # return keys       --- отсортированный список ключей, где последний ключ - целевая матрица
    def __make_training_sample(self):
        # Формируем словарь с матрицами и список с ключами, в котором они хранятся в строгом порядке
        history_files = os.listdir(self.History_path)
        dictionary = {}
        keys = []
        for file in history_files:
            key_path = os.path.join(self.History_path, file) # Путь до конкрентной матрицы в обучающей выборке
            key = file[:-4] # Ключ для матрицы
            matrix = np.load(key_path)

            # Если есть более main_threshold доли пикселей, которые не были сняты в этот момент времени, то матрица не включается в анализ
            amount_na = (matrix == self.nodata).sum()
            shape = matrix.shape
            threshold = amount_na/(shape[0]*shape[1])
            if threshold > self.main_threshold:
                pass
            else:
                keys.append(key)
                dictionary.update({key: matrix})

        # Отсортируем список с ключами
        keys.sort()
        return(dictionary, keys)

    # Приватный метод - подготавливает датасет, обучает модель, записывает результат (в виде файла .npy) в указанную папку
    # dictionary - словарь, ключ - время съемки, значение - матрица; все слои, кроме последнего - нужны для обучения
    # keys - список ключей, где последний ключ относится к матрице, в которой необходимо заполнить пропуски
    # method - название алгоритма (Lasso, RandomForest, ExtraTrees, Knn, SVR)
    # predictor_configuration - подбор предикторов (All, Random, Biome)
    # hyperparameters - выбор гиперпараметров (RandomGridSearch, GridSearch, Custom)
    # params - если выбран аргумент "Custom", то параметры модели передаются через аргумент params
    # add_outputs - будут ли добавляться заполненные алгоритмом слои в обучающую выборку
    def __learning_and_fill(self, dictionary, keys, extra_matrix, method, predictor_configuration, hyperparameters, params, add_outputs):
        # Задаем все необходимые функции
        # Лассо регрессия
        def Lasso_regression(X_train, y_train, X_test, params):
            # Поиск по сетке для Lasso в виду малого количества гиперпараметров 'GridSearch' и 'RandomGridSearch' - одно и то же
            if hyperparameters == 'RandomGridSearch' or hyperparameters == 'GridSearch':
                # Осуществляем поиск по сетке с кросс-валидацией (число фолдов равно 3)
                alphas = np.arange(1, 800, 50)
                param_grid = {'alpha': alphas}
                # Задаем модель, которую будем обучать
                estimator = Lasso()
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = GridSearchCV(estimator, param_grid, iid = 'deprecated', cv = 3, scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = Lasso()
                # Задаем нужные параметры
                estimator.set_params(**params)

                # Проверка по кросс-валидации
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')

                # Обучаем модель уже на всех данных
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # Случайный лес
        def Random_forest_regression(X_train, y_train, X_test, params):
            # Случайный поиск по сетке
            if hyperparameters == 'RandomGridSearch':
                # Осуществляем поиск по сетке с кросс-валидацией (число фолдов равно 3)
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_leaf_nodes': max_leaf_nodes}
                # Задаем модель, которую будем обучать
                estimator = RandomForestRegressor(n_estimators = 50, n_jobs = -1)
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Полный поиск по сетке
            elif hyperparameters == 'GridSearch':
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split,
                              'max_leaf_nodes': max_leaf_nodes}
                # Задаем модель, которую будем обучать
                estimator = RandomForestRegressor(n_estimators = 50, n_jobs = -1)
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = RandomForestRegressor()
                # Задаем нужные параметры
                estimator.set_params(**params)

                # Проверка по кросс-валидации
                fold = KFold(n_splits = 3, shuffle=True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')

                # Обучаем модель уже на всех данных
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # Сверхслучайные леса
        def Extra_trees_regression(X_train, y_train, X_test, params):
            # Случайный поиск по сетке
            if hyperparameters == 'RandomGridSearch':
                # Осуществляем поиск по сетке с кросс-валидацией (число фолдов равно 3)
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_leaf_nodes': max_leaf_nodes}
                # Задаем модель, которую будем обучать
                estimator = ExtraTreesRegressor(n_estimators = 50, n_jobs = -1)
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Полный поиск по сетке
            elif hyperparameters == 'GridSearch':
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split,'max_leaf_nodes': max_leaf_nodes}
                # Задаем модель, которую будем обучать
                estimator = ExtraTreesRegressor(n_estimators = 50, n_jobs = -1)
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = ExtraTreesRegressor()
                # Задаем нужные параметры
                estimator.set_params(**params)

                # Проверка по кросс-валидации
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')

                # Обучаем модель уже на всех данных
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # К-ближайших соседей
        def KNN_regression(X_train, y_train, X_test, params):
            # Случайный поиск по сетке
            if hyperparameters == 'RandomGridSearch':
                # Осуществляем поиск по сетке с кросс-валидацией (число фолдов равно 3)
                weights = ['uniform', 'distance']
                algorithm = ['auto', 'kd_tree']
                n_neighbors = [2, 5,10,15,20]
                param_grid = {'weights': weights, 'n_neighbors': n_neighbors, 'algorithm': algorithm}
                # Задаем модель, которую будем обучать
                estimator = KNeighborsRegressor()
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Полный поиск по сетке
            elif hyperparameters == 'GridSearch':
                weights = ['uniform', 'distance']
                algorithm = ['auto', 'kd_tree']
                n_neighbors = [2, 5, 10, 15, 20]
                param_grid = {'weights': weights, 'n_neighbors': n_neighbors, 'algorithm': algorithm}
                # Задаем модель, которую будем обучать
                estimator = KNeighborsRegressor()
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid='deprecated', scoring='neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = KNeighborsRegressor()
                # Задаем нужные параметры
                estimator.set_params(**params)

                # Проверка по кросс-валидации
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')

                # Обучаем модель уже на всех данных
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # Метод опорных векторов
        def SVM_regression(X_train, y_train, X_test, params):
            # Соединим нашу выборку для процедуры стандартизации
            sample = np.vstack((X_train, X_test))

            # Стандартизуем выборку и снова разделяем
            sample = preprocessing.scale(sample)
            X_train = sample[:-1, :]
            X_test = sample[-1:, :]

            # Случайный поиск по сетке
            if hyperparameters == 'RandomGridSearch':
                # Осуществляем поиск по сетке с кросс-валидацией (число фолдов равно 3)
                Cs = [0.001, 0.01, 0.1, 1, 10]
                epsilons = [0.1, 0.4, 0.7, 1.0]
                param_grid = {'C': Cs, 'epsilon': epsilons}
                # Задаем модель, которую будем обучать
                estimator = SVR(kernel = 'linear', gamma = 'scale')
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Полный поиск по сетке
            elif hyperparameters == 'GridSearch':
                Cs = [0.001, 0.01, 0.1, 1, 10]
                epsilons = [0.1, 0.4, 0.7, 1.0]
                param_grid = {'C': Cs, 'epsilon': epsilons}
                # Задаем модель, которую будем обучать
                estimator = SVR(kernel = 'linear', gamma = 'scale')
                # Производим обучение модели с заданными вариантами параметров (осуществляем поиск по сетке)
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = SVR()
                # Задаем нужные параметры
                estimator.set_params(**params)

                # Проверка по кросс-валидации
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = np.ravel(y_train), cv = fold, scoring = 'neg_mean_absolute_error')

                # Обучаем модель уже на всех данных
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)

            return(predicted, validation_score)

        def all_points(coord_row, coord_column, final_matrix):
            # Индексы всех точек, которые не закрыты облаками (в том числе пиксели со значением skip, nodata)
            coords = np.argwhere(final_matrix != self.gap)
            coords = list(coords)
            coords.append([coord_row, coord_column])
            coords = np.array(coords)

            # Для каждой матрицы из словаря по известным координатам (индексам) снимаем значения параметра и записываем в датасет
            # Последняя строка в датасете - значения для матрицы, которую необходимо загэпфилить :)
            dataframe = []
            for key in keys:
                matrix = dictionary.get(key)
                values = []
                for i, j in coords:
                    values.append(matrix[i, j])
                values = np.array(values)
                dataframe.append(values)
            dataframe = pd.DataFrame(dataframe)
            return (dataframe)

        def random_points(coord_row, coord_column, final_matrix):
            # Осуществляем случайный выбор точек
            shape = final_matrix.shape
            n_strings = shape[0]
            n_columns = shape[1]

            coords = []
            # Если значение в случайной точке равно gap, skip или nodata, то оно не добавляется
            number_iter = 0
            while number_iter <= 100:
                random_i = random.randint(0, n_strings - 1)
                random_j = random.randint(0, n_columns - 1)
                coordinates = [random_i, random_j]
                if final_matrix[random_i, random_j] == self.gap:
                    pass
                elif final_matrix[random_i, random_j] == self.skip:
                    pass
                elif final_matrix[random_i, random_j] == self.nodata:
                    pass
                # Если у нас уже есть такая пара в списке, то мы не добавляем его в список
                elif any(tuple(coordinates) == tuple(element) for element in coords):
                    pass
                else:
                    coords.append(coordinates)
                    number_iter += 1

            # Добавляем координаты точки, для которой подбираются опорные точки (она всегда занимает последнее место в датасете)
            coords.append([coord_row, coord_column])
            coords = np.array(coords)

            # Для каждой матрицы из словаря по известным координатам (индексам) снимаем значения параметра и записываем в датасет
            # Последняя строка в датасете - значения для матрицы, которую необходимо загэпфилить :)
            dataframe = []
            for key in keys:
                matrix = dictionary.get(key)
                values = []
                for i, j in coords:
                    values.append(matrix[i, j])
                values = np.array(values)
                dataframe.append(values)
            dataframe = pd.DataFrame(dataframe)
            return (dataframe)

        def biome_points(coord_row, coord_column, final_matrix, extra_matrix):
            # Индекс строки и столбца для пикселя, который необходимо заполнить
            extra_code = extra_matrix[coord_row, coord_column]  # Код биома (группы пикселей) для пикселя, который мы хотим узнать

            # Важный момент - мы присваиваем всем значениям пикселей в копии матрицы биомов, если они закрыты облаком, значение -100.0
            new_extra_matrix = np.copy(extra_matrix)
            new_extra_matrix[final_matrix == self.gap] = self.gap

            # Индексы точек, которые попадают в данный биом и при этом в данный момент не являются пропусками
            coords = np.argwhere(new_extra_matrix == extra_code)
            if len(coords) > 41:
                ''' Биом подходит в качестве кластера '''

                # Рассчет расстояния целевого пикселя от тех, которые были выбраны в качестве предикторов (по индексам)
                target_pixel = np.array([[coord_row, coord_column]])
                # Вектор из рассчитаных расстояний от целевого пикселя до всех остальных
                distances = scipy.spatial.distance.cdist(target_pixel, coords)[0]
                new_coords = []
                for iter in range(0,40):
                    # Какой индекс в массиве coords имеет элемент с наименьшим расстоянием от целевого пикселя
                    index_min_dist = np.argmin(distances)

                    # Координата данного пикселя в матрице
                    new_coord = coords[index_min_dist]

                    # Добавляем координаты
                    new_coords.append(new_coord)

                    # Заменяем минимальный элемент из массива на очень большое число
                    distances[index_min_dist] = np.inf

                # Добавляем к нашем координатам в самый конец индекс пикселя, для которого строится модель
                coords = list(new_coords)
                coords.append([coord_row, coord_column])
                coords = np.array(coords)
            else:
                '''Недостаточно объектов в данном биоме! Выбираем 100 случайных точек.'''
                # Осуществляем случайный выбор точек
                shape = final_matrix.shape
                n_strings = shape[0]
                n_columns = shape[1]

                coords = []
                # Если значение в случайной точке равно gap, skip или nodata, то оно не добавляется
                number_iter = 0
                while number_iter <= 100:
                    random_i = random.randint(0, n_strings - 1)
                    random_j = random.randint(0, n_columns - 1)
                    coordinates = [random_i, random_j]
                    if final_matrix[random_i, random_j] == self.gap:
                        pass
                    elif final_matrix[random_i, random_j] == self.skip:
                        pass
                    elif final_matrix[random_i, random_j] == self.nodata:
                        pass
                    # Если у нас уже есть такая пара в списке, то мы не добавляем его в список
                    elif any(tuple(coordinates) == tuple(element) for element in coords):
                        pass
                    else:
                        coords.append(coordinates)
                        number_iter += 1
                coords = np.array(coords)

                # Рассчет расстояния целевого пикселя от тех, которые были выбраны в качестве предикторов (по индексам)
                target_pixel = np.array([[coord_row, coord_column]])
                # Вектор из рассчитаных расстояний от целевого пикселя до всех остальных
                distances = scipy.spatial.distance.cdist(target_pixel, coords)[0]
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

                # Добавляем координаты точки, для которой подбираются опорные точки (она всегда занимает последнее место в датасете)
                coords = list(new_coords)
                coords.append([coord_row, coord_column])
                coords = np.array(coords)

            # Для каждой матрицы из словаря по известным координатам (индексам) снимаем значения парметра и записываем в датасет
            # Последняя строка в датасете - значения для матрицы, которую необходимо загэпфилить :)
            dataframe = []
            for key in keys:
                matrix = dictionary.get(key)
                values = []
                for i, j in coords:
                    values.append(matrix[i, j])
                values = np.array(values)
                dataframe.append(values)
            dataframe = pd.DataFrame(dataframe)
            return(dataframe)

        final_matrix = dictionary.get(keys[-1]) # Та матрица, которую необходимо заполнить
        # Отмечаем индексы точек, которые необходимо заполнить
        gaps = np.argwhere(final_matrix == self.gap)
        print('Number of gap pixels -', len(gaps))

        # Делаем копию матрицы - в неё мы будем записывать значения после применения алгоритма
        filled_matrix = np.copy(final_matrix)

        # МОДЕЛЬ ОБУЧАЕТСЯ ПОПИКСЕЛЬНО
        # scores - массив, который будет содержать значения ошибок при кросс-валидации
        scores =[]
        for gap_pixel in gaps:
            coord_row = gap_pixel[0]
            coord_column = gap_pixel[1]

            # Составляем датасет для обучения модели
            if predictor_configuration == 'Biome':
                dataframe = biome_points(coord_row, coord_column, final_matrix = final_matrix, extra_matrix = extra_matrix)
            elif predictor_configuration == 'All':
                dataframe = all_points(coord_row, coord_column, final_matrix = final_matrix)
            # Подразумевается, что при незаданом методе, по умолчанию используется случайный выбор предикторов
            else:
                dataframe = random_points(coord_row, coord_column, final_matrix = final_matrix)

            # Осуществляем подготовку данных для датасета
            # Если в крайнем правом столбце есть хотя бы одно значение skip, то данный пиксель не будет тронут алгоритмом
            # Автоматически присваивается значение skip
            if any(value == self.skip for value in np.array(dataframe)[:,-1]):
                predicted = self.skip
            else:
                # Необходимо удалить те столбцы, в которых есть хотя бы одно значение skip (те значения, которые не нужно заполнять)
                dataframe.replace(self.skip, np.nan, inplace=True)
                dataframe.dropna(axis = 'columns', inplace = True)

                # Зададим новые названия столбцов
                new_columns = range(0, len(dataframe.columns))
                col_names = []
                for i in new_columns:
                    col_names.append(str(i))
                dataframe.set_axis(col_names, axis = 1, inplace = True)

                # Присваиваем оставшимся флагам значение Nan
                dataframe.replace(self.nodata, np.nan, inplace = True)
                dataframe.replace(self.gap, np.nan, inplace = True)

                dataframe = dataframe.dropna(how = 'all')  # Удаляем те строки, где по всем признакам имеем пропуски (облако закрыло всю территорию)

                last_string = np.array(dataframe.iloc[-1:, :-1])  # Берем последнюю строку из датасета (кроме последнего элемента в ней)
                last_string = np.ravel(last_string)
                last_string_na = np.ravel(np.isnan(last_string))  # Выставляем True, где есть пропуски
                indexes_na = np.ravel(np.argwhere(last_string_na == True))  # Получаем список из индексов столбцов, в которых есть пропуски в последней строке
                indexes_na_str = []
                for i in indexes_na:
                    indexes_na_str.append(str(i))

                # Если в последней строчке датасета есть пропуски, то удаляем столбцы с пропусками
                if len(indexes_na_str) > 0:
                    for i in indexes_na_str:
                        dataframe.drop([i], axis = 1, inplace = True)
                    # Необходимо заново перезадать индексы для столбцов
                    new_names = range(0, len(dataframe.columns))
                    new = []
                    for i in new_names:
                        new.append(str(i))
                    dataframe.set_axis(new, axis = 1, inplace = True)
                else:
                    pass

                # Заполняем в датафрейме медианным значением по столбцу
                def col_median(index):
                    sample = np.array(dataframe[index].dropna()) # Ищем медианное значение по столбцу
                    median = np.median(sample)
                    return(median)

                for i in range(0,len(dataframe.columns)-1): # В последнем столбце целевая функция, поэтому в нем пропуски мы пока не трогаем
                    i = str(i)
                    dataframe[i].fillna(col_median(i), inplace = True)

                # Разделим датафрейм на обучающую выборку и на объект, для которого необходимо предсказать значение
                train = dataframe.iloc[:-1, :]
                test = dataframe.iloc[-1:, :]

                # Из обучающей выборки исключаем все объекты с пропуском в целевой функции
                train = train.dropna()

                X_train = np.array(train.iloc[:, :-1])
                y_train = np.array(train.iloc[:, -1:])

                X_test = np.array(test.iloc[:, :-1])

                # В зависимости от выбранной опции - в работу включается соответствующий метод
                if method == 'RandomForest':
                    predicted, score = Random_forest_regression(X_train, np.ravel(y_train), X_test, params = params)
                elif method == 'ExtraTrees':
                    predicted, score = Extra_trees_regression(X_train, np.ravel(y_train), X_test, params = params)
                elif method == 'Knn':
                    predicted, score = KNN_regression(X_train, y_train, X_test, params = params)
                elif method == 'SVR':
                    predicted, score = SVM_regression(X_train, y_train, X_test, params = params)
                # Подразумевается, что при незаданом методе, по     умолчанию используется Лассо
                else:
                    predicted, score = Lasso_regression(X_train, y_train, X_test, params = params)

                # Значение ошибки в тесте при кросс-валидации записываем в отдельный массив
                scores.append(abs(score))

            # В матрицу записываем предсказанное алгоритмом значение
            filled_matrix[coord_row, coord_column] = predicted

        npy_name = str(keys[-1]) + '.npy'
        filled_matrix_npy = os.path.join(self.Outputs_path, npy_name)
        np.save(filled_matrix_npy, filled_matrix)
        # Если параметр "add_outputs" принимает значение True, то заполненный моделью слой включается в обучяющую выборку
        if add_outputs == True:
            filled_matrix_history_npy = os.path.join(self.History_path, npy_name)
            np.save(filled_matrix_history_npy, filled_matrix)

        # Теперь займемся ошибками полученными по кросс-валидации
        scores = np.array(scores)
        mean_score = np.mean(scores)
        print('Mean absolute error for cross-validation -', mean_score)
        # Заполняем метаданные необходимыми сведениями: насколько хорошо данный алгоритм отработал на данной матрице
        self.metadata.update({npy_name: mean_score})

    # Обертка над представленными выше методами, запускает алгоритм
    # method - название алгоритма (Lasso, RandomForest, ExtraTrees, Knn, SVR)
    # predictor_configuration - подбор предикторов (All, Random, Biome)
    # hyperparameters - выбор гиперпараметров (RandomGridSearch, GridSearch, Custom)
    # params - если выбран аргумент "Custom", то параметры модели передаются через аргумент params
    # add_outputs - будут ли добавляться заполненные алгоритмом слои в обучающую выборку
    # key_values - словарь с обозначениями пропусков, нерелевантных и отсутствующих значений
    # В папке проекта "Outputs" создаются матрицы с заполненными пропусками в формате .npy
    # Формируется JSON файл с оценкой качества работы алгоритма на каждой матрице
    def fill_gaps(self, method = 'Lasso', predictor_configuration = 'Random', hyperparameters = 'RandomGridSearch',
                     params = None, add_outputs = False, key_values = {'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}):
        # Определяем флаги для пропусков, значений, которые заполнять не нужно ошибок проецирования
        self.gap = key_values.get('gap')
        self.skip = key_values.get('skip')
        self.nodata = key_values.get('NoData')

        if predictor_configuration == 'Biome':
            # Получаем матрицу для разбиения пикселей на группы
            Extra_file = os.path.join(self.Extra_path, 'Extra.npy')
            extra_matrix = np.load(Extra_file)
        else:
            extra_matrix = None

        # Файлы, в которых необходимо заполнить пропуски
        inputs_files = os.listdir(self.Inputs_path)
        inputs_files.sort()
        for input in inputs_files:
            start = timeit.default_timer()  # Засекаем время
            # Применим метод для сведения матриц в ассоциативный массив, и так как словарь неупорядочен, то запишем ключи в строгом порядке в список keys
            dictionary, keys = self.__make_training_sample() # Пока в словаре только матрицы из обучающей выборки

            input_path = os.path.join(self.Inputs_path, input)  # Путь до конкрентной матрицы в папке "Inputs"
            key = input[:-4]  # Ключ для матрицы
            matrix = np.load(input_path)

            # Если мы имеем менее 101 незакрытых пикселей на сцене, то рассчет не производится
            shape = matrix.shape
            all_pixels = shape[0] * shape[1] # Все пиксели в матрице
            if all_pixels - ((matrix == self.gap).sum() + (matrix == self.skip).sum() + (matrix == self.nodata).sum()) <= 101:
                print('No calculation for matrix', key)
            # Если на снимке нет пропусков, то он сохранается как заполненный без применения алгоритма
            elif (matrix == self.gap).sum() == 0:
                print('No gaps in matrix', key)
                npy_name = str(key) + '.npy'
                filled_matrix_npy = os.path.join(self.Outputs_path, npy_name)
                np.save(filled_matrix_npy, matrix)

                if add_outputs == True:
                    filled_matrix_npy = os.path.join(self.History_path, npy_name)
                    np.save(filled_matrix_npy, matrix)
                # Заполняем словарь с метаданными
                self.metadata.update({npy_name: 0.0})
            else:
                print('Calculations for matrix', key)
                # Теперь в массиве есть матрица, для которой необходимо заполнить пропуски
                keys.append(key)
                dictionary.update({key: matrix})

                # Начинает работать модель
                self.__learning_and_fill(dictionary, keys, extra_matrix = extra_matrix, method = method, predictor_configuration = predictor_configuration,
                                         hyperparameters = hyperparameters, params = params, add_outputs = add_outputs)
            print('Runtime -', timeit.default_timer() - start, '\n')  # Время работы алгоритма

        # Сохраняем сформированный словарь с метаданными в файл JSON
        Outputs_path = os.path.join(self.directory, 'Outputs')
        json_path = os.path.join(Outputs_path, 'Metadata.json')
        with open(json_path, 'w') as json_file:
            json.dump(self.metadata, json_file)

    # Функция, которая позволяет заполнять пропуски с помощью метода интерполяции ближайшим соседом
    def nn_interpolation(self, key_values = {'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}):

        # Определяем флаги для пропусков, значений, которые заполнять не нужно ошибок проецирования
        self.gap = key_values.get('gap')
        self.skip = key_values.get('skip')
        self.nodata = key_values.get('NoData')

        files = os.listdir(self.Inputs_path)
        files.sort()

        for file in files:
            start = timeit.default_timer()  # Засекаем время
            matrix = np.load(os.path.join(self.Inputs_path, file))

            shape = matrix.shape
            all_pixels = shape[0] * shape[1]
            # Если все значения в матрице - это пропуски, и т.д., то слой не заполняется
            if all_pixels - ((matrix == self.gap).sum() + (matrix == self.skip).sum() + (matrix == self.nodata).sum()) < 5:
                print('No calculation for matrix', file[:-4])
            # Если на снимке нет пропусков, то он сохранается как заполненный без применения алгоритма
            elif (matrix == self.gap).sum() == 0:
                print('No gaps in matrix', file[:-4])
                where_to_save = os.path.join(self.Outputs_path, file)
                np.save(where_to_save, matrix)
            else:
                print('Calculations for matrix', file[:-4])
                # Копия матрицы для нанесения всех масок
                copy_matrix = np.copy(matrix)

                # Интерполяция производится для всех флагов - их необходимо пометить как пропуски
                matrix[matrix == self.skip] = self.gap
                matrix[matrix == self.nodata] = self.gap

                # Смотрим как выглядит матрица в данный момент
                masked_array = np.ma.masked_where(matrix == self.gap, matrix)

                #        Interpolation        #
                x = np.arange(0, len(matrix[0]))
                y = np.arange(0, len(matrix))
                Gap_matrix = masked_array
                xx, yy = np.meshgrid(x, y)
                x1 = xx[~Gap_matrix.mask]
                y1 = yy[~Gap_matrix.mask]
                newarr = Gap_matrix[~Gap_matrix.mask]
                GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')

                # Возвращение ранее убранных флагов
                GD1[copy_matrix == self.skip] = self.skip
                GD1[copy_matrix == self.nodata] = self.nodata

                # Сохранение матрицы в папку outputs
                where_to_save = os.path.join(self.Outputs_path, file)
                np.save(where_to_save, GD1)
                print('Runtime -', timeit.default_timer() - start, '\n') # Время работы алгоритма