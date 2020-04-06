# SSGP-toolbox

SimpleSpatialGapfiller - python class for filling gaps in matrices based on machine learing techniques. Main purpose is to provide convenient and simple instruments for modeling geophysical parameters, derived with Earth Remote Sensing, under clouds. But it also could be used for any matrices.


## Requirements
    'python>=3.7',
    'gdal>=2.4',
    'numpy',
    'scikit-learn==0.21.3',
    'pandas',
    'scipy',
    'netCDF4',
    'pyproj' 

## Install module

```python
pip install git+https://github.com/Dreamlone/SSGP-toolbox
```

## Modules

For now SSGT-toolbox is presented with:
 - Gapfiller class
 - Several preparators: for Sentinel 3 LST data; for MODIS LST products; for MODIS NDVI based on reflectance product.
 
 By the way, you can prepare any data by yourself, it must be in binary numpy matrices format (.npy) and organized in several directories, as shown in docs.

## Documentation and examples

All documentation and examples for now are described in Jupyter Notebooks:
 - [Theoretical basis](https://nbviewer.jupyter.org/github/Dreamlone/SSGP-toolbox/blob/master/Notebooks/Theoretical_basis.ipynb)
 - [Gapfiller class, how to organize data and how to use](https://nbviewer.jupyter.org/github/Dreamlone/SSGP-toolbox/blob/master/Notebooks/Gapfilling.ipynb)
 - [Sentinel 3 LST preparation](https://nbviewer.jupyter.org/github/Dreamlone/SSGP-toolbox/blob/master/Notebooks/Prepare_S3LST.ipynb)
 - [MODIS LST preparation](https://nbviewer.jupyter.org/github/Dreamlone/SSGP-toolbox/blob/master/Notebooks/Prepare_MODIS_LST.ipynb)

## Contacts

Feel free to contact us:

Mikhail Sarafanov (maintainer) | mik_sar@mail.ru

Eduard Kazakov | ee.kazakov@gmail.com

======================
## TODO: move all this stuff to notebooks, add new

## Data preparation

Для корректной работы алгоритма необходимо подготовить обучающую выборку и данные для заполнения пропусков.
Формируется директория (LST, NDVI или любое другое название, которое пользователь может задать самостоятельно), в которой расположены папки (названия папок фиксированы): 

![Database.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_1_Database.png)

### 1. History - папка с матрицами в формате .npy, которые являются обучающей выборкой.

Названия файлов должны быть формата - "20190625T185030.npy", где 2019.. - год, ..06.. - месяц, ..25.. - день, ..T185030 - время - часы минуты секунды (format = '%Y%m%dT%H%M%S'). Матрицы в обучающей выборке могут содержать пропуски. При обучении алгоритм будет либо удалять эти пропуски из обучающей выборки, либо заменять их медианой по временному ряду для данного пикселя.


### 2. Inputs - папка с имеющими пропуски матрицами в формате .npy, которые необходимо заполнить.

Названия файлов должны быть формата - "20190625T185030.npy", где 2019.. - год, ..06.. - месяц, ..25.. - день, ..T185030 - время - часы минуты секунды (format = '%Y%m%dT%H%M%S')


### 3. Extra - папка с матрицей в формате .npy, которая позволяет разделить ячейки матриц на группы. Название файла должно быть формата - "Extra.npy"

Матрица может выглядеть следующим образом:
![Biomes.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_2_Biomes.png)

В качестве значений в данной матрице должны быть записаны целые числа.

## Значения в матрицах

1) gap     --- значение в пикселях, которые необходимо заполнить (по умолчанию "-100.0")

2) skip    --- NoData в пикселях, которые заполнять не нужно, например, морская вода, когда заполнять следует только пиксели со значениями температуры поверхности земли. Алгоритм будет ретроспективно оценивать, был ли каждый конкретный пиксель занять значением skip, и если был, то предсказанное моделью значение в данном пикселе будет равно skip. (по умолчанию "-200.0")

3) NoData  --- значение в пикселях, которые не попали в экстенд снимка, также данным значением могут быть помечены ошибки при проецировании растровых изображений. Если количество пикселей с данным значением в матрице из папки 'History' превышает определенное количество процентов на снимке (self.main_threshold = 0.05), то данная матрица не будет включена в обучающую выборку. (по умолчанию "-32768.0")

Алгоритм заполняет только пиксели со значениями gap.


В случае, если матрица, которую необходимо заполнить, имеет менее 101 незакрытого пикселя (то есть НЕ gap, НЕ skip и НЕ NoData), то алгоритм её не заполняет, на экран выводится сообщение 'No calculation for matrix NAME_OF_MATRIX'. Матрица не добавляется в папу "Outputs".


Если матрица не имеет пропусков, то на экране появится сообщение 'No gaps in matrix NAME_OF_MATRIX'. Матрица автоматически добавляется в папу "Outputs".


Таким образом:
- Обучающая выборка размещается в папке "History"
- Матрицы, которые необходимо заполнить, следует разместить в папке "Inputs"
- Папка "Extra" является опциональной и в случае создания содержит одну матрицу
- Папка "Outputs" формируется во время работы алгоритма

В результате работы алгоритма формируется папка 'Outputs', в которой заполненные алгоритмом матрицы сохраняются в формате .npy, а также создается файл .json 
со значенями оценки точности работы алгоритма для каждого слоя. Точность оценивается по кросс-валидации на данных из обучающей выборки.

## Examples

Выбранный метод - метод опорных векторов. Стратегия выбора предикторов - "биомы". Подбор гиперпараметров - пользовательская настройка в виде словаря. 
Параметры "add_outputs" и "key_values" - по умолчанию.

```python
Gapfiller = SimpleSpatialGapfiller(directory = '/media/test/LST')
Gapfiller.fill_gaps(method = 'SVR', predictor_configuration = 'Biome',
                    hyperparameters = 'Custom', 
                    params = {'kernel': 'linear', 'gamma': 'scale', 'C': 1000, 'epsilon': 1})
```

Пример применения алгоритма. Выбранный метод - LASSO регрессия. Стратегия выбора предикторов - "случайные 100 точек". 
Подбор гиперпараметров - полный поиск по сетке. Заполненные алгоритмом матрицы будут включаться в обучающую выборку для последующих слоев.


```python
Gapfiller = SimpleSpatialGapfiller(directory = '/media/test/LST')
Gapfiller.fill_gaps(method = 'Lasso', predictor_configuration = 'Random',
                    hyperparameters = 'GridSearch', add_outputs = True,
                    key_values = {'gap': -1.0, 'skip': -10.0, 'NoData': -100.0})
```

## Параметры

### Выбор алгоритма заполнения пропусков - method
- ПО УМОЛЧАНИЮ 'Lasso' - Лассо регрессия 
- 'RandomForest' - случайный лес
- 'ExtraTrees' - сверхслуйчаный лес
- 'Knn' - k-ближайших соседей
- 'SVR' - метод опорных векторов

### Стратегии подбора предикторов - predictor_configuration
- ПО УМОЛЧАНИЮ 'Random' - Случано выбранные 100 точек на снимке
- 'All' - предикторы - все известные точки на снимке
- 'Biome' - в качестве предикторов выбираются 40 наиболее близких (по Евклидовой метрике) пикселей из того же биома, что и пропуск

### Варианты настройки гиперпараметров - hyperparameters
- ПО УМОЛЧАНИЮ 'RandomGridSearch' - случайный поиск по сетке 
- 'GridSearch' - полный перебор по сетке
- 'Custom' - пользовательская настройка в виде словаря

### Словарь с гиперпараметрами (если hyperparameters = 'Custom') - params
- ПО УМОЛЧАНИЮ - None. Если hyperparameters != 'Custom', то игнорируется

### Возможность использования заполненных слоев - add_outputs
- ПО УМОЛЧАНИЮ - False, т.е. заполненные слои не добавляются в обучающую выборку
- True - в таком случае заполненные алгоритмом матрицы включаются в обучающую выборку

### Cловарь с обозначениями пропусков, нерелевантных и отсутствующих значений - key_values
- ПО УМОЛЧАНИЮ - {'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}


## Theoretical basis

Алгоритм опирается на значения в известных пикселях той же матрицы, что и пропуск, для того, чтобы расчитать значение в пропуске.

### Принцип работы

Алгоритм строит отдельно для каждого пикселя свою модель, затем в зависимости от заданных параметров применяет различные 
способы подбора предикторов. На иллюстрации приведен упрощенный пример того, как алгоритм формирует обучающую выборку.

![Construction.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_3_Construction.png)

Программная реализация содержит 1 класс - SimpleSpatialGapfiller().
#### Приватные методы:
- __make_training_sample  --- формирование обучающей выборки из матриц в папке "History"
- __learning_and_fill     --- заполнение пропусков на конкретной матрице, запись результата в папку "Outputs"

#### Публичные методы:
- fill_gaps               --- применение метода __learning_and_fill для каждой из матриц в папке "Inputs", 
создание файла с метаданными о качестве работы алгоритма

По результатам проведенных экспериментов с моделью, установлено, что временная сложность алгоритма сублинейная.

![Complexity.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_4_Complexity.png)

Для алгоритма была прозведена верефикация на тепловых данных дистанционного зондирования со спутниковой системы Sentinel-3. 
Были отобраны 6 разновременных снимков одной территории. Для каждого из 6 снимков генерировался пропуск определенного размера и формы. 
Всего было сгенерировано 8 типов пропусков. Результаты проведенных тестов можно увидеть ниже. В болшинстве случаев алгоритм ошибался 
менее чем на 1 градус, при том, что средний разброс значений температуры в пропуске составлял около 10. Резултаты верификации
алгоритма можно увидеть ниже.

![Results.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_5_Results.png)

Из графика видно, что точность восстановления данных зависит больше от распределения поля температуры на снимке,
чем от размера пропуска.



