'''

class SimpleSpatialGapfiller --- a class that allows to fill in gaps in matrices based on machine learning method

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

    # When initializing the class, we must specify
    # directory --- the location of the project folder: "History", "Inputs" and "Extra"
    def __init__(self, directory):
        # Threshold value for not including layers in the training selection when exceeded (changes from 0.0 to 1.0)
        self.main_threshold = 0.05
        self.directory = directory

        # Creating the 'Outputs' folder; if there is, use the existing one
        self.Outputs_path = os.path.join(self.directory, 'Outputs')
        if os.path.isdir(self.Outputs_path) == False:
            os.makedirs(self.Outputs_path)

        # Access to all remaining folders in the project
        self.Inputs_path = os.path.join(self.directory, 'Inputs')
        self.Extra_path = os.path.join(self.directory, 'Extra')
        self.History_path = os.path.join(self.directory, 'History')

        # Creating a dictionary with data that will be filled in as the algorithm works
        self.metadata = {}

    # The private method generates a training sample from the matrices in the "History" folder
    # return dictionary --- dictionary where the key (file name) corresponds to the matrix
    # return keys       --- sorted list of keys, where the last key is the target matrix
    def __make_training_sample(self):
        # Creating a dictionary with matrices and a list with keys, where they are stored in strict order
        history_files = os.listdir(self.History_path)
        dictionary = {}
        keys = []
        for file in history_files:
            key_path = os.path.join(self.History_path, file) # Path to a specific matrix in the training sample
            key = file[:-4] # Key for the matrix
            matrix = np.load(key_path)

            # If there is more than a main_threshold percentage of pixels that were not taken at this time, the matrix is not included in the analysis
            amount_na = (matrix == self.nodata).sum()
            shape = matrix.shape
            threshold = amount_na/(shape[0]*shape[1])
            if threshold > self.main_threshold:
                pass
            else:
                keys.append(key)
                dictionary.update({key: matrix})

        # Sorting the list with keys
        keys.sort()
        return(dictionary, keys)

    # Private method - prepares the dataset, trains the model, and writes the result (as a .npy file) to the specified folder
    # dictionary              --- dictionary, key - timestamp of image, value - matrix; all layers except the last one are needed for training
    # keys                    --- list of keys where the last key belongs to the matrix to fill in the gaps in
    # method                  --- name of the algorithm (Lasso, RandomForest, ExtraTrees, Knn, SVR)
    # predictor_configuration --- selection of predictors (All, Random, Biome)
    # hyperparameters         --- the choice of hyperparameters (RandomGridSearch, GridSearch, Custom)
    # params                  --- if the argument is selected "Custom", then the model parameters are passed via the params argument
    # add_outputs             --- will the layers filled in by the algorithm be added to the training selection
    def __learning_and_fill(self, dictionary, keys, extra_matrix, method, predictor_configuration, hyperparameters, params, add_outputs):

        # Lasso
        def Lasso_regression(X_train, y_train, X_test, params):
            # Grid search for Lasso due to the small number of hyperparameters 'GridSearch' and 'RandomGridSearch' are the same
            if hyperparameters == 'RandomGridSearch' or hyperparameters == 'GridSearch':
                # We search the grid with cross-validation (the number of folds is 3)
                alphas = np.arange(1, 800, 50)
                param_grid = {'alpha': alphas}
                # Setting the model to train
                estimator = Lasso()
                # We train the model with the specified parameter options (we search the grid)
                optimizer = GridSearchCV(estimator, param_grid, iid = 'deprecated', cv = 3, scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = Lasso()
                # Setting the necessary parameters
                estimator.set_params(**params)

                # Cross-validation check
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')

                # Training the model already on all data
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # Random forest
        def Random_forest_regression(X_train, y_train, X_test, params):
            # Random grid search
            if hyperparameters == 'RandomGridSearch':
                # Carry out a random grid search with cross-validation (the number of folds is 3)
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_leaf_nodes': max_leaf_nodes}
                # Set the model to be trained
                estimator = RandomForestRegressor(n_estimators = 50, n_jobs = -1)
                # Train the model with the given options of parameters
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Grid search
            elif hyperparameters == 'GridSearch':
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_leaf_nodes': max_leaf_nodes}
                # Set the model to be trained
                estimator = RandomForestRegressor(n_estimators = 50, n_jobs = -1)
                # Train the model with the given options of parameters
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = RandomForestRegressor()
                # Set the params
                estimator.set_params(**params)
                # Cross-validation
                fold = KFold(n_splits = 3, shuffle=True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # Extra trees
        def Extra_trees_regression(X_train, y_train, X_test, params):
            # Random grid search
            if hyperparameters == 'RandomGridSearch':
                # Carry out a random grid search with cross-validation (the number of folds is 3)
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_leaf_nodes': max_leaf_nodes}
                # Set the model to be trained
                estimator = ExtraTreesRegressor(n_estimators = 50, n_jobs = -1)
                # Train the model with the given options of parameters
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Full grid search
            elif hyperparameters == 'GridSearch':
                max_depth = [5, 10, 15, 20, 25]
                min_samples_split = [2, 5, 10]
                max_leaf_nodes = [10, 50, 100]
                param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split,'max_leaf_nodes': max_leaf_nodes}
                # Set the model to be trained
                estimator = ExtraTreesRegressor(n_estimators = 50, n_jobs = -1)
                # Train the model with the given options of parameters
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = ExtraTreesRegressor()
                # Set the params
                estimator.set_params(**params)

                # Cross-validation
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # К-nearest neighbors
        def KNN_regression(X_train, y_train, X_test, params):
            # Random grid search
            if hyperparameters == 'RandomGridSearch':
                # Carry out a random grid search with cross-validation (the number of folds is 3)
                weights = ['uniform', 'distance']
                algorithm = ['auto', 'kd_tree']
                n_neighbors = [2, 5,10,15,20]
                param_grid = {'weights': weights, 'n_neighbors': n_neighbors, 'algorithm': algorithm}
                # Set the model to be trained
                estimator = KNeighborsRegressor()
                # Train the model with the given options of parameters
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Full grid search
            elif hyperparameters == 'GridSearch':
                weights = ['uniform', 'distance']
                algorithm = ['auto', 'kd_tree']
                n_neighbors = [2, 5, 10, 15, 20]
                param_grid = {'weights': weights, 'n_neighbors': n_neighbors, 'algorithm': algorithm}
                # Set the model to be trained
                estimator = KNeighborsRegressor()
                # Train the model with the given options of parameters
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid='deprecated', scoring='neg_mean_absolute_error')
                optimizer.fit(X_train, y_train)
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = KNeighborsRegressor()
                # Set the params
                estimator.set_params(**params)

                # Cross-validation
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv = fold, scoring = 'neg_mean_absolute_error')
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)
            return(predicted, validation_score)

        # Support Vector Machine
        def SVM_regression(X_train, y_train, X_test, params):
            # Combine our sample for the standardization procedure
            sample = np.vstack((X_train, X_test))

            # Standardize the sample and split again
            sample = preprocessing.scale(sample)
            X_train = sample[:-1, :]
            X_test = sample[-1:, :]

            # Random grid search
            if hyperparameters == 'RandomGridSearch':
                # Carry out a random grid search with cross-validation (the number of folds is 3)
                Cs = [0.001, 0.01, 0.1, 1, 10]
                epsilons = [0.1, 0.4, 0.7, 1.0]
                param_grid = {'C': Cs, 'epsilon': epsilons}
                # Set the model to be trained
                estimator = SVR(kernel = 'linear', gamma = 'scale')
                # Train the model with the given options of parameters
                optimizer = RandomizedSearchCV(estimator, param_grid, n_iter = 5, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            # Full grid search
            elif hyperparameters == 'GridSearch':
                Cs = [0.001, 0.01, 0.1, 1, 10]
                epsilons = [0.1, 0.4, 0.7, 1.0]
                param_grid = {'C': Cs, 'epsilon': epsilons}
                # Set the model to be trained
                estimator = SVR(kernel = 'linear', gamma = 'scale')
                # Train the model with the given options of parameters
                optimizer = GridSearchCV(estimator, param_grid, cv = 3, iid = 'deprecated', scoring = 'neg_mean_absolute_error')
                optimizer.fit(X_train, np.ravel(y_train))
                regression = optimizer.best_estimator_
                predicted = regression.predict(X_test)
                validation_score = optimizer.best_score_
            elif hyperparameters == 'Custom':
                estimator = SVR()
                # Set the params
                estimator.set_params(**params)

                # Cross-validation
                fold = KFold(n_splits = 3, shuffle = True)
                validation_score = cross_val_score(estimator = estimator, X = X_train, y = np.ravel(y_train), cv = fold, scoring = 'neg_mean_absolute_error')
                estimator.fit(X_train, np.ravel(y_train))
                predicted = estimator.predict(X_test)

            return(predicted, validation_score)

        def all_points(coord_row, coord_column, final_matrix):
            # The indices of all points that are not covered by clouds (including pixels with a value skip, nodata)
            coords = np.argwhere(final_matrix != self.gap)
            coords = list(coords)
            coords.append([coord_row, coord_column])
            coords = np.array(coords)

            # For each matrix from the dictionary by known coordinates (indexes), we remove the parameter
            # values and write them to the dataset
            # The last row in the dataset is the values for the matrix to be filled in
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
            # Make a random selection of points
            shape = final_matrix.shape
            n_strings = shape[0]
            n_columns = shape[1]

            coords = []
            # If the value at a random point is equal to gap, skip, or nodata, it is not added
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
                # If we already have such a pair in the list, we don't add it to the list
                elif any(tuple(coordinates) == tuple(element) for element in coords):
                    pass
                else:
                    coords.append(coordinates)
                    number_iter += 1

            # Adding the coordinates of the point for which reference points are selected (it always takes the last place in the dataset)
            coords.append([coord_row, coord_column])
            coords = np.array(coords)

            # For each matrix from the dictionary by known coordinates (indexes), we remove the parameter
            # values and write them to the dataset
            # The last row in the dataset is the values for the matrix to be filled in
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
            # Index of the row and column for the pixel to be filled in
            extra_code = extra_matrix[coord_row, coord_column]  # Code of the biome (group of pixels) for the pixel we want to find out

            # An important moment is that we assign all pixel values in the copy of the biome matrix, if they are covered by the cloud, the value gap
            new_extra_matrix = np.copy(extra_matrix)
            new_extra_matrix[final_matrix == self.gap] = self.gap

            # Indexes of points that fall into this biome and are not currently omitted
            coords = np.argwhere(new_extra_matrix == extra_code)
            if len(coords) > 41:
                ''' The biome is suitable as a cluster '''

                # Рассчет расстояния целевого пикселя от тех, которые были выбраны в качестве предикторов (по индексам)
                target_pixel = np.array([[coord_row, coord_column]])
                # Vector of calculated distances from the target pixel to all other pixels
                distances = scipy.spatial.distance.cdist(target_pixel, coords)[0]
                new_coords = []
                for iter in range(0,40):
                    # Which index in the coords array has the element with the smallest distance from the target pixel
                    index_min_dist = np.argmin(distances)

                    # The coordinate of this pixel in the matrix
                    new_coord = coords[index_min_dist]
                    new_coords.append(new_coord)

                    # Replacing the minimum element from the array with a very large number
                    distances[index_min_dist] = np.inf

                # Adding the index of the pixel for which the model is being built
                coords = list(new_coords)
                coords.append([coord_row, coord_column])
                coords = np.array(coords)
            else:
                '''There are not enough objects in this biome! Choose 100 random points'''

                # Make a random selection of points
                shape = final_matrix.shape
                n_strings = shape[0]
                n_columns = shape[1]

                coords = []
                # If the value at a random point is equal to gap, skip, or nodata, it is not added
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
                    # If we already have such a pair in the list, we don't add it to the list
                    elif any(tuple(coordinates) == tuple(element) for element in coords):
                        pass
                    else:
                        coords.append(coordinates)
                        number_iter += 1
                coords = np.array(coords)

                # The calculation of the distance of the target pixel from those that were selected as predictors (index)
                target_pixel = np.array([[coord_row, coord_column]])
                # Vector of calculated distances from the target pixel to all other pixels
                distances = scipy.spatial.distance.cdist(target_pixel, coords)[0]
                new_coords = []
                for iter in range(0, 40):
                    # Which index in the coords array has the element with the smallest distance from the target pixel
                    index_min_dist = np.argmin(distances)

                    # The coordinate of this pixel in the matrix
                    new_coord = coords[index_min_dist]
                    new_coords.append(new_coord)

                    # Replacing the minimum element from the array with a very large number
                    distances[index_min_dist] = np.inf

                # Adding the coordinates of the point for which reference points are selected (it always takes the last place in the dataset)
                coords = list(new_coords)
                coords.append([coord_row, coord_column])
                coords = np.array(coords)

            # For each matrix from the dictionary by known coordinates (indexes), we remove the parameter
            # values and write them to the dataset
            # The last row in the dataset is the values for the matrix to be filled in
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

        final_matrix = dictionary.get(keys[-1]) # The matrix to fill in
        # Mark the indexes of points that need to be filled in
        gaps = np.argwhere(final_matrix == self.gap)
        print('Number of gap pixels -', len(gaps))

        # Make a copy of the matrix - we will write the values to it after applying the algorithm
        filled_matrix = np.copy(final_matrix)

        # THE MODEL IS TRAINED PIXEL BY PIXEL
        # scores - an array that will contain error values during cross-validation
        scores = []
        for gap_pixel in gaps:
            coord_row = gap_pixel[0]
            coord_column = gap_pixel[1]

            # Creating a dataset for model training
            if predictor_configuration == 'Biome':
                dataframe = biome_points(coord_row, coord_column, final_matrix = final_matrix, extra_matrix = extra_matrix)
            elif predictor_configuration == 'All':
                dataframe = all_points(coord_row, coord_column, final_matrix = final_matrix)
            # If the method is not set, a random selection of predictors is used by default
            else:
                dataframe = random_points(coord_row, coord_column, final_matrix = final_matrix)

            # Preparing data for the dataset
            # If there is at least one skip value in the rightmost column, this pixel will not be affected by the algorithm
            # The skip value is automatically assigned
            if any(value == self.skip for value in np.array(dataframe)[:,-1]):
                predicted = self.skip
            else:
                # We must delete those columns that have at least one skip value (those values that do not need to be filled in)
                dataframe.replace(self.skip, np.nan, inplace=True)
                dataframe.dropna(axis = 'columns', inplace = True)

                # Setting new column names
                new_columns = range(0, len(dataframe.columns))
                col_names = []
                for i in new_columns:
                    col_names.append(str(i))
                dataframe.set_axis(col_names, axis = 1, inplace = True)

                # Assign the remaining flags the value Nan
                dataframe.replace(self.nodata, np.nan, inplace = True)
                dataframe.replace(self.gap, np.nan, inplace = True)

                dataframe = dataframe.dropna(how = 'all')  # We delete those lines where we have omissions for all signs (the cloud has closed the entire territory)

                last_string = np.array(dataframe.iloc[-1:, :-1])  # Take the last row from the dataset (except for the last element in it)
                last_string = np.ravel(last_string)
                last_string_na = np.ravel(np.isnan(last_string))  # Setting True where there are gaps
                indexes_na = np.ravel(np.argwhere(last_string_na == True))  # Get a list of column indexes that have gaps in the last row
                indexes_na_str = []
                for i in indexes_na:
                    indexes_na_str.append(str(i))

                # If there are gaps in the last row of the dataset, then delete the columns with gaps
                if len(indexes_na_str) > 0:
                    for i in indexes_na_str:
                        dataframe.drop([i], axis = 1, inplace = True)
                    # You must re-set the indexes for the columns
                    new_names = range(0, len(dataframe.columns))
                    new = []
                    for i in new_names:
                        new.append(str(i))
                    dataframe.set_axis(new, axis = 1, inplace = True)
                else:
                    pass

                # Fill in the dataframe with the median value for the column
                def col_median(index):
                    sample = np.array(dataframe[index].dropna()) # Looking for the median value by column
                    median = np.median(sample)
                    return(median)

                for i in range(0,len(dataframe.columns)-1): # The last column contains the target, so we don't touch the gaps in it yet
                    i = str(i)
                    dataframe[i].fillna(col_median(i), inplace = True)

                # Divide the dataframe into a training sample and an object to predict the value for
                train = dataframe.iloc[:-1, :]
                test = dataframe.iloc[-1:, :]

                # We exclude all objects with omission in the target function from the training sample
                train = train.dropna()

                X_train = np.array(train.iloc[:, :-1])
                y_train = np.array(train.iloc[:, -1:])

                X_test = np.array(test.iloc[:, :-1])

                # Depending on the selected option, the corresponding method is enabled
                if method == 'RandomForest':
                    predicted, score = Random_forest_regression(X_train, np.ravel(y_train), X_test, params = params)
                elif method == 'ExtraTrees':
                    predicted, score = Extra_trees_regression(X_train, np.ravel(y_train), X_test, params = params)
                elif method == 'Knn':
                    predicted, score = KNN_regression(X_train, y_train, X_test, params = params)
                elif method == 'SVR':
                    predicted, score = SVM_regression(X_train, y_train, X_test, params = params)
                # If the method is not set, the Lasso-regression is used by default
                else:
                    predicted, score = Lasso_regression(X_train, y_train, X_test, params = params)

                # The value of the error in the test during cross-validation is recorded in a separate array
                scores.append(abs(score))

            # Write the value predicted by the algorithm to the matrix
            filled_matrix[coord_row, coord_column] = predicted

        npy_name = str(keys[-1]) + '.npy'
        filled_matrix_npy = os.path.join(self.Outputs_path, npy_name)
        np.save(filled_matrix_npy, filled_matrix)
        # If the "add_outputs" parameter is set to True, the layer filled in by the model is included in the training selection
        if add_outputs == True:
            filled_matrix_history_npy = os.path.join(self.History_path, npy_name)
            np.save(filled_matrix_history_npy, filled_matrix)

        # Now let's look at the errors received from cross-validation
        scores = np.array(scores)
        mean_score = np.mean(scores)
        print('Mean absolute error for cross-validation -', mean_score)
        # Fill in the metadata with the necessary information: how well this algorithm worked on this matrix
        self.metadata.update({npy_name: mean_score})

    # Wrapper over the methods presented above, starts the algorithm for filling in gaps
    # method - the name of the algorithm (Lasso, RandomForest, ExtraTrees, Knn, SVR)
    # predictor_configuration - selection of predictors (All, Random, Biome)
    # hyperparameters - the choice of hyperparameters (RandomGridSearch, GridSearch, Custom)
    # params - if the "Custom" argument is selected, the model parameters are passed through the params argument
    # add_outputs - will the layers filled in by the algorithm be added to the training sample
    # key_values - dictionary with omissions, irrelevant and missing values
    # In the project folder "Outputs", matrices with filled-in gaps are created in the .npy format
    # A JSON file with quality metrics for each matrix
    def fill_gaps(self, method = 'Lasso', predictor_configuration = 'Random', hyperparameters = 'RandomGridSearch',
                     params = None, add_outputs = False, key_values = {'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}):
        # Defining flags for gaps, skips and NoData
        self.gap = key_values.get('gap')
        self.skip = key_values.get('skip')
        self.nodata = key_values.get('NoData')

        if predictor_configuration == 'Biome':
            # We get a matrix for dividing pixels into groups
            Extra_file = os.path.join(self.Extra_path, 'Extra.npy')
            extra_matrix = np.load(Extra_file)
        else:
            extra_matrix = None

        # Files where we need to fill in the gaps
        inputs_files = os.listdir(self.Inputs_path)
        inputs_files.sort()
        for input in inputs_files:
            start = timeit.default_timer()
            # We apply the method to reduce matrices to an associative array, and since the dictionary is unordered, we write the keys in strict order in the keys list
            dictionary, keys = self.__make_training_sample() # So far, the dictionary only contains matrices from the training sample

            input_path = os.path.join(self.Inputs_path, input)  # Path to the concrete matrix in the "Inputs" folder
            key = input[:-4]  # Key for the matrix
            matrix = np.load(input_path)

            # If we have less than 101 unclosed pixels on the scene, the calculation is not performed
            shape = matrix.shape
            all_pixels = shape[0] * shape[1]
            if all_pixels - ((matrix == self.gap).sum() + (matrix == self.skip).sum() + (matrix == self.nodata).sum()) <= 101:
                print('No calculation for matrix', key)
            # If there are no gaps in the image, it is saved as filled in without using the algorithm
            elif (matrix == self.gap).sum() == 0:
                print('No gaps in matrix', key)
                npy_name = str(key) + '.npy'
                filled_matrix_npy = os.path.join(self.Outputs_path, npy_name)
                np.save(filled_matrix_npy, matrix)

                if add_outputs == True:
                    filled_matrix_npy = os.path.join(self.History_path, npy_name)
                    np.save(filled_matrix_npy, matrix)
                # Filling in the dictionary with metadata
                self.metadata.update({npy_name: 0.0})
            else:
                print('Calculations for matrix', key)
                # Now there is a matrix in the array for which you need to fill in the gaps
                keys.append(key)
                dictionary.update({key: matrix})

                # Start the model
                self.__learning_and_fill(dictionary, keys, extra_matrix = extra_matrix, method = method, predictor_configuration = predictor_configuration,
                                         hyperparameters = hyperparameters, params = params, add_outputs = add_outputs)
            print('Runtime -', timeit.default_timer() - start, 'sec. \n')

        # Saving the generated dictionary with metadata to a JSON file
        Outputs_path = os.path.join(self.directory, 'Outputs')
        json_path = os.path.join(Outputs_path, 'Metadata.json')
        with open(json_path, 'w') as json_file:
            json.dump(self.metadata, json_file)

    # A function that allows to fill in gaps using the nearest neighbor interpolation method
    def nn_interpolation(self, key_values = {'gap': -100.0, 'skip': -200.0, 'NoData': -32768.0}):

        # Defining flags for gaps, skips and NoData
        self.gap = key_values.get('gap')
        self.skip = key_values.get('skip')
        self.nodata = key_values.get('NoData')

        files = os.listdir(self.Inputs_path)
        files.sort()

        for file in files:
            start = timeit.default_timer()
            matrix = np.load(os.path.join(self.Inputs_path, file))

            shape = matrix.shape
            all_pixels = shape[0] * shape[1]
            # If all the values in the matrix are gaps, etc., then the layer is not filled in
            if all_pixels - ((matrix == self.gap).sum() + (matrix == self.skip).sum() + (matrix == self.nodata).sum()) <= 10:
                print('No calculation for matrix', file[:-4])
            # If there are no gaps in the image, it is saved as filled in without using the algorithm
            elif (matrix == self.gap).sum() == 0:
                print('No gaps in matrix', file[:-4])
                where_to_save = os.path.join(self.Outputs_path, file)
                np.save(where_to_save, matrix)
            else:
                print('Calculations for matrix', file[:-4])
                # Copy of the matrix for applying all masks
                copy_matrix = np.copy(matrix)

                # Interpolation is performed for all flags - they must be marked as gaps
                matrix[matrix == self.skip] = self.gap
                matrix[matrix == self.nodata] = self.gap

                # Let's see what the matrix looks like at the moment
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

                # The return of previously removed flags
                GD1[copy_matrix == self.skip] = self.skip
                GD1[copy_matrix == self.nodata] = self.nodata

                # Saving the matrix to the outputs folder
                where_to_save = os.path.join(self.Outputs_path, file)
                np.save(where_to_save, GD1)
                print('Runtime -', timeit.default_timer() - start, 'sec. \n')