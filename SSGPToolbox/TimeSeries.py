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

    # When initializing the class, we must specify
    # directory  --- the directory where the layers to be placed in the time series are located
    # key_values --- dictionary with omissions and irrelevant values
    # averaging  --- is it needed to average layers that fall within the same time interval ('None', 'weighted', 'simple')
    def __init__(self, directory, key_values = {'gap': -100.0, 'skip': -200.0}, averaging = 'None'):
        self.directory = directory
        self.averaging = averaging

        # All skip values in the matrices will be perceived as those that do not need to be filled in
        self.skip = key_values.get('skip')
        # If the gaps in the time series are not filled in, the gaps will be filled in with the gap value
        self.gap = key_values.get('gap')

        # Let's make a sorted list of all the files that are in the folder
        layers = os.listdir(self.directory)
        layers.sort()

        # Creating a dictionary with matri
        matrices_dictionary = {}
        keys = []
        for layer in layers:
            key = datetime.datetime.strptime(layer[:-4], '%Y%m%dT%H%M%S')
            keys.append(key)
            matrix_path = os.path.join(self.directory, layer)

            matrix = np.load(matrix_path)
            matrices_dictionary.update({key: matrix})

        # An ordered list of file names
        self.keys = keys
        # Dictionary with matrices
        self.matrices_dictionary = matrices_dictionary

    # The private method allows to bring data to the specified time step
    # timestep      --- the time interval after which layers will be placed on the time series grid
    # return tensor --- a multidimensional matrix in which each layer takes its place on the time series
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

        # Time series synthesis with regular margins
        times = pd.date_range(start = start_date, end = end_date, freq = timestep)
        tensor = []
        tensor_timesteps = []
        for i in range(0, len(times) - 1):
            # We take the middle of this time period
            time_interval = (times[i + 1] - times[i])/2
            centroid = times[i] + time_interval

            # We are considering which layers may be suitable for this interval
            suitable_keys = []
            for key in self.keys:
                if key >= times[i] and key < times[i + 1]:
                    suitable_keys.append(key)

            if len(suitable_keys) == 0:
                # First check: if we couldn't find a layer for the last segment, then we shouldn't generate it
                if i == len(times) - 2:
                    break
                else:
                    # In this case, we will generate the layers ourselves while we fill the "blank" with the gap value
                    matrix = np.full((rows, cols), self.gap)
            elif len(suitable_keys) == 1:
                # If there is only one layer, it is added
                main_key = suitable_keys[0]
                matrix = self.matrices_dictionary.get(main_key)
            else:
                # If averaging is not required, select the layer that is closest to the time interval
                if self.averaging == 'None':
                    # Select a single layer that is closest to the interval we are interested in
                    distances = []
                    for element in suitable_keys:
                        if element < centroid:
                            distances.append(centroid - element)
                        elif element > centroid:
                            distances.append(element - centroid)
                        else:
                            distances.append(0)
                    distances = np.array(distances)
                    # Looking for the index of the smallest element
                    min_index = np.argmin(distances)
                    # We get the appropriate layer based on the index
                    main_key = suitable_keys[min_index]
                    matrix = self.matrices_dictionary.get(main_key)

                # If the averaging procedure with the 'simple' parameter is selected, simple averaging will be performed
                elif self.averaging == 'simple':
                    # Creating a small matrix for this time interval
                    matrix_batch = []
                    for element in suitable_keys:
                        step_matrix = self.matrices_dictionary.get(element)
                        matrix_batch.append(step_matrix)
                    matrix_batch = np.array(matrix_batch)

                    # Creating a "dummy" matrix filled with zeros
                    matrix = np.zeros((rows, cols))
                    # Perform the averaging procedure for each pixel
                    for row_index in range(0, matrix_batch[0].shape[0]):
                        for col_index in range(0, matrix_batch[0].shape[1]):
                            mean_value = np.mean(matrix_batch[:, row_index, col_index])
                            # Imputing the value
                            matrix[row_index, col_index] = mean_value

                # If the averaging procedure with the "weighted" parameter is selected,
                # all layers that fall within the time interval are averaged with weights
                elif self.averaging == 'weighted':

                    # Creating a small matrix for this time interval
                    matrix_batch = []
                    for element in suitable_keys:
                        step_matrix = self.matrices_dictionary.get(element)
                        matrix_batch.append(step_matrix)
                    matrix_batch = np.array(matrix_batch)

                    # It is needed to determine how close the layers are to the timestamp
                    distances = []
                    for element in suitable_keys:
                        if element < centroid:
                            distances.append(centroid - element)
                        elif element > centroid:
                            distances.append(element - centroid)
                        else:
                            distances.append(0)
                    distances = np.array(distances)

                    # Let's use a function that returns an array of element indexes if we sort the distances array
                    distances_id_sorted = np.argsort(distances)
                    # Now that we know what place each distance occupies in the array by value, we will set the weights
                    weights = np.copy(distances)
                    weight = len(distances)
                    # Weights are assigned to each element depending on the distance (the closer the element is, the greater the weight)
                    for index in distances_id_sorted:
                        weights[index] = weight
                        weight -= 1

                    # Creating a "dummy" matrix filled with zeros
                    matrix = np.zeros((rows, cols))
                    # Perform the averaging procedure for each pixel
                    for row_index in range(0, matrix_batch[0].shape[0]):
                        for col_index in range(0, matrix_batch[0].shape[1]):
                            mean_value = np.average(matrix_batch[:, row_index, col_index], weights = weights)
                            # Imputing the value
                            matrix[row_index, col_index] = mean_value

            # Adding a matrix
            tensor.append(matrix)
            tensor_timesteps.append(centroid)
        tensor = np.array(tensor)
        return(tensor, tensor_timesteps)

    # Private method for filling in gaps in a time series
    def __gap_process(self, timeseries, filling_method, n_neighbors = 5):
        # Indexes of points on the time series to be filled in
        i_gaps = np.argwhere(timeseries == self.gap)
        i_gaps = np.ravel(i_gaps)

        # Depending on the chosen strategy for filling in time series gaps
        # Gaps in the time series are not filled in
        if filling_method == 'None':
            pass
        elif filling_method == None:
            pass
        # The gaps are filled in with local medians
        elif filling_method == 'median':
            # For each pass in the time series, we find n_neighbors " known neighbors"
            for gap_index in i_gaps:
                # Indexes of known elements (updated at each iteration)
                i_known = np.argwhere(timeseries != self.gap)
                i_known = np.ravel(i_known)

                # Based on the indexes we calculate how far from the gap the known values are located
                id_distances = np.abs(i_known - gap_index)

                # Now we know the indices of the smallest values in the array, so sort indexes
                sorted_idx = np.argsort(id_distances)
                # n_neighbors nearest known values to gap
                nearest_values = []
                for i in sorted_idx[:n_neighbors]:
                    # Getting the index value for the series
                    time_index = i_known[i]
                    # Using this index, we get the value of each of the "neighbors"
                    nearest_values.append(timeseries[time_index])
                nearest_values = np.array(nearest_values)

                est_value = np.nanmedian(nearest_values)
                timeseries[gap_index] = est_value

        elif filling_method == 'poly':
            # For each gap, we build our own low-degree polynomial
            for gap_index in i_gaps:
                # Indexes of known elements (updated at each iteration)
                i_known = np.argwhere(timeseries != self.gap)
                i_known = np.ravel(i_known)

                # Based on the indexes we calculate how far from the gap the known values are located
                id_distances = np.abs(i_known - gap_index)

                # Now we know the indices of the smallest values in the array, so sort indexes
                sorted_idx = np.argsort(id_distances)
                # Nearest known values to gap
                nearest_values = []
                # And their indexes
                nearest_indices = []
                for i in sorted_idx[:n_neighbors]:
                    # Getting the index value for the series - timeseries
                    time_index = i_known[i]
                    # Using this index, we get the value of each of the "neighbors"
                    nearest_values.append(timeseries[time_index])
                    nearest_indices.append(time_index)
                nearest_values = np.array(nearest_values)
                nearest_indices = np.array(nearest_indices)

                # Local approximation by an n-th degree polynomial
                local_coefs = np.polyfit(nearest_indices, nearest_values, 2)

                # We estimate our point according to the selected coefficients
                est_value = np.polyval(local_coefs, gap_index)
                timeseries[gap_index] = est_value

        return(timeseries)

    # timestep --- the time interval after which layers will be placed on the time series grid
    def make_time_series(self, timestep = '12H', filling_method = 'None'):
        # From the specified directory and using the selected time step, we form a multidimensional matrix and time steps
        tensor, tensor_timesteps = self.__sampling(timestep = timestep)

        # We build our own model for each pixel in a row
        for row_index in range(0, tensor[0].shape[0]):
            for col_index in range(0, tensor[0].shape[1]):
                # Getting a time series for a specific pixel
                pixel_timeseries = tensor[:, row_index, col_index]

                # If there is a gap value in the time series, it will be written to all cells
                if any(value == self.skip for value in pixel_timeseries):
                    pixel_timeseries = np.full(len(pixel_timeseries), self.skip)

                # If there is at least one gap in the time series, it must be filled in
                elif any(value == self.gap for value in pixel_timeseries):
                    # Using the private __gap_process method, we fill in the time series gap
                    pixel_timeseries = self.__gap_process(timeseries = pixel_timeseries, filling_method = filling_method)

                # If there are no gaps in the row, then you don't need to fill in anything
                else:
                    pass

                # The time series is filled, so we write the filled time series to a multidimensional matrix
                tensor[:, row_index, col_index] = pixel_timeseries

        return(tensor, tensor_timesteps)


    # A method that allows to save the results as .npy matrices
    # save_path --- the folder to which you want to store the result
    def save_npy(self, tensor, tensor_timesteps, save_path):
        # Create folder 'Discretisation_output'; if there is, then use the existing one
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

        # Now filled in matrices will be saved to this folder
        for index in range(0, len(tensor)):
            matrix = tensor[index]
            time = tensor_timesteps[index]
            # Converting the datetime format to a string
            time = time.strftime('%Y%m%dT%H%M%S')

            npy_name = os.path.join(save_path, time)
            np.save(npy_name, matrix)


    # A method that allows to save the results as a netCDF file
    # save_path --- the folder to which you want to store the result
    def save_netcdf(self, tensor, tensor_timesteps, save_path):

        # Create folder 'Discretisation_output'; if there is, then use the existing one
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)

        netCDF_name = os.path.join(save_path, 'Result.nc')

        # Converting timestamps to a string data type
        str_tensor_timesteps = []
        for time in tensor_timesteps:
            # Converting the datetime format to a string
            str_tensor_timesteps.append(time.strftime('%Y%m%dT%H%M%S'))
        str_tensor_timesteps = np.array(str_tensor_timesteps)

        # Generates a netCDF file
        root_grp = Dataset(netCDF_name, 'w', format='NETCDF4')
        root_grp.description = 'Discretized matrices'

        # Dimensions for the data to be written to the file
        dim_tensor = tensor.shape
        root_grp.createDimension('time', len(str_tensor_timesteps))
        root_grp.createDimension('row', dim_tensor[1])
        root_grp.createDimension('col', dim_tensor[2])

        # Writing data to a file
        time = root_grp.createVariable('time', 'S2', ('time',))
        data = root_grp.createVariable('matrices', 'f4', ('time', 'row', 'col'))

        data[:] = tensor
        time[:] = str_tensor_timesteps

        root_grp.close()