import gdal, json, os, osr
import numpy as np
import scipy.spatial
import random

def reconstruct_geotiff (npy_path, metadata_path, output_path):
    """
    Function for converting a matrix from npy format to geotiff

    :param npy_path: path to the file with the npy extension
    :param metadata_path: path to the file with the geotiff extension
    :return: generates a geotiff file in the specified folder
    """

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

def cellular_expand (matrix, biome_matrix, gap = -100.0, iter = 10):
    """
    "Expansion" of the cloud territory using a probabilistic cellular automaton
    The goal of the algorithm is to identify pixels shaded by the cloud in the image

    :param matrix: numpy matrix with land surface temperature
    :param biome_matrix: matrix which allows us to divide matrix cells into groups
    :param gap: gap code in the matrix
    :param iter: number of iterations
    :return: matrix with extended cloud boundaries marked with a gap code
    """

    # A function that defines all the transformations that must occur with the matrix
    def step(matrix, biome_matrix, gap = gap):
        # We assign a gap value to the matrix with biomes in those places where there are clouds at the momen
        biome_matrix[matrix == gap] = gap
        next_matrix = np.copy(matrix)

        # The calculation of the threshold
        masked_array = np.ma.masked_where(matrix == gap, matrix)
        minimum = np.min(masked_array)
        maximum = np.max(masked_array)
        # The temperature range in the image - this will be required in the future for normalization
        amplitide = maximum - minimum

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # If the pixel is located in the top row
                if i == 0:
                    # The leftmost cell
                    if j == 0:
                        # Truncated Moore's neighborhood
                        arr = matrix[i: i + 2, j: j + 2]
                    # The rightmost cell
                    elif j == (matrix.shape[1] - 1):
                        arr = matrix[i: i + 2, j - 1: j + 1]
                    # The rest of the row
                    else:
                        arr = matrix[i: i + 2, j - 1: j + 2]

                # If the pixel is located in the left column
                elif j == 0:
                    # Bottom cell
                    if i == (matrix.shape[0] - 1):
                        arr = matrix[i - 1: i + 1, j: j + 2]
                    # The rest of the row
                    else:
                        arr = matrix[i - 1: i + 2, j: j + 2]

                # If the pixel is located in the bottom line
                elif i == (matrix.shape[0] - 1):
                    # The rightmost cell
                    if j == (matrix.shape[1] - 1):
                        arr = matrix[i - 1: i + 1, j - 1: j + 1]
                    # The rest of the row
                    else:
                        arr = matrix[i - 1: i + 1, j - 1: j + 2]

                # If the pixel is located in the right column
                elif j == (matrix.shape[1] - 1):
                    arr = matrix[i - 1: i + 2, j - 1: j + 1]

                # If the pixel is covered by a cloud
                elif matrix[i, j] == gap:
                    arr = np.zeros((2, 2))
                else:
                    # Moore's neighborhood
                    arr = matrix[i - 1: i + 2, j - 1: j + 2]

                # Checking if there is a cloud in the neighborhood
                id_cloud = np.argwhere(arr == gap)
                # If there is and the pixel itself is not covered by the cloud
                # then we compare the pixel temperature with the average temperature of the neighborhood
                if len(id_cloud) != 0 and matrix[i, j] != gap:

                    ######################################
                    #  A probabilistic approach is used  #
                    ######################################

                    # Generating a random number in the range from 0 to 1
                    prob_number = random.random()

                    # Select a number such that if prob_number exceeds it, the pixel will become cloudy in the next step
                    if len(id_cloud) >= 8:
                        # The closer the fact_number value is to 0, the more likely it is that the pixel will become a "cloud"
                        fact_number = 0.8
                    elif len(id_cloud) == 7:
                        fact_number = 0.85
                    elif len(id_cloud) == 6:
                        fact_number = 0.9
                    elif len(id_cloud) == 5:
                        fact_number = 0.95
                    else:
                        fact_number = 0.99

                    # We add a certain value to the initially defined probability value
                    # depending on the temperature of the considering pixel

                    # Biome (pixel group) code for a cell
                    biome_code = biome_matrix[i, j]
                    # Indexes of points that fall into this biome and are not currently omitted
                    coords = np.argwhere(biome_matrix == biome_code)
                    # If there are not enough points in the BIOS, then we take all known points
                    if len(coords) < 41:
                        coords = np.argwhere(matrix != gap)
                    else:
                        pass

                    # Coordinates of the pixel from which we calculate distances
                    target_pixel = np.array([[i, j]])
                    # Vector of calculated distances from the target pixel to all other pixels
                    distances = scipy.spatial.distance.cdist(target_pixel, coords)[0]

                    # Selecting the nearest 40 pixels
                    new_coords = []
                    for iter in range(0, 40):
                        # Which index in the coords array has the element with the smallest distance from the target pixel
                        index_min_dist = np.argmin(distances)

                        # Coordinates of this pixel in the matrix
                        new_coord = coords[index_min_dist]

                        # Adding coordinates
                        new_coords.append(new_coord)

                        # Replacing the minimum element from the array with a very large number
                        distances[index_min_dist] = np.inf

                    # Recording the temperature of these nearest 15 pixels
                    list_tmp = []
                    for coord in new_coords:
                        list_tmp.append(matrix[coord[0], coord[1]])
                    list_tmp = np.array(list_tmp)

                    # Take the median value
                    median_local_tmp = np.median(list_tmp)

                    # The normalized temperature difference (to the amplitude)
                    value = ((matrix[i, j] - median_local_tmp) / amplitide)
                    # If the temperature is higher than the white median, then do nothing
                    if value >= 0:
                        pass
                    # If the number of neighboring cloud pixels is strictly less than 3, then we also do not consider this pixel
                    elif len(id_cloud) < 3:
                        pass
                    # The lower the temperature, the more likely it is that a given pixel will become a "cloud" in the next step
                    else:
                        fact_number = fact_number + value

                        # If the generated number is greater than fact_number, the pixel becomes cloudy
                        if prob_number >= fact_number:
                            next_matrix[i, j] = gap

        # Return the matrix that will be obtained in the next step
        return (next_matrix)

    # The specified number of iterations applies the "cloud expansion" procedure"
    for iteration in range(0, iter):
        matrix = step(matrix, biome_matrix)
    return(matrix)