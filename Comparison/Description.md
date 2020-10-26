
# Comparison of gap-filling algorithms

The "Comparison" folder contains land surface temperature (LST) data for two test territories: Madrid (Spain) and Vladivostok (Russia). LST data received from the MODIS sensor - product MOD11A1, daily composites from Terra satellite. The temperature in the matrices in degrees Kelvin.

Spatial coverage of such images is 1 degree of latitude per 1 degree of longitude for each area:
* "Madrid": 5 – 4 °W, 39 – 40 °N;
* "Vladivostok": 132 – 133 °E, 44 – 45 °N;

For Vladivostok case, 21 layers were prepared (data from September 12 to 18 for 2017, 2018, 2019), for the territory of Madrid - 28 (data from August 31 to September 6 for 2017, 2018, 2019, 2020). Validation was performed on the image for September 15, 2019 for Vladivostok and for September 3, 2019 for Madrid. Each image generated 8 types of gaps ranging in size from 5 to 94%.

## Description of files and folders

The set of folders and files in the "Madrid" and "Vladivostok" folders is similar. Let's look at the file location using the "Madrid" folder as an example:
* actual_matrix - folder where the matrix with actual LST values is located without gaps;
* additional_matrices - additional matrices for this territory that may be useful (biomes matrix, elevation, etc.)
* inputs - matrices with gaps to fill in. The filename "20190903T000000_5_percent.npy" means that this matrix has a gap that occupies 5 percent of the territory. Missing pixels are marked with the value -100.0;
* model_outputs - matrices without gaps, results of applying algorithms for each type of gap;
> * CRAN_gapfill - layers without gaps that were filled in using the "CRAN gapfill" algorithm, the filename "20190903T000000_5_percent.npy" means that this matrix was obtained by restoring a gap that occupied 5 percent of the territory;
> * SSGP-toolbox - layers without gaps that were filled in using the "SSGP-toolbox" algorithm;
> * gapfilling_rasters - layers without gaps that were filled in using the "gapfilling_rasters" algorithm;
* training_sample - matrices that can be used to train the algorithm.

Biomes matrix and elevation were derived from additional layers of the Sentinel-3 LST SLSTR level 2 product.

SSGP-toolbox configuration for filling this layers: 
> method = 'SVR', predictor_configuration = 'Biome', hyperparameters = 'RandomGridSearch', add_outputs = False

## Comparison result

#### Table for Madrid case. Mean Absolute Error, Celsius degree
| Algorithm / Gap size | 5%  | 8%  |  17%  |  29%  |  39%  |  50%  |   78%  |  94%  |
| :-----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| SSGP-toolbox | 0,53 | 0,89 | 0,76 | 0,79 | 0,69 | 0,84 | 1,04 | 0,97 |
| CRAN gapfill | 1,03 | 1,19 | 1,39 | 1,17 | 1,11 | 1,19 | 1,32 | 1,42 |
| Gapfilling rasters | 1,37 | 1,7 | 1,56 | 1,57 | 1,76 | 2,15 | 2,66 | 2,94 |

#### Figure for Madrid case
![Madrid.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/madrid_case.png)

#### Table for Vladivostok case. Mean Absolute Error, Celsius degree
| Algorithm / Gap size | 5% | 10% | 15% | 28% | 44% | 50% | 74% | 93% |
| :-----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| SSGP-toolbox | 0,3  |	0,31 |	0,36 |	0,32 |	0,47 |	0,36 |	0,5  | 0,68 |
| CRAN gapfill | 0,47 |	0,36 |	0,58 |	0,43 |	0,59 |	0,55 |	0,84 |	0,73 |
| Gapfilling rasters | 0,67	| 0,63 | 0,66 |0,72 | 0,77 | 0,81 |	0,85 |	1,24 |

#### Figure for Vladivostok case
![Vladivostok.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/vladivostok_case.png)
