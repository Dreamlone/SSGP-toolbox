# Theoretical basis

## Spatial Gapfilling
[class - Gapfiller](https://github.com/Dreamlone/SSGP-toolbox/blob/master/SSGPToolbox/Gapfiller.py)

The algorithm relies on values in known pixels of the same matrix as the gap in order to estimate the value in the gap.

The algorithm builds its own model separately for each pixel, then uses different methods for selecting predictors, depending on the set parameters. The illustration shows a simplified example of how the algorithm generates a training sample.

![Construction.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_3_Construction.png)

In the training sample can also be present in the gaps. In this case, the missing values are filled in with the median value for a time series that is compiled for each pixel separately.

### Inaccuracies information 

Each pixel uses its own machine learning algorithm and generates its own training sample. K Fold cross validation is performed for this sample. The resulting cross-validation accuracy value for all filled pixels is averaged for each matrix with gaps.

### Time complexity

Based on the results of experiments with the model, it was found that the time complexity of the algorithm is sublinear.

![Complexity.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_4_Complexity.png)

### Verification

The algorithm was verified using thermal remote sensing data from the Sentinel-3 satellite system. 6 multi-time images of the same territory were selected. A gap of a certain size and shape was generated for each of the 6 images. A total of 8 types of gaps were generated. The results of these tests can be seen below. In most cases, the algorithm was wrong by less than 1 Celsius degree, despite the fact that the average temperature spread in the gap was about 10 degrees. You can see the results of algorithm verification below.

![Results.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_5_Results.png)

You can see from the graph that the accuracy of data recovery depends more on the distribution of the temperature field in the image than on the size of the gap.

## Time Series Gapfilling
[class - Discretizator](https://github.com/Dreamlone/SSGP-toolbox/blob/master/SSGPToolbox/TimeSeries.py)

To fill in the gaps in the time series, we use a local approximation by a polynomial of the 2nd degree over the known 5 neighboring points. An illustration of how the algorithm works can be seen below.
![TimeSeries1.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_7_TS.png)

### Verification
This approach of filling in gaps is most effective in the case of smooth time series. So, local approximation by polynomials restores NDVI fluctuations very well.
![TimeSeries1.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_8_TS.png)

## Identification of distorted values

###### The algorithm was developed to identify anomalies in temperature fields (therefore, using it in fields with other parameters may lead to incorrect results)

There is a feature in the processing of Land surface temperature data. Pixels may be shadowed by a cloud, resulting in their temperature being slightly lower than the actual temperature in the image. Also, the temperature in pixels close to the cloud can be explained as follows. The satellite signal captures the temperature in a layer of 1-2 cm of the earth's surface. This layer of soil heats up and cools down quickly. If the sunshine does not fall on it for a few minutes, then the recorded temperature during these minutes also changes quite quickly. Clouds are moving objects, block the flow of solar radiation for some pixels for some time during their movement. 

Thus, the cloud moving over the territory can close pixels for several minutes or hours along the path of its movement. Pixels that were in the path of the cloud may lose several degrees. The satellite system takes a picture. After that, the cloud moves on, and the pixels begin to heat up as quickly as they cooled down, and after a few minutes or hours they return to their "former" state. Since the implemented gapfilling algorithm relies on known pixel values in the image, when using "shaded" pixels as predictors, the restored values in the gap will also be underestimated. 

If the study is to fill in gaps in order to get images that characterize the average temperature distribution in the absence of clouds, rather than at the specific time, then it is appropriate to use an approach to exclude pixels shaded by clouds. A cellular automaton is used to detect such shaded pixels. The algorithm is auxiliary.

<p align="center">
    
![Cellular_schema.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_9_cellular.png)

</p>


A probabilistic approach is used to determine shaded pixels. The probability of assigning a pixel to a shaded one is proportional to the number of neighboring (Moore's neighborhood) pixels covered by the cloud. The probability of assigning a pixel to a shaded one is greater for those pixels whose temperature is lower than the median temperature value for pixels from the same biome.

![Cellular_app.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_10_cellular.png)

![Cellular_reason.png](https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/master/Supplementary/images/rm_11_cellular.png)
