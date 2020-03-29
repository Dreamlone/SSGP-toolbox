import gdal, json, os, osr
import numpy as np

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