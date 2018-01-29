from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv
import shapely.wkt
import numpy as np
import cv2
import osr
import time
from osgeo import gdal
import os
import scipy
path_main = 'D:\building\Paris_DataSpaceNet\AOI_3_Paris_Test_public\AOI_3_Paris_Test_public'
import glob, os
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
core_of_computer = multiprocessing.cpu_count()
os.chdir(path_main+'\RGB-PanSharpen')
def cal_8band(file):
    print(file)
    ds = gdal.Open(path_main+r'\RGB-PanSharpen'+"\\"+file)
    data = ds.ReadAsArray()
    data3 = data.copy()
    # plt.imshow(grayImg)
    # plt.show()
    # img2 = np.array([datax,datax,datax,datax]).swapaxes(0,1).swapaxes(1,2)
    img = np.array(data3).swapaxes(0,1).swapaxes(1,2)
    # hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # r_min = min(data3[0])
    # r_max = max(data3[0])
    # g_min = min(data3[1])
    # g_max = max(data3[1])
    # b_min = min(data3[2])
    # b_max = max(data3[2])
    # hsv_image = rgb_to_hsv(img)
    # h, s, v = cv2.split(hsv_image)
    # band_gray = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
    bandstats = {k: dict(max=0, min=0) for k in range(3)}
    for i in range(3):
        bandstats[i]['min'] = scipy.percentile(data3[i], 2)
        bandstats[i]['max'] = scipy.percentile(data3[i], 98)
    
    for chan_i in range(3):
        min_val = bandstats[chan_i]['min']
        max_val = bandstats[chan_i]['max']
        data3[chan_i] = np.clip(data3[chan_i], min_val, max_val)
        data3[chan_i] = (data3[chan_i] - min_val)
    tgi_band = (data3[1] - 0.39*data3[0] - 0.61*data3[2])*10
    grayImg = 0.0722*data3[0] + 0.7152*data3[1] + 0.2126*data3[2]
    # plt.imshow(tgi_band)
    # plt.show()
    data4 = data3.copy()
    data4 = np.asarray(data4, dtype=np.float32)
    for chan_i in range(3):
        min_val = bandstats[chan_i]['min']
        max_val = bandstats[chan_i]['max']
        data4[chan_i] = (data4[chan_i]/(max_val-min_val))
    img_1 = np.array(data4).swapaxes(0,1).swapaxes(1,2)
    hsv_image = rgb_to_hsv(img_1)
    # plt.imshow(hsv_image)
    # plt.show()
    data5 = np.array(hsv_image).swapaxes(2,1).swapaxes(1,0)
    img2 = np.array([data3[0],data3[1],data3[2],data5[0]*360,data5[1]*100,data5[2]*100,tgi_band,grayImg]).swapaxes(0,1).swapaxes(1,2)

    output = path_main+r'\MUL-PanSharpen_2'+"\\"+file
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output,ds.RasterXSize,ds.RasterYSize,(img2.shape[2]),gdal.GDT_UInt16)#gdal.GDT_Byte/GDT_UInt16
    for i in range(1,img2.shape[2]+1):
        dst_ds.GetRasterBand(i).WriteArray(img2[:,:,i-1])
        dst_ds.GetRasterBand(i).ComputeStatistics(False)
    dst_ds.SetProjection(ds.GetProjection())
    dst_ds.SetGeoTransform(ds.GetGeoTransform())
    return 0
def main():
    list_file = []
    for file in glob.glob("*.tif"):
        list_file.append(file)
    p_cnt = Pool(processes=core_of_computer)
    result = p_cnt.map(partial(cal_8band), list_file)
    p_cnt.close()
    p_cnt.join()       
    

if __name__=="__main__":
    main()