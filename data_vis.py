import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from datetime import timedelta
import pickle
from random import sample 
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
plt.style.use(astropy_mpl_style)

from dataset_prep import train_test_split, valid_flare_events, batch_gen_train, batch_gen
from utils import print_ar_stats, remove_c_flares, small_data, print_flare_stats, find_intersection

class dataset_struct():
    def __init__(self):
        self.path_goes = r'C:\datasets\SMARP\\'
        self.len_seq = 12 * 5 * 10 # 10 hours of features
        self.len_pred = 24 * 60 # prediction for 24 hours after
        self.batch_size = 16
        
dataset_mdi = dataset_struct()
dataset_mdi.path_harp =  r'C:\datasets\SMARP\header\\'
dataset_mdi.features = ['USFLUX', 'MEANGBZ', 'R_VALUE']
dataset_mdi.cadence = 96


valid_flare_events(dataset_mdi)
train_test_split(dataset_mdi)

dataset_mdi.mean = 0
dataset_mdi.std = 1

batch = batch_gen(dataset_mdi.valid_events[0:100], [[False, True]] * 100, dataset_mdi, dataset_mdi)




path_smarp_img = r'C:\datasets\SMARP\image\\'
path_sharp_img = r'Z:\data2\SHARP_720s\image\\'
path_img = r'C:\datasets\magnetogram_data\magnetogram_data\image\\'
path_goes = r'C:\datasets\SMARP\\'
path_harp = r'C:\datasets\magnetogram_data\magnetogram_data\header\\'
path_harp_mdi = r'C:\datasets\SMARP\header\\'

features_mdi = ['USFLUX', 'MEANGBZ', 'R_VALUE']

goes = pd.read_csv(path_goes + 'GOES.csv', delimiter=',')
goes['peak_time'] = pd.to_datetime(goes['peak_time'])
goes = goes.dropna(subset=['goes_class'])
goes = goes.loc[(goes['goes_class'].str.len() == 4)]
goes = goes.drop(goes.iloc[np.where(goes['goes_class'].str[:1] == 'A')[0]].index)

intensity_letter = goes['goes_class'].str[:1]
intensity = np.zeros(len(intensity_letter))
intensity[np.where(intensity_letter == 'B')[0]] = 1e-7
intensity[np.where(intensity_letter == 'C')[0]] = 1e-6
intensity[np.where(intensity_letter == 'M')[0]] = 1e-5
intensity[np.where(intensity_letter == 'X')[0]] = 1e-4
intensity = intensity * goes['goes_class'].str[1:].astype(float)
intensity = np.log(intensity)
goes['intensity'] = intensity




mdi_harp_times = []
for file in os.listdir(path_harp_mdi):
    harp = pd.read_csv(path_harp_mdi + file, delimiter=',')
    harp['T_REC'] = harp['T_REC'].str[:-4]
    harp['T_REC'] = harp['T_REC'].str.replace('_', ' ')
    start_time = pd.to_datetime(harp['T_REC'].iloc[0])
    end_time = pd.to_datetime(harp['T_REC'].iloc[-1])
    mdi_harp_times.append([file, start_time, end_time])
    
hmi_harp_times = []
for file in os.listdir(path_harp):
    harp = pd.read_csv(path_harp + file, delimiter=',')
    harp['T_REC'] = harp['T_REC'].str[:-4]
    harp['T_REC'] = harp['T_REC'].str.replace('_', ' ')
    start_time = pd.to_datetime(harp['T_REC'].iloc[0])
    end_time = pd.to_datetime(harp['T_REC'].iloc[-1])
    hmi_harp_times.append([file, start_time, end_time])
    
    


if True:
    image_file = fits.open(path_smarp_img + '000328' + '\\' + 'su_mbobra.smarp_cea_96m.328.19970413_111200_TAI.magnetogram.fits')
    data = image_file[1].data
    plt.figure()
    plt.imshow(data, cmap='gray')
    
    
    
    #plt.colorbar()


if False:
    video = np.load(path_img + 'HARP000040.npy')
    img = video[0,:,:,2]
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img, cmap = 'gray')
    
    
    
    image_file = fits.open(path_sharp_img + '000040' + '\\' + 'hmi.sharp_cea_720s.40.20100528_071200_TAI.magnetogram.fits')
    data = image_file[1].data
    
    res = []
    for dr in os.listdir(path_sharp_img)[940:]:
        file = os.listdir(path_sharp_img + dr)[0]
        image_file = fits.open(path_sharp_img + dr + '\\'  + file)
        res.append(list(image_file[1].data.shape))
        print(dr)
        
    res = np.array(res)
    mean = np.mean(res, axis = 0)   # 313, 643
    std = np.std(res, axis = 0)     # 188, 387
    min_res = np.min(res, axis = 0)  # 21, 54
    max_res = np.max(res, axis = 0)  # 2860, 3359
    
    
    plt.figure()
    plt.imshow(data, cmap='gray')
    
    
    # res_list = []
    # for file in os.listdir(path_img):
    #     video = np.load(path_img + file)
    #     res_list.append([video.shape[1], video.shape[2]])
    # res_list = np.array(res_list)
    # Mean -> (37, 74)
    
    
harp = pd.read_csv(path_harp + 'HARP007075_ATTRS.csv', delimiter=',')

noaa = harp['NOAA_AR'].unique()
if len(noaa) == 1:
    noaa = noaa[0]



if False:
    feature = 'SAVNCPP'
    plt.figure()
    harp_vis = harp[['T_REC', feature]]
    harp_vis['T_REC'] = harp_vis['T_REC'].str[:-4]
    harp_vis['T_REC'] = harp_vis['T_REC'].str.replace('_', ' ')
    harp_vis['T_REC'] = pd.to_datetime(harp_vis['T_REC'])
    plt.plot(harp_vis['T_REC'], harp_vis[feature])
    
    goes_vis = goes.loc[goes['NOAA_ar_num'] == noaa]
    b = goes_vis['intensity'].min()
    a = harp_vis[feature].max() / (goes_vis['intensity'].max() - goes_vis['intensity'].min())
    
    goes_vis_filt = []
    delta = timedelta(hours=12)
    for i in range(len(goes_vis)):
        start = goes_vis.iloc[i]['peak_time'] - delta
        end = goes_vis.iloc[i]['peak_time'] + delta
        neighbors = goes_vis[(goes_vis['peak_time'] >= start) & (goes_vis['peak_time'] < end)]
        neighbors.drop(goes_vis.iloc[i].name, inplace = True)
        if all(goes_vis.iloc[i]['intensity'] > neighbors['intensity']):
            goes_vis_filt.append(goes_vis.iloc[i])
    goes_vis = pd.DataFrame(goes_vis_filt)
        
    
    
    ind = goes_vis['class'].str[:1] == 'B'
    plt.scatter(goes_vis.loc[ind]['peak_time'], (goes_vis.loc[ind]['intensity'] - b) * a, c = 'blue')
    ind = goes_vis['class'].str[:1] == 'C'
    plt.scatter(goes_vis.loc[ind]['peak_time'], (goes_vis.loc[ind]['intensity'] - b) * a, c = 'green')
    ind = (goes_vis['class'].str[:1] == 'M') | (goes_vis['class'].str[:1] == 'X')
    plt.scatter(goes_vis.loc[ind]['peak_time'], (goes_vis.loc[ind]['intensity'] - b) * a, c = 'red')
    plt.xticks(rotation=45)
    plt.show()


