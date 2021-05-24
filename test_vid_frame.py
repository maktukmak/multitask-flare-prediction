import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataset_prep import dataset_struct
from model_rnn import model_rnn
import tensorflow as tf
from utils import  print_flare_stats,  print_intersect_flare_stats
from utils_training import trainer_struct
from random import sample
import os.path

import seaborn as sns
sns.set_theme()

tf.autograph.set_verbosity(0)

tf.config.experimental_run_functions_eagerly(True)

class dataset_cfg_struct():
    def __init__(self):
        self.use_cached_preprocessing = True
        self.use_cached_harps = True
        
        self.len_seq = 12 * 5 * 10 # 10 hours of features
        self.len_pred = 24 * 60 # prediction for 24 hours after
        self.max_flare_filtering = True  # Filter flares within 24 hours window 
        self.max_flare_window_drop = False # If true: Drop filtered flares else: replace with max flare
        self.remove_C = True
        
        self.hmi_split_opt = 'date'  #'date', 'random', 'no'
        self.mdi_split_opt = 'random'  #'date', 'random', 'no'


dataset_cfg = dataset_cfg_struct()

dataset = dataset_struct(dataset_cfg)
dataset.load_datasets()
dataset.preprocess_datasets()


dataset.dataset_hmi.img_on = True
dataset.dataset_hmi.img_res = [128, 128]


#def valid_flare_events_check_images



# Remove events without image snaps
if True:
    file_vec = []
    for i, data in enumerate(dataset.dataset_hmi.valid_events):
        file = dataset.data_to_image_file(data, dataset.dataset_hmi)
        path = dataset.dataset_hmi.path_img + data[0][4:10]
        if not os.path.exists(path + '\\' + file):
            file_vec.append(file)
            dataset.dataset_hmi.valid_events.remove(data)
        print(i)
    

par, img = dataset.read_features(dataset.dataset_hmi.valid_events[96], dataset.dataset_hmi)

# Read images and parameters
img_vec = []
par_vec = []
for i, data in enumerate(dataset.dataset_hmi.valid_events):
    par, img = dataset.read_features(data, dataset.dataset_hmi)
    img_vec.append(img)
    par_vec.append(par)  
    print(i)

    
    
    
    
    
    
    




img_smp = sample(img_vec, 5)
for img in img_smp:
    plt.imshow(img, cmap = 'gray')
    plt.show()
    
    



# cnt = 0
# tmp = []
# for file in dataset.dataset_hmi.harp_data.keys():
#     harp, _, _, video = dataset.read_harp(file, dataset.dataset_hmi, img_read = True)
#     vid_ind = harp['frame'].str[5:].astype(int)
#     if vid_ind[0] != 0:
#         cnt += 1
#         tmp.append(file)
    #img_features = video[vid_ind, :, :, 2]
