import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataset_prep import dataset_struct
from model_rnn import model_rnn
import tensorflow as tf
from utils import  print_flare_stats,  print_intersect_flare_stats
from utils_training import trainer_struct
from datetime import timedelta

import seaborn as sns
sns.set_theme()

tf.autograph.set_verbosity(0)

tf.config.experimental_run_functions_eagerly(False)

class dataset_cfg_struct():
    def __init__(self):
        self.use_cached_preprocessing = True
        self.use_cached_harps = True
        
        self.len_seq = 12 * 5 * 10 # 10 hours of features
        self.len_pred = 24 * 60 # prediction for 24 hours after
        self.max_flare_filtering = True  # Filter flares within 24 hours window 
        self.max_flare_window_drop = False # If true: Drop filtered flares else: replace with max flare
        self.remove_C = True


dataset_cfg = dataset_cfg_struct()

dataset = dataset_struct(dataset_cfg)
dataset.load_datasets()
dataset.preprocess_datasets()

delta2 = timedelta(minutes=dataset.cfg.len_pred)
t = dataset.dataset_hmi.valid_events_train[0][2] - delta2

path = r'Z:\data2\SHARP_720s\image\\'
path + dataset.dataset_hmi.valid_events_train[0][0][4:10] + 
