import os
os.environ['PATH']

import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataset_prep import dataset_struct
from model_rnn import model_rnn
import tensorflow as tf
from utils import  print_flare_stats,  print_intersect_flare_stats
from utils_training import trainer_struct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
        self.max_flare_filtering = False  # Filter flares within 24 hours window 
        self.max_flare_window_drop = False # If true: Drop filtered flares else: replace with max flare
        self.remove_C = False
        
        self.hmi_split_opt = 'date'  #'date', 'random', 'no'
        self.mdi_split_opt = 'random'  #'date', 'random', 'no'

class model_cfg_struct():
    def __init__(self):
        self.hmi_img_on = False
        self.hmi_par_on = True
        
        self.gen = True
        self.prior = True
        self.mdi_on = None
        self.network_type = 'recurrent'   # 'recurrent', 'conv', 'dense'
        self.z_dim = 2
        self.x_dim = [50, 22]   #!!!!!!
        self.x_img_dim = [50, 36, 72]   #!!!!!!
        self.x2_dim = [6, 4]
        self.y_dim = 2
        
        self.downsample_img = 8
        
        self.lr = 0.0001
        self.y_coef = 50
        
class training_cfg_struct():
    def __init__(self):
        self.use_cached_train_data = True
        self.use_cached_test_data = True
        
        self.hmi_par_on = True
        self.img_on = False
        self.mdi_on = None
        self.log_steps=10
        self.Neps_train=2000
        self.test_batch_size = 64

dataset_cfg = dataset_cfg_struct()
model_cfg = model_cfg_struct()
training_cfg = training_cfg_struct()

dataset = dataset_struct(dataset_cfg)
dataset.load_datasets()
dataset.preprocess_datasets()

exp_no = 1
results_vec = []
for hmi_on, mdi_on in [[True, True]]:#, [True, True]]:
    
    #tf.config.experimental_run_functions_eagerly(mdi_on)
    
    for exp in range(0, exp_no):
        
        
        model_cfg.mdi_on = mdi_on
        training_cfg.mdi_on = mdi_on
        
        model_cfg.hmi_par_on = hmi_on
        training_cfg.hmi_par_on = hmi_on
        
        model = model_rnn(model_cfg)
        
        # if not 'trainer' in locals():
        trainer = trainer_struct(training_cfg, dataset)
        # else:
        #     trainer.results = trainer.result_struct()
        
        trainer.train_model(model, dataset)
        results_vec.append(trainer.results)
    
#trainer.dataset_train_cache = []

if False:
    with open("results_rec.txt", "wb") as fp:
        pickle.dump(results_vec[1], fp)
    with open("results_snap.txt", "wb") as fp:
        pickle.dump(results_vec[0], fp)
    
if False:
    with open("results_rec.txt", "rb") as fp:
        results_both = pickle.load(fp) 
    with open("results_snap.txt", "rb") as fp:
        results_hmi = pickle.load(fp) 
    
if False: #Performance plots HMI vs HMI-MDI
    plt.plot(np.mean(np.array([results_vec[i+exp_no].f1_test for i in range(exp_no)]), axis = 0) , label = 'Hmi + Mdi')
    plt.plot(np.mean(np.array([results_vec[i].f1_test for i in range(exp_no)]), axis = 0) , label = 'Hmi')
    plt.xlabel('Iteration')
    plt.ylabel('F1-score')
    plt.legend()
    plt.show()
    
    plt.plot(np.mean(np.array([results_vec[i+exp_no].acc_test for i in range(exp_no)]), axis = 0) , label = 'Hmi + Mdi')
    plt.plot(np.mean(np.array([results_vec[i].acc_test for i in range(exp_no)]), axis = 0) , label = 'Hmi')
    plt.xlabel('Iteration')
    plt.ylabel('Classification Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(np.mean(np.array([results_vec[i+exp_no].hss2_test for i in range(exp_no)]), axis = 0) , label = 'Hmi + Mdi')
    plt.plot(np.mean(np.array([results_vec[i].hss2_test for i in range(exp_no)]), axis = 0) , label = 'Hmi')
    plt.xlabel('Iteration')
    plt.ylabel('HSS2')
    plt.legend()
    plt.show()
    
    plt.plot(np.mean(np.array([results_vec[i+exp_no].tss_test for i in range(exp_no)]), axis = 0) , label = 'Hmi + Mdi')
    plt.plot(np.mean(np.array([results_vec[i].tss_test for i in range(exp_no)]), axis = 0) , label = 'Hmi')
    plt.xlabel('Iteration')
    plt.ylabel('TSS')
    plt.legend()
    plt.show()

if False: # Image reconstruction
    i = 0
    x_img = trainer.dataset_test_cache[0][1][i:i+1, :, :, :, np.newaxis]
    x_img = tf.image.per_image_standardization(x_img)
    m,s = model.x_img_encode(x_img)
    x_img_rec = model.x_img_decode(m)
    
    plt.imshow(x_img[0][-1])
    plt.show()
    x_img_rec = tf.image.per_image_standardization(x_img_rec)
    plt.imshow(x_img_rec[0])
    plt.show()
    
    
if True: # Visualize train data embeddings
    m_vec_all, r_vec_all = trainer.vis_train_data(model, dataset)
    
    
    # # HMI-MDI
    # m_vec = np.array(m_vec_all[2])
    # m_vec = m_vec.reshape((-1, m_vec.shape[-1]))
    # r_vec = np.array(r_vec_all[2]) 
    # r_vec = r_vec.reshape((-1, r_vec.shape[-1]))
    # colors = ['b', 'g']
    # labels = ['B', 'M/X']
    
    # for i, label in enumerate(labels):
    #     ind = np.argmax(r_vec, axis = 1)
    #     plt.scatter(m_vec[np.where(ind == i)[0],0], m_vec[np.where(ind == i)[0],1], s= 2,c=colors[i], label = label)
    #     plt.legend()
    # plt.show()
    
    
    
    m_vec = np.array(m_vec_all[0])
    m_vec = m_vec.reshape((-1, m_vec.shape[-1]))
    r_vec = np.array(r_vec_all[0]) 
    r_vec = r_vec.reshape((-1, r_vec.shape[-1]))
    
    m_vec_comb = [m_vec]
    r_vec_comb = [r_vec]
    
    m_vec = np.array(m_vec_all[1])
    m_vec = m_vec.reshape((-1, m_vec.shape[-1]))
    r_vec = np.array(r_vec_all[1]) 
    r_vec = r_vec.reshape((-1, r_vec.shape[-1]))
    
    m_vec_comb.append(m_vec)
    r_vec_comb.append(r_vec)
    
    
    W = model.y_dec.weights[0].numpy()
    b = model.y_dec.weights[1].numpy()
    
    x = np.array([min(min(m_vec_comb[0][:,0]), min(m_vec_comb[1][:,0])), max(max(m_vec_comb[0][:,0]), max(m_vec_comb[1][:,0]))])
    #x = x * W[0][1] / W[0][0]
    y = [-(b[0] + W[0][0]*(x[0]))/W[0][1], -(b[0] + W[0][0]*(x[1]))/W[0][1]]
    
    colors = ['b', 'g', 'r', 'k']
    labels = ['HMI B', 'HMI M/X', 'MDI B', 'MDI M/X']
    
    for i, label in enumerate(labels):
        ind = np.argmax(r_vec_comb[int(i/2)], axis = 1)
        #plt.plot(x, y)
        plt.scatter(m_vec_comb[int(i/2)][np.where(ind == i % 2)[0],0], m_vec_comb[int(i/2)][np.where(ind == i % 2)[0],1], s= 1, c=colors[i], label = labels[i])
        plt.legend()
    plt.show()
    
    data = np.vstack(m_vec_comb)
    resp = np.vstack(r_vec_comb)
    data = m_vec_comb[1]
    resp = r_vec_comb[1]
    #clf = LogisticRegression(random_state=0).fit(data, np.argmax(resp, axis = 1))
    clf = KNeighborsClassifier(n_neighbors=20).fit(data, np.argmax(resp, axis = 1))
    
    
if True: # Visualize test data embeddings

    trainer.cfg.use_cached_test_data = False
    m_vec_all, r_vec_all = trainer.vis_test_data(model, dataset)
    m_vec = np.array(m_vec_all[0])
    m_vec = m_vec.reshape((-1, m_vec.shape[-1]))
    r_vec = np.array(r_vec_all[0]) 
    r_vec = r_vec.reshape((-1, r_vec.shape[-1]))
    
    m_vec_comb_test = [m_vec]
    r_vec_comb_test = [r_vec]
    
    m_vec = np.array(m_vec_all[1])
    m_vec = m_vec.reshape((-1, m_vec.shape[-1]))
    r_vec = np.array(r_vec_all[1]) 
    r_vec = r_vec.reshape((-1, r_vec.shape[-1]))
    
    m_vec_comb_test.append(m_vec)
    r_vec_comb_test.append(r_vec)
    
    W = model.y_dec.weights[0].numpy()
    b = model.y_dec.weights[1].numpy()
    x = np.array([min(min(m_vec_comb[0][:,0]), min(m_vec_comb_test[1][:,0])), max(max(m_vec_comb_test[0][:,0]), max(m_vec_comb[1][:,0]))])
    y = [-(b[0] + W[0][0]*(-1.5))/W[0][1], -(b[0] + W[0][0]*(1.5))/W[0][1]]
    
    colors = ['b', 'g', 'r', 'k']
    labels = ['HMI B', 'HMI M/X', 'MDI B', 'MDI M/X']
    
    for i, label in enumerate(labels):
        ind = np.argmax(r_vec_comb_test[int(i/2)], axis = 1)
        #plt.plot(x, y)
        if i>=2:
            plt.scatter(m_vec_comb_test[int(i/2)][np.where(ind == i % 2)[0],0], m_vec_comb_test[int(i/2)][np.where(ind == i % 2)[0],1], s= 1, c=colors[i], label = labels[i])
        plt.legend()
    plt.show()
    
    #y_pred = clf.predict(m_vec_comb_test[0])
    y_pred = np.argmax(model.y_decode(m_vec_comb_test[0]), axis = 1)
    tn, fp, fn, tp = confusion_matrix(np.argmax(r_vec_comb_test[0], axis = 1), y_pred).ravel()
    rec = tp/(tp + fn)
    prec = tp/(tp + fp)
    f1  = (2 * prec * rec) / (prec + rec)
    acc = (tp + tn) / (tn + fp + fn + tp)
    hss2 = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
    
    print('Acc:', acc)
    print('F1:', f1)
    print('Hss2:', hss2)
    
    
    
    


    
    



    
