import numpy as np
import tensorflow as tf
import time
import sys
import pickle
from random import sample, choices
from sklearn.metrics import confusion_matrix

from datetime import timedelta

class trainer_struct():
    
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        
        self.data_train = tf.data.Dataset.from_generator(
                lambda: map(tuple, self.batch_gen_train(dataset)),
                (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool))
        
        self.iterator = iter(self.data_train.repeat())
        self.results = self.result_struct()
        
        if self.cfg.use_cached_train_data:
            #if self.cfg.mdi_on:
                if self.cfg.img_on:
                    with open(dataset.path_cache + "cache_train_img_on.txt", "rb") as fp:
                        self.dataset_train_cache = pickle.load(fp) 
                else:
                    with open(dataset.path_cache + "cache_train_img_off.txt", "rb") as fp:
                        self.dataset_train_cache = pickle.load(fp) 
            # else:
            #     with open(dataset.path_cache + "cache_train_mdi_off.txt", "rb") as fp:
            #         self.dataset_train_cache = pickle.load(fp) 
                    
        if self.cfg.use_cached_test_data:
            with open(dataset.path_cache + "cache_test.txt", "rb") as fp:
                self.dataset_test_cache = pickle.load(fp) 
        
    class result_struct():
        def __init__(self):
            self.loss = [0]
            self.acc_test = [0]
            self.f1_test = [0]
            self.rec_test = [0]
            self.prec_test = [0]
            self.hss2_test = [0]
            self.tss_test = [0]
    
    def batch_gen_train(self, dataset):
        
        dataset_hmi = dataset.dataset_hmi
        dataset_mdi = dataset.dataset_mdi
    
        #if self.cfg.mdi_on:
        batch_bc = []
        mask = []
        
        N = 16
        ind_BC = [idx for idx, element in enumerate(dataset_hmi.valid_events_train) if element[1][:1] == 'B' or element[1][:1] == 'C']
        ind_MX = [idx for idx, element in enumerate(dataset_hmi.valid_events_train) if element[1][:1] == 'M' or element[1][:1] == 'X']
        batch_bc += [dataset_hmi.valid_events_train[i] for i in sample(ind_BC, int(N/2))] + [dataset_hmi.valid_events_train[i] for i in sample(ind_MX, int(N/2))]
        mask += [[True, False, True]] * N
        
        N = 16
        ind_BC = [idx for idx, element in enumerate(dataset_mdi.valid_events_train) if element[1][:1] == 'B' or element[1][:1] == 'C']
        ind_MX = [idx for idx, element in enumerate(dataset_mdi.valid_events_train) if element[1][:1] == 'M' or element[1][:1] == 'X']
        batch_bc += [dataset_mdi.valid_events_train[i] for i in sample(ind_BC, int(N/2))] + [dataset_mdi.valid_events_train[i] for i in sample(ind_MX, int(N/2))]
        mask += [[False, True, True]] * N
        
        # N = 16
        # ind_BC = [idx for idx, element in enumerate(dataset_hmi.valid_events_train_intersect) if element[2][:1] == 'B' or element[2][:1] == 'C']
        # ind_MX = [idx for idx, element in enumerate(dataset_hmi.valid_events_train_intersect) if element[2][:1] == 'M' or element[2][:1] == 'X']
        # batch_bc += [dataset_hmi.valid_events_train_intersect[i] for i in sample(ind_BC, int(N/2))] + [dataset_hmi.valid_events_train_intersect[i] for i in choices(ind_MX, k = int(N/2))]  #!!!!
        # mask += [[True, True, True]] * N
        
        N = 16
        batch_bc += [dataset.sample_overlap_data() for i in range(N)]
        mask += [[True, True, False]] * N
        
        # else:
            
        #     N = 48
        #     ind_BC = [idx for idx, element in enumerate(dataset_hmi.valid_events_train) if element[1][:1] == 'B' or element[1][:1] == 'C']
        #     ind_MX = [idx for idx, element in enumerate(dataset_hmi.valid_events_train) if element[1][:1] == 'M' or element[1][:1] == 'X']
        #     batch_bc = [dataset_hmi.valid_events_train[i] for i in sample(ind_BC, int(N/2))] + [dataset_hmi.valid_events_train[i] for i in sample(ind_MX, int(N/2))]
        #     mask = [[True, False, True]] * N
            
        
        inp_hmi, img_hmi, inp_mdi, img_mdi, resp, mask  = self.batch_gen(batch_bc, mask, dataset)

        yield inp_hmi, img_hmi, inp_mdi, img_mdi, resp, mask 
        #return inp_hmi, img_hmi, inp_mdi, img_mdi, resp, mask 
        
    def batch_gen(self, data_list, mask, dataset):
        
        resp = []
        
        hmi_batch = []
        mdi_batch = []
        for i, data in enumerate(data_list):
            
            if mask[i][0:2] == [True, True]:
                hmi_batch.append(dataset.read_features([data[1], data[2], data[3]], dataset.dataset_hmi))
                mdi_batch.append(dataset.read_features([data[0], data[2], data[3]], dataset.dataset_mdi))
            else:
                if mask[i][0] == True:
                    hmi_batch.append(dataset.read_features(data, dataset.dataset_hmi))
                else:
                    dummy_par = np.zeros((int(dataset.cfg.len_seq/dataset.dataset_hmi.cadence), len(dataset.dataset_hmi.features)))
                    #dummy_img = np.zeros((int(dataset.cfg.len_seq/dataset.dataset_hmi.cadence), dataset.dataset_hmi.img_res[0], dataset.dataset_hmi.img_res[1]))
                    dummy_img = np.zeros((int(dataset.cfg.len_seq/dataset.dataset_hmi.cadence), 1,1))
                    hmi_batch.append([dummy_par, dummy_img])
                    
                if mask[i][1] == True:
                    mdi_batch.append(dataset.read_features(data, dataset.dataset_mdi))
                else:
                    dummy_par = np.zeros((int(dataset.cfg.len_seq/dataset.dataset_mdi.cadence), len(dataset.dataset_mdi.features)))
                    #dummy_img = np.zeros((int(dataset.cfg.len_seq/dataset.dataset_mdi.cadence), dataset.dataset_mdi.img_res[0], dataset.dataset_mdi.img_res[1]))
                    dummy_img = np.zeros((int(dataset.cfg.len_seq/dataset.dataset_mdi.cadence), 1, 1))
                    mdi_batch.append([dummy_par, dummy_img])
                
            if (data[-2][:1] == 'B') or (data[-2][:1] == 'C'):
                resp.append(0)
            else:
                resp.append(1)
            
        inp_hmi = np.array([hmi_batch[i][0] for i in range(len(hmi_batch))], dtype = np.float32)
        img_hmi = np.array([hmi_batch[i][1] for i in range(len(hmi_batch))], dtype = np.float32)
        inp_mdi = np.array([mdi_batch[i][0] for i in range(len(mdi_batch))], dtype = np.float32)
        img_mdi = np.array([mdi_batch[i][1] for i in range(len(mdi_batch))], dtype = np.float32)
        resp = np.eye(2)[resp]
        resp = np.array(resp).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        
        return inp_hmi, img_hmi, inp_mdi, img_mdi, resp, mask 
    
    def get_next_train(self, epoch):
        if self.cfg.use_cached_train_data:
            data = self.dataset_train_cache[epoch]
        else:
            data = self.iterator.get_next()
        return data
    
    def get_next_test(self, epoch, dataset, test_dataset):
        if self.cfg.use_cached_test_data:
            data = self.dataset_test_cache[epoch]
        else:
            
            index = epoch * self.cfg.test_batch_size
            
            if test_dataset == 'hmi':
                mask = [[True, False, True]] * self.cfg.test_batch_size
                data = self.batch_gen(dataset.dataset_hmi.valid_events_test[index : index + self.cfg.test_batch_size], mask, dataset)
            elif test_dataset == 'mdi':
                mask = [[False, True, True]] * self.cfg.test_batch_size
                data = self.batch_gen(dataset.dataset_mdi.valid_events_test[index : index + self.cfg.test_batch_size], mask, dataset)
        
        
        return data

    def train_model(self, model, dataset):
        
        dataset_cache = []
        
        start = 0
        total_loss = []
        for epoch in range(0,self.cfg.Neps_train):
            
            data = self.get_next_train(epoch)
            dataset_cache.append(data)
            
            if not self.cfg.mdi_on:
                data = [data[i][0:16] for i in range(len(data))]
            if not self.cfg.hmi_par_on:
                data = [data[i][16:32] for i in range(len(data))]
            
            loss, y_pred = model.train_batch_classifier(data, train = True)
            
            total_loss.append(loss)
            if epoch % self.cfg.log_steps == 0:
                
                self.test_model(model, dataset)
                
                print('Step {}: loss = {:.2f}, acc_test = {:.3f}, f1_test = {:.3f}, hss2_test = {:.3f}, tss_test = {:.3f}, Time to run {} steps = {:.2f} seconds'.format(
                    epoch, np.mean(total_loss),
                    self.results.acc_test[-1],
                    self.results.f1_test[-1],
                    self.results.hss2_test[-1],
                    self.results.tss_test[-1],
                    self.cfg.log_steps, time.time() - start))
                
                start = time.time()
                sys.stdout.flush()
                
                self.results.loss.append(np.mean(total_loss))
                
                total_loss = []
                
        if not self.cfg.use_cached_train_data:
            #if self.cfg.mdi_on:
                if self.cfg.img_on:
                    with open(dataset.path_cache + "cache_train_img_on.txt", "wb") as fp:
                        pickle.dump(dataset_cache, fp)
                else:
                    with open(dataset.path_cache + "cache_train_img_off.txt", "wb") as fp:
                        pickle.dump(dataset_cache, fp)
                
            #else:
            #    with open(dataset.path_cache + "cache_train_mdi_off.txt", "wb") as fp:
            #        pickle.dump(dataset_cache, fp)
            
    def vis_test_data(self, model, dataset):
        
        m_vec_all = []
        r_vec_all = []
        
        m_vec = []
        r_vec = []
        for i in range(int(len(dataset.dataset_hmi.valid_events_test) /  self.cfg.test_batch_size)):
            data = self.get_next_test(i, dataset, 'hmi')
            _,_, m = model.compute_loss(data, train = False)
            m_vec.append(m)
            r_vec.append(data[4])
        m_vec_all.append(m_vec)
        r_vec_all.append(r_vec)
        
        m_vec = []
        r_vec = []
        for i in range(int(len(dataset.dataset_mdi.valid_events_test) /  self.cfg.test_batch_size)):
            data = self.get_next_test(i, dataset, 'mdi')
            _,_, m = model.compute_loss(data, train = False)
            m_vec.append(m)
            r_vec.append(data[4])
        m_vec_all.append(m_vec)
        r_vec_all.append(r_vec)
            
        return m_vec_all, r_vec_all
    
    def vis_train_data(self, model, dataset):
        
        m_vec_all = []
        r_vec_all = []
        
        # HMI dataset
        m_vec = []
        r_vec = []
        N = 32
        for i in range(int(len(dataset.dataset_hmi.valid_events_train) /  N)):
            data_list = dataset.dataset_hmi.valid_events_train[i*N : (i+1)*N]
            mask = [[True, False, True]] * N
            data  = self.batch_gen(data_list, mask, dataset)
            _,_, m = model.compute_loss(data, train = False)
            m_vec.append(m)
            r_vec.append(data[4])
        m_vec_all.append(m_vec)
        r_vec_all.append(r_vec)
        
        # MDI dataset
        m_vec = []
        r_vec = []
        N = 32
        for i in range(int(len(dataset.dataset_mdi.valid_events_train) /  N)):
            data_list = dataset.dataset_mdi.valid_events_train[i*N : (i+1)*N]
            mask = [[False, True, True]] * N
            data  = self.batch_gen(data_list, mask, dataset)
            _,_, m = model.compute_loss(data, train = False)
            m_vec.append(m)
            r_vec.append(data[4])
        m_vec_all.append(m_vec)
        r_vec_all.append(r_vec)
        
        # Overlapping region
        m_vec = []
        r_vec = []
        N = 1
        for i in range(int(len(dataset.dataset_mdi.valid_events_train_intersect) /  N)):
            data_list = dataset.dataset_mdi.valid_events_train_intersect[i*N : (i+1)*N]
            mask = [[True, True, True]] * N
            data  = self.batch_gen(data_list, mask, dataset)
            _,_, m = model.compute_loss(data, train = False)
            m_vec.append(m)
            r_vec.append(data[4])
        m_vec_all.append(m_vec)
        r_vec_all.append(r_vec)
            
            
        return m_vec_all, r_vec_all
        
    
    def test_model(self, model, dataset):
        
        y_pred_all = []
        resp_all = []
        dataset_cache = []
        for i in range(int(len(dataset.dataset_hmi.valid_events_test) /  self.cfg.test_batch_size)):
                        
            data = self.get_next_test(i, dataset, 'hmi')
            dataset_cache.append(data)
                
            _,y_pred, _ = model.compute_loss(data, train = False)
            
            y_pred = tf.argmax(y_pred, axis = 1)
            y_pred = tf.one_hot(y_pred, data[4].shape[1]).numpy()
            y_pred_all.append(y_pred)
            resp_all.append(data[4])
            
        for i in range(int(len(dataset.dataset_mdi.valid_events_test) /  self.cfg.test_batch_size)):
                        
            data = self.get_next_test(i, dataset, 'mdi')
            dataset_cache.append(data)
                
            _,y_pred, _ = model.compute_loss(data, train = False)
            
            y_pred = tf.argmax(y_pred, axis = 1)
            y_pred = tf.one_hot(y_pred, data[4].shape[1]).numpy()
            y_pred_all.append(y_pred)
            resp_all.append(data[4])
        
        resp_all = np.vstack(resp_all)
        y_pred_all = np.vstack(y_pred_all)
        
        tn, fp, fn, tp = confusion_matrix(np.argmax(resp_all, axis = 1), np.argmax(y_pred_all, axis = 1)).ravel()
        rec = tp/(tp + fn)
        prec = tp/(tp + fp)
        f1  = (2 * prec * rec) / (prec + rec)
        acc = (tp + tn) / (tn + fp + fn + tp)
        hss2 = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
        tss = (tp / (tp + fn)) - (fp / (fp + tn))
        
        #prec = (y_pred_all[:,1] * resp_all[:,1]).sum() / y_pred_all[:,1].sum()
        #rec = (y_pred_all[:,1] * resp_all[:,1]).sum() / resp_all[:,1].sum()
        #acc = (y_pred_all * resp_all).sum() / y_pred_all.shape[0]
        
        self.results.acc_test.append(acc)
        self.results.f1_test.append(f1)
        self.results.prec_test.append(prec)
        self.results.rec_test.append(rec)
        self.results.hss2_test.append(hss2)
        self.results.tss_test.append(tss)

        if not self.cfg.use_cached_test_data:
            with open(dataset.path_cache + "cache_test.txt", "wb") as fp:
                pickle.dump(dataset_cache, fp)