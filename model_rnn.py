
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Bidirectional, LSTM, Masking, GRU, RepeatVector, TimeDistributed, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.activations import relu, softplus
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

class model_rnn(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        self.x_dim = self.cfg.x_dim
        self.x2_dim = self.cfg.x2_dim
        self.y_dim = self.cfg.y_dim
        
        self.x_enc_init()
        self.x_dec_init()
        self.x_img_enc_init()
        self.x_img_dec_init()
        self.x2_enc_init()
        self.x2_dec_init()
        self.y_dec_init()
        
        self.loss_y = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.loss_x = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_x_img = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_x2 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr)
        
    class MyRegularizer(regularizers.Regularizer):

        def __init__(self, strength):
            self.strength = strength
    
        def __call__(self, x):
            return self.strength * tf.reduce_sum(tf.square(x))
        
    def x_enc_init(self):
        self.x_enc_hidden1 = LSTM(64, return_state=True)
        self.x_enc_hidden2 = LSTM(64, return_sequences=True)
        self.x_enc_pool = GlobalMaxPooling1D()
        self.x_enc_mean = Dense(self.cfg.z_dim)
        self.x_enc_var = Dense(self.cfg.z_dim)
        
    def x_dec_init(self):
        self.x_dec_rep = RepeatVector(self.x_dim[0])
        self.x_dec_hidden1 = LSTM(64, return_sequences=True)
        self.x_dec_hidden2 = Dense(64, activation='relu')
        self.x_dec_out = TimeDistributed(Dense(self.x_dim[1]))
        
    def x_img_enc_init(self):
        if self.cfg.network_type == 'recurrent':
            self.x_img_enc_conv1 = TimeDistributed(Conv2D(64, 3, (2,2), activation='relu', padding='same'))
            self.x_img_enc_conv2 = TimeDistributed(Conv2D(64, 3, (2,2), activation='relu', padding='same'))
            self.x_img_enc_flatten = TimeDistributed(Flatten())
            self.x_img_enc_hidden1 = LSTM(64, return_state=True)
            self.x_img_enc_hidden2 = LSTM(64, return_sequences=True)
            self.x_img_enc_pool = GlobalMaxPooling1D()
        elif self.cfg.network_type == 'dense':
            self.x_img_enc_conv1 = Conv2D(64, 3, (2,2), activation='relu', padding='same')
            self.x_img_enc_conv2 = Conv2D(64, 3, (2,2), activation='relu', padding='same')
            self.x_img_enc_flatten = Flatten()
        
        self.x_img_enc_mean = Dense(self.cfg.z_dim)
        self.x_img_enc_var = Dense(self.cfg.z_dim)
    def x_img_dec_init(self):
        self.x_img_dec_rep = RepeatVector(int(self.cfg.x_img_dim[0]/self.cfg.downsample_img)+1)
        self.x_img_dec_hidden1 = LSTM(64, return_sequences=True)
        if self.cfg.network_type == 'recurrent':
            self.x_img_dec_dense = TimeDistributed(Dense(int(self.cfg.x_img_dim[1]/4) * int(self.cfg.x_img_dim[2]/4) *64, activation='relu'))
            self.x_img_dec_conv1 = TimeDistributed(Conv2DTranspose(64, 3, 2, activation='relu', padding='same'))
            self.x_img_dec_conv2 = TimeDistributed(Conv2DTranspose(64, 3, 2, activation='relu', padding='same'))
            self.x_img_dec_conv3 = TimeDistributed(Conv2DTranspose(1, 3, 1, padding='same'))
        elif self.cfg.network_type == 'dense':
            self.x_img_dec_dense = Dense(int(self.cfg.x_img_dim[1]/4) * int(self.cfg.x_img_dim[2]/4) *64, activation='relu')
            self.x_img_dec_conv1 = Conv2DTranspose(64, 3, 2, activation='relu', padding='same')
            self.x_img_dec_conv2 = Conv2DTranspose(64, 3, 2, activation='relu', padding='same')
            self.x_img_dec_conv3 = Conv2DTranspose(1, 3, 1, padding='same')
        
        
        
    def x2_enc_init(self):
        self.x2_enc_hidden1 = LSTM(64, return_state=True) #batch_input_shape=(self.batch_size, self.x2_dim[0], self.x2_dim[1]))
        self.x2_enc_mean = Dense(self.cfg.z_dim)
        self.x2_enc_var = Dense(self.cfg.z_dim)
    def x2_dec_init(self):
        self.x2_dec_rep = RepeatVector(self.x2_dim[0])
        self.x2_dec_hidden1 = LSTM(64, return_sequences=True)
        self.x2_dec_hidden2 = Dense(64, activation='relu')
        self.x2_dec_out = TimeDistributed(Dense(self.x2_dim[1]))
        
    @tf.function
    def x_encode(self, x):
        if self.cfg.network_type == 'recurrent':
            zx = self.x_enc_hidden2(x)
            zx, h, c = self.x_enc_hidden1(zx)
            h = tf.concat([h, c], axis = -1)
        elif self.cfg.network_type == 'dense':
            h  = x[:,-1,:]
        
        mean = self.x_enc_mean(h)
        logvar = self.x_enc_var(h)
        return mean, logvar
    @tf.function
    def x_decode(self, z):
        if self.cfg.network_type == 'recurrent':
            z = self.x_dec_rep(z)
            z = self.x_dec_hidden2(z)
            z = self.x_dec_hidden1(z)
            x = self.x_dec_out(z)
        elif self.cfg.network_type == 'dense':
            z = self.x_dec_rep(z)
            x = self.x_dec_out(z)
            
        return x
    
    
    @tf.function
    def x_img_encode(self, x):
        if self.cfg.network_type == 'recurrent':
            x = x[:, 0::self.cfg.downsample_img, :, :, :]
            h = self.x_img_enc_conv1(x)
            h = self.x_img_enc_conv2(h)
            h = self.x_img_enc_flatten(h)
            zx = self.x_img_enc_hidden2(h)
            zx, h, c = self.x_img_enc_hidden1(zx)
            h = tf.concat([h, c], axis = -1)
        elif self.cfg.network_type == 'dense':
            h = x[:,-1,:,:,:]
            h = self.x_img_enc_conv1(h)
            h = self.x_img_enc_conv2(h)
            h = self.x_img_enc_flatten(h)
        mean = self.x_img_enc_mean(h)
        logvar = self.x_img_enc_var(h)
        return mean, logvar
    @tf.function
    def x_img_decode(self, z):
        if self.cfg.network_type == 'recurrent':
            z = self.x_img_dec_rep(z)
            z = self.x_img_dec_hidden1(z)
            h = self.x_img_dec_dense(z)
            h = Reshape((-1, int(self.cfg.x_img_dim[1]/4) , int(self.cfg.x_img_dim[2]/4) , 64))(h)
        elif self.cfg.network_type == 'dense':
            h = self.x_img_dec_dense(z)
            h = Reshape((int(self.cfg.x_img_dim[1]/4) , int(self.cfg.x_img_dim[2]/4) , 64))(h)
        h = self.x_img_dec_conv1(h)
        h = self.x_img_dec_conv2(h)
        x = self.x_img_dec_conv3(h)
        return x
    
    @tf.function
    def x2_encode(self, x):
        if self.cfg.network_type == 'recurrent':
            zx, h, c = self.x2_enc_hidden1(x)
            h = tf.concat([h, c], axis = -1)
        elif self.cfg.network_type == 'dense':
            h = x[:,-1,:]
        mean = self.x2_enc_mean(h)
        logvar = self.x2_enc_var(h)
        return mean, logvar
    @tf.function
    def x2_decode(self, z):
        if self.cfg.network_type == 'recurrent':
            z = self.x2_dec_rep(z)
            z = self.x2_dec_hidden2(z)
            z = self.x2_dec_hidden1(z)
            x = self.x2_dec_out(z)
        elif self.cfg.network_type == 'dense':
            z = self.x2_dec_rep(z)
            x = self.x2_dec_out(z)
        return x
    
    
    def y_dec_init(self):
        self.y_dec1 = Dense(64, activation='relu')
        #self.y_dec2 = Dense(64, activation='relu')
        self.y_dec = Dense(self.y_dim)
        
    @tf.function
    def y_decode(self, z):
        #y = self.y_dec1(z)
        y = self.y_dec(z)
        return y
        
    @tf.function
    def kl_prior(self, mean, logvar, raxis=1):
        return -.5 * tf.reduce_sum( 1 + logvar - mean * mean  - tf.exp(logvar) , axis=raxis)
    @tf.function
    def compute_kl(self, m_0, s_0, m_1, s_1):
        s_0 = tf.reshape(s_0, (-1, 1, self.cfg.z_dim))
        m_0 = tf.reshape(m_0, (-1, 1, self.cfg.z_dim))
        s_1 = tf.reshape(s_1, (1, -1, self.cfg.z_dim))
        m_1 = tf.reshape(m_1, (1, -1, self.cfg.z_dim))
        
        kl = tf.exp(s_0 - s_1)
        kl += s_1 - s_0
        kl += (m_1 - m_0) * (m_1 - m_0) * (1/tf.exp(s_1))
        kl = 0.5 * (tf.reduce_sum(kl, axis = -1) - self.cfg.z_dim)
        
        return kl
    
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    @tf.function
    def train_batch_classifier(self, data, train = True):
    
        with tf.GradientTape() as s_tape:
            loss_s, y_pred,_ = self.compute_loss(data, train = train)
        gradients = s_tape.gradient(loss_s, self.trainable_variables)
        
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)
        
        return loss_s, y_pred
    
    @tf.function
    def compute_euc(self, m_0, m_1):
        
        m_0 = m_0 / tf.norm(m_0, axis = 1, keepdims = True)
        m_1 = m_1 / tf.norm(m_1, axis = 1, keepdims = True)
        
        m_0 = tf.reshape(m_0, (-1, 1, self.cfg.z_dim))
        m_1 = tf.reshape(m_1, (1, -1, self.cfg.z_dim))
        
        euc = (m_1 - m_0) * (m_1 - m_0)
        euc = tf.reduce_sum(euc, axis = -1) 
        return euc
    
    @tf.function
    def compute_dot(self, m_0, m_1):
        m_0 = m_0 / tf.norm(m_0, axis = 1, keepdims = True)
        m_1 = m_1 / tf.norm(m_1, axis = 1, keepdims = True)
        return tf.matmul (m_0, tf.transpose(m_1))

    @tf.function
    def compute_loss(self, data, train = True):
        
        if not self.cfg.mdi_on and train:
            data = [data[i][0:16] for i in range(len(data))]
        
        y = data[0] # HMI Pars
        x = data[1] # HMI Image
        v = data[2] # MDI Pars
        r = data[4] # HMI Response
        mask = tf.cast(data[5], dtype = tf.float32)
        
        x = tf.expand_dims(x, -1)
        x = tf.image.per_image_standardization(x)
        n = r.shape[0]
        
        # Encoding
        mean_z = 0
        prec_z = 0
        if self.cfg.hmi_par_on:
            m_y, s_y = self.x_encode(y)
            prec_z += (1./tf.exp(s_y)) * mask[:,0:1] 
            mean_z += m_y * (1./tf.exp(s_y)) * mask[:,0:1]
        if self.cfg.hmi_img_on:
            m_x, s_x = self.x_img_encode(x)
            prec_z += (1./tf.exp(s_x)) * mask[:,0:1] 
            mean_z += m_x * (1./tf.exp(s_x)) * mask[:,0:1]
        if self.cfg.mdi_on:
            m_v, s_v = self.x2_encode(v)
            prec_z += (1./tf.exp(s_v)) * mask[:,1:2] 
            mean_z += m_v * (1./tf.exp(s_v)) * mask[:,1:2] 
        s_z = 1./prec_z
        m_z = mean_z * s_z
        s_z = tf.math.log(s_z)
        
        z = self.reparameterize(m_z, s_z)
        
        # Ranking
        
        loss_rank = 0
        if train:
            
            # HMI ranking
            dist = self.compute_euc(m_z[0:16], m_z[0:16])
            dist = tf.exp(-dist)
            
            sum_exp_negs = tf.reduce_sum(dist[0:8, 8:], axis = 1, keepdims=True)
            soft = dist[0:8, 0:8] / (dist[0:8, 0:8] + sum_exp_negs)
            soft = tf.linalg.set_diag(soft, tf.ones(soft.shape[0:-1]))
            loss_rank += tf.reduce_sum(-tf.math.log(soft))
            
            sum_exp_negs = tf.reduce_sum(dist[8:, 0:8], axis = 1, keepdims=True)
            soft = dist[8:, 8:] / (dist[8:, 8:] + sum_exp_negs)
            soft = tf.linalg.set_diag(soft, tf.ones(soft.shape[0:-1]))
            loss_rank += tf.reduce_sum(-tf.math.log(soft))
            
            # MDI ranking
            dist = self.compute_euc(m_z[16:32], m_z[16:32])
            dist = tf.exp(-dist)
            
            sum_exp_negs = tf.reduce_sum(dist[0:8, 8:], axis = 1, keepdims=True)
            soft = dist[0:8, 0:8] / (dist[0:8, 0:8] + sum_exp_negs)
            soft = tf.linalg.set_diag(soft, tf.ones(soft.shape[0:-1]))
            loss_rank += tf.reduce_sum(-tf.math.log(soft))
            
            sum_exp_negs = tf.reduce_sum(dist[8:, 0:8], axis = 1, keepdims=True)
            soft = dist[8:, 8:] / (dist[8:, 8:] + sum_exp_negs)
            soft = tf.linalg.set_diag(soft, tf.ones(soft.shape[0:-1]))
            loss_rank += tf.reduce_sum(-tf.math.log(soft))
            
            # Mutual ranking
            dist = self.compute_euc(m_y[32:48], m_v[32:48])
            dist = tf.exp(-dist)
            
            sum_exp_all = tf.reduce_sum(dist, axis = 1, keepdims=True)
            soft = tf.linalg.diag_part(dist) / sum_exp_all
            loss_rank += tf.reduce_sum(-tf.math.log(soft))
            
            
        
        # Reconstruction
        loss_rec = 0
        if self.cfg.hmi_par_on:
            y_rec = self.x_decode(z)
            if self.cfg.network_type == 'recurrent':
                loss_y = tf.reduce_mean(self.loss_x(y_rec, y), axis = -1)
            elif self.cfg.network_type == 'dense':
                loss_y = self.loss_x(y_rec, y)[:,-1]
            loss_rec += tf.reduce_sum(loss_y * mask[:,0])
        
        if self.cfg.hmi_img_on:
            x_rec = self.x_img_decode(z)
            if self.cfg.network_type == 'recurrent':
                loss_x = tf.reduce_sum(self.loss_x_img(x_rec, x[:,0::self.cfg.downsample_img]), axis = [1,2,3])
            elif self.cfg.network_type == 'dense':
                loss_x = tf.reduce_sum(self.loss_x_img(x_rec, x[:,-1]), axis = [1,2])
            loss_rec += tf.reduce_sum(loss_x * mask[:,0])
        
        if self.cfg.mdi_on:
            v_rec = self.x2_decode(z)
            if self.cfg.network_type == 'recurrent':
                loss_v = tf.reduce_mean(self.loss_x2(v_rec, v), axis = -1)
            elif self.cfg.network_type == 'dense':
                loss_v = self.loss_x2(v_rec, v)[:,-1]
            loss_rec += tf.reduce_sum(loss_v * mask[:,1])
        
        # KL prior
        log_prior = self.kl_prior(m_z, s_z)
        loss_prior = tf.reduce_sum(log_prior)
        
        # Regularization terms (JSD)
        if self.cfg.hmi_par_on and self.cfg.mdi_on:
            d = self.compute_kl(m_y, s_y, m_v, s_v)
            d = tf.linalg.diag_part(d)
            d2 = self.compute_kl(m_v, s_v, m_y, s_y)
            d2 = tf.linalg.diag_part(d2)
            loss_reg = tf.reduce_sum((d+d2)* mask[:,0] * mask[:,1]) * 1
        
        # Response decoding
        if train:
            logits = self.y_decode(z)
        else:
            logits = self.y_decode(m_z)
        loss_class = tf.reduce_sum(self.loss_y(r, logits) * mask[:,2]) *  self.cfg.y_coef
        
        loss = 0
        if train:
            loss = 0*loss_rank +  1*loss_rec + 1*loss_prior + 1*loss_class + sum(self.losses)

        return tf.reduce_mean(loss), logits, m_z #/ tf.norm(m_z, axis = 1, keepdims = True)