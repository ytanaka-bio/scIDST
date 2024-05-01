#0 initialization
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

#parse argument
parser = argparse.ArgumentParser(description='Dimensional reduction with variational autoencoder',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-a', help='activation function', default="relu", type=str, dest="func")
parser.add_argument('-e', help='epoch for best hyperparameter configuations', default=10, type=int, dest="epoch")
parser.add_argument('-b', help='batch size', default=64, type=int, dest="batch_size")
parser.add_argument('-f', help='buffer size', default=1024, type=int, dest="buffer_size")
parser.add_argument('-g', help='Learning rate', default=1e-3, dest="learning_rate")
parser.add_argument('-v', help='Beta value',default=12.0,dest="beta")
parser.add_argument('-i', help='Dimension of intermediate layers in encoder and decoder',default=[5000,1000],type=int,dest="intermediate_dim")
parser.add_argument('-l', help='Dimension of latent layers',default=200,type=int,dest="latent_dim")
parser.add_argument('-n', help='number of cells used for autoencoder training',default=20000,type=int,dest="train_num")
parser.add_argument('-s', help='number of split of cells when running autoencoder (larger number will use lower memory)',default=10,type=int,dest="split")
parser.add_argument('-t', help="number of threads",default=1,type=int,dest="thread")
parser.add_argument('-c', help="prefix of column name in output",default="Input_",type=str, dest="prefix")
parser.add_argument('-m', help="Normalization method (0 (Zscore), 1 (MinMax), 2 (Quantile))",default=2,type=int,dest="norm")
parser.add_argument('data_dir',nargs=1,help='cellranger matrix directory',type=str)
parser.add_argument('output',nargs=1,help='output filename',type=str)

args = parser.parse_args()


# construction of variational autoencoder
import os, psutil
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(args.thread)
tf.config.threading.set_intra_op_parallelism_threads(args.thread)
from scipy.io import mmread
from sklearn.preprocessing import MinMaxScaler, normalize, quantile_transform, StandardScaler
from tensorflow.keras import layers
from scipy.stats import zscore
import random




def kl_loss(z_mu,z_rho):
    sigma_squared = tf.math.softplus(z_rho) ** 2
    kl_1d = -0.5 * (1 + tf.math.log(sigma_squared) - z_mu ** 2 - sigma_squared)
    # sum over sample dim, average over batch dim
    kl_batch = tf.reduce_mean(tf.reduce_sum(kl_1d,axis=1))

    return kl_batch

def elbo(z_mu,z_rho,decoded_img,original_img):
    # reconstruction loss
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(original_img - decoded_img),axis=1))
    # kl loss
    kl = kl_loss(z_mu,z_rho)

    return mse,kl


def get_encoder(original_dim, intermediate_dim, latent_dim, func):
    inputs = tf.keras.Input(shape = (original_dim,))
    x = tf.keras.layers.Dense(units=intermediate_dim[0], activation=func)(inputs)
    for i in range(1,len(intermediate_dim)):
        x = tf.keras.layers.Dense(units=intermediate_dim[i], activation=func)(x)
    mu = tf.keras.layers.Dense(units=latent_dim)(x)
    rho = tf.keras.layers.Dense(units=latent_dim)(x)
    Encoder = tf.keras.Model(inputs=inputs,outputs=[mu,rho])
    
    return Encoder

def get_decoder(original_dim, intermediate_dim, latent_dim, func):
    z = tf.keras.Input(shape = (latent_dim,))
    index = len(intermediate_dim)-1
    x = tf.keras.layers.Dense(units=intermediate_dim[index], activation=func)(z)
    for i in range(1, len(intermediate_dim)):
        index = len(intermediate_dim) - 1 - i
        x = tf.keras.layers.Dense(units=intermediate_dim[index], activation=func)(x)
    decoded_img = tf.keras.layers.Dense(units=original_dim)(x)
    Decoder = tf.keras.Model(inputs=z,outputs=[decoded_img])
    
    return Decoder

class VAE(tf.keras.Model):
    def __init__(self,original_dim, intermediate_dim, latent_dim, func):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_block = get_encoder(original_dim, intermediate_dim, latent_dim, func)
        self.decoder_block = get_decoder(original_dim, intermediate_dim, latent_dim, func)

    def call(self,img):
        z_mu,z_rho = self.encoder_block(img)

        epsilon = tf.random.normal(shape=z_mu.shape,mean=0.0,stddev=1.0)
        z = z_mu + tf.math.softplus(z_rho) * epsilon

        decoded_img = self.decoder_block(z)

        return z_mu,z_rho,decoded_img


#read file
mtx_file = args.data_dir[0] + "/matrix.mtx.gz"
mtx = mmread(mtx_file)
#mtx = mmread('/home/ytanaka/scratch/PD_midbrain_v18/matrix/matrix.mtx.gz')
mtx = mtx.transpose()
mtx = normalize(mtx, axis=1,norm='l1')

#Zscore method (default method)
if args.norm == 0:
    scaler = StandardScaler(with_mean=False)
    mtx = scaler.fit_transform(mtx)

#MinMax method is memory efficient, but sensitive to noise
elif args.norm == 1:
    mtx = normalize(mtx, axis=0, norm='max')

#Quantile normalization
elif args.norm == 2:
    mtx = quantile_transform(mtx,axis=0)
 
f_file = args.data_dir[0] + "/features.tsv.gz"
feature = pd.read_csv(f_file,compression='gzip',sep="\t",header=None)
b_file = args.data_dir[0] + "/barcodes.tsv.gz"
barcode = pd.read_csv(b_file,compression='gzip',sep="\t",header=None)
barcode = barcode.loc[:,0]

#set parameters
original_dim = mtx.shape[1]
model = VAE(original_dim, args.intermediate_dim, args.latent_dim,args.func)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')
mse_loss_tracker = tf.keras.metrics.Mean(name='mse_loss')

train_dataset = tf.data.Dataset.from_tensor_slices(mtx.todense())
train_dataset = train_dataset.batch(args.batch_size)

#dimensional reduction
print('Passing the input data into an variational autoencoder')

for epoch in range(args.epoch):
    z_mu_list = None
    beta = args.beta
    

    for step, x_batch_train  in train_dataset.enumerate():
            
        # training loop
        with tf.GradientTape() as tape:
            # forward pass
            z_mu,z_rho,decoded_imgs = model(x_batch_train)
            # compute loss
            decoded_imgs = tf.cast(decoded_imgs,tf.float64)
            mse,kl = elbo(z_mu,z_rho,decoded_imgs,x_batch_train)
            kl = tf.cast(kl,tf.float64)
            loss = mse + beta * kl
            
        # compute gradient
        gradients = tape.gradient(loss,model.variables)

        # update weights
        optimizer.apply_gradients(zip(gradients, model.variables))
            
        # update metrics
        kl_loss_tracker.update_state(beta * kl)
        mse_loss_tracker.update_state(mse)

        # save encoded means and labels for latent space visualization
        if z_mu_list is None:
            z_mu_list = z_mu
        else:
            z_mu_list = np.concatenate((z_mu_list,z_mu),axis=0)

    # display metrics at the end of each epoch.
    epoch_kl,epoch_mse = kl_loss_tracker.result(),mse_loss_tracker.result()
    print(f'epoch: {epoch}, mse: {epoch_mse:.4f}, kl_div: {epoch_kl:.4f}')

dataset = pd.DataFrame(z_mu_list,index=barcode)
dataset.columns = [args.prefix + str(j) for j in range(len(dataset.columns))]
dataset.to_csv(args.output[0])
