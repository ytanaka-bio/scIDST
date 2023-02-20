#0 initialization
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

#parse argument
parser = argparse.ArgumentParser(description='Dimensional reduction with autoencoder',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-e', help='epoch for best hyperparameter configuations', default=10, type=int, dest="epoch")
parser.add_argument('-x', help='Total number of trials for random search tuner', default=100, type=int, dest="max_trial")
parser.add_argument('-d', help='directory for autoencoder optimization', default='./autoencoder', type=str, dest="dir")
parser.add_argument('-v', help='ratio of validation dataset', default=0.2, type=float,dest="ratio_val")
parser.add_argument('-n', help='number of cells used for autoencoder training',default=20000,type=int,dest="train_num")
parser.add_argument('-s', help='number of split of cells when running autoencoder (larger number will use lower memory)',default=10,type=int,dest="split")
parser.add_argument('-t', help="number of threads",default=1,type=int,dest="thread")
parser.add_argument('-r', help="mode (1 if optimize autoencoder, 0 if use existing autoencoder)",default=1, type=int, dest="dim_type")
parser.add_argument('-c', help="prefix of column name in output",default="Input_",type=str, dest="prefix")
parser.add_argument('-m', help="Normalization method (0 (Zscore), 1 (MinMax), 2 (Quantile))",default=2,type=int,dest="norm")
parser.add_argument('data_dir',nargs=1,help='cellranger matrix directory',type=str)
parser.add_argument('output',nargs=1,help='output filename',type=str)
parser.add_argument('model_file', nargs=1,help='directory of autoencoder model (output file with -r 1, input file with -r 0)',type=str)

args = parser.parse_args()

#Type of dimensional reduction
dim_type = args.dim_type 
if dim_type != 0 and dim_type != 1:
    sys.exit("Type of dimensional reduction should be 1 (Autoencoder optimization) or 0 (Existing autoencoder)")

#run main
import os, psutil
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(args.thread)
tf.config.threading.set_intra_op_parallelism_threads(args.thread)
from functools import reduce
from tensorflow import feature_column, keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize, quantile_transform, StandardScaler
import keras_tuner as kt
from scipy.io import mmread
from scipy.stats import zscore
import scipy.sparse as sp
import random


#Read file
mtx_file = args.data_dir[0] + "/matrix.mtx.gz"
mtx = mmread(mtx_file)
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

#select barcode
if len(barcode) > args.train_num:
    select = random.sample(range(0,len(barcode)),args.train_num)
else:
    print("The defined number of cells for autoencoder training (-n) is larger than the number of sample in the input")
    print("Thus, all input data will be used for autoencoder training")
    select = range(0,len(barcode))

barcode_select = barcode.loc[select,0]
mtx_select = mtx.tocsr()[select,]

#define functions
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row,coo.col]).transpose()
    return tf.sparse.SparseTensor(indices,coo.data,coo.shape)

def hypermodel(hp):
    depth = hp.Int('depth', min_value = 0, max_value = 10, step = 1)
    hidden_dim = hp.Int('hidden_dim', min_value = 0, max_value = 1000, step = 200)
    latent_dim = hp.Int('latent_dim', min_value = 100, max_value = 500, step = 100)
    model = build_autoencoder(depth, hidden_dim, latent_dim, input_dim)
    return model

def build_autoencoder(depth, hidden_dim, latent_dim, input_dim):
    # Encoder
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.Input(shape=(input_dim,),sparse=True,dtype=tf.float64))
    encoder.add(Dense(hidden_dim, activation='tanh'))
    for i in range(1, depth):
        encoder.add(Dense(hidden_dim, activation='tanh'))    
    encoder.add(Dense(latent_dim, activation='tanh'))
    # Dencoder
    decoder = tf.keras.Sequential()
    for i in range(1, depth):
        decoder.add(Dense(hidden_dim, activation='tanh'))    
    decoder.add(Dense(input_dim, activation='sigmoid'))
    # Combining
    combined_model = tf.keras.Sequential()
    combined_model.add(encoder)
    combined_model.add(decoder)
    combined_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return combined_model

def autoencoder_reduction_presplit(dataset, proj, dir, epoch = 10, max_trial=10):
    train, val = train_test_split(dataset, test_size=args.ratio_val)
    #train_st = convert_sparse_matrix_to_sparse_tensor(train)
    #val_st = convert_sparse_matrix_to_sparse_tensor(val)
    #dataset_st = convert_sparse_matrix_to_sparse_tensor(dataset)
    
    autoencoder_tuner = kt.RandomSearch(hypermodel, objective = 'val_loss', max_trials = max_trial, executions_per_trial = 2, directory = dir, project_name = proj, overwrite = False)
    autoencoder_tuner.search(train.todense(), train.todense(), epochs = epoch, validation_data = (val.todense(), val.todense()))
    best_hyperparameters = autoencoder_tuner.get_best_hyperparameters(1)[0]
    latent_dim = best_hyperparameters.get('latent_dim')
    hidden_dim = best_hyperparameters.get('hidden_dim')
    depth = best_hyperparameters.get('depth')
    #print(latent_dim,hidden_dim,depth)
    ## AUTOENCODER
    # Autoencoder from TensorFlow.Keras
    class Autoencoder(Model):
        def __init__(self, latent_dim = latent_dim):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            # Encoder
            self.encoder = tf.keras.Sequential()
            self.encoder.add(InputLayer(input_shape=(input_dim,),sparse=True))
            self.encoder.add(Dense(hidden_dim, activation='tanh'))
            for i in range(1, depth):
                self.encoder.add(Dense(hidden_dim, activation='tanh'))    
            self.encoder.add(Dense(latent_dim, activation='tanh'))
            # Dencoder
            self.decoder = tf.keras.Sequential()
            for i in range(1, depth):
                self.decoder.add(Dense(hidden_dim, activation='tanh'))    
            self.decoder.add(Dense(input_dim, activation='sigmoid'))
        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    # print(best_hyperparameters.get('depth'), best_hyperparameters.get('hidden_dim'), best_hyperparameters.get('latent_dim'))
    autoencoder = Autoencoder()
    autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
    autoencoder.fit(dataset.todense(), dataset.todense(), epochs=epoch)
    encoded_df = autoencoder.encoder(np.array(dataset.todense())).numpy()
    # convert to df
    df = pd.DataFrame(encoded_df, columns= [args.prefix + str(j) for j in range(latent_dim)])
    return df, autoencoder

#dimensional reduction
if dim_type == 1:
    print('Passing the input data into an optimized autoencoder')
    input_dim = mtx.shape[1]
    dataset, model = autoencoder_reduction_presplit(mtx_select, epoch = args.epoch, max_trial = args.max_trial, dir = args.dir, proj = args.model_file[0])
    model.save(filepath=args.model_file[0],overwrite=True)

else:
    print('Reading the existing autoencoder')
    model = tf.keras.models.load_model(args.model_file[0])
    
for x in range(0,args.split):
    start = int(x * len(barcode)/args.split)
    if x == (args.split - 1):
        end = len(barcode)
    else:
        end = int((x+1) * len(barcode)/args.split)
    mtx_st = convert_sparse_matrix_to_sparse_tensor(mtx[range(start,end),])
    if x == 0:
        encoded = model.encoder(mtx_st)
    else:
        encoded2 = model.encoder(mtx_st)
        encoded = tf.concat([encoded,encoded2],0)
            
dataset = pd.DataFrame(encoded.numpy(), index = barcode.loc[:,0])
dataset.columns = [args.prefix + str(j) for j in range(len(dataset.columns))]
dataset.to_csv(args.output[0])
