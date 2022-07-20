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
parser.add_argument('-p', help='project name', default='autoencoder', type=str,dest="proj")
parser.add_argument('-v', help='ratio of validation dataset', default=0.2, type=float,dest="ratio_val")
parser.add_argument('-t', help="number of threads",default=1,type=int,dest="thread")
parser.add_argument('-r', help="mode (1 if optimize autoencoder, 0 if use existing autoencoder)",default=1, type=int, dest="dim_type")
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
from scipy.io import mmread
from scipy.stats import zscore

#Read file
mtx_file = args.data_dir[0] + "/matrix.mtx.gz"
mtx = mmread(mtx_file)
mtx_norm = 10000 * mtx.transpose() / mtx.sum(axis=0).transpose()    #normalized by total number of reads and multiplied by 10000
scRNA = pd.DataFrame(mtx_norm)
scRNA = scRNA.apply(zscore)
scRNA = scRNA.replace(np.nan, 0)
f_file = args.data_dir[0] + "/features.tsv.gz"
feature = pd.read_csv(f_file,compression='gzip',sep="\t",header=None)
b_file = args.data_dir[0] + "/barcodes.tsv.gz"
barcode = pd.read_csv(b_file,compression='gzip',sep="\t",header=None)
scRNA.columns = feature.loc[:,1]
scRNA.index = barcode.loc[:,0]


#define functions
def hypermodel(hp):
    depth = hp.Int('depth', min_value = 0, max_value = 10, step = 1)
    hidden_dim = hp.Int('hidden_dim', min_value = 0, max_value = 1000, step = 200)
    latent_dim = hp.Int('latent_dim', min_value = 100, max_value = 500, step = 100)
    model = build_autoencoder(depth, hidden_dim, latent_dim, input_dim)
    return model

def build_autoencoder(depth, hidden_dim, latent_dim, input_dim):
    # Encoder
    encoder = tf.keras.Sequential()
    encoder.add(Dense(hidden_dim, activation='tanh', input_shape=(input_dim,)))
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
    autoencoder_tuner = kt.RandomSearch(hypermodel, objective = 'val_loss', max_trials = max_trial, executions_per_trial = 2, directory = dir, project_name = proj, overwrite = False)
    autoencoder_tuner.search(dataset, dataset, epochs = epoch, validation_split = 0.2)
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
            self.encoder.add(Dense(hidden_dim, activation='tanh', input_shape=(input_dim,)))
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
    autoencoder.fit(dataset, dataset, epochs=epoch)
    encoded_df = autoencoder.encoder(np.array(dataset)).numpy()
    # convert to df
    df = pd.DataFrame(encoded_df, index = dataset.index, columns= ['input_' + str(j) for j in range(latent_dim)])
    return df, autoencoder

#dimensional reduction
if dim_type == 1:
    print('Passing the input data into an optimized autoencoder')
    input_dim = len(scRNA.columns)
    dataset, model = autoencoder_reduction_presplit(scRNA, epoch = args.epoch, max_trial = args.max_trial, dir = args.dir, proj = args.proj)
    reduced_dim = len(dataset.columns)
    dataset.to_csv(args.output[0])
    model.save(filepath=args.model_file[0],overwrite=True)

else:
    print('Reading the existing autoencoder')
    model = tf.keras.models.load_model(args.model_file[0])
    encoded = model.encoder(np.array(scRNA)).numpy()
    dataset = pd.DataFrame(encoded, index = scRNA.index)
    dataset.columns = ['input_' + str(j) for j in range(len(dataset.columns))]
    dataset.to_csv(args.output[0])
