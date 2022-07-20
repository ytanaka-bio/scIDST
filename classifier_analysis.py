#module load cuda
#export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/ytanaka/program/cellranger-7.0.0/external/anaconda/lib/:$LD_LIBRARY_PATH

#0 initialization
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

#0.1 parse argument
parser = argparse.ArgumentParser(description='Run deep learning classifier',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode',help='Choose learning mode',action='store_true')
subparsers = parser.add_subparsers(help='sub-command help')

#0.2 subparameters mode
parser_train = subparsers.add_parser('train',help='training of learning model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_train.add_argument('-i',nargs=1,help='data filename',type=argparse.FileType('r'),dest="data_file")
parser_train.add_argument('-p',nargs=1,help='phenotype filename',type=argparse.FileType('r'),dest="phenotype_file")
parser_train.add_argument('-o',nargs=1,help='output prefix (<prefix>_test.csv (with -l option), <prefix>_predict.csv (with -l option), <prefix>_result.csv, and <prefix>_model/)',type=str,dest="out_file")
parser_train.add_argument('-v', help='ratio of validation dataset', nargs=1, default=0.2, type=float, dest="ratio_val")
parser_train.add_argument('-s', help='ratio of test dataset', nargs=1, default=0.2, type=float, dest="ratio_test")
parser_train.add_argument('-b', help='batch size', nargs=1, default=2048, type=int, dest="batch_size")
parser_train.add_argument('-e', help='epoch',nargs=1,type=int, default=20,dest='epoch')
parser_train.add_argument('-l', help='Evaluate model?',action='store_true',default=False,dest="eval")
parser_train.add_argument('-m', help="Max trial",nargs=1,type=int, default=200,dest="max_trial")
parser_train.add_argument('-t', help="number of threads",default=1,type=int,dest="thread")
parser_train.add_argument('-d', help='directory for model tuning', default='model_autoencoder_hyperparam_tuning_trans', type=str, dest="dir")

parser_predict = subparsers.add_parser('predict',help='prediction by learning model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_predict.add_argument('-i',nargs=1,help='data filename',type=argparse.FileType('r'),dest="data_file")
parser_predict.add_argument('-p',nargs=1,help='temporary phenotype filename',type=argparse.FileType('r'),dest="phenotype_file")
parser_predict.add_argument('-d',nargs=1,help='model directory',type=str,dest="model_file")
parser_predict.add_argument('-o',nargs=1,help='output filename',type=str,dest="out_file")
parser_predict.add_argument('-b', help='batch size', nargs=1, default=2048, type=int, dest="batch_size")
parser_predict.add_argument('-t', help="number of threads",default=1,type=int,dest="thread")


#0.3. get arguments of selected mode
args = parser.parse_args()

#0.4. import required packages
import numpy as np
import pandas as pd

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(args.thread)
tf.config.threading.set_intra_op_parallelism_threads(args.thread)

from tensorflow import feature_column, keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import keras_tuner as kt
from keras.models import load_model

#1 define functions
#1.1. deep learning model
def deephypermodel(hp):
  depth = hp.Int('layer', min_value = 2, max_value = 10, step = 1)
  hidden_dim = hp.Int('units', min_value = 50, max_value = 500, step = 50)
  act_fct = hp.Choice('act', ['relu','sigmoid','tanh'])
  drop = hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)
  opt = hp.Choice('opt', ['adam','sgd'])
  loss_fct = hp.Choice('loss', ['binary_crossentropy','categorical_crossentropy','mse'])
  learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001, 0.0001, 0.00001])
  model = build_model(depth, hidden_dim, act_fct, opt, loss_fct, drop, learning_rate, feature_layer, out_dim)
  return model

#1.2. Build model
def build_model(depth, hidden_dim, act_fct, opt, loss_fct, drop, learning_rate, feature_layer, out_dim):
  model = keras.Sequential(feature_layer) 
  for i in range(1, depth):
    model.add(layers.Dense(hidden_dim, activation = act_fct))
    
  model.add(layers.Dropout(drop))
  
  model.add(layers.Dense(out_dim, activation = 'softmax')) # CHANGE ACCORDING TO OUTPUT WE WANT TO PREDICT
  model.compile(
    optimizer = opt,
    loss = loss_fct,
    metrics = ['accuracy']) #hp.Choice('metric', ['accuracy', 'precision', 'recall'])
  return model

#1.3. convert dataset format
def df_to_dataset(dataframe, dindex, feature, batch_size=2048):
  dataframe = dataframe.copy()
  labels = dataframe.loc[:,feature]
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe.loc[:,dindex]), labels))
  ds = ds.batch(batch_size)
  return ds

#2 main
#2.1. read files
#2.1.1. data file
dataframe = pd.read_csv(args.data_file[0],index_col=0)
#dataframe = dataframe.apply(zscore)
dindex = dataframe.columns
reduced_dim = len(dindex)
output = args.out_file[0]

#2.1.2. phenotype or model file
#2.1.2.1 phenotype file
dataout = pd.read_csv(args.phenotype_file[0],index_col=0)
dataframe = dataframe.combine_first(dataout)
pheno = dataout.columns
out_dim = len(pheno)

#2.2. Prepare dataset
if sys.argv[1] == 'train':
  if args.eval == True:
    train, test = train_test_split(dataframe, test_size=args.ratio_test)
    train, val = train_test_split(train, test_size=args.ratio_val)
    train_ds = df_to_dataset(train, dindex, pheno, batch_size=args.batch_size)  
    val_ds = df_to_dataset(val, dindex, pheno, batch_size=args.batch_size)
    test_ds = df_to_dataset(test, dindex, pheno, batch_size=args.batch_size)

  train2, val2 = train_test_split(dataframe, test_size=args.ratio_val)
  train2_ds = df_to_dataset(train2, dindex, pheno, batch_size=args.batch_size)
  val2_ds = df_to_dataset(val2, dindex, pheno, batch_size=args.batch_size)
  all_ds = df_to_dataset(dataframe, dindex, pheno, batch_size=args.batch_size)
  
else:
  all_ds = df_to_dataset(dataframe, dindex, pheno, batch_size=args.batch_size)

    
#2.3. model ceation
if sys.argv[1] != 'predict':
  #2.3.1. model parameter optimization
  feature_columns = []
  for x in range(0,len(dindex)): feature_columns.append(feature_column.numeric_column(dindex[x]))
  
  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
  tuner = kt.RandomSearch(deephypermodel, objective='val_accuracy', executions_per_trial = 3, max_trials=args.max_trial, directory = args.dir, project_name = args.out_file[0])
  tuner.search(train2_ds, epochs=args.epoch, validation_data=val2_ds) 
  best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
  
  #2.3.2. model build
  model = tuner.hypermodel.build(best_hps)

  if args.eval == True:
    model.fit(train_ds, validation_data=val_ds, epochs=args.epoch)
    y_score = pd.DataFrame(model.predict(test_ds))
    y_test = test.loc[:,pheno]
    y_score.index = y_test.index
    y_score.columns = y_test.columns
    output1 = output + "_test.csv"
    y_test.to_csv(output1)
    output2 = output + "_predict.csv"
    y_score.to_csv(output2)

  model.fit(train2_ds, validation_data=val2_ds, epochs=args.epoch)
  all_score = pd.DataFrame(model.predict(all_ds))
  all_score.index = dataframe.index
  all_score.columns = pheno
  output5 = output + "_result.csv"
  all_score.to_csv(output5)
  
  output3 = output + "_model"
  model.save(filepath=output3,overwrite=True)
  
else:
  model = tf.keras.models.load_model(args.model_file[0])
  all_score = pd.DataFrame(model.predict(all_ds))
  all_score.index = dataframe.index
  all_score.columns = pheno
  all_score.to_csv(output)
  