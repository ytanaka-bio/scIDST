#module load cuda
#export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/ytanaka/program/cellranger-7.0.0/external/anaconda/lib/:$LD_LIBRARY_PATH

#0 initialization
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

#0.1 parse argument
parser = argparse.ArgumentParser(description='Run deep learning with supervised or weekly-supervised mode',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('mode',help='Choose learning mode',action='store_true')
subparsers = parser.add_subparsers(help='sub-command help')

#0.2 subparameters mode
parser_train = subparsers.add_parser('train',help='training of learning model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_train.add_argument('-i',nargs=1,help='data filename',type=argparse.FileType('r'),dest="data_file")
parser_train.add_argument('-p',nargs=1,help='phenotype filename',type=argparse.FileType('r'),dest="phenotype_file")
parser_train.add_argument('-o',nargs=1,help='output prefix (<prefix>_test.csv (with -l option), <prefix>_predict.csv (with -l option), <prefix>_result.csv, and <prefix>_model/)',type=str,dest="out_file")
parser_train.add_argument('-v', help='ratio of validation dataset', default=0.2, type=float, dest="ratio_val")
parser_train.add_argument('-s', help='ratio of test dataset', default=0.2, type=float, dest="ratio_test")
parser_train.add_argument('-b', help='batch size', default=2048, type=int, dest="batch_size")
parser_train.add_argument('-r', help='learning rate', default=0.0001,type=float, dest="learning_rate")
parser_train.add_argument('-z', help='optimizer function', default="Adam",type=str,dest="opt_function")
parser_train.add_argument('-f', help='loss function',default="binary_crossentropy",type=str,dest="loss_function")
parser_train.add_argument('-m', help='metrics',default='accuracy',type=str,dest="metrics")
parser_train.add_argument('-e', help='epoch',type=int, default=100,dest='epoch')
parser_train.add_argument('-a', help='activation function in each hidden layer', nargs='*',type=str,default=["relu","relu","relu"],dest="act_function")
parser_train.add_argument('-n', help='number of node in each hidden layer', nargs='*',type=int,default=[500,250,50],dest="node")
parser_train.add_argument('-u', help='activation function in output layer', type=str,default='softmax',dest="act_out")
parser_train.add_argument('-l', help='Evaluate model?',action='store_true',default=False,dest="eval")
parser_train.add_argument('-t', help="number of threads",type=int,dest="thread")

parser_predict = subparsers.add_parser('predict',help='prediction by learning model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_predict.add_argument('-i',nargs=1,help='data filename',type=argparse.FileType('r'),dest="data_file")
parser_predict.add_argument('-p',nargs=1,help='phenotype filename',type=argparse.FileType('r'),dest="phenotype_file")
parser_predict.add_argument('-d',nargs=1,help='model directory',type=str,dest="model_file")
parser_predict.add_argument('-o',nargs=1,help='output filename',type=str,dest="out_file")
parser_predict.add_argument('-t', help="number of threads",default=1,type=int,dest="thread")

#0.3. get arguments of selected mode
args = parser.parse_args()
if sys.argv[1] == "train":
  if args.act_function != None and args.node != None:
    if len(args.act_function) != len(args.node):
      sys.exit("The number of activation function list (-a) should be same with that of node number list (-n)\n")
      
#0.4. import required packages
import numpy as np
import pandas as pd

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(args.thread)
tf.config.threading.set_intra_op_parallelism_threads(args.thread)

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from keras.models import load_model

#1 define functions
#1.1. model information
###############################################################################
## Please modify this def if you want to test different neural network model ##
def model_creation(input_layer, num_outnode):
  model = tf.keras.Sequential([
    feature_layer
  ])
  if args.act_function != None:
    num_layer = len(args.act_function)
    for i in range(0,num_layer):
      model.add(layers.Dense(args.node[i], activation=args.act_function[i]))
      
  model.add(layers.Dense(num_outnode, activation=args.act_out))   
  return model

###############################################################################

#1.2. convert dataset format
def df_to_dataset(dataframe, dindex, feature, shuffle=False, batch_size=2048):
  dataframe = dataframe.copy()
  labels = dataframe.loc[:,feature]
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe.loc[:,dindex]), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

#1.3. use weekly_supervised mode
def weekly_supervised(dataframe, feature, prob_file):
  dataframe = dataframe.copy()
  prob = pd.read_csv(prob_file)
  prob_mean = prob.mean(axis=1)
  dataframe.loc[:,feature] = np.array(prob_mean)
  return dataframe


#2 main
#2.1. read files
#2.1.1. data file
dataframe = pd.read_csv(args.data_file[0],index_col=0)
dindex = dataframe.columns
output = args.out_file[0]

#2.1.2. phenotype or model file
#2.1.2.1 phenotype file
dataout = pd.read_csv(args.phenotype_file[0],index_col=0)
dataframe = dataframe.combine_first(dataout)
pheno = dataout.columns

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

  model = model_creation(feature_layer, len(pheno))

  #2.3.2. model compile
  model.compile(optimizer=args.opt_function,
                loss=args.loss_function,
                metrics=[args.metrics])

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
  
