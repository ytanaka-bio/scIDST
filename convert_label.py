#0 initialization
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

#0.1 parse argument
parser = argparse.ArgumentParser(description='Rewrite phenotype table with probablistic labels',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f',help='Features with probablistic label',nargs='*',type=str,default=None,dest="feature")
parser.add_argument('-w',help='Files including probablistic label',nargs='*',type=str,default=None,dest="prob_file")
parser.add_argument('-p',nargs=1,help='phenotype filename',type=str,dest="phenotype_file")
parser.add_argument('-o',nargs=1,help='output filename',type=str,dest="out_file")

args = parser.parse_args()

#definition
def convert_label(dataframe, feature, prob_file):
  dataframe = dataframe.copy()
  prob = pd.read_csv(prob_file,index_col=0)
  prob_mean = prob.mean(axis=1)
  dataframe.loc[:,feature] = np.array(prob_mean)
  return dataframe


#run main
dataout = pd.read_csv(args.phenotype_file[0],index_col=0)
if args.feature != None:
    num_feature = len(args.feature)
    for i in range(0,num_feature):
      dataout = convert_label(dataout, args.feature[i], args.prob_file[i])

#output
dataout.to_csv(args.out_file[0])
