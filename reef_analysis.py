#python version 3.8.10
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ytanaka/program/cellranger-7.0.0/external/anaconda/lib/
#modify data/loader.py import cross_validation => from sklearn.model_selection import train_test_split
#modify program_synthesis/label_aggregator.py print * => print(*)
#modify program_synthesis/verifier.py from label_aggregator import Label_Aggregator => from program_synthesis.label_aggregator import *
#modify program_synthesis/synthesizer.py marginals = hf.predict_proba(X[:,feat_combos[i]])[:,1] => marginals = hf.predict_proba(X[:,feat_combos[i]])

import argparse
import warnings
warnings.filterwarnings("ignore")

#parse argument
parser = argparse.ArgumentParser(description='Calculate probablistic label from automatically-generated heuristic',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', help='output file name',default="<feature>_plabel.csv",type=str,dest="output")
parser.add_argument('-t', help='ratio of test dataset', default=0.2, type=float, dest="ratio_test")
parser.add_argument('-v', help='ratio of validation dataset', default=0.2, type=float, dest="ratio_val")
parser.add_argument('-b', help='beta parameter for heuristic generator', default=0.5, type=float, dest="beta")
parser.add_argument('-i', help='max iteration for synthesize-prune-verify process', default=50, type=int,dest="iter")
parser.add_argument('-r', help='Number of runs of reef program', default=10, type=int,dest="run")
parser.add_argument('data_file',nargs=1,help='data filename',type=argparse.FileType('r'))
parser.add_argument('phenotype_file',nargs=1,help='phenotype filename',type=argparse.FileType('r'))
parser.add_argument('feature',nargs=1,help='feature, which want to be converted to probalistic label',type=str)

args = parser.parse_args()

datafile = args.data_file[0]
phenofile = args.phenotype_file[0]
feature = args.feature[0]
if args.output == "<feature>_plabel.csv":
    output = feature + "_plabel.csv"
else:
    output = args.output


#run main
import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv(datafile)
dataframe = dataframe.apply(zscore)
dindex = dataframe.columns
dataout = pd.read_csv(phenofile)
dataframe = dataframe.combine_first(dataout)

def run_reef(dataframe,dindex,label,iter=50, t_size=0.2, v_size=0.2):
    dataframe = dataframe.copy()
    reef_result = dataframe.loc[:,label]
    
    train, test = train_test_split(dataframe, test_size=t_size)
    train, val = train_test_split(train, test_size=v_size)

    train_matrix = train.loc[:,dindex].to_numpy()
    val_matrix = val.loc[:,dindex].to_numpy()
    train_ground = train.loc[:,label].to_numpy()
    val_ground = val.loc[:,label].to_numpy()

    #convert label -1,1 to 0,1
    train_ground = (train_ground * 2) - 1
    val_ground = (val_ground * 2) - 1

    from program_synthesis.heuristic_generator import HeuristicGenerator
    hg = HeuristicGenerator(train_matrix, val_matrix, val_ground, train_ground, b=0.5)

    validation_accuracy = []
    training_accuracy = []
    validation_coverage = []
    training_coverage = []

    training_marginals = []
    idx = None

    for i in range(3,iter+2):
        if (i-2)%5 == 0:
            print("Running iteration: ", str(i-2))        
        #Repeat synthesize-prune-verify at each iterations
        if i == 3:
            hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
        else:
            hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
        hg.run_verifier()    
        #Save evaluation metrics
        va,ta, vc, tc = hg.evaluate()
        validation_accuracy.append(va)
        training_accuracy.append(ta)
        training_marginals.append(hg.vf.train_marginals)
        validation_coverage.append(vc)
        training_coverage.append(tc)    
        #Find low confidence datapoints in the labeled set
        hg.find_feedback()
        idx = hg.feedback_idx 
        #Stop the iterative process when no low confidence labels
        if idx == []:
            break

    print("Program Synthesis Train Accuracy: ", training_accuracy[-1])
    print("Program Synthesis Train Coverage: ", training_coverage[-1])
    print("Program Synthesis Validation Accuracy: ", validation_accuracy[-1])

    reef_result.loc[train.index,] = training_marginals[-1]
    return reef_result

#convert disease and age label to probablistic label
result = []
for x in range(0,args.run):
    result.append(np.array(run_reef(dataframe, dindex, feature, args.iter, args.ratio_test, args.ratio_val)))

pd.DataFrame(result).transpose().to_csv(output)

