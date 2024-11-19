# scIDST (single-cell Identification of Disease progressive STages)
## Introduction
scIDST is designed to identify progressive disease states of individual cells from single-cell/nuclei RNA-seq (scRNA-seq) by weakly-supervised deep learning approach. Over the past decade, single-cell transcriptome profiling has been applied to various patient-derived samples to better understand and counter a variety of diseases. Comparative analysis with healthy donor data is widely implemented to identify potential genes that may be involved in disease progression. However, the patient-derived biospecimen is composed of mixture of cells in different disease stages and even contains a portion of healthy cells. Such high heterogeneity obscures differential expression between patient and healthy donors and hinders identification of bona fide disease-associated gene expression patterns. To overcome the heterogeneous disease states in patient-derived single-cell data, scIDST infers disease progression level of individual cells with weak supervision approach and segregate diseased cells from healthy or early disease stage cells. 
![alt text](Fig/model.png)

## Requirement
The Python package requirements are in the file `requirements.txt`. The program was tested in python 3.10.

## Downloading
```{r eval=FALSE}
git clone https://github.com/ytanaka-bio/scIDST
cd scIDST
pip install -r requirements.txt --user
```
## Preprocessing
Map raw sequence data to reference genome by [CellRanger](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger). File set of sparse matrix generated by CellRanger (matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz) will be used as input of scIDST.

## Get Started
1. Dimensional Reduction by autoencoder (multi thread (-t) is recommended)
```{r eval=FALSE}
$ python autoencoder.py matrix/outs/filtered_feature_bc_matrix/ reduced_data.csv auto_model -x 100 -t 8 -d autoencode
```
2. Convertion of binary label into probablistic label
```{r eval=FALSE}
$ python reef_analysis.py reduced_data.csv label.csv disease -o disease_plabel.csv
$ python reef_analysis.py reduced_data.csv label.csv age -o age_plabel.csv
$ python reef_analysis.py reduced_data.csv label.csv sex -o sex_plabel.csv
$ python convert_label.py -f disease age sex -w disease_plabel.csv age_plabel.csv sex_plabel.csv -p label.csv -o label_ws.csv
```
3. Training a neural network model with probablistic label. train command generates one model directory `<prefix>_model/` and one prediction file `<prefix>_result.csv`.
```{r eval=FALSE}
$ python classifier_analysis.py train -i reduced_data.csv -p label_ws.csv -o train -t 8 #Generate train_model directory and train_result.csv file
```
## Optional
1.1. If you want to perform dimensional reduction with pre-trained autoencoder (e.g. `auto_model/` directory), run autoencoder.py with `-r 0` option:
```{r eval=FALSE}
$ python classifier_analysis.py other_matrix/outs/filtered_feature_bc_matrix/ reduced_data_other.csv auto_model -r 0
```
3.1. Training a neural network model with binary label (Supervised learning mode).
```{r eval=FALSE}
$ python classifier_analysis.py train -i reduced_data.csv -p label.csv -o train_bin -t 8 
```
3.2. Evaluating the performance of the learning model with `-l` option that generates two additional files: `<prefix>_test.csv` and `<prefix>_predict.csv`.
```{r eval=FALSE}
$ python classifier_analysis.py train -i reduced_data.csv -p label_ws.csv -o train -t 8 -l 
```
Then, draw ROC curve using `plot_ROC.R` in `R_utils/` directory.
```{r eval=FALSE}
> source("R_utils/plot_ROC.R")
> plot_ROC("train_test.csv","train_predict.csv")
```
3.3. If you want to predict using pre-trained learning model (e.g. `train_model/` directory), run classifier_analysis.py with `predict` mode:
```{r eval=FALSE}
$ python classifier_analysis.py predict -i reduced_data_other.csv -p label_other.csv -d train_model -o predict_other.csv
```
Note: Please prepare temporary phenotype file (e.g. `label_other.csv`) that is zero matrix by setting cells and phenotypes as row and column.
```{r eval=FALSE}
                     disease age sex
AAACAGCCAACGTGCT-1_1       0   0   0
AAACAGCCACAACAAA-1_1       0   0   0
AAACATGCAATAACGA-1_1       0   0   0
AAACCGAAGGACCGCT-1_1       0   0   0
```
3.4. Since hyperparameter tuning spends large time, users want to test the classification model with defined parameteres. Another python script, `tensor_analysis.py`, can generate artifical neural network with user-defined parameters.
```{r eval=FALSE}
usage: tensor_analysis.py [-h]  {train,predict} ...

Run deep learning with supervised or weekly-supervised mode

positional arguments:
  mode             Choose learning mode
  {train,predict}  sub-command help
    train          training of learning model
    predict        prediction by learning model

optional arguments:
  -h, --help       show this help message and exit
[ytanaka@cdr1793 github]$ python tensor_analysis.py train -h
usage: tensor_analysis.py train [-h] [-i DATA_FILE] [-p PHENOTYPE_FILE]
                                [-o OUT_FILE] [-v RATIO_VAL] [-s RATIO_TEST]
                                [-b BATCH_SIZE] [-r LEARNING_RATE]
                                [-z OPT_FUNCTION] [-f LOSS_FUNCTION]
                                [-m METRICS] [-e EPOCH]
                                [-a [ACT_FUNCTION [ACT_FUNCTION ...]]]
                                [-n [NODE [NODE ...]]] [-l] [-t THREAD]

optional arguments:
  -h, --help            show this help message and exit
  -i DATA_FILE          data filename (default: None)
  -p PHENOTYPE_FILE     phenotype filename (default: None)
  -o OUT_FILE           output prefix (<prefix>_test.csv (with -l option),
                        <prefix>_predict.csv (with -l option),
                        <prefix>_result.csv, and <prefix>_model/) (default:
                        None)
  -v RATIO_VAL          ratio of validation dataset (default: 0.2)
  -s RATIO_TEST         ratio of test dataset (default: 0.2)
  -b BATCH_SIZE         batch size (default: 2048)
  -r LEARNING_RATE      learning rate (default: 0.0001)
  -z OPT_FUNCTION       optimizer function (default: Adam)
  -f LOSS_FUNCTION      loss function (default: binary_crossentropy)
  -m METRICS            metrics (default: accuracy)
  -e EPOCH              epoch (default: 100)
  -a [ACT_FUNCTION [ACT_FUNCTION ...]]
                        activation function in each layer (default: ['relu',
                        'relu', 'relu'])
  -n [NODE [NODE ...]]  number of node in each layer (default: [500, 250, 50])
  -l                    Evaluate model? (default: False)
  -t THREAD             number of threads (default: 1)
```
Instead of `classifier_analysis.py`, train the neural network by `tensor_analysis.py`.
```{r eval=FALSE}
$ python tensor_analysis.py train -i reduced_data.csv -p label_ws.csv -o train_fixedparam 
```

## Tips
### Prepare binary label in R
When label table will be generated by R, please save the table with `col.names=NA` as follow:
```{r eval=FALSE}
> head(label)
                     disease age sex
AAACAGCCAACGTGCT-1_1       0   1   1
AAACAGCCACAACAAA-1_1       0   1   1
AAACATGCAATAACGA-1_1       0   1   1
AAACCGAAGGACCGCT-1_1       0   1   1
AAACCGAAGTGTTGTA-1_1       0   1   1
AAACCGCGTTACATCC-1_1       0   1   1
> write.table(label,"label.csv",sep=",",quote=F,col.names=NA)
```
### Convert Seurat object to the sparse matrix file set
If quality control is performed by Seurat, please generate sparse matrix file set using `sgMatrix_table.R` script in `R_utils` directory.
```{r eval=FALSE}
> source("R_utils/sgMatrix_table.R") #Require Seurat, Matrix, and R.utils libraries.
> sgMatrix_table(seurat,"matrix")
```
### Calculation of probabilistic label
If you encounter error in reef_analysis.py, please adjust beta value threshold with `-b` option. 
```
python reef_analysis.py reduced_data.csv label.csv disease -o disease_plabel.csv -b 0.3
```
For detail of beta value, please see [Reef/Snuba](https://github.com/HazyResearch/reef/blob/master/%5B1%5D%20generate_reef_labels.ipynb).

### ImportError: libgfortran.so.4
If your system does not have `libgfortran.so.4`, you can use this shared library in CellRanger. If you unpack CellRanger in your home directory, please prepend `cellranger-X.X.X/external/anaconda/lib` directory into LD_LIBRARY_PATH as follow:
```
tar xvfz cellranger-X.X.X.tar.gz
export LD_LIBRARY_PATH=~/cellranger-X.X.X/external/anaconda/lib/:$LD_LIBRARY_PATH     # X.X.X corresponds to version of CellRanger, which you downloaded.
```

## Citation
F. Wehbe, L. Adams, J. Babadoudou, S. Yuen, Y.-S. Kim and Y. Tanaka, Inferring Disease Progressive Stages in Single-Cell Transcriptomics Using Weakly-Supervised Deep Learning Approach, ***Genome Research***(doi: 10.1101/gr.278812.123)

## Reference
- L. Adams, M.K. Song, Y. Tanaka# and Y.-S. Kim#, Single-nuclei paired multiomic analysis of young, aged, and Parkinson's disease human midbrain reveals age-associated glial changes and their contribution to Parkinson's disease, ***Nat Aging.*** 2024 Mar;4(3):364-378 [PubMed](https://pubmed.ncbi.nlm.nih.gov/38491288/) #Co-corresponding authors
- P. Varma, C. Re, Snuba: automating weak supervision to label training data. ***Proceedings VLDB Endowment.*** 2018 Nov;12(3):223-236 [PubMed](https://pubmed.ncbi.nlm.nih.gov/31777681/)
