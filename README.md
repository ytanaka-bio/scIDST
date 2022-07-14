# scIDST (single-cell Identificaiton of Disease STate)
## Introduction
scIDST is designed to identify progressive disease states of individual cells from single-cell/nuclei RNA-seq (scRNA-seq) by weekly-supervised deep learning approach. Over the past decade, single-cell transcriptome profiling has been applied to various patient-derived samples to better understand and counter a variety of diseases. Comparative analysis with healthy donor data is widely implemented to identify potential genes that may be involved in disease progression. However, the patient-derived biospecimen is composed of mixture of cells in different disease stages and even contains a portion of healthy cells. Such high heterogeneity obscures differential expression between patient and healthy donors and hinders identification of bona fide disease-associated gene expression patterns. To overcome the heterogeneous disease states in patient-derived single-cell data, scIDST infers disease progression level of individual cells with weak supervision approach and segregate diseased cells from healthy or early disease stage cells. 
![alt text](model.png)

## Requirement
The Python package requirements are in the file `requirements.txt`.

## How to start
```{r eval=FALSE}
git clone https://github.com/ytanaka-bio/scIDST
cd scIDST
pip install -r requirements.txt --user
```

## How to use
1. Dimensional Reduction
2. Convertion of binary label into probablistic label
3. Training a neural network model with probablistic label
3-1. Evaluation
3-2. Training
3-3. Prediction

#Tips
```{r eval=FALSE}
> head(label)
                     disease age sex
AAACAGCCAACGTGCT-1_1       0   1   1
AAACAGCCACAACAAA-1_1       0   1   1
AAACATGCAATAACGA-1_1       0   1   1
AAACCGAAGGACCGCT-1_1       0   1   1
AAACCGAAGTGTTGTA-1_1       0   1   1
AAACCGCGTTACATCC-1_1       0   1   1
> write.table(label,"label.csv",sep=",",quote=F,col.names=F)
```

## Citation
Paper in preparation

## Reference
P. Varma, C. Re, Snuba: automating weak supervision to label training data. Processings of VLDB Endowment 12, 223-236 (2018). [PubMed](https://pubmed.ncbi.nlm.nih.gov/31777681/)
