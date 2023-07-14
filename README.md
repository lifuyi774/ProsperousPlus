# ProsperousPlus: a one-stop and comprehensive platform for accurate protease-specific substrate cleavage prediction and machine-learning model construction.
## introduction

Proteases contribute to a broad spectrum of cellular functions. Given a relatively limited amount of experimental data, development of accurate sequence-based predictors of substrate cleavage sites facilitates better understanding of protease functions and substrate specificity. While many protease-specific predictors of substrate cleavage sites were developed, these efforts are outpaced by the growth of the protease substrate cleavage data. In particular, since data for 100+ protease types are available and this number continues to grow, it becomes impractical to publish predictors for new protease types, and instead it might be better to provide a computational platform that helps users to quickly and efficiently build predictors that address their specific needs. To this end, we conceptualized, developed, tested, and released a versatile bioinformatics platform, ProsperousPlus, that empowers users, even those with no programming or little bioinformatics background, to build fast and accurate predictors of substrate cleavage sites. ProsperousPlus facilitates the use of the rapidly accumulating substrate cleavage data to train, empirically assess and deploy predictive models for user-selected substrate types. Benchmarking tests on test datasets show that our platform produces predictors that on average exceed predictive performance of current state-of-the-art approaches. ProsperousPlus is available as a webserver and a stand-alone software package at http://prosperousplus.unimelb-biotools.cloud.edu.au/.

## Environment
* Anaconda
* python 3.7.13
* JDK 17

## Dependency

* pandas		1.3.5
* numpy		1.19.5
* scikit-learn	0.23.2
* scipy		1.5.4
* pycaret	2.3.10
* shap		0.42.0
* biopython	1.81
* matplotlib	3.5.3
* weblogo	3.7.12
* catboost 1.2
* lightgbm 3.3.5
* xgboost 1.6.2
* Cython 0.29.36
* pymrmr 0.1.11
* redis 4.6.0


## Usage

To get the information the user needs to enter for help, run:
    python ProsperousPlus.py --help
 or
    python ProsperousPlus.py -h

as follows:

>python ProsperousPlus.py -h
usage: it's usage tip.
optional arguments:
  -h, --help            show this help message and exit
  --inputType            FASTA or peptide.
  --config            The config file.
  --trainfile            The path to the training set file containing the protease sequences in fasta(peptide) format, where the length of the sequences is 8/10/12/14/16/18/20.
  --protease            The protease you want to predict cleavage to, eg: A01.001' + '\n'
                'Or if you want to build a new model, please create a name. There should no space in the model name.
  --outputpath            The path of output.
  --testfile            The path to the test data file containing the protease sequences in fasta(peptide) format, where the length of the sequences is 8/10/12/14/16/18/20. If not, it will be divided from the training set.
  --predictfile            The path to the prediction data file containing the protease sequences in fasta(peptide) format, where the length of the sequences is 8/10/12/14/16/18/20.
  --mode            choose the mode, three modes can be used: prediction, TrainYourModel, UseYourOwnModel, only select one mode each time.
  --modelfile            The file of trained model generated from the TrainYourModel mode. eg 0_model
  --SHAP           Select Yes or No to control the program to calculate SHAP.
  --PLOT            Selecting Yes or No to control whether the program computes the visualization of cleavage sites.
  --processNum            The number of processes in the program. Note: Integer values represent the number of processes. --processNum setting can speed up the running efficiency of the program, but it also takes up more computing resources.

## Examples:

### Prediction:
```python ProsperousPlus.py --predictfile predict.fasta --outputpath results --inputType fasta --protease A01.001 --mode prediction --PLOT Yes --processNum 2```
### TrainYourModel:
```python ProsperousPlus.py --trainfile data/A01.001_trainset_1_1.fasta --outputpath resultfile --inputType fasta --protease A01.001 --mode TrainYourModel --SHAP Yes --processNum 2```
### useYourModel:
```python ProsperousPlus.py --predictfile predict.fasta --outputpath resultfile --inputType fasta --protease A01.001 --mode UseYourOwnModel --modelfile modelfile --processNum 2```
## Output:

When the task is prediction or UseYourOwnModel, the result of the program is the test performance of the model; while when the task is TrainYourModel, the result of the program includes the model files, test results, matrix, ROC graph,SHAP(if selected), and the visualization of cleavage sites (if selected).

1. matrix: used to encode the features of the sequence.

## Note:

1. The config file contains the default base model for the program.
2. Under the source code of "shap.summary_plot", add two parameters to enable the saving of the SHAP plot.

"shap.summary_plot(shap_values, X_train,max_display=50,show=False, save=True,path='./shap/%s.png'%(d))"

Add to the bottom of the summary_plot function:
if save:
        pl.savefig(path)
        pl.close()