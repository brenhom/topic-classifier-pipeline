# Topic Classifier Pipeline
Brendan and Jenn

In this repo we recreate and test the results of the paper [Less is More: Parameter-Free Text Classification with Gzip](https://arxiv.org/abs/2212.09410)

## Plans

We need
* A thing that downloads a dataset and splits it
    * We also need to get stats on the dataset
    
* Something that does the training on a model
* A model object
    * Training
    * Classify
    * Hyperparameter tuning (?)
* Evaluation processing/results display
* A preprocessor class


## Datasets
AG News - huggingface
DBpedia - huggingface
YahooAnswers - no huggingface dataset loader https://paperswithcode.com/dataset/yahoo-answers
20News - huggingface
ohsumed - not in huggingface but there is a hugginf
R8 - huggingface
R52 - huggingface
KinyarwandaNews - huggingface
KirundiNews - huggingface
DengueFilipino - huggingface
SwahiliNews - ?
SogouNews - ?

## References
arXiv:2212.09410v1 [cs.CL] 19 Dec 2022