# semeval-ffnn-baseline  
This project presents a baseline system for Task 12 i.e. Named Entity Recognition (NER) and Concept Resolution subtasks that uses a 2-layer feedforward neural network.
  
Dependencies:
1) ```python```
2) ```geonames-services``` for disabmiguation and normalization

Requirements:
1) Directory containing BRAT annotated files i.e. corpus files containing article texts (.txt) and respective annotation files (.ann). You can extract the training files from the provided dataset and place the .ann and .txt files in the ```data/train``` directory.
2) A file containing word embeddings i.e word vectors that can be loaded using the gensim model. You can download word embeddings trained on PubMed and Wikipedia articles(wikipedia-pubmed-and-PMC-w2v.bin) from http://bio.nlplab.org/ and place the bin file in the ```resources``` directory.

Install dependencies:
```
pip install --upgrade -r requirements.txt
```

To train the model:
1) Create the files required for training by running the following command. 
```
python gen_training_files.py
```
To look at more options for input and output paths, run the command with the ```-h``` flag.

2) Train the model by running the following command:
```
python ffnn_train.py
```
To look at more options for input and paths and hyperparameter configurations run the command with the ```-h``` flag.

3) To annotate files using the trained model, run the following command:
```
python ffnn_run.py
```
