# semeval-ffnn-baseline  
This project presents a baseline system for Task 12 i.e. Named Entity Recognition (NER) and Concept Resolution subtasks that uses a 2-layer feedforward neural network.
  
Dependencies:
1) ```python``` tested with versions 2.7 and 3.5
2) [geonames-services](https://github.com/amagge/geonames-service) for disabmiguation and normalization. Alternatively, you can use GeoNames's search API (with a few modifications in this system).

Install python dependencies:
```
pip install --upgrade -r requirements.txt
```

# Disambiguation
If you wish to perform the disambiguation step alone, this baseline implementation does not require any training but it does depend on the geonames-service. Make sure the geonames-service is configured and running. Keep the corpus directory containing the article texts (.txt) and BRAT annotation (.ann) files in ```data/test```. You can then perform disambiguation by running the following command:
```
python run.py dis data/test
```
You fill find the outputs in the ```out``` directory. To look at more options for input and output paths, run the command with the ```-h``` or ```--help``` flag.

# Training for Detection and Resolution
Both Detection and Disambiguation steps require that a trained model for detection is available.

Requirements:
- Directory containing BRAT annotated files i.e. corpus files containing article texts (.txt) and respective annotation files (.ann). You can extract the training files from the provided dataset and place the .ann and .txt files in the ```data/train``` directory.
- A file containing word embeddings i.e word vectors that can be loaded using the gensim model. You can download word embeddings trained on PubMed and Wikipedia articles(wikipedia-pubmed-and-PMC-w2v.bin) from http://bio.nlplab.org/ and place the bin file in the ```resources``` directory.

To train the model:
1) Generate the files required for training by running the following command. 
```
python gen_training_files.py
```
This will generate some files in the ```resources``` directory that are required for training and running the models. To look at more options for input and output paths, run the command with the ```-h``` or ```--help``` flag.

2) Train the model by running the following command:
```
python train.py
```
This will create model files in the model directory. To look at more options for inputs paths and hyperparameter configurations run the command with the ```-h```  or ```--help``` flag.

# Detection
If you wish to perform detection alone using the trained model, complete the steps 1 and 2 above run the following command:
```
python run.py det data/test/
```
You fill find the outputs in the ```out``` directory. To look at more options for input and output paths, run the command with the ```-h``` or ```--help``` flag.

# Resolution
Since normalization performs detection and disambiguation, complete the steps 1 and 2 above and then run the following command:
```
python run.py res data/test/
```
You fill find the outputs in the ```out``` directory. To look at more options for input and output paths, run the command with the ```-h``` or ```--help``` flag.

