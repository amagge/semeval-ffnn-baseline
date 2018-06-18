# ner-topo-ff  
Named Entity Recognizer (NER) for entity extraction using a feedforward deep neural network and distance supervision  
  
Requirements:  
tensorflow  
numpy  
arparse 
gensim  
  
To run :  
python ff_model.py  
  
Argparse prompt  
  
usage: ff_model.py [-h] [--train TRAIN] [--test TEST] [--val VAL]
                   [--dist DIST] [--pubdir PUBDIR] [--outdir OUTDIR]
                   [--emb_loc EMB_LOC] [--embvocab EMBVOCAB]
                   [--hid_dim HID_DIM] [--lrn_rate LRN_RATE]
                   [--feat_cap FEAT_CAP] [--feat_dict FEAT_DICT]
                   [--dropout DROPOUT] [--window_size WINDOW_SIZE]
                   [--dist_epochs DIST_EPOCHS] [--train_epochs TRAIN_EPOCHS]
                   [--eval_interval EVAL_INTERVAL] [--n_classes {2,3}]
                   [--batch_size BATCH_SIZE] [--restore RESTORE] [--save SAVE]
  
optional arguments:  
  -h, --help            show this help message and exit  
  --train _TRAIN_         train file location  
  --test _TEST_           test file location  
  --val _VAL_             val file location  
  --dist _DIST_           distance supervision files dir.  
  --pubdir _PUBDIR_       pubmed files dir. To be production set.  
  --outdir _OUTDIR_       Output dir for ffmodel annotated pubmed files.  
  --emb_loc _EMB_LOC_     word2vec embedding location  
  --embvocab _EMBVOCAB_   load top n words in word emb  
  --hid_dim _HID_DIM_     dimension of hidden layers  
  --lrn_rate _LRN_RATE_   learning rate  
  --feat_cap _FEAT_CAP_   Capitalization feature  
  --feat_dict _FEAT_DICT_ Dictionary feature  
  --dropout _DROPOUT_     dropout probability  
  --window_size _WINDOW_SIZE_ context window size - 3/5/7  
  --dist_epochs _DIST_EPOCHS_ number of distsup epochs  
  --train_epochs _TRAIN_EPOCHS_ number of train epochs  
  --eval_interval _EVAL_INTERVAL_ evaluate once in _ epochs  
  --n_classes _{2,3}_     number of classes  
  --batch_size _BATCH_SIZE_ batch size of training  
  --restore _RESTORE_     path of saved model  
  --save _SAVE_           path to save model  

Input files:

Annotated input expected as a file containing tokens on each line along with their respective annotations B/I/O or I/O separated by tab-spaces.

```
Overall O  
, O  
these O  
results O  
indicate O  
widespread O  
human-to-animal O  
transmission O  
of O  
pandemic O  
( O  
H1N1 O  
) O  
2009 O  
influenza O  
viruses O  
in O  
South B  
Korea I  
. O  
```

