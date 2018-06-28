'''Create smaller efficient embeddings'''
from __future__ import print_function

import argparse
import cPickle as pickle
import random
import re
import sys
from os import listdir, makedirs
from os.path import join, exists
import codecs
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from utils import tokenize_document, Annotation
from utils import UNK_FILENAME, NUM_FILENAME
from ffnn_train import TRAIN_FILE_NAME, VALID_FILE_NAME, TEST_FILE_NAME
LOC_ANN_TAG = "LOC"
PRO_ANN_TAG = "EXT"
WORDEMB_FILENAME = "word-embeddings.pkl"

def read_annotations(doc_path):
    '''Read annotations into annotation object'''
    annotations = []
    with open(doc_path, 'r') as myfile:
        doc_lines = myfile.readlines()
        index = 0
        while index < len(doc_lines):
            line = doc_lines[index].strip()
            parts = line.split("\t") #re.split(r"(\t)", line)
            if len(parts) == 3:
                if parts[1].startswith("Location") or parts[1].startswith("Protein"):
                    if parts[1].startswith("Location"):
                        ann_type = LOC_ANN_TAG
                        offset_text = parts[2]
                        offset_start = int(parts[1].strip().split()[1])
                        offset_end = int(parts[1].strip().split()[-1])
                        index += 2
                    elif parts[1].startswith("Protein"):
                        offset_start = int(parts[1].strip().split()[1])
                        end_parts = doc_lines[index+2].strip().split("\t")
                        offset_end = int(end_parts[1].strip().split()[-1])
                        offset_text = parts[2] + "-" + end_parts[2]
                        ann_type = PRO_ANN_TAG
                        index += 4
                    ann = Annotation(offset_text, offset_start, offset_end, ann_type)
                    annotations.append(ann)
    return annotations

def load_train_data(train_dir):
    """load training data"""
    vocab = set()
    tfile = codecs.open(TRAIN_FILE_NAME, 'w', 'utf-8')
    vfile = codecs.open(VALID_FILE_NAME, 'w', 'utf-8')
    txt_files = [f for f in listdir(train_dir) if f.endswith(".txt")]
    random.shuffle(txt_files)
    for findex, txt_file in enumerate(txt_files):
        print("Reading", txt_file)
        rfile = vfile if findex % 10 == 0 else tfile
        doc_tokens, file_vocab = tokenize_document(join(train_dir, txt_file))
        vocab = vocab.union(file_vocab)
        annotations = read_annotations(join(train_dir, txt_file[:-3]+"ann"))
        for token in doc_tokens:
            ignore_token = False
            for ann in annotations:
                # print(token.text + "\t" + str(token.start) + "\t" + str(ann.start) +
                # "\t" + str(token.end) + "\t" + str(ann.end), file=rfile)
                if token.start >= ann.start and token.end <= ann.end:
                    # Change this for IOB annotations
                    if ann.atype == LOC_ANN_TAG:
                        token.encoding = "I-LOC"
                    if ann.atype == PRO_ANN_TAG:
                        ignore_token = True
                    break
            if not ignore_token:
                print(token.text + "\t" + token.encoding, file=rfile)
    tfile.close()
    vfile.close()
    return vocab

def load_test_data(train_dir):
    """load training data"""
    vocab = set()
    tfile = codecs.open(TEST_FILE_NAME, 'w', 'utf-8')
    txt_files = [f for f in listdir(train_dir) if f.endswith(".txt")]
    for _, txt_file in enumerate(txt_files):
        print("Reading", txt_file)
        doc_tokens, file_vocab = tokenize_document(join(train_dir, txt_file))
        vocab = vocab.union(file_vocab)
        annotations = read_annotations(join(train_dir, txt_file[:-3]+"ann"))
        for token in doc_tokens:
            ignore_token = False
            for ann in annotations:
                if token.start >= ann.start and token.end <= ann.end:
                    # Change this for IOB annotations
                    if ann.atype == LOC_ANN_TAG:
                        token.encoding = "I-LOC"
                    if ann.atype == PRO_ANN_TAG:
                        ignore_token = True
                    break
            if not ignore_token:
                print(token.text + "\t" + token.encoding, file=tfile)
    tfile.close()
    return vocab

def create_embeddings(args):
    '''Create embeddings object and dump pickle for use in subsequent models'''
    vocab = load_train_data(args.train_corpus)
    print("Total vocab:", len(vocab))
    print("Loading word embeddings:", args.emb_loc)
    unk_words = set()
    wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True)
    wemb_dict = {}
    for word in vocab:
        try:
            wemb_dict[word] = wvec[word]
        except KeyError:
            unk_words.add(word)
    print("Number of unknown words:", len(unk_words))
    if not exists(args.out_dir):
        makedirs(args.out_dir)
    # Dump dictionary pickle to disk
    print("Dumping training files to", args.out_dir)
    pickle.dump(wemb_dict, open(join(args.out_dir, WORDEMB_FILENAME), "wb"))
    # Dump unk and num to disk
    unk = np.array([random.random() for _ in range(wvec.vector_size)])
    num = np.array([random.random() for _ in range(wvec.vector_size)])
    pickle.dump(unk, open(join(args.out_dir, UNK_FILENAME), "wb"))
    pickle.dump(num, open(join(args.out_dir, NUM_FILENAME), "wb"))
    print("Done!")

def main():
    '''Main method : parse input arguments and train'''
    parser = argparse.ArgumentParser()
    # Input and Output paths
    parser.add_argument('-t', '--train_corpus', type=str, default='data/train/',
                        help='path to dir where training corpus files are stored')
    parser.add_argument('-e', '--emb_loc', type=str,
                        help='path to the word2vec embedding location')
    parser.add_argument('-o', '--out_dir', type=str, default='resources/',
                        help='output file containing minimal vocabulary')
    args = parser.parse_args()
    print(args)
    create_embeddings(args)

if __name__ == '__main__':
    main()
