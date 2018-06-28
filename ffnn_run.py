'''
A Deep Neural network with two layers for independent classification
'''

from __future__ import print_function
import sys
import argparse
import cPickle as pickle
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np

from ff_model import FFModel
from ffnn_train import HYPRM_FILE_NAME, MODEL_NAME
from utils import WordEmb, tokenize_document, write_pred_and_entities

def get_input_pmc(args, word_emb_model, input_file):
    '''loads files for annotation'''
    window_size = 5 # TODO:add to settings
    n_neighbors = int(window_size/2)
    doc_tokens, _ = tokenize_document(input_file)
    # print("processing file: {} and neighbors = {}".format(input_file, n_neighbors))
    padding = "<s>"
    words = []
    for _ in range(n_neighbors):
        words.append(padding)
    for token in doc_tokens:
        words.append(token.text)
    for _ in range(n_neighbors):
        words.append(padding)
    instances = []
    for i in range(n_neighbors, len(words)-n_neighbors):
        context = []
        for j in range(-n_neighbors, n_neighbors+1):
            context = np.append(context, word_emb_model[words[i+j]])
        instances.append(context)
    assert len(doc_tokens) == len(instances)
    return doc_tokens, instances

def run(args):
    '''Run method'''
    word_emb = WordEmb(args)
    print("Loading model")
    hyperparams = pickle.load(open(join(args.work_dir, HYPRM_FILE_NAME), "rb"))
    model = FFModel(hyperparams)
    print("Starting session")
    save_loc = join(args.save, MODEL_NAME)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(save_loc + '.meta')
        saver.restore(sess, save_loc)
        # load pubmed files
        pub_files = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]
        for _, pubfile in enumerate(pub_files):
            pub_t, pub_v = get_input_pmc(args, word_emb, join(args.dir, pubfile))
            prediction = sess.run(model.pred, feed_dict={model.input_x: np.asarray(pub_v),
                                                         model.dropout: 1.0})
            write_pred_and_entities(args, pub_t, prediction, pubfile.replace(".txt", ""))

def main():
    '''Main method : parse input arguments and train'''
    parser = argparse.ArgumentParser()
    # Input files
    parser.add_argument('dir', type=str,
                        help='Location to dir containing files to be annotated')
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    parser.add_argument('--save', type=str, default="model/", help="path to saved model")
    parser.add_argument('--outdir', type=str, default="out/",
                        help='Output dir for annotated pubmed files.'+
                        'Created in same directory by default.')
    # Word Embeddings
    parser.add_argument('--emb_loc', type=str,
                        default="resources/wikipedia-pubmed-and-PMC-w2v.bin",
                        help='word2vec embedding location')
    parser.add_argument('--embvocab', type=int, default=-1,
                        help='load top n words in word emb. -1 for all.')
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
