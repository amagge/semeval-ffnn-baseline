'''
A Deep Neural network with two layers for independent classification
'''

from __future__ import print_function

import argparse
import pickle
from os.path import join
import numpy as np
import tensorflow as tf

from models import FFModel, ModelHypPrms
from utils import WordEmb, f1score, phrasalf1score

MODEL_NAME = "FFNN"
TRAIN_FILE_NAME = "train-io.txt"
VALID_FILE_NAME = "valid-io.txt"
TEST_FILE_NAME = "test-io.txt"
HYPRM_FILE_NAME = "hyperprms.pkl"

def get_input(args, word_emb_model, input_file):
    '''loads input dataset'''
    n_neighbors = int(args.window_size/2)
    print("processing file: {} and neighbors = {}".format(input_file, n_neighbors))
    padding = "<s>"
    words = []
    labels = []
    for _ in range(n_neighbors):
        words.append(padding)
    for line in open(input_file):
        assert len(line.split()) == 2
        word = line.split()[0]
        label = line.split()[1]
        words.append(word)
        if label.startswith('I'):
            labels.append(np.array([1, 0]))
        elif label == 'O':
            labels.append(np.array([0, 1]))
        else:
            print("Invalid tag {} found for word {}".format(label, word))
    for _ in range(n_neighbors):
        words.append(padding)
    instances = []
    for i in range(n_neighbors, len(words)-n_neighbors):
        context = []
        for j in range(-n_neighbors, n_neighbors+1):
            context = np.append(context, word_emb_model[words[i+j]])
        instances.append(context)
    words = words[n_neighbors:len(words)-n_neighbors]
    assert len(words) == len(instances) == len(labels)
    return words, instances, labels

def train(args):
    '''Training method'''
    # Load Word Embeddings
    word_emb = WordEmb(args)
    # Load training and validation tokens, vector instances (input vector) and labels
    train_t, train_v, train_l = get_input(args, word_emb, join(args.work_dir, TRAIN_FILE_NAME))
    valid_t, valid_v, valid_l = get_input(args, word_emb, join(args.work_dir, VALID_FILE_NAME))
    # test_t, test_v, test_l = get_input(args, word_emb, TEST_FILE_NAME)
    n_input = len(train_v[0])
    print("Input size detected ", n_input)
    n_classes = 2
    hyperparams = ModelHypPrms(n_input, n_classes, args.hid_dim, args.lrn_rate)
    # Save hyperparams to disk
    pickle.dump(hyperparams, open(join(args.work_dir, HYPRM_FILE_NAME), "wb"))
    # Create model
    model = FFModel(hyperparams)
    # Model checkpoint path
    save_loc = join(args.save, MODEL_NAME)
    # Initialize TG variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        def evaluate(tokens, instances, labels, write_result=False):
            '''Evaluate and print results'''
            prediction, target = sess.run([model.pred, model.output_y],
                                          feed_dict={model.input_x: np.asarray(instances),
                                                     model.output_y: np.asarray(labels),
                                                     model.dropout: 1.0})
            prec, recall, f1sc = f1score(n_classes, prediction, target)
            if write_result:
                print("Found MAX")
                print("--Tokenwise P:{:.5f}".format(prec), "R:{:.5f}".format(recall),
                      "F1:{:.5f}".format(f1sc))
                prec, recall, f1sc = phrasalf1score(tokens, prediction, target)
                print("--Phrasal P:{:.5f}".format(prec), "R:{:.5f}".format(recall),
                      "F1:{:.5f}".format(f1sc))
            return f1sc
        # Training cycle
        if args.train_epochs > 0:
            maxf1 = 0.0
            for epoch in range(args.train_epochs):
                avg_cost = 0.
                total_batch = int(len(train_v)/args.batch_size)
                for ptr in range(0, len(train_v), args.batch_size):
                    # Run backprop and cost during training
                    _, epoch_cost = sess.run([model.optimizer, model.cost], feed_dict={
                        model.input_x: np.asarray(train_v[ptr:ptr + args.batch_size]),
                        model.output_y: np.asarray(train_l[ptr:ptr + args.batch_size]),
                        model.dropout: args.dropout})
                    # Compute average loss across batches
                    avg_cost += epoch_cost / total_batch
                print("Epoch:", '%02d' % (epoch+1), "cost=", "{:.5f}".format(avg_cost))
                if epoch % args.eval_interval == 0:
                    train_f1 = evaluate(train_t, train_v, train_l)
                    val_f1 = evaluate(valid_t, valid_v, valid_l)
                    print("-Training : {:.5f}".format(train_f1), "Val : {:.5f}".format(val_f1))
                    if val_f1 > maxf1:
                        maxf1 = val_f1
                        print("Saving model to {}".format(save_loc))
                        saver.save(sess, save_loc)
                        # evaluate(test_t, test_v, test_l, True)
            print("Optimization Finished!")
        # Load best model and evaluate model on the test set before applying to production
        saver = tf.train.import_meta_graph(save_loc + '.meta')
        saver.restore(sess, save_loc)
        print("Model from {} restored.".format(save_loc))
        # evaluate(test_t, test_v, test_l, True)

def main():
    '''Main method : parse input arguments and train'''
    parser = argparse.ArgumentParser()
    # Input files
    parser.add_argument('--work_dir', type=str, default="resources/",
                        help="working directory containing resource files")
    # Word Embeddings
    parser.add_argument('--emb_loc', type=str, default="resources/word-embeddings.pkl",
                        help='word2vec embedding location')
    # Hyperparameters
    parser.add_argument('--hid_dim', type=int, default=100, help='dimension of hidden layers')
    parser.add_argument('--lrn_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    # Settings
    parser.add_argument('--window_size', type=int, default=5, help='context window size - 3/5/7')
    parser.add_argument('--train_epochs', type=int, default=50, help='number of train epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate once in _ epochs')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size of training')
    # Model save and restore paths
    parser.add_argument('--save', type=str, default="model/", help="path to save model")
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
