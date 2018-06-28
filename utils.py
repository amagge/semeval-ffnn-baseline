"""Utility functions for loading datasets and computing performance"""
from __future__ import print_function

import codecs
import cPickle as pickle
import re
import sys
from os.path import isfile, join

import numpy as np
import requests
from gensim.models.keyedvectors import KeyedVectors
from requests.utils import quote

GEONAMES_URL = "http://localhost:8091/location?location="
UNK_FILENAME = "unk.pkl"
NUM_FILENAME = "num.pkl"
SPLIT_REGEX = r"(\s|\,|\.|\"|\(|\)|\\|\-|\'|\?|\!|\/|\:|\;|\_|\+|\`|\[|\]|\#|\*|\%|\<|\>|\=)"

class Token(object):
    '''Token object which contains fields for offsets and annotation'''
    def __init__(self, text, start, end, encoding):
        # placeholders
        self.text = text
        self.start = start
        self.end = end
        self.encoding = encoding

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Annotation(object):
    '''Token object which contains fields for offsets and annotation'''
    def __init__(self, text, start, end, atype, geonameid=-1, lat=0, lon=0):
        # placeholders
        self.text = text
        self.start = start
        self.end = end
        self.atype = atype
        self.geonameid = geonameid
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class WordEmb(object):
    """Loads the word2vec model"""
    def __init__(self, args):
        print('processing corpus ' + str(args.emb_loc))
        if args.emb_loc.endswith("pkl"):
            self.wvec = pickle.load(open(args.emb_loc, "rb"))
        else:
            if args.embvocab > 0:
                self.wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True,
                                                              limit=args.embvocab)
            else:
                self.wvec = KeyedVectors.load_word2vec_format(args.emb_loc, binary=True)
        unk_filename = join(args.save, UNK_FILENAME)
        num_filename = join(args.save, NUM_FILENAME)
        if isfile(unk_filename) and isfile(num_filename):
            print("Loading unk from file")
            self.unk = pickle.load(open(unk_filename, "rb"))
            self.num = pickle.load(open(num_filename, "rb"))
        else:
            print("Can't find unk and num pkl files. Run training steps again.")
            sys.exit(0)
        self.is_case_sensitive = True if (self.wvec['the'] != self.wvec['The']).all() else False
        if not self.is_case_sensitive:
            print("Warning: dictionary is NOT case-sensitive")

    def __getitem__(self, word):
        if not self.is_case_sensitive:
            word = word.lower()
        try:
            word_vec = self.wvec[word]
        except KeyError:
            if word.isdigit():
                word_vec = self.num
            else:
                word_vec = self.unk
        word_vec = np.append(word_vec, np.array(case_feature(word)))
        return word_vec

def case_feature(word):
    '''returns an basic orthographic feature'''
    all_caps = True
    for char in word:
        if not ord('A') <= ord(char) <= ord('Z'):
            all_caps = False
            break
    if all_caps:
        return [1, 0, 0]
    else:
        if ord('A') <= ord(word[0]) <= ord('Z'):
            return [0, 1, 0]
        else:
            return [0, 0, 1]

def tokenize_document(doc_path):
    '''Tokenize the text and preserve offsets'''
    with codecs.open(doc_path, 'r', 'utf-8') as myfile:
        doc_text = myfile.read()
    doc_vocab = set()
    # Split text into words and create Token objects
    doc_tokens = []
    words = re.split(SPLIT_REGEX, doc_text)
    words = [word.strip() for word in words if word.strip() != ""]
    current_offset = 0
    for word in words:
        word_offset = doc_text.index(word, current_offset)
        current_offset = word_offset + len(word)
        doc_token = Token(word, word_offset, word_offset+len(word), "O")
        doc_tokens.append(doc_token)
        doc_vocab.add(word)
    return doc_tokens, doc_vocab

def get_namedentities(args, tokens, prediction):
    '''Get list of named entitiess'''
    assert len(tokens) == len(prediction)
    entities = []
    indices = []
    found = False
    entity = ''
    indstr = ''
    for i, label in enumerate(prediction):
        if label == 0:
            if found:
                if entity != '':
                    entity += "_{}".format(i)
                    try:
                        indstr += "_{}".format(tokens[i].text)
                    except UnicodeError:
                        pass
                    # indstr += "_{}".format(tokens[i].text)
            else:
                if entity == '':
                    entity = "{}".format(i)
                    try:
                        indstr += "_{}".format(tokens[i].text)
                    except UnicodeError:
                        pass
                    # indstr += "_{}".format(tokens[i].text)
                    found = True
        else:
            found = False
            if entity != '':
                entities.append(entity)
                indices.append(indstr)
                entity = ''
                indstr = ''
    assert len(indices) == len(entities)
    return indices, entities

def get_pred_anns(tokens, prediction):
    '''Get list of named entitiess'''
    assert len(tokens) == len(prediction)
    entities = []
    text = ''
    start = -1
    end = -1
    for i, label in enumerate(prediction):
        if label == 0:
            if text != '':
                end = tokens[i].end
                text += " {}".format(tokens[i].text.encode('ascii', 'ignore').decode('ascii'))
            else:
                start = tokens[i].start
                end = tokens[i].end
                text = "{}".format(tokens[i].text.encode('ascii', 'ignore').decode('ascii'))
        else:
            if text != '':
                entity = Annotation(text, start, end, "LOC")
                entities.append(entity)
                text = ''
    return entities

def get_ne_indexes(args, tags):
    '''Get named entities by indices'''
    entities = []
    found = False
    entity = ''
    for i, label in enumerate(tags):
        if label == 0:
            if found:
                if entity != '':
                    entity += "_{}".format(i)
            else:
                if entity == '':
                    entity = "{}".format(i)
                    found = True
        else:
            found = False
            if entity != '':
                entities.append(entity)
                entity = ''
    return entities

def write_errors(tokens, true_pos, false_pos, false_neg, fname='results.txt'):
    '''Write the named entities into a file for error analysis'''
    print("TP {} FP {} FN {}".format(len(true_pos), len(false_pos), len(false_neg)))
    rfile = open(fname, 'w')
    print("TP {} FP {} FN {}".format(len(true_pos), len(false_pos), len(false_neg)), file=rfile)
    print("--TP--", file=rfile)
    for i, item in enumerate(true_pos):
        for index in item.split('_'):
            print("{}\t{}\t{}".format(i, item, tokens[int(index)]), file=rfile)
    print("--FP--", file=rfile)
    for i, item in enumerate(false_pos):
        for index in item.split('_'):
            print("{}\t{}\t{}".format(i, item, tokens[int(index)]), file=rfile)
    print("--FN--", file=rfile)
    for i, item in enumerate(false_neg):
        for index in item.split('_'):
            print("{}\t{}\t{}".format(i, item, tokens[int(index)]), file=rfile)
    rfile.close()

def phrasalf1score(args, tokens, prediction, target, write_err=False):
    '''Compute phrasal F1 score for the results'''
    gold_entities = get_ne_indexes(args, np.argmax(target, 1))
    pred_entities = get_ne_indexes(args, np.argmax(prediction, 1))
    # inefficient but easy to understand
    true_pos = [x for x in pred_entities if x in gold_entities]
    false_pos = [x for x in pred_entities if x not in gold_entities]
    false_neg = [x for x in gold_entities if x not in pred_entities]
    precision = 1.0 * len(true_pos)/(len(true_pos) + len(false_pos))
    recall = 1.0 * len(true_pos)/(len(true_pos) + len(false_neg))
    f1sc = 2.0 * precision * recall / (precision + recall)
    if write_err:
        write_errors(tokens, true_pos, false_pos, false_neg, "runs/ne_{:.5f}".format(f1sc)+".txt")
    return precision, recall, f1sc

def write_results(words, prediction, target, fname='results.txt'):
    '''Write results to file'''
    target = np.argmax(target, 1)
    prediction = np.argmax(prediction, 1)
    rfile = open(fname, 'w')
    for i, label in enumerate(target):
        print("{}\t{}\t{}".format(words[i], prediction[i], label), file=rfile)
    rfile.close()

def get_ent_concepts(entities):
    '''Get entity geoname ids'''
    for entity in entities:
        url = GEONAMES_URL+quote(entity.text)
        # print(url)
        response = requests.get(url)
        jsondata = response.json()
        # print(jsondata)
        if jsondata and int(jsondata["retrieved"]) > 0:
            # print("Updating")
            record = jsondata["records"][0]
            entity.geonameid = record["GeonameId"]
            entity.lat = record["Latitude"]
            entity.lon = record["Longitude"]
    return entities

def write_pred_and_entities(args, tokens, prediction, pmid):
    '''Write results to file'''
    prediction = np.argmax(prediction, 1)
    fname = join(args.outdir, pmid + '_pred')
    rfile = codecs.open(fname, 'w', 'utf-8')
    for i, label in enumerate(prediction):
        label = 'I' if label == 0 else 'O'
        try:
            print("{}\t{}".format(tokens[i].text, label), file=rfile)
        except UnicodeError:
            pass
    rfile.close()
    fname = join(args.outdir, pmid + '_nes')
    rfile = codecs.open(fname, 'w', 'utf-8')
    entities = get_pred_anns(tokens, prediction)
    entities = get_ent_concepts(entities)
    for index, entity in enumerate(entities):
        # print("{}".format(entity), file=rfile)
        print("T{}\tLocation {} {}\t{}".format(index, entity.start, entity.end, entity.text), file=rfile)
        print("#{}\tAnnotatorNotes T{}\t<latlng>{},{}</latlng><geoID>{}</geoID>".format(index, index, entity.lat, entity.lon, entity.geonameid), file=rfile)
    rfile.close()
    print("{}\t{} entities found".format(pmid, len(entities)))

def f1score(class_size, prediction, target):
    '''Compute F1 score for the results'''
    true_pos = np.array([0] * (class_size + 1))
    false_pos = np.array([0] * (class_size + 1))
    false_neg = np.array([0] * (class_size + 1))
    target = np.argmax(target, 1)
    prediction = np.argmax(prediction, 1)
    for i, label in enumerate(target):
        if label == prediction[i]:
            true_pos[label] += 1
        else:
            false_pos[label] += 1
            false_neg[prediction[i]] += 1
    unnamed_entity = class_size - 1
    for i in range(class_size):
        if i != unnamed_entity:
            true_pos[class_size] += true_pos[i]
            false_pos[class_size] += false_pos[i]
            false_neg[class_size] += false_neg[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size + 1):
        precision.append(true_pos[i] * 1.0 / (true_pos[i] + false_pos[i]))
        recall.append(true_pos[i] * 1.0 / (true_pos[i] + false_neg[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    return precision[class_size], recall[class_size], fscore[class_size]

def make_unicode(input_data):
    '''Returns unicode string'''
    if type(input_data) != unicode:
        input_data = input_data.decode('utf-8')
        return input_data
    else:
        return input
