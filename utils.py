"""Utility functions for loading datasets and computing performance"""
from __future__ import print_function

import codecs
import pickle
import re
import sys
from os import makedirs
from os.path import exists, isfile, join

import requests
from requests.utils import quote
from gensim.models.keyedvectors import KeyedVectors

import numpy as np

# Fix for unicode encoding errors
# Python isn't the best for this
reload(sys)
sys.setdefaultencoding('utf8')

EXT_GLL_REGEX = r"<latl.?ng>(.*)</latl.?ng>"
EXT_GID_REGEX = r"<geoID>(.*)</geoID>"

GEONAMES_URL = "http://localhost:8091/location?location="
UNK_FILENAME = "unk.pkl"
NUM_FILENAME = "num.pkl"
SPLIT_REGEX = r"(\s|\,|\.|\"|\(|\)|\\|\-|\'|\?|\!|\/|\:|\;|\_|\+|\`|\[|\]|\#|\*|\%|\<|\>|\=)"
LOC_ANN_TAG = "LOC"
PRO_ANN_TAG = "EXT"

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
        unk_filename = join(args.work_dir, UNK_FILENAME)
        num_filename = join(args.work_dir, NUM_FILENAME)
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

def read_annotations(doc_path):
    '''Read annotations into annotation object'''
    annotations = []
    with open(doc_path, 'r') as myfile:
        print("Reading", doc_path)
        doc_lines = myfile.readlines()
        index = 0
        span_map = {}
        while index < len(doc_lines):
            line = doc_lines[index].strip()
            parts = line.split("\t")
            # There can exist more than 2 tab spaces if the text itself has tab spaces
            if len(parts) >= 3:
                if parts[0].startswith("T") or parts[0].startswith("#"):
                    if parts[0].startswith("T")  and parts[1].startswith("Location"):
                        ann_id = parts[0].strip()
                        offset_text = ""
                        # for cases when text has tab spaces
                        for part in parts[2:]:
                            offset_text += part
                        # for cases when text has new lines
                        while index+1 < len(doc_lines) and len(doc_lines[index+1].split("\t")) < 3:
                            offset_text += doc_lines[index+1]
                            index += 1
                        ann_type = LOC_ANN_TAG
                        offset_text = offset_text.strip()
                        offset_start = int(parts[1].strip().split()[1])
                        offset_end = int(parts[1].strip().split()[-1])
                        ann = Annotation(offset_text, offset_start, offset_end, ann_type)
                        span_map[ann_id] = ann
                        index += 1
                    elif parts[0].startswith("#") and parts[1].startswith("AnnotatorNotes"):
                        # Extract geonameid
                        id_parts = doc_lines[index].strip().split("\t")
                        ann_id = (id_parts[1].split(" ")[1]).strip()
                        id_search = re.search(EXT_GID_REGEX, id_parts[2])
                        geonameid = id_search.group(1) if id_search else "-1"
                        # if not id_search:
                        #     print(doc_path+"\t"+ann_id+"\tNO GID")
                        id_search = re.search(EXT_GLL_REGEX, id_parts[2])
                        if id_search:
                            latlng = id_search.group(1)
                            lat = latlng.split(",")[0] if latlng != "NA" else "0"
                            lon = latlng.split(",")[1] if latlng != "NA" else "0"
                        else:
                            print("ERROR: Could not extract Lat Long parts "+
                                  ann_id + " " + id_parts[2] + " in " + doc_path)
                        if ann_id in span_map:
                            ann = span_map[ann_id]
                            ann = Annotation(ann.text, ann.start, ann.end, ann.atype,
                                             geonameid, lat, lon)
                            annotations.append(ann)
                        else:
                            print("ERROR:", ann_id, "not found")
                        index += 1
                    elif parts[0].startswith("T") and parts[1].startswith("Protein"):
                        # Extract exclusion section. We will not consider text from this span
                        offset_start = int(parts[1].strip().split()[1])
                        # Check if next line ends with BEGIN
                        if not doc_lines[index+1].strip().endswith("BEGIN"):
                            print("Error: invalid protein", index+1, "Entity:", parts[1])
                        end_parts = doc_lines[index+2].strip().split("\t")
                        offset_end = int(end_parts[1].strip().split()[-1])
                        offset_text = parts[2] + "-" + end_parts[2]
                        # Check if next line ends with END
                        if not doc_lines[index+3].strip().endswith("END"):
                            print("Error: invalid protein", index+3, "Entity:", parts[1])
                        ann_type = PRO_ANN_TAG
                        index += 4
                        ann = Annotation(offset_text, offset_start, offset_end, ann_type)
                        annotations.append(ann)
                    else:
                        print("Error: invalid line", index, "Entity:", parts[1])
                        index += 1
                else:
                    print("Error: invalid line", index, "Entity:", parts[1])
                    index += 1
            else:
                print("Error: invalid line", index, "Tab separated entries:", len(parts))
                index += 1
        # Now check
    return annotations

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

def get_pred_anns(tokens, prediction, doc_path):
    '''Get list of named entitiess'''
    with codecs.open(doc_path, 'r', 'utf-8') as myfile:
        # Read text directly from file
        doc_text = myfile.read()
    assert len(tokens) == len(prediction)
    entities = []
    text = ''
    start = -1
    end = -1
    for i, label in enumerate(prediction):
        if label == 0:
            if text != '':
                end = tokens[i].end
                text += u" {}".format(tokens[i].text.encode('utf-8'))
            else:
                start = tokens[i].start
                end = tokens[i].end
                text = u"{}".format(tokens[i].text.encode('utf-8'))
        else:
            if text != '':
                text = doc_text[start:end]
                entity = Annotation(text, start, end, LOC_ANN_TAG)
                entities.append(entity)
                text = ''
    return entities

def get_ne_indexes(tags):
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

def phrasalf1score(tokens, prediction, target, write_err=False):
    '''Compute phrasal F1 score for the results'''
    gold_entities = get_ne_indexes(np.argmax(target, 1))
    pred_entities = get_ne_indexes(np.argmax(prediction, 1))
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

def get_ent_concepts(entities):
    '''Get entity geoname ids from geoname services project'''
    try:
        for entity in entities:
            if entity.atype == LOC_ANN_TAG:
                loc = entity.text
                url = GEONAMES_URL+quote(loc.encode('utf8'))
                response = requests.get(url)
                jsondata = response.json()
                # print(jsondata)
                if jsondata and int(jsondata["retrieved"]) > 0:
                    # print("Updating")
                    record = jsondata["records"][0]
                    entity.geonameid = record["GeonameId"]
                    entity.lat = record["Latitude"]
                    entity.lon = record["Longitude"]
    except ConnectionError as e:
        # Needs geonames-service installation
        print("ERROR:", e, "Can't connect to geonames service.", GEONAMES_URL)
    return entities

def get_entity_annotations(outdir, tokens, prediction, pmid, pmfile, write_tokens=False):
    '''Get entity annotations from predictions and write to file for debugging'''
    prediction = np.argmax(prediction, 1)
    if write_tokens:
        if not exists(outdir):
            makedirs(outdir)
        fname = join(outdir, pmid + '_debug.txt')
        rfile = codecs.open(fname, 'w', 'utf-8')
        for i, label in enumerate(prediction):
            label = 'I' if label == 0 else 'O'
            try:
                print("{}\t{}".format(tokens[i].text, label), file=rfile)
            except UnicodeError:
                print("{}\t{}".format("ERR-TOKEN", label), file=rfile)
        rfile.close()
    entities = get_pred_anns(tokens, prediction, pmfile)
    return entities

def write_annotations(outdir, entities, pmid, normalize=False):
    '''Write annotations to file in BRAT format'''
    if not exists(outdir):
        makedirs(outdir)
    print("writing results to", outdir)
    fname = join(outdir, pmid + '.ann')
    rfile = codecs.open(fname, 'w', encoding="utf-8")
    if normalize:
        entities = get_ent_concepts(entities)
    for index, entity in enumerate(entities):
        if entity.atype == LOC_ANN_TAG:
            print(u"T{}\tLocation {} {}\t{}".format(index, entity.start, entity.end, entity.text),
                file=rfile)
            print("#{}\tAnnotatorNotes T{}\t<latlng>{},{}</latlng><geoID>{}</geoID>".format(
                index, index, entity.lat, entity.lon, entity.geonameid), file=rfile)
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
        return input_data
