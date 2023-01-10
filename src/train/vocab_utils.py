# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import re

# import math


class Vocab(object):
  def __init__(self, vec_path=None, dim=20, fileformat='bin', voc=None,
               word2id=None, word_vecs=None, unk_mapping_path=None):
    self.unk_label = '<unk>'
    self.stoplist = None
    if fileformat == 'txt2':
      self.fromText_format2(vec_path,voc=voc)
    else:
      self.fromVocabualry(voc, dim=dim)

    self.__unk_mapping = None
    if unk_mapping_path is not None:
      self.__unk_mapping = {}
      in_file = open(unk_mapping_path, 'rt')
      for line in in_file:
        items = re.split('\t', line)
        self.__unk_mapping[items[0]] = items[1]
      in_file.close()

  def fromVocabualry(self, voc, dim=20):
    # load freq table and build index for each word
    self.word2id = {}
    self.id2word = {}
    self.word2id[self.unk_label] = 0
    self.vocab_size = len(voc)
    self.word_dim = dim
    for word in voc:
      cur_index = len(self.word2id)
      self.word2id[word] = cur_index
      self.id2word[cur_index] = word
    shape = (self.vocab_size+1, self.word_dim)
    scale = 0.05
    self.word_vecs = np.array(np.random.uniform(
      low=-scale, high=scale, size=shape), dtype=np.float32)


  def fromText_format2(self, vec_path,voc=None,pre_word_vecs=None):
    # load freq table and build index for each word
    self.word2id = {}
    self.id2word = {}
    
    vec_file = open(vec_path, 'rt')
    word_vecs = {}
    for line in vec_file:
      line = line.strip()
      parts = line.split('\t')
      cur_index = int(parts[0])
      word = parts[1]
      vector = np.array([float(x) for x in re.split('\\s+', parts[2])], dtype='float32')
      self.word2id[word] = cur_index 
      self.id2word[cur_index] = word
      word_vecs[cur_index] = vector
      self.word_dim = vector.size
    vec_file.close()

    self.vocab_size = len(self.word2id)

    if pre_word_vecs is not None:
      self.word_vecs = pre_word_vecs
    else:
      self.word_vecs = np.zeros((self.vocab_size+1, self.word_dim), dtype=np.float32) # the last dimension is all zero
      for cur_index in range(self.vocab_size):
        self.word_vecs[cur_index] = word_vecs[cur_index]

  def setWordvec(self, word_vecs):
    self.word_vecs = word_vecs

  def hasWord(self, word):
    return word in self.word2id

  def size(self):
    return len(self.word2id)

  def getIndex(self, word):
    if(word in self.word2id):
      return self.word2id.get(word)
    else:
      return 0

  def getWord(self, idx):
    return self.id2word.get(idx)

  def getVector(self, word):
    if(word in self.word2id):
      idx = self.word2id.get(word)
      return self.word_vecs[idx]
    return None

  def to_index_sequence(self, sentence):
    sentence = sentence.strip()
    seq = []
    for word in re.split('/', sentence):
      idx = self.getIndex(word)
      if idx is None and self.__unk_mapping is not None and word in self.__unk_mapping:
        simWord = self.__unk_mapping[word]
        idx = self.getIndex(simWord)
      if idx is None:
        idx = self.vocab_size
      seq.append(idx)
    return seq

  def to_index_sequence_for_list(self, words):
    seq = []
    for word in words:
      idx = self.getIndex(word)
      if idx is None and self.__unk_mapping is not None and word in self.__unk_mapping:
        simWord = self.__unk_mapping[word]
        idx = self.getIndex(simWord)
      if idx is None:
        idx = self.vocab_size
      seq.append(idx)
    return seq

  def to_character_matrix(self, sentence, max_char_per_word=-1):
    sentence = sentence.strip()
    seq = []
    for word in re.split('/', sentence):
      cur_seq = []
      for i in range(len(word)):
        cur_char = word[i]
        idx = self.getIndex(cur_char)
        if idx is None and self.__unk_mapping is not None and cur_char in self.__unk_mapping:
          simWord = self.__unk_mapping[cur_char]
          idx = self.getIndex(simWord)
        if idx is None:
          idx = self.vocab_size
        cur_seq.append(idx)
      if max_char_per_word != -1 and len(cur_seq) > max_char_per_word:
        cur_seq = cur_seq[:max_char_per_word]
      seq.append(cur_seq)
    return seq

  def to_index_sequence4binary_features(self, sentence):
    sentence = sentence.strip().lower()
    seq = []
    for word in re.split(' ', sentence):
      idx = self.getIndex(word)
      if idx is None:
        continue
      seq.append(idx)
    return seq

  def to_char_ngram_index_sequence(self, sentence):
    sentence = sentence.strip().lower()
    seq = []
    words = re.split(' ', sentence)
    for word in words:
      sub_words = collect_char_ngram(word)
      for sub_word in sub_words:
        idx = self.getIndex(sub_word)
        if idx is None:
          continue
        seq.append(idx)
    return seq

  def to_sparse_feature_sequence(self, sentence1, sentence2):
    words1 = set(re.split(' ', sentence1.strip().lower()))
    words2 = set(re.split(' ', sentence2.strip().lower()))
    intersection_words = words1.intersection(words2)
    seq = []
    for word in intersection_words:
      idx = self.getIndex(word)
      if idx is None:
        continue
      seq.append(idx)
    return seq

  def get_sentence_vector(self, sentence):
    sent_vec = np.zeros((self.word_dim,), dtype='float32')
    sentence = sentence.strip().lower()
    total = 0.0
    for word in re.split(' ', sentence):
      cur_vec = self.getVector(word)
      if cur_vec is None:
        continue
      sent_vec += cur_vec
      total += 1.0
    if total != 0.0:
      sent_vec /= total
    return sent_vec

  def dump_to_txt2(self, outpath):
    outfile = open(outpath, 'wt')
    for word in self.word2id.keys():
      cur_id = self.word2id[word]
      cur_vector = self.getVector(word)

      outline = "{}\t{}\t{}".format(cur_id, word, vec2string(cur_vector))
      outfile.write(outline + "\n")
    outfile.close()

  def dump_to_txt3(self, outpath):
    outfile = open(outpath, 'wt')
    for word in self.word2id.keys():
      cur_vector = self.getVector(word)
      outline = word + " {}".format(vec2string(cur_vector))
      outfile.write(outline + "\n")
    outfile.close()


def vec2string(val):
  result = ""
  for v in val:
    result += " {}".format(v)
  return result.strip()


def collect_all_ngram(words, n=2):
  all_ngrams = set()
  for i in range(len(words)-n):
    cur_ngram = words[i:i+n]
    all_ngrams.add(' '.join(cur_ngram))
  return all_ngrams


def collect_char_ngram(word, n=3):
  all_words = []
  if len(word) <= n:
    all_words.append(word)
  else:
    for i in range(len(word)-n+1):
      cur_word = word[i:i+3]
      all_words.append(cur_word)
  return all_words


def to_char_ngram_sequence(sentence, n=3):
  seq = []
  words = re.split(' ', sentence)
  for word in words:
    sub_words = collect_char_ngram(word)
    seq.extend(sub_words)
  return ' '.join(seq)


def collectVoc(trainpath):
  vocab = set()
  inputFile = file(trainpath, 'rt')
  for line in inputFile:
    line = line.strip()
    label, sentence = re.split('\t', line)
    sentence = sentence.lower()
    for word in re.split(' ', sentence):
      vocab.add(word)
  inputFile.close()
  return vocab


if __name__ == '__main__':
  print('DONE!')
