from __future__ import (
  absolute_import, division, print_function, unicode_literals)
from configs.config import (
    batch_size, emb_size, rawDataPath, trainCsv, alpha, 
    testCsv, valCsv, char_emb_dim, token_emb_dim, element_size
)
import numpy as np
import imageio
import sys
import re
import os
import csv
from time import time
from scipy import misc
from PIL import Image, ImageOps, ImageDraw

from collections import defaultdict
import random
from tensorflow.keras.utils import Sequence
from vocab_utils import Vocab
from utils import prewhiten

old_package = [
]
 
image_cache = {}
global_count_ = 0
def read_all_instances(csvFile):
  ''' reformat the csv file
    Input: A csv file having format of [label example1 example2]
    Ouput: Triplet format [anchor positive negative]
  '''
  xpath_same_count=0
  neg_same_xpath_count=0
  instances, rows = [], []
  pos_idxs, neg_idxs = read_neg_pos(csvFile)
  with open(csvFile, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
      rows.append(row)
  count = -1
  for (anchor_pos, neg) in triplet_idices_generator(pos_idxs, neg_idxs):
    count += 1
    anchor_row = rows[anchor_pos]
    neg_row = rows[neg]
    instances.append((anchor_row, neg_row))
    if anchor_row[16] == anchor_row[33]:
      xpath_same_count +=1
    if neg_row[16] == neg_row[33]:
      neg_same_xpath_count +=1
    if count > 100: break
  print('same xpath ratio: ', xpath_same_count/len(instances))
  print('neg same xpath ratio: ', neg_same_xpath_count/len(instances))
  return instances


def collect_vocabs(trainCsv):
  ''' Collect info of textual parts: vocabularies, max_len...
  '''
  all_tokens = set()
  count = -1
  max_len = 0
  with open(trainCsv, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
      count += 1
      (matched, screen_matching, package, prime_deivce_name,
        prime_device_api_level, prime_screen_size,
        prime_device_density, prime_model, prime_pixel_ratio,
        prime_stat_bar_height, prime_screen_name, revisit_screen_name,
        prime_activity_name, prime_session_id, prime_element_name,
        prime_locating_id, prime_xpath, prime_left, prime_top, prime_right,
        prime_bottom, prime_clickable, revisit_device_name,
        revisit_api_level, revisit_screen_size, revisit_screen_density,
        revisit_model, revisit_pixel_ratio, revisit_stat_bar_height,
        revisit_activity_name, revisit_session_id, revisit_element_name,
        revisit_locating_id, revisit_xpath, revisit_left, revisit_top,
        revisit_right, revisit_bottom, revisit_clickable) = row
      prime_xpath_tokens = prime_xpath.split('/')
      revisit_xpath_tokens = revisit_xpath.split('/')
      all_tokens.update(prime_xpath_tokens)
      all_tokens.update(revisit_xpath_tokens)
      max_len = max(max_len, len(prime_xpath_tokens),
              len(revisit_xpath_tokens))
    all_chars = set()
    for token in all_tokens:
      for char in token:
        all_chars.add(char)
  return (max_len, all_tokens, all_chars)


def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
  ''' pad a 2d-array to a tensor
  '''
  out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
  if dim1_size > len(in_vals):
    dim1_size = len(in_vals)
  for i in range(dim1_size):
    cur_in_vals = in_vals[i]
    cur_dim2_size = dim2_size
    if cur_dim2_size > len(cur_in_vals):
      cur_dim2_size = len(cur_in_vals)
    out_val[i, :cur_dim2_size] = cur_in_vals[:cur_dim2_size]
  return out_val
def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
  out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
  if dim1_size > len(in_vals): dim1_size = len(in_vals)
  for i in range(dim1_size):
    in_vals_i = in_vals[i]
    cur_dim2_size = dim2_size
    if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
    for j in range(cur_dim2_size):
      in_vals_ij = in_vals_i[j]
      cur_dim3_size = dim3_size
      if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
      out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
  return out_val

def convert_row_2_embs(
  row, rawDataPath, token_vocab, char_vocab,
  screenShotOutputSize=299, elementOutputSize=60, use_element_img=True,
  use_vertical_cut=False, use_horizontal_cut=False, use_xpath=False):
  ''' Convert raw data to numpy array

    Input: a row from csv file
    Output: embeddings array (only element image, screenshot image and xpath for now)
  '''
  (matched, screen_matching, package, prime_deivce_name,
    prime_device_api_level, prime_screen_size,
    prime_device_density, prime_model, prime_pixel_ratio,
    prime_stat_bar_height, prime_screen_name, revisit_screen_name,
    prime_activity_name, prime_session_id, prime_element_name,
    prime_locating_id, prime_xpath, prime_left, prime_top, prime_right,
    prime_bottom, prime_clickable, revisit_device_name, revisit_api_level,
    revisit_screen_size, revisit_screen_density, revisit_model,
    revisit_pixel_ratio, revisit_stat_bar_height, revisit_activity_name,
    revisit_session_id, revisit_element_name, revisit_locating_id,
    revisit_xpath, revisit_left, revisit_top, revisit_right, revisit_bottom,
    revisit_clickable) = row

  if(screen_matching == '1'):
    return None  # skip screen-level matching for now
  # prime_screen_img = read_and_patch(rawDataPath + 'screenshots/'
  # + prime_screen_name + '.jpg', screenShotOutputSize)
  # revisit_screen_img = read_and_patch(rawDataPath + 'screenshots/'
  # + revisit_screen_name + '.jpg', screenShotOutputSize)
  if use_element_img:
    '''prime_ele_img = read_and_resize(
      rawDataPath + 'element-images/' +
      prime_element_name + '.jpg', elementOutputSize)
    revisit_ele_img = read_and_resize(
      rawDataPath + 'element-images/' + revisit_element_name + '.jpg',
      elementOutputSize)'''

    prime_cut_box   = [prime_left, prime_top, prime_right, prime_bottom]
    revisit_cut_box = [revisit_left, revisit_top, revisit_right, revisit_bottom]
    prime_ele_img = cut_element(
            rawDataPath + 'screenshots/' + prime_screen_name +
            '.jpg', [int(x) for x in prime_cut_box],
            output_size=elementOutputSize)
    revisit_ele_img = cut_element(
            rawDataPath + 'screenshots/' + revisit_screen_name +
            '.jpg', [int(x) for x in revisit_cut_box],
            output_size=elementOutputSize)
    if prime_ele_img is None or revisit_ele_img is None:
      return None
  else:
    prime_ele_img = None
    revisit_ele_img = None

  if use_horizontal_cut:
    prime_cut_box   = [prime_left, prime_top, prime_right, prime_bottom]
    revisit_cut_box = [revisit_left, revisit_top, revisit_right, revisit_bottom]
    prime_horizontal_cut_img = get_horizontal_cut(
            rawDataPath + 'screenshots/' + prime_screen_name +
            '.jpg', [int(x) for x in prime_cut_box])
    revisit_horizontal_cut_img = get_horizontal_cut(
            rawDataPath + 'screenshots/' + revisit_screen_name +
            '.jpg', [int(x) for x in revisit_cut_box])
    # print('get_horizontal_cut takes %ss'%(time()-start))
  else:
    prime_horizontal_cut_img = None
    revisit_horizontal_cut_img = None

  if use_vertical_cut:
    prime_cut_box   = [prime_left, prime_top, prime_right, prime_bottom]
    revisit_cut_box = [revisit_left, revisit_top, revisit_right, revisit_bottom]
    prime_vertical_cut_img = get_vertical_cut(
            rawDataPath + 'screenshots/' + prime_screen_name +
            '.jpg', [int(x) for x in prime_cut_box])
    revisit_vertical_cut_img = get_vertical_cut(
            rawDataPath + 'screenshots/' + revisit_screen_name +
            '.jpg', [int(x) for x in revisit_cut_box])
  else:
    prime_vertical_cut_img = None
    revisit_vertical_cut_img = None

  if use_xpath:
    prime_token_idxs = token_vocab.to_index_sequence(prime_xpath)
    revisit_token_idxs = token_vocab.to_index_sequence(revisit_xpath)
    prime_chars_matrix = char_vocab.to_character_matrix(prime_xpath)
    revisit_chars_matrix = char_vocab.to_character_matrix(revisit_xpath)
  else:
    prime_token_idxs = None
    revisit_token_idxs = None
    prime_chars_matrix = None
    revisit_chars_matrix = None

  label = (matched == '1')
  # if prime_screen_img is None or prime_ele_img is None or revisit_screen_img
  # is None or revisit_ele_img is None:

  return [prime_element_name, prime_ele_img, prime_horizontal_cut_img, 
          prime_vertical_cut_img, prime_token_idxs, prime_chars_matrix,
          revisit_element_name, revisit_ele_img, revisit_horizontal_cut_img,
          revisit_vertical_cut_img, revisit_token_idxs, revisit_chars_matrix]


def read_neg_pos(csvFile):
  '''For each element on a screen, find all of its negative, positive examples
  Input: a csv file
  output: 2 dictionary
  '''
  neg, pos = defaultdict(list), defaultdict(list)
  count = -1
  with open(csvFile, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
      count += 1
      try:
        (matched, screen_matching, package, prime_deivce_name,
         prime_device_api_level, prime_screen_size, prime_device_density,
         prime_model, prime_pixel_ratio, prime_stat_bar_height,
         prime_screen_name, revisit_screen_name, prime_activity_name,
         prime_session_id, prime_element_name, prime_locating_id, prime_xpath,
         prime_left, prime_top, prime_right, prime_bottom, prime_clickable,
         revisit_device_name, revisit_api_level, revisit_screen_size,
         revisit_screen_density, revisit_model, revisit_pixel_ratio,
         revisit_stat_bar_height, revisit_activity_name, revisit_session_id,
         revisit_element_name, revisit_locating_id,
         revisit_xpath, revisit_left, revisit_top, revisit_right,
         revisit_bottom, revisit_clickable) = row
      except:
        print(row)
        print('count: ', count)
      if screen_matching == '1':
        continue
      if 'EditText' in prime_xpath and package in old_package:
        continue
      if '-1' == matched:
        neg[prime_element_name].append(count)
      else:
        pos[prime_element_name].append(count)
  return (neg, pos)


def triplet_idices_generator(neg, pos):
  ''' all the triplets for training. Each contains an anchor,
      a positive and a negative example
  Input: Negative and positive dictionary
  Output: A list of all the triplets
  '''
  triplets = []
  allElementNames = pos.keys()
  random.shuffle(list(allElementNames))
  for elementName in allElementNames:
    # print('elementName: ', elementName)
    itsNegs = neg[elementName]
    itsPos = pos[elementName]
    # print('%s has %s negs and %s pos: '%(elementName, len(itsNegs), len(itsPos)))
    if len(itsNegs) == 0 or len(itsPos) < 2:
      continue
    for i in range(len(itsNegs)):
      # randomly get 2 positive:
      anchor_positive = random.choice(itsPos)
      negative = itsNegs[i]
      yield (anchor_positive, negative)


def read_and_resize(im_pth, outputSize):
  ''' resize an image to a squared 2d array
  '''
  try:
    img = misc.imread(os.path.expanduser(im_pth), mode='RGB')
  except:
    print('Not found: ', im_pth)
    return None
  if img.shape[0] < 30 or img.shape[1] < 30:
    return None
  img = misc.imresize(img, (outputSize, outputSize), interp='bilinear')
  prewhitened = prewhiten(img)
  return prewhitened


def read_and_patch(im_pth, outputSize):
  ''' The file is pad to keep the original ratio
  '''
  im = Image.open(im_pth)
  old_size = im.size  # old_size[0] is in (width, height) format

  ratio = float(outputSize)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  if new_size[0] < 15 or new_size[1] < 15:
    return None
  try:
    im = im.resize(new_size, Image.ANTIALIAS)
  except Exception as err:
    print(err)
    return None
  new_im = Image.new("RGB", (outputSize, outputSize))
  start = ((outputSize-new_size[0])//2,
       (outputSize-new_size[1])//2)
  new_im.paste(im, start)
  #draw = ImageDraw.Draw(new_im)
  #draw.rectangle(((rec[0] + start[0],rec[1] + start[1]), (rec[2] + start[0],rec[3] + start[1])), fill="red")
  # new_im.show()
  #im = None

  return prewhiten(np.array(new_im))


def cut_element(im_pth, box, output_size=80, margin=20):
  global global_count_
  try:
    if(im_pth in image_cache):
      im = image_cache[im_pth]
    elif global_count_ < 1000:
      im = imageio.imread(
        os.path.expanduser(im_pth), as_gray=False, pilmode="RGB")
      image_cache[im_pth] = im
    else:
      global_count_ = 0
      image_cache.clear()
      im = imageio.imread(
        os.path.expanduser(im_pth), as_gray=False, pilmode="RGB")
      image_cache[im_pth] = im
  except:
    # print('error: cannot read: ', im_pth)
    return None
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  if (box[2] - box[0] < 20 or box[3] - box[1] < 20):
    return None
  new_im = im[max(box[1] - margin, 0):min(box[3] + margin, s_height),
              box[0]:box[2],:]
  new_im = Image.fromarray(new_im).resize((output_size, output_size))
  return prewhiten(np.array(new_im))


def get_horizontal_cut(im_pth, box, output_size=60, margin=20):
  ''' get a horizontal cut of the screenshot containing the element
      This gives more context for element matching
    Input: the path to the image and the element bounding box
        im_pth: str
        box: [left, top, right, bottom]
        output_size: resize the cut
        margin: cut larger than the image a bit to get more context
    Output: a 2D numpy array
  '''
  global global_count_
  try:
    if(im_pth in image_cache):
      im = image_cache[im_pth]
    elif global_count_ < 1000:
      im = imageio.imread(
        os.path.expanduser(im_pth), as_gray=False, pilmode="RGB")
      image_cache[im_pth] = im
      global_count_ +=1
    else:
      image_cache.clear()
      global_count_ = 0
      im = imageio.imread(
        os.path.expanduser(im_pth), as_gray=False, pilmode="RGB")
      image_cache[im_pth] = im
  except:
    print('error reading: ', im_pth)

  (s_height, s_width) = np.asarray(im.shape)[0:2]
  new_im = im[max(box[1] - margin, 0):min(s_height, 
              box[3] + margin), 0:s_width,:]
  new_im = Image.fromarray(new_im).resize((output_size, output_size))
  return prewhiten(np.array(new_im))

def get_vertical_cut(im_pth, box, output_size=60, margin=20):
  ''' get a vertical cut of the screenshot containing the element
      This gives more context for element matching
    Input: the path to the image and the element bounding box
        im_pth: str
        box: [left, top, right, bottom]
        output_size: resize the cut
        margin: cut larger than the image a bit to get more context
    Output: a 2D numpy array
  '''
  global global_count_
  if(im_pth in image_cache):
    im = image_cache[im_pth]
  elif global_count_ < 1000:
    im = imageio.imread(
      os.path.expanduser(im_pth), as_gray=False, pilmode="RGB")
    image_cache[im_pth] = im
  else:
    global_count_ = 0
    image_cache.clear()
    im = imageio.imread(
      os.path.expanduser(im_pth), as_gray=False, pilmode="RGB")
    image_cache[im_pth] = im
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  cut_area = (box[0], max(box[1] - 100, 0), box[2], min(box[3] + 100, s_height))
  new_im = im[max(box[1] - 100, 0):min(box[3] + 100, s_height), box[0]:box[2],:]
  new_im = Image.fromarray(new_im).resize((output_size, output_size))
  #new_im.show()
  return prewhiten(np.array(new_im))

def parse_screen_size(screenSize):
  ''' 
  input: string like 1080x2560
  output: [1080, 2560]
  '''
  return [int(s) for s in screenSize.split('x')]


def make_batches(size, batch_size):
  nb_batch = int(np.ceil(size/float(batch_size)))
  return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


class DataGenSequence(Sequence):
  ''' A batched data generator class, give a batch of data whenever __getitem__ is called
  '''
  def __init__(self, usage, token_vocab, char_vocab, 
               max_len=100, use_xpath=False, use_element_img=True,
               use_vertical_cut=False, use_horizontal_cut=False):
    self.usage = usage
    self.is_training = False
    self.max_len = max_len
    self.use_xpath = use_xpath
    self.use_element_img = use_element_img
    self.token_vocab = token_vocab
    self.char_vocab = char_vocab
    self.use_vertical_cut = use_vertical_cut
    self.use_horizontal_cut = use_horizontal_cut
    if usage == 'train':
      self.instances = read_all_instances(trainCsv)
      self.hard_cases_map = {}
      self.train_idx2_data_idx = {}
      self.train_data_count = 0
      self.hard_instances = self.instances
    elif usage == 'test':
      self.instances = read_all_instances(testCsv)
    elif usage == 'validation':
      self.instances = read_all_instances(valCsv)
    #print('%s has %d instances'%(usage, len(self.instances),))

  def __len__(self):
    if self.is_training:
      return int(np.ceil(len(self.hard_instances) / float(batch_size)))
    return int(np.ceil(len(self.instances) / float(batch_size)))

  def instancesSize(self):
    return len(self.instances)
  
  def set_training(self, is_training):
    self.is_training = is_training

  def __getitem__(self, idx):
    ''' Return a batch of data. 
      This get called by the fit_generator method in the train scripts
    '''
    #print('Reading batch: ', idx)
    i = idx * batch_size
    if self.is_training:
      real_batch_size = min(batch_size, (len(self.hard_instances) - i))  
    else:
      real_batch_size = min(batch_size, (len(self.instances) - i))
    batch_anchor_screens, batch_anchor_elements = [], []
    batch_anchor_horizontal, batch_anchor_vertical = [], []
    batch_positive_screens, batch_positive_elements = [], []
    batch_pos_horizontal, batch_pos_vertical = [], []
    batch_negative_screens, batch_negative_elements = [], []
    batch_neg_horizontal, batch_neg_vertical = [], []
    batch_dummy_target = None
    (batch_anchor_token_lens, batch_positive_token_lens, 
     batch_negative_token_lens) = ([], [], [])  # xpath tokens len
    (batch_anchor_tokens, batch_positive_tokens,
     batch_negative_tokens) = [], [], []  # xpath tokens
    batch_anchor_chars, batch_pos_chars, batch_neg_chars = ([],[],[])
    (batch_anchor_char_lens, batch_pos_char_lens,
     batch_neg_char_lens) = ([],[],[])
    max_token_len_batch = 0
    max_char_len_batch = 0
    for i_batch in range(real_batch_size):
      if self.is_training:
        instance = self.hard_instances[i + i_batch]
      else:
        instance = self.instances[i + i_batch]
      (anchor_row, neg_row) = instance

      anchor_data = convert_row_2_embs(
        anchor_row, rawDataPath, self.token_vocab, 
        self.char_vocab, 
        use_element_img=self.use_element_img,
        use_horizontal_cut=self.use_horizontal_cut,
        use_vertical_cut=self.use_vertical_cut,
        use_xpath=self.use_xpath,
        elementOutputSize=element_size)
      neg_data = convert_row_2_embs(
        neg_row,    rawDataPath, self.token_vocab, self.char_vocab, 
        use_element_img=self.use_element_img,
        use_horizontal_cut=self.use_horizontal_cut,
        use_vertical_cut=self.use_vertical_cut,
        use_xpath=self.use_xpath,
        elementOutputSize=element_size)

      if anchor_data is None or neg_data is None:
        continue
      # get just the second half of the neg row
      anchor_data.extend(neg_data[int(len(neg_data)/2):])

      (anchor_ele_name, anchor_ele_img, anchor_horizontal_cut, 
       anchor_vertical_cut, anchor_tokens, anchor_char_matrix,
       pos_ele_name, pos_ele_img, pos_horizontal_cut, 
       pos_vertical_cut, pos_tokens, pos_char_matrix,
       neg_ele_name, neg_ele_img, neg_horizontal_cut, 
       neg_vertical_cut, neg_tokens, neg_char_matrix) = anchor_data

      # batch_anchor_screens.append(anchor_screen_img)
      # #[batch_size, screen_size, screen_size, channels]
      # batch_positive_screens.append(pos_screen_img)
      # batch_negative_screens.append(neg_screen_img)
      # [batch_size, element_size, element_size, channels]
      batch_anchor_elements.append(anchor_ele_img)
      batch_positive_elements.append(pos_ele_img)
      batch_negative_elements.append(neg_ele_img)
      batch_anchor_horizontal.append(anchor_horizontal_cut)
      batch_anchor_vertical.append(anchor_vertical_cut)
      batch_pos_horizontal.append(pos_horizontal_cut)
      batch_pos_vertical.append(pos_vertical_cut)
      batch_neg_horizontal.append(neg_horizontal_cut)
      batch_neg_vertical.append(neg_vertical_cut)
      # [batch_size, xpath_len]
      if self.use_xpath:
        batch_anchor_tokens.append(anchor_tokens)
        batch_positive_tokens.append(pos_tokens)
        batch_negative_tokens.append(neg_tokens)
        batch_anchor_token_lens.append(min(len(anchor_tokens), self.max_len))
        batch_positive_token_lens.append(min(len(pos_tokens), self.max_len))
        batch_negative_token_lens.append(min(len(neg_tokens), self.max_len))
        max_token_len_batch = max(len(anchor_tokens), len(pos_tokens), 
                                   len(neg_tokens), max_token_len_batch)
        batch_anchor_chars.append(anchor_char_matrix)
        batch_pos_chars.append(pos_char_matrix)
        batch_neg_chars.append(neg_char_matrix)
        batch_anchor_char_lens.append([len(cur_char_idx) for cur_char_idx in anchor_char_matrix])
        batch_pos_char_lens.append([len(cur_char_idx) for cur_char_idx in pos_char_matrix])
        batch_neg_char_lens.append([len(cur_char_idx) for cur_char_idx in neg_char_matrix])

        
      
      if not self.is_training and self.usage == 'train':
        self.train_idx2_data_idx[self.train_data_count] = i + i_batch
        self.train_data_count +=1

    # patch xpath tokens to max_len
    if self.use_xpath:
      real_batch_size = len(batch_anchor_tokens)  # ignored some None data
      batch_dummy_target = np.zeros(
        (real_batch_size, emb_size * 3), dtype=np.float32)
      batch_anchor_tokens = pad_2d_vals(
        batch_anchor_tokens, real_batch_size, max_token_len_batch)
      batch_positive_tokens = pad_2d_vals(
        batch_positive_tokens, real_batch_size, max_token_len_batch)
      batch_negative_tokens = pad_2d_vals(
        batch_negative_tokens, real_batch_size, max_token_len_batch)
      if max_token_len_batch > self.max_len:
        max_token_len_batch = self.max_len
      

      # convert len arrays to numpy ones
      batch_anchor_token_lens = np.array(batch_anchor_token_lens)
      batch_positive_token_lens = np.array(batch_positive_token_lens)
      batch_negative_token_lens = np.array(batch_negative_token_lens)
      # characters
      max_char_len_anchor = np.max([np.max(aa) for aa in batch_anchor_char_lens])
      max_char_len_pos = np.max([np.max(aa) for aa in batch_pos_char_lens])
      max_char_len_neg = np.max([np.max(aa) for aa in batch_neg_char_lens])
      max_char_len = max(max_char_len_anchor, max_char_len_pos, max_char_len_neg)
      batch_anchor_chars = pad_3d_vals(
                          batch_anchor_chars, real_batch_size, 
                          max_token_len_batch, max_char_len, dtype=np.int32)
      batch_pos_chars = pad_3d_vals(
                          batch_pos_chars, real_batch_size, 
                          max_token_len_batch, max_char_len, dtype=np.int32)
      batch_neg_chars = pad_3d_vals(
                          batch_neg_chars, real_batch_size, 
                          max_token_len_batch, max_char_len, dtype=np.int32)
      batch_anchor_char_lens = pad_2d_vals(batch_anchor_char_lens, real_batch_size,  max_token_len_batch)
      batch_pos_char_lens = pad_2d_vals(batch_pos_char_lens, real_batch_size,  max_token_len_batch)
      batch_neg_char_lens = pad_2d_vals(batch_neg_char_lens, real_batch_size,  max_token_len_batch)




    output = []
    if self.use_element_img or True:
      output.extend(
        [batch_anchor_elements, batch_positive_elements, batch_negative_elements])
    if self.use_horizontal_cut:
      output.extend(
        [batch_anchor_horizontal, batch_pos_horizontal, batch_neg_horizontal])
    if self.use_vertical_cut:
      output.extend(
        [batch_anchor_vertical, batch_pos_vertical, batch_neg_vertical])
    if self.use_xpath:
      output.extend(
        [batch_anchor_tokens, batch_positive_tokens, batch_negative_tokens])
      output.extend(
        [batch_anchor_token_lens, batch_positive_token_lens, batch_negative_token_lens])

      output.extend(
        [batch_anchor_chars, batch_pos_chars, batch_neg_chars])
      output.extend(
        [batch_anchor_char_lens, batch_pos_char_lens, batch_neg_char_lens])
      

    return output, batch_dummy_target

  def on_epoch_end(self):
    #np.random.shuffle(self.instances)
    print('on_epoch_end')
    return None
  
  def filter_hard_cases(self, pos_dist, neg_dist):
    #print('len pos_dist: ', len(pos_dist))
    #print('training count: ', self.train_data_count)
    #print('instance count: ', len(self.instances))
    hard_idx = neg_dist < pos_dist + alpha
    #print('pos_dist: ', pos_dist, '\n\n')
    #print('neg_dist: ', neg_dist, '\n\n')
    hard_idx = hard_idx.nonzero()
    hard_train_idx = hard_idx[0].tolist()
    #print('hard_train_idx: ', hard_train_idx)
    hard_data_idx = [self.train_idx2_data_idx[i] for i in hard_train_idx]
    self.hard_instances = [self.instances[i] for i in hard_data_idx]
    print('hard_instances: ', len(self.hard_instances), '\n')
    self.train_idx2_data_idx = {}
    self.train_data_count = 0

  


if __name__ == '__main__':
  csvFile = '/Users/hoavu/workplace/crawler/python-scripts/final-data/dataset.csv'
  rawData = '/Users/hoavu/workplace/tools-ml/tools/common-tools/src/ml-models/data/screenshots/'
  screen_name = 'SM-N950U1_8.0.0_ionic.storm.alert.centinela_.LoginActivity_-544726065.jpg'

  # max_len, all_tokens, all_chars = collect_vocabs(trainCsv)
  # char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=char_emb_dim)
  # token_vocab = Vocab(fileformat='voc', voc=all_tokens, dim=token_emb_dim)
  # instances = read_all_instances(csvFile, rawData, token_vocab, char_vocab)
  # print('len: ', len(instances))
  # pos, neg = read_neg_pos(csvFile)
  # triplet_idices_generator(pos, neg)
  cut_element(rawData+screen_name, (42,889,1038,1014))
