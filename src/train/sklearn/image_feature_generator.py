from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from scipy import misc
from sk_utils import load_model, prewhiten


class ImageGenerator(object):
  def __init__(self, model_path, size=160):
    self.cache = {}
    self.size = size
    self.sess = tf.compat.v1.Session()
    load_model(self.sess, model_path)
    self.not_found = 0

    #read file back to cache
    if os.path.exists('features.txt'):
      with open('features.txt', 'r') as f:
        for line in f:
          try:
            line = line.strip()
            parts = line.split('\t')
            array_string = parts[1]
            emb = np.array(list(map(float, array_string.split(' '))))
            self.cache[parts[0]] = emb
          except Exception as e:
            print('Error: ', e, '\t at line: ', line)


  def read_image(self, path):
    try:
      img = misc.imread(os.path.expanduser(path), mode='RGB')
    except Exception as e:
      print(e)
      print('eror path: ', path)
      return None

    if img.shape[0] < 30 or img.shape[1] < 30:
      return None
    img = misc.imresize(img, (self.size, self.size), interp='bilinear')
    prewhitened = prewhiten(img)
    return prewhitened


  def extract_emb(self, path):
    if path in self.cache:
      return self.cache[path]

    img = self.read_image(path)
    inputs = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    feed_dict = {
      inputs: [img],
      phase_train_placeholder : False
    }
    [anchor_emb] = self.sess.run([embeddings], feed_dict=feed_dict)
    self.cache[path] = anchor_emb[0]
    return anchor_emb[0]


  def has_key(self, key):
    return key in self.cache


  def get_emb(self, key, img):
    ''' extract emb given an np array
        key: for caching
        img: np array [size, size, 3]
    '''
    if key in self.cache:
      return self.cache[key]

    inputs = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
    feed_dict = {
      inputs: [img],
      phase_train_placeholder : False
    }
    [anchor_emb] = self.sess.run([embeddings], feed_dict=feed_dict)
    self.cache[key] = anchor_emb[0]
    return anchor_emb[0]


  def store_cache(self):
    with open('features.txt', 'w') as f:
      for key, value in self.cache.items():
        feat_str = ' '.join(map(str, value.tolist()))
        f.write('%s\t%s\n'%(key, feat_str))


def main(args):
  gen = ImageGenerator(args.model_dir)
  paths = glob.glob(args.img_dir + '*.jpg')
  for path in paths:
    key = path[len(args.img_dir):-4]
    img = gen.read_image(path)
    gen.get_emb(key, img)
    break

  gen.store_cache()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='path to our facenet pretrain model', type=str)
  parser.add_argument('img_dir', help='path to the image dir', type=str)
  args = parser.parse_args()
  main(args)
