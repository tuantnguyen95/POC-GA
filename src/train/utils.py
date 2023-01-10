from configs import config
from configs.config import alpha, emb_size, alpha
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys
import os
import math
import time
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate


def tf_triplet_loss(anchor, positive, negative, alpha):
    """ Tripletloss in tensorflow
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

def tf_center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def triplet_loss(y_true, y_pred):
  a_pred = y_pred[:, 0:1*emb_size]
  p_pred = y_pred[:, 1*emb_size:1*2*emb_size]
  n_pred = y_pred[:, 1*2*emb_size:1*3*emb_size]

  positive_distance = K.sum(K.square(a_pred - p_pred), axis=1, keepdims=True)
  negative_distance = K.sum(K.square(a_pred - n_pred), axis=1, keepdims=True)
  loss = K.sum(K.maximum(0.0, positive_distance -
               negative_distance + alpha))
  return loss


def triplet_loss_dist(y_true, y_pred):
  margin = K.constant(alpha)
  positive_dist = y_pred[:,0,0]
  negative_dist = y_pred[:,1,0]
  return K.mean(K.maximum(K.constant(0), positive_dist - negative_dist + margin))


def euclidean_distance(vects):
  x, y = vects
  return K.sqrt(
    K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def accuracy(y_true, y_pred):
  return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])


def distance(embeddings1, embeddings2, distance_metric=0):
  if distance_metric == 0:
    # Euclidian distance
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
  elif distance_metric == 1:
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * \
      np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
  else:
    raise 'Undefined distance metric %d' % distance_metric
  return dist


def prewhiten(x):
  return (x-127.5)*0.0078125


def evaluate(embeddings, actual_issame, subtract_mean=False,
             distance_metric=0, threshold=0.8):
  ''' Calculate evaluation metrics
    embeddings :     [2*batch_size, embedding_size]
        2*anchors+positives+negatives
    actual_issame :  [batch_size] = numOf(positives+negatives)
    threshold   : fixed for now
  '''
  assert(len(embeddings) == 2*len(actual_issame))
  embeddings1, embeddings2 = np.split(embeddings, 2)
  if subtract_mean:
    mean = np.mean(embeddings, axis=0)
  else:
    mean = 0.0
  dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
  print('dist: ', dist)
  pred = None
  thresholds = np.arange(0, 4, 0.001)
  accuracies = np.zeros(len(thresholds))
  for idx, threshold in enumerate(thresholds):
    if distance_metric == 0:
      pred = dist <= threshold
    else:
      pred = dist >= threshold
    equals = np.equal(pred, actual_issame)
    totalCorrect = np.sum(equals.astype(int))
    accuracies[idx] = totalCorrect/len(actual_issame)
  print('accuracies: ', accuracies)
  max_idx = np.argmax(accuracies)
  print('accuracies: %.2f\t threshold: %.2f' %
      (accuracies[max_idx], thresholds[max_idx]))


def get_smallest_loss():
  import re
  pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
  p = re.compile(pattern)
  losses = [float(p.match(f).groups()[1])
        for f in os.listdir('models/') if p.match(f)]
  if len(losses) == 0:
    import sys
    return sys.float_info.max
  else:
    return np.min(losses)


def get_latest_model():
  import glob
  import os
  files = glob.glob('models/*.hdf5')
  files.sort(key=os.path.getmtime)
  if len(files) > 0:
    return files[-1]
  else:
    return None


def get_best_model():
  import re
  pattern = 'model.(?P<epoch>\d+)-(?P<val_loss>[0-9]*\.?[0-9]*).hdf5'
  p = re.compile(pattern)
  files = [f for f in os.listdir('models/') if p.match(f)]
  filename = None
  if len(files) > 0:
    losses = [float(p.match(f).groups()[1]) for f in files]
    best_index = int(np.argmin(losses))
    filename = os.path.join('models', files[best_index])
    print('loading best model: {}'.format(filename))
  return filename


def save_variables_and_metagraph(
  sess, saver, model_dir, model_name,
  step, logfile, inputs={}, outputs={},
  char_vocab=None, token_vocab=None,
  char_path=None, token_path=None):
    # Save the model checkpoint
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    print('Saving variables, metagraph: ',
          checkpoint_path + ' step: ' + str(step))
    with open ('train/logs/' + logfile, 'a') as f:
      f.write('Saving variables, metagraph: ' +
              checkpoint_path + ' step: ' + str(step) + '\n')
    save_time_variables = time.time() - start_time
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time

    tf.saved_model.simple_save(
      sess, model_dir.replace('facenet', 'saved_model') + str(step),
      inputs=inputs,
      outputs=outputs)
    if char_vocab:
      char_vocab.dump_to_txt2(char_path)
    if token_vocab:
      token_vocab.dump_to_txt2(token_path)


def tf_evaluate(
      anchor_embs, positive_embs, negative_embs):
  anchor_embs = np.array(anchor_embs)
  positive_embs = np.array(positive_embs)
  negative_embs = np.array(negative_embs)
  diff = np.subtract(anchor_embs, positive_embs)
  pos_dist = np.sum(np.square(diff),1)
  #print('pos_dist: ', pos_dist.tolist())
  diff = np.subtract(anchor_embs, negative_embs)
  neg_dist = np.sum(np.square(diff),1)
  #print('\n\n')
  #print('neg_dist: ', neg_dist.tolist())
  thresholds = np.arange(0, 4, 0.001)
  accuracies = np.zeros(len(thresholds))
  for idx, threshold in enumerate(thresholds):
    correct_count = 0
    for i in range(len(pos_dist)):
      if threshold > pos_dist[i] and threshold < neg_dist[i]:
        correct_count += 1
    accuracies[idx] = correct_count/len(pos_dist)
  max_idx = np.argmax(accuracies)
  return accuracies[max_idx], thresholds[max_idx], pos_dist, neg_dist


if __name__ == "__main__":
  embeddings = np.random.random((128,  9*emb_size))
  emb_size = 128
  a_pred = embeddings[:,  0:3*emb_size]  # [batch_size, 3*emb_size]
  p_pred = embeddings[:,  3*emb_size:6*emb_size]  # [batch_size, 3*emb_size]
  n_pred = embeddings[:,  6*emb_size:9*emb_size]
  embeddings1 = np.concatenate((a_pred, a_pred))
  embeddings2 = np.concatenate((p_pred, n_pred))

  actual_issame = np.concatenate((np.ones(128), np.zeros(128)))
  print('embeddings: ', embeddings)
  evaluate(np.concatenate((embeddings1, embeddings2)),
       actual_issame, distance_metric=1)
