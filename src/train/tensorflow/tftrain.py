import tensorflow as tf
import numpy as np
import os
import sys
import time
from datetime import datetime
import os.path
import importlib
from .tfmodels import lenet
from ..data import DataGenSequence, collect_vocabs
from ..utils import tf_triplet_loss, tf_evaluate, save_variables_and_metagraph
from configs.config import (
  epochs, trainCsv, char_emb_dim, token_emb_dim,
  char_path, token_path, patience, batch_size, emb_size, fine_tune_at,
  element_size, use_xpath, use_element_img, use_horizontal_cut, 
  use_vertical_cut, alpha, drop_rate, weight_decay)
if __name__ == '__main__':
  np.random.seed(1)
  with tf.Graph().as_default():
    tf.set_random_seed(1)

    # Data generators
    token_vocab = None
    char_vocab = None
    max_len = 20
    train_generator = DataGenSequence(
      'train', token_vocab, char_vocab, max_len, use_xpath=use_xpath,
      use_element_img=use_element_img, use_horizontal_cut=use_horizontal_cut,
      use_vertical_cut=use_vertical_cut)
    valid_generator = DataGenSequence(
      'validation', token_vocab, char_vocab, max_len, use_xpath=use_xpath,
      use_element_img=use_element_img, use_horizontal_cut=use_horizontal_cut,
      use_vertical_cut=use_vertical_cut)
    test_generator = DataGenSequence(
      'test', token_vocab, char_vocab, max_len, use_xpath=use_xpath,
      use_element_img=use_element_img, use_horizontal_cut=use_horizontal_cut,
      use_vertical_cut=use_vertical_cut)
    print('length train batches', len(train_generator))
    print('length valid batches', len(valid_generator))

    batch_idx = tf.Variable(0, dtype=tf.float32)
    phase_train = tf.placeholder_with_default(
      False, shape=(), name='phase_train')
    img_placeholder = tf.placeholder(
      tf.float32, shape=(None, element_size, element_size, 3), name='inputs')
    network = importlib.import_module('src.nets.inception_resnet_v2')
    pre_logits, _ = network.inference(
      img_placeholder, 1-drop_rate, phase_train=phase_train,
      bottleneck_layer_size=emb_size, weight_decay=weight_decay)

    embeddings = tf.nn.l2_normalize(pre_logits, 1, 1e-10, name='embeddings')
    anchor, positive, negative = tf.split(embeddings, num_or_size_splits=3, axis=0)

    # Calculate the total losses
    triplet_loss = tf_triplet_loss(anchor, positive, negative, alpha)
    regularization_losses = tf.get_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([triplet_loss] +
                          regularization_losses, name='total_loss')
    learning_rate_base = 0.02
    learning_rate = tf.train.exponential_decay(
                  learning_rate_base,
                  batch_idx * batch_size,
                  1*len(train_generator)*batch_size,
                  0.9, staircase=True)
    opt = tf.train.AdamOptimizer(
                    learning_rate, beta1=0.9, 
                    beta2=0.999, epsilon=0.1)
    train_op = opt.minimize(total_loss, global_step=batch_idx)
    init_op = tf.initialize_all_variables()
    sess = tf.Session()

    # Create a saver
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(os.path.expanduser('models/facenet'), subdir)
    best_acc = 0
    best_train_acc = 0
    best_thres = 0

    # Actually intialize the variables
    sess.run(init_op)
    for epoch in range(500):
      anchor_embs, negative_embs, positive_embs = [], [], []
      sumLoss = 0
      start = time.time()
      train_generator.set_training(True)
      for idx, batch in enumerate(train_generator):
        batch = batch[0]  # ignore those dummy labels
        batch_images = np.concatenate((batch[0], batch[1], batch[2]))
        _, loss, lr, batch_idx_ = sess.run([
                      train_op, total_loss, learning_rate, batch_idx],
                      feed_dict={
                        img_placeholder: batch_images,
                        phase_train: True})
        sumLoss += loss
        print('Batch %d/%d - loss %.3f - learning_rate %.4f - took %ds' %
              (idx+1, len(train_generator), loss, lr,  time.time() - start),
              end='\r')
        if idx > 100: 
          break
      sumLoss = sumLoss/len(train_generator)
      #evaluate on the training set and select hard, semi-hard cases
      start = time.time()
      print()
      train_generator.set_training(False)
      for idx, batch in enumerate(train_generator): 
        batch = batch[0]  # ignore those dummy labels
        batch_images = np.concatenate((batch[0], batch[1], batch[2]))
        a_embs, p_embs, n_embs = sess.run(
              [anchor, positive, negative], feed_dict={
                img_placeholder: batch_images,
                phase_train: False})
        anchor_embs.extend(a_embs)
        positive_embs.extend(p_embs)
        negative_embs.extend(n_embs)
        print('Evaluating training batch %d/%d - took %ds'
              % (idx+1, len(train_generator), time.time() - start), end='\r')

      train_acc, thres, pos_dist, neg_dist = tf_evaluate(
                        anchor_embs, positive_embs, negative_embs)
      train_generator.filter_hard_cases(pos_dist, neg_dist)
      print('\nsumLoss: %.2f' % sumLoss)
      print('train accuracy: %.2f\t train threshold: %.4f\n' %
            (train_acc, thres))
      with open('logs/' + subdir, 'a') as f:
        f.write('train loss: %.2f\n' % sumLoss)
        f.write('train accuracy: %.2f\t train threshold: %.4f\n' %
                (train_acc, thres))

      # evaluate on test set
      anchor_embs, negative_embs, positive_embs = [], [], []
      for idx, batch in enumerate(test_generator): 
        batch = batch[0]  # ignore those dummy labels
        batch_images = np.concatenate((batch[0], batch[1], batch[2]))
        a_embs, p_embs, n_embs = sess.run(
          [anchor, positive, negative], feed_dict={
                img_placeholder: batch_images,
                phase_train: False})
        anchor_embs.extend(a_embs)
        positive_embs.extend(p_embs)
        negative_embs.extend(n_embs)

      acc, thres, _, _ = tf_evaluate(anchor_embs, positive_embs, negative_embs)
      print('test accuracy: %.3f\t test threshold: %.4f\n' %
            (acc, thres))
      with open('logs/' + subdir, 'a') as f:
        f.write('test accuracy: %.3f\t test threshold: %.4f\n' %
                (acc, thres))
      if(acc > best_acc or acc +
         train_acc > best_acc + best_train_acc) and acc > 0.75:
        best_acc = acc
        best_train_acc = train_acc
        best_thres = thres
        save_variables_and_metagraph(
                        sess, saver, model_dir, subdir,
                        epoch, logfile=subdir,
                        inputs={'inputs': img_placeholder},
                        outputs={'embeddings': embeddings})

