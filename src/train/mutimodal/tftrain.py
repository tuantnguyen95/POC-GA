import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import sys
import time
from datetime import datetime
import os.path
import importlib
from ..vocab_utils import Vocab
from .tfmodels import lenet, token_encoder, char_encoder
from ..data import DataGenSequence, collect_vocabs
from ..utils import tf_triplet_loss, tf_evaluate, save_variables_and_metagraph
from ..configs.config import (
  epochs, trainCsv, char_emb_dim, token_emb_dim, max_xpath_len, 
  char_path, token_path, patience, batch_size, emb_size, fine_tune_at,
  element_size, use_xpath, use_element_img, use_horizontal_cut, 
  use_vertical_cut, alpha, drop_rate, weight_decay)


if __name__ == '__main__':
  np.random.seed(1)
  with tf.Graph().as_default():
    tf.set_random_seed(1)

    # Data generators
    max_len, all_tokens, all_chars = collect_vocabs(trainCsv)
    char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=char_emb_dim)
    token_vocab = Vocab(fileformat='voc', voc=all_tokens, dim=token_emb_dim)
    #char_vocab.dump_to_txt2(char_path)
    #token_vocab.dump_to_txt2(token_path)
    if max_len > max_xpath_len:
      max_len = max_xpath_len
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

    pre_logits = None
    batch_idx = tf.Variable(0, dtype=tf.float32)
    phase_train = tf.placeholder_with_default(
      False, shape=(), name='phase_train')
    
    if use_element_img:
      img_placeholder = tf.placeholder(
        tf.float32, shape=(None, element_size, element_size, 3), name='element_img')
      network = importlib.import_module('train.nets.dummy')
      pre_logits, _ = network.inference(  # [batch_size, emb_size]
        img_placeholder, 1-drop_rate, phase_train=phase_train,
        bottleneck_layer_size=emb_size, weight_decay=weight_decay)

    if use_xpath:
      tokens_placeholder = tf.placeholder(tf.int32, shape=(None, None), name='xpath_tokens')
      token_lens = tf.placeholder(tf.int32, [None])
      char_lens = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
      chars_placeholder = tf.placeholder(tf.int32, [None, None, None])

      train_char_emb = char_encoder(  # [batch_size, emb_size]
        chars_placeholder, char_lens, char_vocab, 
        is_training=True, dropout_rate=drop_rate, 
        dim_size=char_emb_dim)
      test_char_emb = char_encoder(  # [batch_size, emb_size]
        chars_placeholder, char_lens, char_vocab, 
        is_training=False, dropout_rate=drop_rate,
        dim_size=char_emb_dim)
      train_token_emb = token_encoder(  # [batch_size, emb_size]
        tokens_placeholder, token_lens,
        train_char_emb, token_vocab, 
        is_training=True, dropout_rate=drop_rate, 
        dim_size=token_emb_dim)
      test_token_emb = token_encoder(  # [batch_size, emb_size]
        tokens_placeholder, token_lens, 
        test_char_emb, token_vocab, 
        is_training=False, dropout_rate=drop_rate,
        dim_size=token_emb_dim)
      #train_token_emb = tf.concat(axis=-1, values=[train_char_emb, train_token_emb])
      #test_token_emb = tf.concat(axis=-1, values=[test_char_emb, test_token_emb])
      def f1(): return train_token_emb
      def f2(): return test_token_emb
      token_embs = tf.cond(
        tf.equal(phase_train, tf.constant(True)), f1, f2)
      if pre_logits is not None:
        token_embs = slim.fully_connected(token_embs, emb_size, activation_fn=None, reuse=False)
        pre_logits = tf.concat(axis=-1, values=[pre_logits, token_embs])
      else:
        pre_logits = token_embs
      pre_logits = slim.fully_connected(pre_logits, emb_size, activation_fn=None, reuse=False)
      

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
    model_dir = os.path.join(os.path.expanduser('train/models/facenet'), subdir)
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
        #print('components in batch: ', len(batch))
        feed_dict={phase_train: True}
        if use_element_img:
          batch_images    = np.concatenate((batch[0], batch[1], batch[2]))
          feed_dict[img_placeholder] = batch_images
        if use_xpath: 
          batch_tokens    = np.concatenate((batch[3], batch[4], batch[5]))
          batch_token_len = np.concatenate((batch[6], batch[7], batch[8]))
          batch_chars     = np.concatenate((batch[9], batch[10], batch[11]))
          batch_char_lens = np.concatenate((batch[12], batch[13], batch[14]))

          feed_dict[tokens_placeholder] = batch_tokens
          feed_dict[token_lens] = batch_token_len
          feed_dict[chars_placeholder] = batch_chars
          feed_dict[char_lens] = batch_char_lens
        _, loss, lr, batch_idx_ = sess.run([
                      train_op, total_loss, learning_rate, batch_idx],
                      feed_dict = feed_dict)
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
        batch = batch[0]
        feed_dict={phase_train: False}
        if use_element_img:
          batch_images    = np.concatenate((batch[0], batch[1], batch[2]))
          feed_dict[img_placeholder] = batch_images
        if use_xpath: 
          batch_tokens    = np.concatenate((batch[3], batch[4], batch[5]))
          batch_token_len = np.concatenate((batch[6], batch[7], batch[8]))
          batch_chars     = np.concatenate((batch[9], batch[10], batch[11]))
          batch_char_lens = np.concatenate((batch[12], batch[13], batch[14]))

          feed_dict[tokens_placeholder] = batch_tokens
          feed_dict[token_lens] = batch_token_len
          feed_dict[chars_placeholder] = batch_chars
          feed_dict[char_lens] = batch_char_lens
        a_embs, p_embs, n_embs = sess.run(
              [anchor, positive, negative], feed_dict=feed_dict)
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
      with open('train/logs/' + subdir, 'a') as f:
        f.write('train loss: %.2f\n' % sumLoss)
        f.write('train accuracy: %.2f\t train threshold: %.4f\n' %
                (train_acc, thres))

      # evaluate on test set
      anchor_embs, negative_embs, positive_embs = [], [], []
      for idx, batch in enumerate(test_generator): 
        batch = batch[0]  # ignore those dummy labels
        feed_dict={phase_train: False}
        if use_element_img:
          batch_images    = np.concatenate((batch[0], batch[1], batch[2]))
          feed_dict[img_placeholder] = batch_images
        if use_xpath: 
          batch_tokens    = np.concatenate((batch[3], batch[4], batch[5]))
          batch_token_len = np.concatenate((batch[6], batch[7], batch[8]))
          batch_chars     = np.concatenate((batch[9], batch[10], batch[11]))
          batch_char_lens = np.concatenate((batch[12], batch[13], batch[14]))

          feed_dict[tokens_placeholder] = batch_tokens
          feed_dict[token_lens] = batch_token_len
          feed_dict[chars_placeholder] = batch_chars
          feed_dict[char_lens] = batch_char_lens
        a_embs, p_embs, n_embs = sess.run(
          [anchor, positive, negative], feed_dict=feed_dict)
        anchor_embs.extend(a_embs)
        positive_embs.extend(p_embs)
        negative_embs.extend(n_embs)

      acc, thres, _, _ = tf_evaluate(anchor_embs, positive_embs, negative_embs)
      print('test accuracy: %.3f\t test threshold: %.4f\n' %
            (acc, thres))
      with open('train/logs/' + subdir, 'a') as f:
        f.write('test accuracy: %.3f\t test threshold: %.4f\n' %
                (acc, thres))
      if acc > best_acc and acc > 0.7:
        best_acc = acc
        best_train_acc = train_acc
        best_thres = thres
        inputs={}
        if use_xpath:
          inputs['tokens'] = tokens_placeholder
          inputs['token_lens'] = token_lens
          inputs['chars'] = chars_placeholder
          inputs['char_lens'] =  char_lens
        if use_element_img:
          inputs['element_img'] = img_placeholder
        save_variables_and_metagraph(
                        sess, saver, model_dir, subdir,
                        epoch, logfile=subdir,
                        char_path=char_path, token_path=token_path,
                        inputs=inputs,
                        outputs={'embeddings': embeddings})

