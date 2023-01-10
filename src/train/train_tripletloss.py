from __future__ import (
  absolute_import, division, print_function, unicode_literals)
import os
import sys
from tensorflow import keras
from tensorflow.keras.callbacks import (
  ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
import tensorflow as tf
import numpy as np
import math
from .models import multimodal_net
from .data import DataGenSequence, collect_vocabs
from .utils import (
  get_best_model, get_smallest_loss, evaluate, accuracy, triplet_loss_dist)
from configs.config import (
      epochs, trainCsv, char_emb_dim, token_emb_dim,
      char_path, token_path, patience, batch_size, emb_size, fine_tune_at,
      element_size, use_xpath, use_element_img, use_horizontal_cut, 
      use_vertical_cut)
from .vocab_utils import Vocab


if __name__ == '__main__':
  checkpoint_models_path = 'models/'
  max_len, all_tokens, all_chars = collect_vocabs(trainCsv)
  char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=char_emb_dim)
  token_vocab = Vocab(fileformat='voc', voc=all_tokens, dim=token_emb_dim)
  model, visual_module = multimodal_net(
    'cnn', use_xpath=use_xpath, max_xpath_len=max_len,
    use_element_img=use_element_img, use_horizontal_cut=use_horizontal_cut, 
    use_vertical_cut=use_vertical_cut)
  print('max_len: ', max_len)
  char_vocab.dump_to_txt2(char_path)
  token_vocab.dump_to_txt2(token_path)

  # Callbacks
  tensor_board = keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, 
    write_images=True)
  model_names = checkpoint_models_path + \
    'model.{epoch:02d}-{val_loss:.4f}.hdf5'
  model_checkpoint = ModelCheckpoint(
    model_names, monitor='val_loss', verbose=1, save_best_only=True)
  early_stop = EarlyStopping('val_loss', patience=patience)
  reduce_lr = ReduceLROnPlateau(
    'val_loss', factor=0.5, patience=int(2), verbose=1
  )
  callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

  # Data generators
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

  # start training
  adam = keras.optimizers.Adam(lr=0.00001)
  model.compile(optimizer=adam, loss=triplet_loss_dist, metrics=[accuracy])
  model.fit_generator(train_generator,
                      steps_per_epoch=len(train_generator),
                      validation_data=valid_generator,
                      validation_steps=len(valid_generator),
                      epochs=3,
                      verbose=1,
                      shuffle=False,
                      use_multiprocessing=False,
                      callbacks=callbacks,
                      max_queue_size=300)

  # Test
  pretrained_path = get_best_model()
  print('pretrained_path: ', pretrained_path)
  model.load_weights(get_best_model())
  preds = []
  for idx, batch in enumerate(test_generator):
    # [batch_size, 3] [positive_dist, negative_dist]
    y_pred = model.predict(batch[0])
    y_pred = np.reshape(y_pred, (-1, 2))
    preds.append(y_pred)

  preds = np.concatenate(preds)  # [size, 3]
  print(preds)
  thresholds = np.arange(0, 4, 0.0001)
  accuracies = np.zeros(len(thresholds))
  for idx, threshold in enumerate(thresholds):
    correct_count = 0
    for pred in preds:
      if threshold > pred[0] and threshold < pred[1]:
        correct_count += 1
    accuracies[idx] = correct_count/preds.shape[0]
  max_idx = np.argmax(accuracies)
  print('accuracies: %.2f\t threshold: %.4f' %
      (accuracies[max_idx], thresholds[max_idx]))
