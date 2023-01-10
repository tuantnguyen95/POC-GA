from __future__ import (
  absolute_import, division, print_function, unicode_literals
)
import tensorflow as tf

import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
  Input, Dense, concatenate, Lambda, LSTM, Embedding, Dropout)
from tensorflow.keras.layers import (
  GlobalAveragePooling2D, Conv2D, Flatten, MaxPooling2D)
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from .utils import euclidean_distance


def multimodal_net(net, use_element_img=True, use_xpath=False,
                   use_horizontal_cut=True, use_vertical_cut=True, screen_size=299,
                   element_size=128, channel=3, emb_size=128, xpath_emb_size=8,
                   max_xpath_len=100, token_vocab_size=500, fine_tune_at=100,
                   dropout_rate=0.4, cut_input_size=128):
  ''' Encode the visual parts, namely the screenshot and the element image
    triplet: [anchor_info, positive_info, negative_info]
    Input:
    net: cnn, mobilenet or inceptionResnet

    Output:
    distances: pos_anchor, neg_anchor, neg_pos
  '''
  # Define the network for the visual part
  element_base_model = None
  if use_element_img:
    if net == 'cnn':
      element_base_model = cnn_net(
          input_size=element_size, num_channels=3,
          emb_size=emb_size, dropout_rate=dropout_rate)
    elif net == 'mobilenet':
      element_base_model = tf.keras.applications.MobileNetV2(
          include_top=False, weights='imagenet',
          input_shape=(element_size, element_size, channel), pooling='max')
    else:
      element_base_model = tf.keras.applications.InceptionResNetV2(
          include_top=False, weights='imagenet',
          input_shape=(element_size, element_size, channel), pooling='max')

    element_encoder = tf.keras.Sequential([
        element_base_model,
        keras.layers.Dense(emb_size, activation='relu'),
        Dropout(dropout_rate)
    ])

  #define the network for encoding vertical cut:
  if use_vertical_cut:
    vertical_cut_base = cnn_net(
        input_size=cut_input_size, num_channels=3,
        emb_size=emb_size, dropout_rate=dropout_rate)
    vertical_cut_encoder = tf.keras.Sequential([
        vertical_cut_base,
        keras.layers.Dense(emb_size, activation='relu'),
        Dropout(dropout_rate)
    ])

  #define the network for encoding horizontal cut:
  if use_horizontal_cut:
    horizontal_cut_base  = cnn_net(
        input_size=cut_input_size, num_channels=3,
        emb_size=emb_size, dropout_rate=dropout_rate)
    horizontal_cut_encoder = tf.keras.Sequential([
        horizontal_cut_base,
        keras.layers.Dense(emb_size, activation='relu'),
        Dropout(dropout_rate)
    ])

  # embedding and LSTM for xpath tokens
  if use_xpath:
    emb_layer = Embedding(
        output_dim=xpath_emb_size,
        input_dim=token_vocab_size, input_length=max_xpath_len)
    shared_lstm = LSTM(xpath_emb_size)
    xpath_encoder = tf.keras.Sequential([
        emb_layer,
        shared_lstm,
        Dense(xpath_emb_size, activation='relu'),
        Dropout(dropout_rate/2)
    ])


  # input nodes
  model_inputs = []
  if use_element_img:
    input_a_element = Input(
        (element_size, element_size, channel), name='anchor_element')
    input_p_element = Input(
        (element_size, element_size, channel), name='positive_element')
    input_n_element = Input(
        (element_size, element_size, channel), name='negative_element')
    model_inputs.extend(
        [input_a_element, input_p_element, input_n_element])

  if use_horizontal_cut:
    input_a_horizontal = Input(
        (cut_input_size, cut_input_size, channel), name='anchor_horizontal_cut')
    input_p_horizontal = Input(
        (cut_input_size, cut_input_size, channel), name='positive_horizontal_cut')
    input_n_horizontal = Input(
        (cut_input_size, cut_input_size, channel), name='negative_horizontal_cut')
    model_inputs.extend(
        [input_a_horizontal, input_p_horizontal, input_n_horizontal])

  if use_vertical_cut:
    input_a_vertical = Input(
        (cut_input_size, cut_input_size, channel), name='anchor_vertical_cut')
    input_p_vertical= Input(
        (cut_input_size, cut_input_size, channel), name='positive_vertical_cut')
    input_n_vertical = Input(
        (cut_input_size, cut_input_size, channel), name='negative_vertical_cut')
    model_inputs.extend(
        [input_a_vertical, input_p_vertical, input_n_vertical])

  if use_xpath:
    input_a_tokens = Input(shape=(max_xpath_len,),
                           dtype='int32', name='anchor_xpath_tokens')
    input_p_tokens = Input(shape=(max_xpath_len,),
                           dtype='int32', name='positive_xpath_tokens')
    input_n_tokens = Input(shape=(max_xpath_len,),
                           dtype='int32', name='negative_xpath_tokens')
    model_inputs.extend([input_a_tokens, input_p_tokens, input_n_tokens])

  #encoding
  total_embs_size = 0
  all_anchor_embs = []
  all_positive_embs = []
  all_negative_embs = []
  if use_element_img:
    a_element_embs = element_encoder(input_a_element)
    p_element_embs = element_encoder(input_p_element)
    n_element_embs = element_encoder(input_n_element)
    all_anchor_embs.append(a_element_embs)
    all_positive_embs.append(p_element_embs)
    all_negative_embs.append(n_element_embs)
    total_embs_size += emb_size

  if use_horizontal_cut:
    a_horizonal_embs = horizontal_cut_encoder(input_a_horizontal)
    p_horizonal_embs = horizontal_cut_encoder(input_p_horizontal)
    n_horizonal_embs = horizontal_cut_encoder(input_n_horizontal)
    all_anchor_embs.append(a_horizonal_embs)
    all_positive_embs.append(p_horizonal_embs)
    all_negative_embs.append(n_horizonal_embs)
    total_embs_size += emb_size

  if use_vertical_cut:
    a_vertical_embs = vertical_cut_encoder(input_a_vertical)
    p_vertical_embs = vertical_cut_encoder(input_p_vertical)
    n_vertical_embs = vertical_cut_encoder(input_n_vertical)
    all_anchor_embs.append(a_vertical_embs)
    all_positive_embs.append(p_vertical_embs)
    all_negative_embs.append(n_vertical_embs)
    total_embs_size += emb_size

  if use_xpath:
    a_xpath_emb = xpath_encoder(input_a_tokens)
    p_xpath_emb = xpath_encoder(input_p_tokens)
    n_xpath_emb = xpath_encoder(input_n_tokens)
    all_anchor_embs.append(a_xpath_emb)
    all_positive_embs.append(p_xpath_emb)
    all_negative_embs.append(n_xpath_emb)
    total_embs_size += xpath_emb_size

  if len(all_anchor_embs) == 1:
    print(all_anchor_embs)
    a_embs = all_anchor_embs[0] 
    p_embs = all_positive_embs[0]
    n_embs = all_negative_embs[0]
  else:
    print('total_embs_size: ', total_embs_size)
    merge_layer = tf.keras.Sequential([
        Dense(emb_size, input_shape=(total_embs_size,), activation='relu'),
        Dropout(dropout_rate)
    ])
    a_embs = merge_layer(
        concatenate(all_anchor_embs, axis=-1))
    p_embs = merge_layer(
        concatenate(all_positive_embs, axis=-1))
    n_embs = merge_layer(
        concatenate(all_negative_embs, axis=-1))

  normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')

  a_embs = normalize(a_embs)
  p_embs = normalize(p_embs)
  n_embs = normalize(n_embs)

  # define distances between the anchor, the positive and the negative embeddings
  positive_dist = Lambda(
      euclidean_distance, name='pos_dist')([a_embs, p_embs])
  negative_dist = Lambda(
      euclidean_distance, name='neg_dist')([a_embs, n_embs])

  all_dists = Lambda(
      lambda vects: K.stack(vects, axis=1),
      name='all_dists'
  )([positive_dist, negative_dist])
  model = keras.models.Model(
      model_inputs, all_dists, name='triplet_siamese')

  return model, element_base_model


def cnn_net(input_size=128, num_channels=3, emb_size=128, dropout_rate=0.4):
  ''' A straight forward convolutional neural net
    input: An input image of (input_size, input_size, num_channels)
  '''
  model = Sequential()
  model.add(Conv2D(16, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(input_size, input_size, num_channels)))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(emb_size, activation='relu'))
  model.add(Dropout(dropout_rate))
  return model


def main():
  with tf.device("/cpu:0"):
    model, _ = multimodal_net('cnn')
  print(model.summary())
  plot_model(model, to_file='model.svg',
             show_layer_names=True, show_shapes=True)
  K.clear_session()


if __name__ == '__main__':
  main()
