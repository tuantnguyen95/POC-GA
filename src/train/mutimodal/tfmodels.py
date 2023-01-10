import tensorflow as tf
import tensorflow.contrib.slim as slim
from . import layer_utils
def model(img=None, horizontal_cut=None, vertical_cut=None, 
          use_imge=True, use_horizontal_cut=True, use_vertical_cut=True):

  return None

def lenet(X, img_size=128, num_channels=3, dim_size=128 ,phase_train=True):
    """
    X: a placeholder containing image data, shape [None, 160, 160, 3]
    """
    with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
      #reshape X as it is flat array in mnist dataset
      #X = tf.reshape(X, [-1, img_size,img_size,num_channels])
      #first conv layer, same padding by default
      conv1 = slim.conv2d(X, 6, [3, 3], scope='conv1')
      mp1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
      #second conv layer
      conv2 = slim.conv2d(mp1, 16, [3, 3], scope='conv2')
      mp2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
      #third conv layer
      conv3 = slim.conv2d(mp2, 32, [5, 5], scope='conv3')
      mp3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
      #last fully connected layer
      mp3_flatten = tf.contrib.layers.flatten(mp3)
      fc2 = slim.fully_connected(mp3_flatten, dim_size,
              activation_fn=None, scope='fc3') #shape[None, dim_size]
      out = slim.dropout(fc2, 0.4, is_training=phase_train, scope='dropout2')
      return out


def token_encoder(tokens, token_lens, char_embs, word_vocab, dim_size=20,
                  phase_train=True, dropout_rate=0.4, is_training=False):
  ''' 
  char_embs [batch_size, token_len, dim]
  '''
  with tf.variable_scope("token_encoder", reuse=tf.AUTO_REUSE):
    word_embedding = tf.get_variable(
      "word_embedding", trainable=True,
      initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
    token_repres = tf.nn.embedding_lookup(word_embedding, tokens) # [batch_size, question_len, word_dim]
    token_repres = tf.concat(axis=-1, values=[token_repres, char_embs])
    (token_outputs_fw, token_outputs_bw, _) = layer_utils.my_lstm_layer(token_repres, dim_size,
                    input_lengths=token_lens, scope_name="token_lstm", reuse=False,
                    is_training=is_training, dropout_rate=dropout_rate)
    token_outputs_fw  = layer_utils.collect_final_step_of_lstm(token_outputs_fw, token_lens - 1)
    token_outputs_bw  = token_outputs_bw[:,0,:]
    token_outputs_emb = tf.concat(axis=1, values=[token_outputs_fw, token_outputs_bw])

    if is_training:
      token_outputs_emb = tf.nn.dropout(token_outputs_emb, (1 - dropout_rate))

    return token_outputs_emb


def char_encoder(char_matrix, char_lens, char_vocab, dim_size=20,
                  phase_train=True, dropout_rate=0.4, is_training=False):
  input_shape = tf.shape(char_matrix)
  batch_size = input_shape[0]
  token_len = input_shape[1]
  char_len = input_shape[2]

  with tf.variable_scope("char_encoder", reuse=tf.AUTO_REUSE):
    char_embedding = tf.get_variable(
      "char_embedding", trainable=True,
      initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
    char_repres = tf.nn.embedding_lookup(char_embedding, char_matrix) # [batch_size, question_len, word_dim]
    char_repres = tf.reshape(char_repres, shape=[-1, char_len, dim_size])
    char_lens = tf.reshape(char_lens, [-1])
    char_mask = tf.sequence_mask(char_lens, char_len, dtype=tf.float32)
    char_repres = tf.multiply(char_repres, tf.expand_dims(char_mask, axis=-1))


    (char_outputs_fw, char_outputs_bw, _) = layer_utils.my_lstm_layer(char_repres, dim_size,
                    input_lengths=char_lens, scope_name="char_lstm", reuse=False,
                    is_training=is_training, dropout_rate=dropout_rate)
    char_outputs_fw  = layer_utils.collect_final_step_of_lstm(char_outputs_fw, char_lens - 1)
    char_outputs_bw  = char_outputs_bw[:,0,:]
    char_outputs_emb = tf.concat(axis=1, values=[char_outputs_fw, char_outputs_bw])
    char_outputs_emb = tf.reshape(char_outputs_emb, [batch_size, token_len, 2*dim_size])


    if is_training:
      char_outputs_emb = tf.nn.dropout(char_outputs_emb, (1 - dropout_rate))

    return char_outputs_emb
