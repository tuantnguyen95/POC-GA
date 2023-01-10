import tensorflow as tf
import tensorflow.contrib.slim as slim
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

