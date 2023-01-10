import tensorflow as tf
import sys, os
import json
import random
import re
import numpy as np
from numpy.linalg import norm
from .data import read_and_resize, DataGenSequence
from tensorflow.python.platform import gfile
from .utils import distance, tf_evaluate
def read_vector_from_file (fname):
  json_array = json.load(open(fname))
  return np.asarray(json_array)

def load_model(model, input_map=None):
    # Check if the model is a model directory 
    # (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(
          os.path.join(model_exp, meta_file), 
          input_map=input_map)
        saver.restore(tf.get_default_session(), 
                      os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
          'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
          'There should not be more than one meta '
          'file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


np.random.seed(1)
tf.set_random_seed(1)
test_generator = DataGenSequence(
      'test', None, None, 30, use_xpath=False,
      use_element_img=True, use_horizontal_cut=False,
      use_vertical_cut=False)
print('test batch: ', len(test_generator))
print('test instance: ', test_generator.instancesSize())

sess=tf.Session() 

with tf.Graph().as_default():

  with tf.Session() as sess:

    # Load the model
    load_model(sys.argv[1])

    # Get input and output tensors
    # names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # for name in names:
    #   if 'input' in name:
    #     print(name)

    inputs = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    anchor_embs, negative_embs, positive_embs = [],[],[]
    for idx, batch in enumerate(test_generator): 
      batch = batch[0] # ignore those dummy labels
      #batch_images = np.concatenate((batch[0], batch[1], batch[2]))
      feed_dict = {
        inputs: batch[0],
        phase_train_placeholder : False
      }
      [anchor_emb] = sess.run([embeddings], feed_dict=feed_dict)
      anchor_embs.extend(anchor_emb)
      feed_dict = {
        inputs: batch[1],
        phase_train_placeholder : False
      }
      [pos_emb] = sess.run([embeddings], feed_dict=feed_dict)
      positive_embs.extend(pos_emb)
      feed_dict = {
        inputs: batch[2],
        phase_train_placeholder : False
      }
      [neg_emb] = sess.run([embeddings], feed_dict=feed_dict)
      negative_embs.extend(neg_emb)
    acc, thres, _, _ = tf_evaluate(anchor_embs, positive_embs, negative_embs)
print('test accuracy: %.3f\t test threshold: %.4f\n' %
                (acc, thres))