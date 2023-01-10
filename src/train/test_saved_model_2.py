import tensorflow as tf
import time
import sys
import json
import numpy as np
from numpy.linalg import norm
from data import read_and_resize
from utils import distance

def read_vector_from_file (fname):
  json_array = json.load(open(fname))
  return np.asarray(json_array)



sess=tf.Session() 
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'input'
output_key = 'embeddings'

export_path =  sys.argv[1]
img_paths = sys.argv[2:]
imgs = [read_and_resize(img_path, 160) for img_path in img_paths]
imgs = np.array(imgs)
imgs = [imgs for i in range(40)]
imgs = np.concatenate(imgs)
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature = meta_graph_def.signature_def

input_node_name = signature[signature_key].inputs[input_key].name
embs_node_name = signature[signature_key].outputs[output_key].name

inputs = sess.graph.get_tensor_by_name(input_node_name)
embeddings = sess.graph.get_tensor_by_name(embs_node_name)

start = time.time()

feed_dict = {
  inputs: imgs
}

[embs] = sess.run([embeddings], feed_dict=feed_dict)

print('forwarding took: ', time.time() - start)