import tensorflow as tf
import sys
import json
import numpy as np
from numpy.linalg import norm
from .data import read_and_resize, DataGenSequence
from .utils import distance, tf_evaluate
def read_vector_from_file (fname):
  json_array = json.load(open(fname))
  return np.asarray(json_array)

test_generator = DataGenSequence(
      'test', None, None, 30, use_xpath=False,
      use_element_img=True, use_horizontal_cut=False,
      use_vertical_cut=False)
print('test batch: ', len(test_generator))
print('test instance: ', test_generator.instancesSize())

sess=tf.Session() 
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'inputs' # [1, 80, 80 , 3]
is_training= 'is_training'
output_key = 'embeddings'

export_path =  sys.argv[1]
img_paths = sys.argv[2:]
imgs = [read_and_resize(img_path, 80) for img_path in img_paths]
imgs = np.array(imgs)# np.expand_dims(img, axis=0)
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature = meta_graph_def.signature_def

print(signature[signature_key].inputs)
input_node_name = signature[signature_key].inputs[input_key].name
embs_node_name = signature[signature_key].outputs[output_key].name

inputs = sess.graph.get_tensor_by_name(input_node_name)
embeddings = sess.graph.get_tensor_by_name(embs_node_name)


anchor_embs, negative_embs, positive_embs = [],[],[]
for idx, batch in enumerate(test_generator): 
  batch = batch[0] # ignore those dummy labels
  #batch_images = np.concatenate((batch[0], batch[1], batch[2]))
  feed_dict = {
    inputs: batch[0]
  }
  [anchor_emb] = sess.run([embeddings], feed_dict=feed_dict)
  anchor_embs.extend(anchor_emb)
  feed_dict = {
    inputs: batch[1]
  }
  [pos_emb] = sess.run([embeddings], feed_dict=feed_dict)
  positive_embs.extend(pos_emb)
  feed_dict = {
    inputs: batch[2]
  }
  [neg_emb] = sess.run([embeddings], feed_dict=feed_dict)
  negative_embs.extend(neg_emb)
acc, thres, _, _ = tf_evaluate(anchor_embs, positive_embs, negative_embs)
print('test accuracy: %.3f\t test threshold: %.4f\n' %
                (acc, thres))