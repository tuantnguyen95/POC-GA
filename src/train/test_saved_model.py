import tensorflow as tf
import sys
import json
import numpy as np
from numpy.linalg import norm
from .data import read_and_resize, pad_2d_vals, pad_3d_vals
from .utils import distance
from .vocab_utils import Vocab

def read_vector_from_file (fname):
  json_array = json.load(open(fname))
  return np.asarray(json_array)



sess=tf.Session() 
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# 'tokens': tokens_placeholder,
# 'token_lens': token_lens,
# 'chars': chars_placeholder,
# 'char_lens': char_lens
tokens_input_key="tokens"
token_lens_input_key="token_lens" 
chars_input_key="chars"
char_lens_input_key="char_lens"
embedding_output_key="embeddings"

export_path =  sys.argv[1]
# img_path = sys.argv[2]
# img = read_and_resize(img_path, 60)
# img = np.expand_dims(img, axis=0)
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature = meta_graph_def.signature_def

print(signature[signature_key])

tokens_input_name     = signature[signature_key].inputs[tokens_input_key].name
token_lens_input_name = signature[signature_key].inputs[token_lens_input_key].name
chars_input_name      = signature[signature_key].inputs[chars_input_key].name
char_lens_input_name  = signature[signature_key].inputs[char_lens_input_key].name
embedding_output_name  = signature[signature_key].outputs[embedding_output_key].name

tokens_input = sess.graph.get_tensor_by_name(tokens_input_name)
token_lens_input = sess.graph.get_tensor_by_name(token_lens_input_name)
chars_input = sess.graph.get_tensor_by_name(chars_input_name)
char_lens_input = sess.graph.get_tensor_by_name(char_lens_input_name)
embedding_output = sess.graph.get_tensor_by_name(embedding_output_name)

xpath=("/hierarchy/android.widget.FrameLayout/android.widget"
      ".LinearLayout/android.widget.FrameLayout/android.widget"
      ".LinearLayout/android.widget.FrameLayout/android.widget"
      ".RelativeLayout/android.widget.FrameLayout/android.widget"
      ".LinearLayout/android.widget.RelativeLayout/android.webkit"
      ".WebView/android.webkit.WebView/android.view.View[3]/android"
      ".view.View/android.view.View[2]/android.view.View/android.widget.Spinner")
char_vocab_path = '/Users/hoavu/workplace/tools-ml/tools/common-tools/src/ml-models/models/char.vocab'
token_vocab_path = '/Users/hoavu/workplace/tools-ml/tools/common-tools/src/ml-models/models/token.vocab'
char_vocab  = Vocab(vec_path=char_vocab_path, fileformat='txt2')
token_vocab = Vocab(vec_path=token_vocab_path, fileformat='txt2')
max_token_len = 60

token_idxs = token_vocab.to_index_sequence(xpath)
token_len = len(token_idxs)
chars_matrix = char_vocab.to_character_matrix(xpath)
char_lens = [len(cur_token) for cur_token in chars_matrix]
#print (chars_matrix)
#print('char_lens: ', char_lens)
if len(token_idxs) > max_token_len:
  token_len  = max_token_len
  token_idxs = token_idxs[:max_token_len]
  chars_matrix = chars_matrix [:max_token_len, :]
max_char_len = max(char_lens)
chars_matrix = pad_3d_vals([chars_matrix], 1, token_len, max_char_len)
char_lens    = pad_2d_vals([char_lens], 1, len(char_lens))
token_idxs   = pad_2d_vals([token_idxs], 1, len(token_idxs))
token_lens = [token_len]
  
  
feed_dict = {
  tokens_input: token_idxs,
  token_lens_input: token_lens,
  chars_input: chars_matrix,
  char_lens_input: char_lens
}

embs = sess.run([embedding_output], feed_dict=feed_dict)
embs = np.squeeze(embs)
js_array = embs
anchor = embs
print ('anchor: ', embs)
# print ('positive: ', positive)
# print ('negative: ', negative)
print('js_array: ', js_array)
cos_sim = np.dot(anchor, js_array)/(norm(anchor)*norm(js_array))
print('dist: ', cos_sim)