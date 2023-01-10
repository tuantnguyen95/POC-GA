import torch
import numpy as np
import string
from service.luna.string_embedding.networks import TwoLayerCNN

data_file = 'service/luna/string_embedding/model/'
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

max_length = 30 # A word maximum length is required by Embedding model
alphabet = 'achensrdvkuiobftglmyjzpwqx'
char_count = len(alphabet)

net = TwoLayerCNN(char_count, max_length, embedding=128, channel=8, mtc_input=False).to(device)
net.load_state_dict(torch.load(data_file + 'embedding_model.torch'))
net.eval()

map_digits = ['zero','one','two','three','four','five','six','seven','eight','nine']

def replace_digit(word):
  text = word
  for digit in string.digits:
    if digit in text:
      text = text.replace(digit, map_digits[int(digit)])

  if len(text) > max_length:
    for digit in string.digits:
      if digit in word:
        word = word.replace(digit, ' ' + map_digits[int(digit)] + ' ')
    return word
  else:
    return text


def embedding(text):
  text = text.lower()

  text = text.replace('\n',' ')
  for p in string.punctuation:
    if p in text:
      text = text.replace(p,' ')

  # Handle case digit in text
  # will remove when retrain embedding model with digit data
  words = []
  for w in text.split():
    new_w = replace_digit(w)
    if ' ' in new_w:
      words.extend(new_w.split())
    else:
      words.append(new_w)

  if len(words) == 0:
    return np.zeros((128))

  embeddings = []
  for w in words:
    ip = np.zeros((1, char_count, max_length))
    for i, c in enumerate(w):
      idx = alphabet.find(c)
      if idx == -1:
        continue
      if i >= max_length:
        break
      ip[0, idx, i] = 1.
    input_tensor = torch.from_numpy(ip)
    embeddings.append(net(input_tensor.float().to(device)).cpu().data.numpy())
  embs = np.concatenate(embeddings)
  emb_v = np.mean(embs, axis=0)

  return emb_v
