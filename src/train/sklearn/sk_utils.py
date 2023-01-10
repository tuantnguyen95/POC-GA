import os
import re
import traceback
import argparse
import tensorflow as tf
import numpy as np
import pytesseract
import cv2
import imageio
from lxml import etree
from PIL import Image
from tensorflow.python.platform import gfile
from numpy.linalg import norm


# an workaround for saving error
tf.compat.v1.disable_eager_execution()


# to be added if a class is missing
IOS_TEXT_ATTRS = {
  "XCUIElementTypeAlert": "label",
  "XCUIElementTypeButton": "label",
  "XCUIElementTypeStaticText": "value",
  "XCUIElementTypeTextView": "value"
}


def get_id(element, platform):
  if platform.lower() == 'ios':
    if 'name' in element:
      return element['name']
  # Or else on Android
  else:
    if 'accessibility-id' in element:
      return element['accessibility-id']
    if 'resource-id' in element:
      return element['resource-id']
  return ''


def get_recursive_text(element, platform):
  ''' its text and its children text recursively
  '''
  text = get_element_text(element, platform)
  children_text = ''
  for child in list(element):  # all children
    children_text += get_recursive_text(child, platform) + ' '
  return text + children_text.strip()


def get_element_text(element, platform):
  ''' text attribute of the element
  '''
  if platform.lower() == 'ios':
    field = 'label'
    ele_type = element.attrib['type']
    if ele_type in IOS_TEXT_ATTRS:
      field = IOS_TEXT_ATTRS[ele_type]
  else:
    field = 'text'
  if field in element.attrib:
    return element.attrib[field]
  return ''


def get_ele_info_by_xpath(tree, xpath):
  ele_finder = etree.XPath(xpath)
  candidates = ele_finder(tree)
  if len(candidates) == 0:
    return None
  ele = dict(candidates[0].attrib)
  return ele


def get_ele_by_xpath(tree, xpath):
  #print('get_ele_by_xpath: ' + xpath)
  ele_finder = etree.XPath(xpath)
  candidates = ele_finder(tree)
  if len(candidates) == 0:
    return None
  return candidates[0]


def tree_from_file(raw_path, prime_screen_name):
  return etree.parse(raw_path + '/xmls/' + prime_screen_name + '.xml')


def load_model(sess, model, input_map=None):
  # Check if the model is a model directory 
  # (containing a metagraph and a checkpoint file)
  #  or if it is a protobuf file with a frozen graph
  model_exp = os.path.expanduser(model)
  if os.path.isfile(model_exp):
    print('Model filename: %s' % model_exp)
    with gfile.FastGFile(model_exp, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, input_map=input_map, name='')
  else:
    print('Model directory: %s' % model_exp)
    meta_file, ckpt_file = get_model_filenames(model_exp)

    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)

    saver = tf.compat.v1.train.import_meta_graph(
      os.path.join(model_exp, meta_file), 
      input_map=input_map)
    saver.restore(sess, os.path.join(model_exp, ckpt_file))


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


def screen_file_name(screen_name, raw_dir):
  return "%s/screenshots/%s.jpg" % (raw_dir, screen_name)


def prewhiten(x):
  return (x-127.5)*0.0078125


def is_empty(s):
  if not isinstance(s, str):
    return True
  return s is None or len(s.strip()) == 0


def cosine(x, y):
  return np.dot(x, y)/(norm(x) * norm(y))


def ocr_path_text(img_pth):
  image = cv2.imread(img_pth)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  text = pytesseract.image_to_string(image)
  return text


def ocr_img_text(img):
  if img is None:
    return ""
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  text = pytesseract.image_to_string(img)
  return text


def lev_distance(s1, s2):
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2+1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1],distances[i1 + 1],
                                   distances_[-1])))
    distances = distances_
  return distances[-1]


def lev_sim(str1, str2):
  return 1 - lev_distance(str1, str2)/max(len(str1), len(str2))


def class_name_from_xpath(xpath):
  if is_empty(xpath): 
    return None
  return xpath.split('/')[-1].split('[')[0]


def idx2onehot(idx, size):
  a = np.zeros(size)
  a[idx] = 1
  return a.tolist()


def screen_size_string2area(ssize):
  w, h = list(map(int,ssize.split('x')))
  return w*h


def area(left, top, right, bottom):
  return (right -left) * (bottom -top)


def element_shape_ratio(left, top, right, bottom):
  return (right - left) / (bottom - top)


def cast_dataframe_2int(df, columns):
  for col in columns:
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype(int)
  return df


def str_to_bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def overall_similarity(
    visual_sim=0, h_visual_sim=0, h_visual_small_sim=0,
    v_visual_sim=0, v_visual_small_sim=0,
    prime_ocr_exist=0, ocr_sim=0, revisit_ocr_exist=0,
    prime_id_exist=0, id_sim=0, revisit_id_exist=0,
    prime_text_exist=0, text_sim=0, revisit_text_exist=0,
    prime_recur_text_exist=0, recur_text_sim=0, revisit_recur_text_exist=0,
    xpath_sim=0, classname_sim=0, scrollable_element=False):
  ''' A man-made similarity with
      all the weights got from a perceptron
  '''
  if prime_ocr_exist:
    ocr_coff = 2.708
  else:
    ocr_coff = 0.708

  if prime_text_exist:
    text_coff = 6.26
  else:
    text_coff = 0.26

  if prime_recur_text_exist:
    recur_text_coff = 5.1632
  else:
    recur_text_coff = 0.1632

  if prime_id_exist:
    id_coff = 0.8826
  else:
    id_coff = 0.1826

  if scrollable_element:
    coff = [0.5, 0.5]
    return np.average([xpath_sim, classname_sim], weights=coff)
  coff = [10.02, 0.85, 3.09, 0.174,
          7.22, ocr_coff, 0.94, 0.408,
          id_coff, text_coff, recur_text_coff]
  return np.average([visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim,
                     v_visual_small_sim, ocr_sim, xpath_sim, classname_sim, 
                     id_sim, text_sim, recur_text_sim], weights=coff)


def decode_bound(element, platform, pixel_map=None):
  ''' a string like [0,72][168,240] on Android
      x, y, width, height on iOS
      output: the same format [left, top, right, bottom]
  '''
  if platform == 'ios':
    fields = ['x', 'y', 'width', 'height']
    left, top, width, height = [int(element.attrib[f]) for f in fields]
    if not pixel_map:
      return [left, top, left + width, top + height]

    # pre-scale mapping on some devices
    if pixel_map.horizontal_pre_scale:
      left *= pixel_map.horizontal_pre_scale
      width *= pixel_map.horizontal_pre_scale
    if pixel_map.vertical_pre_scale:
      top *= pixel_map.vertical_pre_scale
      height *= pixel_map.vertical_pre_scale

    left = int((left + pixel_map.x_offset) * pixel_map.horizontal_scale)
    top = int((top + pixel_map.y_offset) * pixel_map.vertical_scale)
    width = int(width * pixel_map.horizontal_scale)
    height = int(height * pixel_map.vertical_scale)
    return [left, top, left + width, top + height]


  # Or else on Android
  bounds = element.get('bounds')
  parts = bounds[1:-1].replace('][', ',').split(',')
  return list(map(int, parts))


def exact_cut(screen_name, raw_dir, box, output_size=160):
  ''' get an exact cut of the element along the bound
  '''
  img_path = screen_file_name(screen_name, raw_dir)
  try:
    im = imageio.imread(
      os.path.expanduser(img_path), as_gray=False, pilmode="RGB")
    if box[2] - box[0] < 20 or box[3] - box[1] < 20:
      return None
    new_im = im[box[1]: box[3], box[0] : box[2], :]
    new_im = Image.fromarray(new_im)
    new_im = new_im.resize((output_size, output_size))
    return prewhiten(np.array(new_im))
  except Exception as e:
    print(e)
    traceback.print_exc()
  return None


def ocr_cut(screen_name, raw_dir, box):
  ''' get an exact cut without resizing and standardlize
  '''
  img_path = screen_file_name(screen_name, raw_dir)
  try:
    im = imageio.imread(
      os.path.expanduser(img_path), as_gray=False, pilmode="RGB")
    if box[2] - box[0] < 20 or box[3] - box[1] < 20:
      return None
    new_im = im[box[1]: box[3], box[0] : box[2], :]
    return new_im
  except Exception as e:
    print(e)
    traceback.print_exc()
  return None


def horizonal_cut(screen_name, raw_dir, box, output_size=160, expand_ratio=1):
  ''' get a horizontal cut of the screenshot containing the element
      This gives more context for element matching
    Input: the path to the image and the element bounding box
        im_pth: str
        box: [left, top, right, bottom]
        output_size: resize the cut
    Output: a 2D numpy array
  '''
  img_path = screen_file_name(screen_name, raw_dir)
  try:
    im = imageio.imread(
      os.path.expanduser(img_path), as_gray=False, pilmode="RGB")
    (_, s_width) = np.asarray(im.shape)[0:2]
    new_im = im[box[1]: box[3], 0 : s_width, :]
    new_im = Image.fromarray(new_im)
    new_im = new_im.resize((output_size, output_size))
    return prewhiten(np.array(new_im))
  except Exception as e:
    print(e)
    traceback.print_exc()
  return None


def horizonal_cut_small(screen_name, raw_dir, box, output_size=160, expand_ratio=1):
  ''' Expand the element cut horizontally a bit to get more context.
      The expanded size is two times the element's width
    Input: the path to the image and the element bounding box
        im_pth: str
        box: [left, top, right, bottom]
        output_size: resize the cut
        margin: cut larger than the image a bit to get more context
    Output: a 2D numpy array
  '''
  img_path = screen_file_name(screen_name, raw_dir)
  try:
    im = imageio.imread(
      os.path.expanduser(img_path), as_gray=False, pilmode="RGB")
    e_width = box[2] - box[0]
    (_, s_width) = np.asarray(im.shape)[0:2]
    margin = 2 * e_width

    new_im = im[box[1]: box[3], max(0, box[0] - margin) : min(box[2] + margin, s_width), :]
    new_im = Image.fromarray(new_im)
    new_im = new_im.resize((output_size, output_size))
    return prewhiten(np.array(new_im))
  except Exception as e:
    print(e)
    traceback.print_exc()
  return None


def vertical_cut(screen_name, raw_dir, box, output_size=160, expand_ratio=1):
  ''' get a horizontal cut of the screenshot containing the element
      This gives more context for element matching
    Input: the path to the image and the element bounding box
        im_pth: str
        box: [left, top, right, bottom]
        output_size: resize the cut
    Output: a 2D numpy array
  '''
  img_path = screen_file_name(screen_name, raw_dir)
  try:
    im = imageio.imread(
      os.path.expanduser(img_path), as_gray=False, pilmode="RGB")
    (s_height, _) = np.asarray(im.shape)[0:2]

    new_im = im[0: s_height, box[0]: box[2], :]
    new_im = Image.fromarray(new_im)
    new_im = new_im.resize((output_size, output_size))
    return prewhiten(np.array(new_im))
  except Exception as e:
    print(e)
    traceback.print_exc()
  return None


def vertical_cut_small(screen_name, raw_dir, box, output_size=160, expand_ratio=1):
  ''' Expand the element cut vertially a bit to get more context
      The expanded size is twice element size.
    Input: the path to the image and the element bounding box
        im_pth: str
        box: [left, top, right, bottom]
        output_size: resize the cut
        margin: cut larger than the image a bit to get more context
    Output: a 2D numpy array
  '''
  img_path = screen_file_name(screen_name, raw_dir)
  try:
    im = imageio.imread(
      os.path.expanduser(img_path), as_gray=False, pilmode="RGB")
    e_height = box[3] - box[1]
    (s_height, _) = np.asarray(im.shape)[0:2]
    margin = 2 * e_height

    new_im = im[max(box[1] - margin, 0): min(box[3] + margin, s_height), box[0]: box[2], :]
    new_im = Image.fromarray(new_im)
    new_im = new_im.resize((output_size, output_size))
    return prewhiten(np.array(new_im))
  except Exception as e:
    print(e)
    traceback.print_exc()
  return None


def calculate_thresholds(similarities, labels):
  precisions, recalls, f1s = [], [], []
  thresholds = np.arange(0, 1, 0.01)
  for idx, threshold in enumerate(thresholds):
    precision, recall, f1 = calculate_accuracy_figures(similarities, labels, threshold)
    # print('threshold, precision, recall, f1: ', threshold, precision, recall, f1)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

  # EXACT_THRES is the lowest thres giving the best precision
  precisions = np.array(precisions)
  max_prec_idxs = np.nonzero(precisions > 0.97)
  exact_thres = thresholds[np.min(max_prec_idxs)]
  print('EXACT_THRES: %f -> precesion %f, recall %f, f1 %f'%
        (exact_thres, precisions[np.min(max_prec_idxs)],
         recalls[np.min(max_prec_idxs)], f1s[np.min(max_prec_idxs)]))

  # PRECISE_THRES is the lowest thres giving the best F1 score
  f1s = np.array(f1s)
  max_f1_idxs = np.nonzero(f1s == np.max(f1s))
  precise_thres = thresholds[np.min(max_f1_idxs)]
  print('PRECISE_THRES: %f -> precesion %f, recall %f, f1 %f' %
        (precise_thres, precisions[np.min(max_f1_idxs)],
         recalls[np.min(max_f1_idxs)], np.max(f1s)))

  # LOOSE_THRES is the highest thres giving the best recall
  recalls = np.array(recalls)
  max_recall_idxs = np.nonzero(recalls >= 0.98)
  loose_thres = thresholds[np.max(max_recall_idxs)]
  print('LOOSE_THRES: %f -> precesion %f, recall %f, f1 %f'%
        (loose_thres, precisions[np.max(max_recall_idxs)],
         recalls[np.max(max_recall_idxs)], f1s[np.max(max_recall_idxs)]))


def calculate_accuracy_figures(similarities, labels, threshold):
  fn, tn, fp, tp = 0, 0, 0, 0
  for idx, (sim, label) in enumerate(zip(similarities, labels)):
    if label == 1:
      if sim >= threshold:
        tp += 1
      else:
        fn += 1
    else:
      if sim < threshold:
        tn += 1
      else:
        fp += 1
  if tp == 0:
    return 0.0, 0.0, 0.0
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2*(recall * precision) / (recall + precision)
  return precision, recall, f1
