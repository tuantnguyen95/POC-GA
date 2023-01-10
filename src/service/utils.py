import os
import re
import traceback
import time
import numpy as np
import cv2
import math
import consul
import argparse
from collections import Counter
import pytesseract
from Levenshtein import distance as lev_distance
from lxml import etree
from lxml.html import etree as html
from numpy.linalg import norm
from service import constants
from colormath.color_objects import XYZColor, sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000 as color_distance
from unidecode import unidecode_expect_nonascii
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from service.logger import LoggingWrapper


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--logstash_host", type=str, help="", default="")
  parser.add_argument("--deploy_env", type=str, help="", default="test")
  parser.add_argument(
    'serving_host', type=str, help='ip adress of the serving api', default='0.0.0.0')
  parser.add_argument(
    'serving_port', type=str, help='port of the serving api', default='8500')
  parser.add_argument(
    'deploy_host', type=str, default='0.0.0.0', help='host to deploy this api')
  parser.add_argument(
    'deploy_port', type=str, default='5000', help='port to deploy this api')
  return parser.parse_args()


def get_kobiton_ocr_service_info():
  try:
    consul_host = os.environ.get('KOBITON_CONSUL_HOST')
    consul_port = os.environ.get('KOBITON_CONSUL_REST_API_PORT')
    ocr_service_id = os.getenv('KOBITON_AI_OCR_SERVICE_ID', 'ita-ocr-service')
    ocr_url_key = 'grpc_service_private_url'
    if consul_host is not None and consul_port is not None:
      c = consul.Consul(host=consul_host, port=consul_port)
      _, metadata = c.catalog.service(ocr_service_id)
      if metadata:
        ocr_metadata = metadata[0]['ServiceMeta']
        if ocr_url_key in ocr_metadata:
          url = ocr_metadata[ocr_url_key].split(':')
          ocr_host = url[0]
          ocr_port = url[1]
          return ocr_host, ocr_port
  except Exception:
    trace = traceback.format_exc()
    logger.error(
      "Cannot get OCR service info from Consul",
      extras={
        "error": trace,
      },
    )
  return None, None


def exist_xpath(xml_tree, xpath):
  ele_finder = etree.XPath(xpath)
  return ele_finder and len(ele_finder(xml_tree)) > 0


def get_ele_info_by_xpath(tree, xpath):
  ele_finder = etree.XPath(xpath)
  candidates = ele_finder(tree)
  if len(candidates) == 0:
    return None
  ele = dict(candidates[0].attrib)
  return ele


def get_ele_by_xpath(tree, xpath):
  ele_finder = etree.XPath(xpath)
  candidates = ele_finder(tree)
  if len(candidates) == 0:
    return None
  return candidates[0]


def classname_from_xpath(xpath):
  if not xpath or len(xpath) < 1:
    return None
  class_name = xpath.split('/')[-1].split('[')[0]
  return class_name


def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
  ''' pad a 2d-array to a tensor
  '''
  out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
  if dim1_size > len(in_vals):
    dim1_size = len(in_vals)
  for i in range(dim1_size):
    cur_in_vals = in_vals[i]
    cur_dim2_size = dim2_size
    if cur_dim2_size > len(cur_in_vals):
      cur_dim2_size = len(cur_in_vals)
    out_val[i, :cur_dim2_size] = cur_in_vals[:cur_dim2_size]
  return out_val


def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
  out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
  if dim1_size > len(in_vals):
    dim1_size = len(in_vals)
  for i in range(dim1_size):
    in_vals_i = in_vals[i]
    cur_dim2_size = dim2_size
    if cur_dim2_size > len(in_vals_i):
      cur_dim2_size = len(in_vals_i)
    for j in range(cur_dim2_size):
      in_vals_ij = in_vals_i[j]
      cur_dim3_size = dim3_size
      if cur_dim3_size > len(in_vals_ij):
        cur_dim3_size = len(in_vals_ij)
      out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
  return out_val


def decode_bound(element, platform, pixel_map=None, screen_shape=None):
  """
  A string like [0,72][168,240] on Android
  x, y, width, height on iOS
  output: the same format [left, top, right, bottom]
  """
  bounding_box = [0] * 4
  try:
    if 'bounds' in element.attrib:
      # On Android or Webview
      bounds = element.get('bounds')
      parts = bounds[1:-1].replace('][', ',').split(',')
      bounding_box = list(map(int, parts))
    elif 'x' in element.attrib and 'y' in element.attrib and 'width' in element.attrib and 'height' in element.attrib:
      # On iOS
      fields = ['x', 'y', 'width', 'height']
      left, top, width, height = [int(element.get(f)) for f in fields]
      bounding_box = [left, top, left+width, top+height]
  except Exception:
    bounding_box = [0] * 4

  if pixel_map and any(bounding_box) and platform == constants.Platform.IOS:
    # pre-scale mapping on some devices
    if pixel_map.horizontal_pre_scale:
      bounding_box[0] *= pixel_map.horizontal_pre_scale
      bounding_box[2] *= pixel_map.horizontal_pre_scale
    if pixel_map.vertical_pre_scale:
      bounding_box[1] *= pixel_map.vertical_pre_scale
      bounding_box[3] *= pixel_map.vertical_pre_scale

    bounding_box[0] = int((bounding_box[0] + pixel_map.x_offset) * pixel_map.horizontal_scale)
    bounding_box[1] = int((bounding_box[1] + pixel_map.y_offset) * pixel_map.vertical_scale)
    bounding_box[2] = int((bounding_box[2] + pixel_map.x_offset) * pixel_map.horizontal_scale)
    bounding_box[3] = int((bounding_box[3] + pixel_map.y_offset) * pixel_map.vertical_scale)

  if screen_shape:
    for i in range(4):
      bounding_box[i] = max(bounding_box[i], 0)
      if i == 0 or i == 2:
        #left, right vs screen width
        bounding_box[i] = min(bounding_box[i], screen_shape[0]-1)
      else:
        #top, bottom vs screen height
        bounding_box[i] = min(bounding_box[i], screen_shape[1]-1)
  return bounding_box


def prewhiten(tensor):
  return (tensor - 127.5) * 0.0078125


def crop_img(im, box):
  width = box[2] - box[0]
  height = box[3] - box[1]
  if width > 0 and height > 0:
    new_im = im[box[1]:box[3], box[0]:box[2]]
    return new_im[..., :3]
  return None


def reisze_tf_image(img, cut_box, new_size):
  cut_img = crop_img(img, cut_box)
  if cut_img is not None:
    new_im = cv2.resize(cut_img, new_size)
    return prewhiten(np.array(new_im))
  return None


def cut_element(im, box, output_size=160, use_margin=False, skip_small=False):
  """
  Get a cut of the screenshot with margin
  im: np array [w, h, c]
  """
  im = np.asarray(im)
  im = im[..., :3]
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  if skip_small and (box[2] - box[0] < 10 or box[3] - box[1] < 10):
    return None

  # calculate margin: extends 30% on each size:
  v_margin, h_margin = 0, 0
  if use_margin:
    h_margin = int((box[2] - box[0]) * 3 / 10)
    v_margin = int((box[3] - box[1]) * 3 / 10)

  cut_box = [max(box[0] - h_margin, 0), max(box[1] - v_margin, 0),
            min(box[2] + h_margin, s_width), min(box[3] + v_margin, s_height)]
  return reisze_tf_image(im, cut_box, (output_size, output_size))


def get_horizontal_cut(im, box, output_size=160):
  """
  Get a horizontal cut of the screenshot containing the element
    The cut has the same width as the screenshot
  Input: the path to the image and the element bounding box
      im: str
      box: [left, top, right, bottom]
      output_size: resize the cut
  Output: a 2D numpy array
  """
  im = np.asarray(im)
  im = im[..., :3]
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  cut_box = [0, max(box[1], 0), s_width, min(box[3], s_height)]
  return reisze_tf_image(im, cut_box, (output_size, output_size))


def get_horizontal_small_cut(im, box, output_size=160):
  """
  Get a horizontally expanded cut of the screenshot containing the element
  The expanding size is twice as large as the element's width
  Input: the path to the image and the element bounding box
      im: str
      box: [left, top, right, bottom]
      output_size: resize the cut
  Output: a 2D numpy array
  """
  im = np.asarray(im)
  im = im[..., :3]
  e_width = box[2] - box[0]
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  margin = 2 * e_width
  cut_box = [max(0, box[0] - margin), max(box[1], 0),
             min(box[2] + margin, s_width), min(box[3], s_height)]
  return reisze_tf_image(im, cut_box, (output_size, output_size))


def get_vertical_small_cut(im, box, output_size=160):
  """
  Get a vertically expanded cut of the screenshot containing the element
    The expanding size is twice as large as the element's height
  Input: the path to the image and the element bounding box
      im: str
      box: [left, top, right, bottom]
      output_size: resize the cut
  Output: a 2D numpy array
  """
  im = np.asarray(im)
  im = im[..., :3]
  e_height = box[3] - box[1]
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  margin = 2 * e_height
  cut_box = [max(box[0], 0), max(box[1] - margin, 0),
             min(box[2], s_width), min(box[3] + margin, s_height)]
  return reisze_tf_image(im, cut_box, (output_size, output_size))


def get_vertical_cut(im, box, output_size=160):
  """
  Get a vertical cut of the screenshot containing the element
    The cut has the same height as the screenshot.
  Input: the path to the image and the element bounding box
      im: str
      box: [left, top, right, bottom]
      output_size: resize the cut
  Output: a 2D numpy array
  """
  im = np.asarray(im)
  im = im[..., :3]
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  cut_box = [max(box[0], 0), 0, min(box[2], s_width), s_height]
  return reisze_tf_image(im, cut_box, (output_size, output_size))


def ocr_img_text(img, scrollable_element=False):
  start_time = time.time()
  if scrollable_element:
    return '', 0
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  text = pytesseract.image_to_string(img)
  text = text.lower().strip()
  if text:
    text = ascii_normalize(text)
  return text, time.time() - start_time


def visual_cosine_sim(prime_emb, revisit_embs):
  sims = np.dot(revisit_embs, prime_emb) / (norm(prime_emb) * norm(revisit_embs, axis=1))
  return sims.tolist()


def emb_cosine_sim(emb1, emb2):
  sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
  return sim


def is_empty_string(st):
  return not st or not st.strip()


def text_similarities(prime_text, revisit_texts):
  if is_empty_string(prime_text):
    return [0 for t in revisit_texts]
  return [1 - lev_distance(prime_text, t) / max(len(prime_text), len(t)) for t in revisit_texts]


def text_similarity(text1, text2):
  return 1 - lev_distance(text1, text2) / max(len(text1), len(text2))


def xpath_similarity(prime_xpath, revisit_xpaths):
  return [1 - lev_distance(prime_xpath, t) / max(len(prime_xpath), len(t)) for t in revisit_xpaths]


def ios_overall_similarity(
    visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim,
    v_visual_small_sim, ocr_sim, xpath_sim, classname_sim,
    id_sim, text_sim, recur_text_sim, scrollable_element=False):
  """
  A man-made similarity with all the weights got from a perceptron
  """
  if ocr_sim > 0.2:
    ocr_coff = 6.5995
  else:
    ocr_coff = 0.5995

  if text_sim > 0.2:
    text_coff = 6.5995
  else:
    text_coff = 0.5995

  if recur_text_sim > 0.2:
    recur_text_coff = 6.5995
  else:
    recur_text_coff = 0.5995

  if id_sim > 0.2:
    id_coff = 3.455
  else:
    id_coff = 0.455

  if scrollable_element:
    coff = [0.5, 0.5]
    return np.average([xpath_sim, classname_sim], weights=coff) * 0.90

  coff = [13.56, 1.84, 10.358, 0.672, 4.6955, ocr_coff, 0.1824, 1.355,
          id_coff, text_coff, recur_text_coff]
  return np.average([visual_sim, h_visual_sim, h_visual_small_sim,
                     v_visual_sim, v_visual_small_sim, ocr_sim, xpath_sim,
                     classname_sim, id_sim, text_sim, recur_text_sim], weights=coff)


def android_overall_similarity(
    visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim,
    v_visual_small_sim, ocr_sim, xpath_sim, classname_sim,
    id_sim, text_sim, recur_text_sim, scrollable_element=False):
  """
  A man-made similarity with all the weights got from a perceptron
  """
  if ocr_sim > 0.1:
    ocr_coff = 5.72
  else:
    ocr_coff = 0.72

  if text_sim > 0.1:
    text_coff = 1.2662
  else:
    text_coff = 0.2662

  if recur_text_sim > 0.2:
    recur_text_coff = 5.0993
  else:
    recur_text_coff = 0.50993

  if id_sim > 0.2:
    id_coff = 3.455
  else:
    id_coff = 0.455

  if scrollable_element:
    coff = [0.45, 0.45, 0.06, 0.02, 0.02]
    return np.average([xpath_sim, classname_sim, visual_sim, h_visual_sim, v_visual_sim], weights=coff)

  coff = [18.3, 1.84, 4.339, 0.672, 6.955, ocr_coff, 0.1824, 1.355,
          id_coff, text_coff, recur_text_coff]
  return np.average([visual_sim, h_visual_sim, h_visual_small_sim,
                     v_visual_sim, v_visual_small_sim, ocr_sim, xpath_sim,
                     classname_sim, id_sim, text_sim, recur_text_sim], weights=coff)


def idx2onehot(idx, size):
  onehot_vec = np.zeros(size)
  onehot_vec[idx] = 1
  return onehot_vec.tolist()


def screen_size_string2area(ssize):
  width, height = list(map(int, ssize.split('x')))
  return width * height


def get_area(left, top, right, bottom):
  return (right - left) * (bottom - top)


def screen_size(im):
  im = np.asarray(im)
  (s_height, s_width) = np.asarray(im.shape)[0:2]
  return '%sx%s' % (s_width, s_height)


def ios_scale_factor(im, tree):
  im = np.asarray(im)
  real_width = np.asarray(im.shape)[1]
  logical_width = tree.getroot().getchildren()[0].attrib['width']
  logical_width = int(logical_width)
  return real_width / logical_width


def get_screen_size_in_xml(tree):
  root = tree.getroot()
  width = root.get('width')
  height = root.get('height')
  if not width or not height:
    ele_finder = etree.XPath("//*[@bounds]")
    elements = ele_finder(tree)
    for element in elements:
      bound = decode_bound(element, constants.Platform.ANDROID)
      if bound:
        return get_area(*bound)
  return int(width) * int(height)


def get_image_size_in_xml(tree):
  root = tree.getroot()
  width = root.get('width')
  height = root.get('height')
  if not width or not height:
    length_of_root = len(root)
    if length_of_root > 1: ## For dC devices
        child = root[length_of_root-1]
    elif length_of_root > 0:
        child = root[0]
    width = child.get('width')
    height = child.get('height')
  return int(width), int(height)


def ios_element_off_screen(im, bound):
  """
  Check if the element is completely out of the viewport
  bound: the projected bound: left, top, right, bottom
  """
  (left, top, right, bottom) = bound
  im = np.asarray(im)
  height, width = im.shape[:2]
  # return left < 0 and top < 0 and right > width and bottom > height
  return left >= width or right <= 0 or bottom <= 0 or top >= height


def partially_beyond_viewport(bound, screen, ocr_text, text, platform):
  """
  Check if an element has a part lying outside the viewport
  - On Android, we count element having one edge on the screen boundary or beyond
  - On iOS, we only count those having an edge lying beyond screen boundaries.
  """
  (left, top, right, bottom) = bound
  height, width = screen.shape[:2]
  if constants.Platform.ANDROID == platform:
    if top == 0 or left == 0 or right == width or bottom == height:
      if ocr_text is None:
        return True
    return top < 0 or left < 0 or right > width or bottom > height
  # On iOS
  return top < 0 or left < 0 or right > width or bottom > height


def is_scrollable_view(xpath, tree, platform, scrollable_classes):
  ele_finder = etree.XPath(xpath)
  candidates = ele_finder(tree)
  if len(candidates) == 0:
    return False
  if platform == constants.Platform.IOS and '/html' not in xpath:
    classname = candidates[0].get('type')
    return classname in scrollable_classes
  # html
  scrollable = candidates[0].get('scrollable', default='false') == 'true'
  if '/html' in xpath:
    return scrollable
  # Android
  area_thresh = 0.5  # Threshold of area to pick scrollable large view.
  except_cls_set = {'android.view.View', 'android.widget.Button'}
  classname = classname_from_xpath(xpath)
  # Consider exception case with large View
  if classname == 'android.view.View':
    ele_bound = decode_bound(candidates[0], platform)
    ele_area = get_area(*ele_bound)
    screen_area = get_screen_size_in_xml(tree)
    if ele_area / screen_area > area_thresh:
      except_cls_set.discard('android.view.View')
  has_scrollable_class = False
  for name in scrollable_classes:
    if name in classname:
      has_scrollable_class = True
      break
  return (scrollable or has_scrollable_class) and classname not in except_cls_set


def get_locating_id(element, platform):
  if platform == constants.Platform.IOS:
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
  """
  Element's text and its children' text recursively
  """
  if element is None:
    return ''
  text = get_element_text(element, platform)
  children_text = ''
  for child in list(element):  # all children
    children_text += get_recursive_text(child, platform) + ' '
  return text + children_text.strip()


def get_element_text(element, platform):
  """
  Text attribute of the element
  """
  if element is not None:
    if platform == constants.Platform.IOS and 'type' in element.attrib:
      field = 'label'
      ele_type = element.attrib['type']
      if ele_type in constants.IOS_TEXT_ATTRS:
        field = constants.IOS_TEXT_ATTRS[ele_type]
    else:
      field = 'text'
    if field in element.attrib:
      return element.attrib[field]
  return ''


def decode_img_bytes(img_bytes):
  bgr_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
  return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


# TODO: Similarity for merged
def overall_similarity(visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim,
                       v_visual_small_sim, ocr_sim, xpath_sim, classname_sim,
                       id_sim, text_sim, recur_text_sim, prime_platform, revisit_platform,
                       scrollable_element=False):
  return ios_overall_similarity(  # Just temporary
    visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim,
    v_visual_small_sim, ocr_sim, xpath_sim, classname_sim,
    id_sim, text_sim, recur_text_sim, scrollable_element=scrollable_element)


def random_color(seed=None):
  if seed:
    np.random.seed(0)
  return list(np.random.choice(range(256), size=3))


def draw_debug_info(bound, screen, st, color, idx):
  """
  Visualize assertions info for QA.
  """
  color = (int(color[0]), int(color[1]), int(color[2]))
  (left, top, right, bottom) = bound
  cv2.rectangle(screen, (left, top), (right, bottom), color, 3)
  text_pos = (int((right + left) / 2) - 50, top)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(screen, str(idx) + ": " + st, text_pos, font, 1, color, 3)


def save_screen_debug(prime_screen, revisit_screen, request_id, level):
  """
  Put the prime and revisit screens side by side
  """
  prime_screen = np.asarray(prime_screen)
  p_height, _ = prime_screen.shape[:2]
  revisit_screen = np.asarray(revisit_screen)
  r_height, r_width = revisit_screen.shape[:2]
  new_r_width = int(r_width * p_height / r_height)
  revisit_screen = cv2.resize(revisit_screen, (new_r_width, p_height))
  com = np.concatenate((prime_screen, revisit_screen), axis=1)

  img_dir = "text_assertion_images"
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)
  cv2.imwrite(img_dir + '/%s-%s.jpg' % (request_id, level), com)


def preprocess_ocr(img):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  return gray


def detect_texts(img, box_level=5, threshold=0.):
  preprocessed_img = preprocess_ocr(img)
  data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT, config='--psm 12')
  level = np.asarray(data['level']).astype('int8')
  score = np.asarray([int(float(x)) for x in data['conf']])
  texts = np.asarray(data['text'])
  bounds = np.vstack([data['left'], data['top'], np.asarray(data['left']) + np.asarray(data['width']),
                      np.asarray(data['top']) + np.asarray(data['height'])])
  bounds = bounds.T
  valid_idx = np.argwhere((level == box_level) & (score >= threshold)).flatten()
  return texts[valid_idx], bounds[valid_idx], score[valid_idx]


def get_text_mask(img):
  """
  Segment the text in an image
  """
  img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh_val, mask = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  if mask[mask == 0].shape[0] < mask[mask == 255].shape[0]:
    mask = cv2.bitwise_not(mask)
  return thresh_val, mask.astype(bool)


def get_background_mask(img):
  """
  Segment the text in an image
  """
  img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh_val, mask = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  if mask[mask == 0].shape[0] > mask[mask == 255].shape[0]:
    mask = cv2.bitwise_not(mask)
  return thresh_val, mask.astype(bool)


def get_mean_color_by_mask(img, mask):
  pixel_val = img[mask]
  mean_color = np.mean(pixel_val, axis=0)
  mean_color = mean_color.astype('uint8')
  return mean_color


def compare_colors_in_rgb(color1, color2):
  rgb_1 = sRGBColor(*color1, is_upscaled=True)
  rgb_2 = sRGBColor(*color2, is_upscaled=True)
  xyz_1 = convert_color(rgb_1, XYZColor)
  xyz_2 = convert_color(rgb_2, XYZColor)
  lab_1 = convert_color(xyz_1, LabColor)
  lab_2 = convert_color(xyz_2, LabColor)
  return color_distance(lab_1, lab_2)


def compare_colors_in_lab(color1, color2):
  lab_1 = LabColor(*color1)
  lab_2 = LabColor(*color2)
  return color_distance(lab_1, lab_2)


def get_text_color(img):
  _, mask = get_text_mask(img)
  dst = cv2.medianBlur(img, 3)

  lab_img = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)
  lab_pixel_val = lab_img[mask]
  rgb_pixel_val = img[mask]

  uniques, counts = np.unique(lab_pixel_val, return_counts=True, axis=0)
  if len(counts) == 0:
    return None, None
  idx = np.argsort(counts)
  lab_color_1 = uniques[idx[-1]]
  rgb_color_1 = np.median(rgb_pixel_val[np.where(lab_pixel_val == lab_color_1)[0]], axis=0)

  # Some texts are displayed in gradient color but the biggest color group is incorrect
  # (not caught by human eyes). Let's pick the second.
  if len(counts) > 1 and abs(counts[idx[-1]] - counts[idx[-2]])/lab_pixel_val.shape[0] < 0.01:
    lab_color_2 = uniques[idx[-2]]
    rgb_color_2 = np.median(rgb_pixel_val[np.where(lab_pixel_val == lab_color_2)[0]], axis=0)
    bg_color = get_background_color(img)
    if compare_colors_in_rgb(rgb_color_2, bg_color) > compare_colors_in_rgb(rgb_color_1, bg_color):
      return lab_color_2, rgb_color_2.astype('uint8')
  return lab_color_1, rgb_color_1.astype('uint8')


def get_background_color(img):
  _, mask = get_background_mask(img)
  return get_mean_color_by_mask(img, mask)


def is_different_color(fg_color, bg_color):
  return any([i != j for i, j in zip(fg_color, bg_color)])


def get_linear_color(c: int):
  """
  Calculate linear color.
  Ref: https://github.com/google/Accessibility-Test-Framework-for-Android/blob/7ab5fdb5e2cb675edb752c0d0d9cae3986c0bb0c/src/main/java/com/google/android/apps/common/testing/accessibility/framework/utils/contrast/ContrastUtils.java#L61
  """
  sRGB = c/255.0
  if sRGB <= 0.03928:
    return sRGB/12.92
  else:
    return pow((sRGB+0.055)/1.055, 2.4)


def calculate_luminance(r: int, g: int, b: int):
  """
  Calculate luminance value.
  Ref: https://github.com/google/Accessibility-Test-Framework-for-Android/blob/7ab5fdb5e2cb675edb752c0d0d9cae3986c0bb0c/src/main/java/com/google/android/apps/common/testing/accessibility/framework/utils/contrast/ContrastUtils.java#L54
  """
  r = get_linear_color(r)
  g = get_linear_color(g)
  b = get_linear_color(b)
  return 0.2126*r + 0.7152*g + 0.0722*b


def calculate_contrast_ratio(fg_color, bg_color):
  """
  Calculate contrast ratio.
  Ref: https://github.com/google/Accessibility-Test-Framework-for-Android/blob/7ab5fdb5e2cb675edb752c0d0d9cae3986c0bb0c/src/main/java/com/google/android/apps/common/testing/accessibility/framework/utils/contrast/ContrastSwatch.java#L270
  """
  fr, fg, fb = fg_color
  br, bg, bb = bg_color
  f_luminance = calculate_luminance(fr, fg, fb)
  b_luminance = calculate_luminance(br, bg, bb)
  luminance_ratio = (max(f_luminance, b_luminance)+0.05) / (min(f_luminance, b_luminance)+0.05)
  return round(luminance_ratio, 2)


def is_truth_attribute(attributes, k):
  return attributes and attributes.get(k) == 'true'


def get_attributes_dict(attributes):
  return dict(zip(attributes.keys(), attributes.values()))


def get_ele_img_by_xpath(xpath, xml_tree, screen, platform, pixel_map):
  element = get_ele_info_by_xpath(xml_tree, xpath)
  ele_bound = decode_bound(element, platform, pixel_map)
  return crop_img(screen, ele_bound)


def is_bound1_inside_bound2(bound1, bound2):
  return bound1[0] >= bound2[0] and bound1[1] >= bound2[1] and bound1[2] <= bound2[2] and bound1[3] <= bound2[3]


def is_in_the_same_horizontal_line(bound1, bound2, bound_resize_ratio=0):
  if((bound2[1] <= bound1[3] and bound2[1] >= bound1[1]) \
    or (bound2[3] <= bound1[3] and bound2[3] >= bound1[1]) \
    or (bound2[3] >= bound1[3] and bound2[1] <= bound1[1])):
    return True
  else:
    return False


def extract_ocr_text_by_bound(ele_bound, bounds, texts):
  lines = []
  line = []
  anchor_bound = None

  for idx, (bound, text) in enumerate(zip(bounds, texts)):
    if is_bound1_inside_bound2(bound, ele_bound):
      if len(line) == 0:
        line.append(text)
        anchor_bound = bound
      else:
        if anchor_bound is not None:
          if is_in_the_same_horizontal_line(bound, anchor_bound):
            line.append(text)
          else:
            line_text = ' '.join(line)
            lines.append(line_text.strip())
            line = [text]
            anchor_bound = bound

    if idx == len(bounds) - 1:
      line_text = ' '.join(line)
      lines.append(line_text.strip())
  return '\n'.join(lines)


def extract_ocr_text_by_bound_iou(ele_bound, bounds, texts, iou_thresh = 0.9):
  lines = []
  line = []
  anchor_bound = None
  text_bounds = []

  for idx, (bound, text) in enumerate(zip(bounds, texts)):
    if intersect(bound, ele_bound) >= iou_thresh:
      if len(line) == 0:
        line.append(text)
        anchor_bound = bound
      else:
        if anchor_bound is not None:
          if is_in_the_same_horizontal_line(bound, anchor_bound):
            line.append(text)
          else:
            line_text = ' '.join(line)
            lines.append(line_text.strip())
            line = [text]
            anchor_bound = bound

      text_width = anchor_bound[2] - anchor_bound[0]
      text_height = anchor_bound[3] - anchor_bound[1]
      text_bound = [0] * 4
      text_bound[0] = anchor_bound[0] - ele_bound[0]
      text_bound[1] = anchor_bound[1] - ele_bound[1]
      text_bound[2] = text_bound[0] + text_width
      text_bound[3] = text_bound[1] + text_height
      text_bounds.append(text_bound)

    if idx == len(bounds) - 1:
      line_text = ' '.join(line)
      lines.append(line_text.strip())
  return '\n'.join(lines), text_bounds


def is_in_scroll_view(xpath, xml_tree, platform):
  if platform == constants.Platform.IOS:
    for class_name in constants.IOS_SCROLLABLE_CLASSES:
      if class_name in xpath:
        return True
  elif platform == constants.Platform.ANDROID:
    class_names = xpath.split('/')
    current_xpath = ''
    for class_name in class_names:
      if class_name:
        current_xpath += '/' + class_name
        if is_scrollable_view(current_xpath, xml_tree, platform, constants.ANDROID_SCROLLABLE_CLASSES):
          return True
  return False


def read_file_bytes(filename):
  with open(filename, 'rb') as f:
    byte_data = f.read()
    return byte_data


def convert_image_numpy2bytes(img):
  img = img[:, :, :3]  # Just get value in RGB channels.
  converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  success, encoded_img = cv2.imencode('.jpeg', converted_img)
  return encoded_img.tobytes()


def retry_call_n_times(func, logger, logger_params, *args, n=3):
  """
  Retry a function n times. This is used to assure the results gotten from APIs.
  :param func: a callable function
  :param logger: logger of the service
  :param logger_params: a dictionary that has keys 'log' and 'extras_log'
  :param n: number of time to retry
  :return: result of the function, None if unsuccessful.
  """
  retry = 0
  while retry < n:
    try:
      return func(*args)
    except Exception:
      retry += 1
      extras = logger_params.get('extras', {})
      extras['error'] = traceback.format_exc()
      logger.error("Retry {} times:\n".format(retry) + logger_params.get('log', ''),
                   extras=extras)
  return None


def is_google_vision_enabled():
  return os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


def pad_img_by_width(img, width):
  img_width = img.shape[1]
  if img_width > width:
    raise ValueError('Width of image need to be smaller than the defined width')

  width_diff = width - img_width
  pad_width = int(width_diff / 2)
  pad_width_left = pad_width
  pad_width_right = width_diff - pad_width

  clone_img = img.copy()
  background_color = get_background_color(img)
  return np.stack([np.pad(clone_img[:, :, i], [(0, constants.GOOGLE_VISION_IMAGE_PADDING),
                                               (pad_width_left, pad_width_right)],
                          constant_values=background_color[i])
                   for i in range(3)],
                  axis=2)


def concat_imgs_vertically_by_padding_to_same_width(imgs):
  max_img_width = max([img.shape[1] for img in imgs])
  padded_imgs = [pad_img_by_width(img, max_img_width) for img in imgs]
  return np.vstack(padded_imgs)


def right_strip_text(text):
  text = text.strip()
  for c in constants.CUT_OFF_CHARS:
    text = text.rstrip(c)
  return text


def rgb2hex(rgb):
  r, g, b = rgb
  return "#{:02x}{:02x}{:02x}".format(r, g, b).upper()


def rgb2grayscale(rgb):
  r,g,b = rgb
  return (0.3 * r) + (0.59 * g) + (0.11 * b)


def is_probably_latin(text, count_thresh=5):
  latin_count = 0
  unicode_count = 0
  for char in text:
    if char.isascii():
      latin_count += 1
    else:
      unicode_count += 1
    if latin_count > count_thresh:
      return True
    elif unicode_count > count_thresh:
      return False
  if latin_count > unicode_count:
    return True
  return False


def ascii_normalize(text):
  if not text.isascii() and is_probably_latin(text):
    text = unidecode_expect_nonascii(text)
  return text


def is_substring(text, full_str):
  if not text or not full_str:
    return False
  return text in full_str


def get_first_word(text):
  idx = -1
  for char in text:
    if char not in [' ', '\n']:
      idx += 1
    else:
      break
  return text[:idx + 1]


def remove_icon_chars(ocr_text, text):
  """
  Icons are detected based on these following assumptions:
    - A single character (except 'a') at the beginning and ending of sentence.
    - A 2-letter word at the beginning of sentence is uncommon in English.
    - A first character if it makes the first word at the beginning of sentence is not meaningful in English.
    - The first word at the beginning of sentence if it has 3 letters and is not meaningful in English.
  """
  if not ocr_text:
    return ''
  initial_length = len(ocr_text)
  first_word = get_first_word(ocr_text)

  first_word_pattern = r'^[^a]\s+'
  last_word_pattern = r'(^[^a]\s+)|(\s+.$)'
  if len(first_word) == 2 and is_probably_latin(ocr_text):
    first_word_pattern = r'(^(?!(of|to|in|it|is|be|as|at|so|we|he|by|or|on|do|if|me|my|up|an|go|no|us|am)).{2}\s+)'
  ocr_text = re.sub(first_word_pattern, '', ocr_text, flags=re.IGNORECASE)
  ocr_text = re.sub(last_word_pattern, '', ocr_text, flags=re.IGNORECASE)
  if len(ocr_text) < initial_length:
    return ocr_text

  if is_probably_latin(ocr_text):
    # Refines a work like `KHome`
    if not is_substring(first_word, text) and is_substring(first_word[1:], text):
      return ocr_text[1:]
    # Ignores a word like `oOo Content`
    elif len(first_word) == 3 and not is_substring(first_word, text):
      striped_first_word_text = ocr_text[3:].lstrip()
      if striped_first_word_text and is_substring(striped_first_word_text, text):
        return striped_first_word_text
      elif not striped_first_word_text:
        return striped_first_word_text
  return ocr_text


def get_ios_keyboard(tree):
  """
  Find the keyboard in ios screen if any.
  """
  ele_finder = etree.XPath('//XCUIElementTypeKeyboard')
  candidates = ele_finder(tree)
  return candidates[0] if len(candidates) else None


def get_html(byte):
  pattern = re.compile(r'\<html.*\>.*\<\/html\>'.encode('ascii'), re.DOTALL)
  html_bytes = pattern.findall(byte)
  if html_bytes:
    return html_bytes[0]
  return None


def parse_html_tree(html_str):
  return html.fromstring(html_str).getroottree()


def parse_xml_tree(xml_str):
  return etree.fromstring(xml_str).getroottree()


def get_webview_contain_html(tree, platform):
  ele_finder = etree.XPath('//html')
  candidates = ele_finder(tree)
  return candidates[0].getparent() if len(candidates) else None


def get_screen_resolution_in_xml(tree, platform, pixel_map=None):
  """
  Return the width x height pixels of the screen.
  """
  ele_finder = etree.XPath("//XCUIElementTypeApplication")
  elements = ele_finder(tree)
  element = elements[0] if elements else None

  if element is None:
    raise ValueError('Cannot parse width and height value in xml')

  left, top, right, bot = decode_bound(element, platform, pixel_map=pixel_map)
  return right - left, bot - top


def match_by_hungarian(feats_1, feats_2, threshold=0.19685, metric='l2'):
  """
  Use hungarian algorithm to match element.
  The algorithm assures that 2 different elements not able to pair with a same element.
  @param feats_1: List of features of elements
  @param feats_2: List of features of elements
  @param threshold: Remove pairs which have similarity is higher than the threshold
  @param metric: 'l2' or 'cosine'
  @return: List of pair matched index. Exp: [(id1, id2), ...]
  """
  feats_1 = np.vstack(feats_1)
  feats_2 = np.vstack(feats_2)
  cost_mat = pairwise_distances(feats_1, feats_2, metric=metric)
  ids_1, ids_2 = linear_sum_assignment(cost_mat)
  result = []
  for id1, id2 in zip(ids_1, ids_2):
    length_diff = cost_mat[id1, id2]
    if length_diff < threshold:
      result.append((id1, id2))
  return result


def calculate_padding(element_bound, parent_bound):
  return [abs(coord_1 - coord_2) for coord_1, coord_2 in zip(parent_bound, element_bound)]


def get_element_margin(element, platform, pixel_map):
  parent = element.getparent()
  if parent is None:
    return [0, 0, 0, 0], None
  element_bound = decode_bound(element, platform, pixel_map)
  parent_bound = decode_bound(parent, platform, pixel_map)
  if any(parent_bound) and parent_bound != element_bound:
    return [abs(coord_1 - coord_2) for coord_1, coord_2 in zip(parent_bound, element_bound)], parent
  else:
    return get_element_margin(parent, platform, pixel_map)


def index_of(value, ls):
  for idx, item in enumerate(ls):
    if value == item:
      return idx
  return -1

def intersect(bound_1, bound_2):
  intersect_left = max(bound_1[0], bound_2[0])
  intersect_top = max(bound_1[1], bound_2[1])
  intersect_right = min(bound_1[2], bound_2[2])
  intersect_bottom = min(bound_1[3], bound_2[3])
  if intersect_right < intersect_left or intersect_bottom < intersect_top:
    return 0.0

  intersect_area = get_area(intersect_left, intersect_top, intersect_right, intersect_bottom)
  bound_1_area = (bound_1[2]-bound_1[0]) * (bound_1[3]-bound_1[1])
  return intersect_area/bound_1_area


def get_iou(bound_1, bound_2):
  intersect_left = max(bound_1[0], bound_2[0])
  intersect_top = max(bound_1[1], bound_2[1])
  intersect_right = min(bound_1[2], bound_2[2])
  intersect_bottom = min(bound_1[3], bound_2[3])
  if intersect_right < intersect_left or intersect_bottom < intersect_top:
    return 0.0
  intersect_area = get_area(intersect_left, intersect_top, intersect_right, intersect_bottom)
  union_area = get_area(*bound_1) + get_area(*bound_2) - intersect_area
  return intersect_area / union_area


def is_overlap(bound_1, bound_2, threshold=0.1):
  return get_iou(bound_1, bound_2) > threshold


def is_overlap_within(bound_1, bounds, threshold=0.1):
  for bound_2 in bounds:
    if is_overlap(bound_1, bound_2, threshold=threshold):
      return True
  return False


def is_blank_img(img, bg_color):
  gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  height, width = img.shape[:2]
  area = width * height
  common = Counter(gray_img.reshape((-1))).most_common(1)[0][0]
  diff = np.abs(gray_img-common)
  mask = diff >= 1
  return np.sum(mask*1)/area <= 1e-3 and abs(int(common) - int(bg_color)) <= 10


def is_visual_box(element_img, image_shape, element_xpath, bg_color=255):
  if element_img is None:
    return False
  height, width = element_img.shape[:2]
  min_height = image_shape[0] * 0.015
  min_width = image_shape[1] * 0.015
  if width < min_width or height < min_height:
     return False
  class_name = classname_from_xpath(element_xpath).split('.')[-1]
  return not is_blank_img(element_img, bg_color) or class_name in constants.EDITABLE_ELEMENTS


def get_interval_name(number, dict, name):
  bins = dict['bins_' + name]
  labels = dict['labels_' + name]

  if number > bins[-1]:
    return labels[-1]
  if number < bins[0]:
    return labels[0]

  for i in range(len(bins)-1):
   lower = bins[i]
   upper = bins[i+1]
   if lower <= number <= upper:
     return labels[i]


def filter_text(texts, bounds, scores, threshold):
  idx = np.where(scores >= threshold)[0]
  return texts[idx], bounds[idx], scores[idx]


def get_common_bound(bounds):
  lefts = []
  tops = []
  rights = []
  bottoms = []
  for bb in bounds:
    lefts.append(bb[0])
    tops.append(bb[1])
    rights.append(bb[2])
    bottoms.append(bb[3])

  return [int(np.min(lefts)), int(np.min(tops)), int(np.max(rights)), int(np.max(bottoms))]


def remove_large_elements(session, elements):
  half_screen_area = (session.screen_img.shape[0] * session.screen_img.shape[1])/2
  for index in reversed(range(len(elements))):
    val = elements[index]
    if get_area(val.bound[0], val.bound[1], val.bound[2], val.bound[3]) >= half_screen_area:
      del elements[index]
  return elements


def is_point_inside_bound(point, bound, pad=0):
  if bound[0] + pad <= point[0] <= bound[2] - pad and bound[1] + pad <= point[1] <= bound[3] - pad:
    return True
  return False


def get_center_point_of_bound(bound):
  return (int((bound[0] + bound[2]) / 2), int((bound[1] + bound[3]) / 2))


def get_distance_two_points(point1, point2):
  return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def remove_overlap_elements(elements):
  areas = []
  for element in elements:
    areas.append(get_area(*element.bound))

  sorted_idx = np.argsort(areas)
  number_elements = len(sorted_idx)
  elements_groups = {}

  for i, index in enumerate(sorted_idx):
    child = elements[index]
    child_area = areas[index]
    if child.xpath not in elements_groups:
      is_child = False
      for j in reversed(range(i+1, number_elements)):
        parent = elements[sorted_idx[j]]
        parent_area = areas[sorted_idx[j]]
        if intersect(child.bound, parent.bound) > 0.99:
          if child_area == parent_area:
            is_child = True
          else:
            if parent.xpath not in elements_groups:
              elements_groups[parent.xpath] = {'element': parent, 'children': []}
            elements_groups[parent.xpath]['children'].append(child)
            is_child = True
          break
      if not is_child:
        elements_groups[child.xpath] = {'element': child, 'children': []}

  matching_candidates = []
  for parent_xpath, v in elements_groups.items():
    parent = v['element']
    matching_candidates.append(parent)
    children = v['children']
    parent_center_point = get_center_point_of_bound(parent.bound)
    if len(children) > 1 or (len(children) == 1 and not is_point_inside_bound(parent_center_point, children[0].bound, pad=constants.TOUCH_PADDING)):
      matching_candidates.extend(children)
  return matching_candidates


def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_bg_color_beside_element(element_bound, gray):
  default_bg = 255
  if element_bound is None or element_bound[2] - element_bound[0] == 0 or element_bound[3] - element_bound[1] == 0:
    return default_bg
  left, top, right, bottom = element_bound
  height, width = gray.shape[:2]
  pad = 5
  if left - pad > 0:
    return gray[top, left - pad]
  if top - pad > 0:
    return gray[top - pad, left]
  if right + pad < width:
    return gray[bottom, right + pad]
  if bottom + pad < height:
    return gray[bottom + pad, right]
  return default_bg

def get_texts_of_images_from_ocr_response(text, bound, imgs, padding):
  current_top = 0
  max_img_width = max([img.shape[1] for img in imgs])
  texts = []
  for img in imgs:
    img_height = img.shape[0]
    current_bot = current_top + img_height
    ele_bound = [0, current_top, max_img_width, current_bot]
    texts.append(extract_ocr_text_by_bound(ele_bound, gg_bound, gg_text))
    current_top = current_bot + padding
  return texts

def get_ocr_text_for_text_element(element_bound, google_bounds, google_texts):
    lines = []
    line = []
    anchor_bound = None
    for idx, (bound, text) in enumerate(zip(google_bounds, google_texts)):
      if get_area(bound[0], bound[1], bound[2], bound[3]) == 0:
        continue
      if intersect(bound, element_bound) >= 0.8:
        if len(line) == 0:
          line.append(text)
          anchor_bound = bound
        else:
          if anchor_bound is not None:
            if is_in_the_same_horizontal_line(bound, anchor_bound, bound_resize_ratio=0.1):
              line.append(text)
            else:
              line_text = ' '.join(line)
              lines.append(line_text.strip())
              line = [text]
              anchor_bound = bound

      if idx == len(google_bounds) - 1:
        line_text = ' '.join(line)
        lines.append(line_text.strip())
    return '\n'.join(lines)

def get_lines_in_keyboard(keyboard_elements):
  lines = []
  for element in keyboard_elements:
    bottom = element.bound[3]
    if bottom not in lines:
      lines.append(bottom)
  if lines:
    lines.sort(reverse=True)
  return lines

def get_elements_in_line_of_keyboard(line_index, keyboard_elements):
  lines = get_lines_in_keyboard(keyboard_elements)
  candidates = []
  for element in keyboard_elements:
    if element.bound[3] == lines[line_index]:
      candidates.append(element)
  return sorted(candidates, key=lambda x: x.bound[0])

def get_index_of_elements_in_line(element, line):
  order_left_bound = []
  for ele in line:
    if ele.bound[0] not in order_left_bound:
      order_left_bound.append(ele.bound[0])
  if order_left_bound:
    order_left_bound.sort()
    return order_left_bound.index(element.bound[0])
  return None


config = parse_arguments()
logger = LoggingWrapper(
  logstash_host=config.logstash_host,
  environment=config.deploy_env,
  namespace=constants.NAMESPACE % (os.getpid(),), )
kobiton_ocr_service_info = get_kobiton_ocr_service_info()