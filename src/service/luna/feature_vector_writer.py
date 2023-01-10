# match_screens.py
#################################################
# Imports and function definitions
#################################################
# For running inference on the TF-Hub module with Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

import cv2

import json

# For saving 'feature vectors' into a txt file
import numpy as np
import sys
import glob
from collections import Counter

from service.luna import string_embed as string_embed
from service import utils

from lxml import etree

import pytesseract
from pytesseract import Output

from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

border = 30
ocr_buffer = 2

if os.environ.get('KOBITON_AI_TF_MODEL'):
  print("Loading mobilenet from local...")
  module = hub.load(os.environ.get('KOBITON_AI_TF_MODEL'))
else:
  print("Loading mobilenet from internet...")
  module = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4")

min_text_eval_len = 2


#################################################
# This function:
# Loads the mobilenet model in TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_image_feature_vectors_pad(imgs):
    # Resizes the image to 224 x 224 x 3 shape tensor
    images_tf = []
    for img in imgs:
        img_tf = tf.image.resize_with_pad(img, 224, 224, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        images_tf.append(img_tf)
    # Converts the data type of uint8 to float32 by adding a new axis
    # img becomes batch x 224 x 224 x 3 tensor with data type of float32
    # This is required for the mobilenet model we are using

    images_tf = tf.image.convert_image_dtype(images_tf, tf.float32)

    # Calculate the image feature vector of the img
    features = module(images_tf)
    return features


def get_image_feature_vectors_nopad(imgs):
    # Resizes the image to 224 x 224 x 3 shape tensor
    images_tf = []
    for img in imgs:
        img_tf = tf.image.resize(img, (224, 224))
        images_tf.append(img_tf)
    # Converts the data type of uint8 to float32 by adding a new axis
    # img becomes batch x 224 x 224 x 3 tensor with data type of float32
    # This is required for the mobilenet model we are using

    images_tf = tf.image.convert_image_dtype(images_tf, tf.float32)

    # Calculate the image feature vector of the img
    features = module(images_tf)

    return features


#################################################
# This function cuts off the blank space of
# element image
#################################################
def preprocess_ocr(gray):
    h, w = gray.shape[:2]
    padding = 10

    # column
    blank = True
    c = w - 1
    while blank and c > 0:
        col = gray[:, c]
        common = Counter(col).most_common(1)[0][0]
        diff = np.abs(col - common)
        mask = diff >= 10
        if np.sum(mask * 1) <= 10:
            c -= 1
        else:
            blank = False
    gray1 = gray[:, :min(c + padding, w)]

    # row
    blank = True
    r = h - 1
    while blank and r > 0:
        row = gray1[r, :]
        common = Counter(row).most_common(1)[0][0]
        diff = np.abs(row - common)
        mask = diff >= 10
        if np.sum(mask * 1) <= 10:
            r -= 1
        else:
            blank = False
    gray1 = gray1[:min(r + padding, h), :]
    return gray1


def gray_scale_image(img):
    # Detect background and text color first
    background_color = utils.get_background_color(img)
    _, text_color = utils.get_text_color(img)
    background_color_in_grayscale = utils.rgb2grayscale(background_color)
    text_color_in_grayscale = 100
    gap = 30
    if text_color is not None:
      text_color_in_grayscale = utils.rgb2grayscale(text_color)
    if background_color_in_grayscale <= text_color_in_grayscale:
        # This case is white text and black background
        gry1 = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        color_threshold = int(255 - background_color_in_grayscale) - gap
    else:
        # This case is black text and white background
        gry1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        color_threshold = int(background_color_in_grayscale) - gap

    if color_threshold >= 255 - gap:
        color_threshold = 127
    _, thresh = cv2.threshold(gry1, color_threshold, 255, cv2.THRESH_BINARY)
    return thresh, gry1

def image_to_data_light(img):
  thresh1, gry1 = gray_scale_image(img)
  d = pytesseract.image_to_data(thresh1, config='--psm 12', output_type=Output.DICT)
  return d, True

def image_to_data(img):
    is_wiped = True
    threshold = 60
    min_size_image = 20
    thresh1, gry1 = gray_scale_image(img)
    d = pytesseract.image_to_data(thresh1, config='--psm 12', output_type=Output.DICT)
    idx = np.where(np.array(d['conf']).astype(float) >= threshold)[0]
    if len(idx) == 0:
        # Cut blank col/row
        gry2 = preprocess_ocr(gry1)
        # Case color background, no text
        if gry2.shape[0] == min_size_image or gry2.shape[0] == min_size_image:
            return d, is_wiped
        if gry2.shape != gry1.shape:
          thresh2, gry2 = gray_scale_image(img[:gry2.shape[0], :gry2.shape[1]])
          d = pytesseract.image_to_data(thresh2, config='--psm 12', output_type=Output.DICT)
          idx = np.where(np.array(d['conf']).astype(float) >= threshold)[0]
        if len(idx) == 0:
            # Resize original image
            default_dimension = 160
            h, w = gry2.shape[:2]
            is_wiped = False
            if h<w:
                ratio = w*1./h
                gry3 = cv2.resize(gry1, (int(default_dimension*ratio), default_dimension))
            else:
                ratio = h*1./w
                gry3 = cv2.resize(gry1, (default_dimension, int(default_dimension*ratio)))
            d = pytesseract.image_to_data(gry3, config='--psm 12', output_type=Output.DICT)
            idx = np.where(np.array(d['conf']).astype(float) >= threshold)[0]
            if len(idx) == 0:
                # Assume a single uniform block of text
                d = pytesseract.image_to_data(gry3, config='--psm 6', output_type=Output.DICT)
    return d, is_wiped


def get_min_max_height(h, min_height, max_height):
    if h > max_height:
        max_height = h
    if h < min_height:
        min_height = h
    return [min_height, max_height]


def get_best_text(d):
    thresholds = [60]
    text = ''
    for threshold in thresholds:
        text = ' '.join(np.array(d['text'])[np.where(np.array(d['conf']).astype(float) >= threshold)[0]])
        if text != '':
            return text
    return text


#################################################
# This function:
# Takes an img, which represents a cropped area of mobile
# screen, extracts the text from the img using tesseract,
# uses the loaded universal-sentence-encoder model from TF.HUB
# to create a text embedding into a 512 x 1 tensor of the
# extracted text. It returns the text feature embeddings
# and the cropped image with the text area cut out.
# For 2 thresholds: 0, 50
#################################################
def get_text_feature_vector_assertion(img, shape, get_color=True, lighter=False):
    if lighter:
      d, is_wiped = image_to_data_light(img)
    else:
      d, is_wiped = image_to_data(img)

    text_50 = ""
    text_0 = ""
    n_boxes = len(d['level'])

    mask_50 = np.zeros((img.shape[:2]), dtype=np.uint8)
    mask_0 = np.zeros((img.shape[:2]), dtype=np.uint8)

    font_height_50 = [0.0, img.shape[0]]
    text_width_50 = 0.0

    font_height_0 = [0.0, img.shape[0]]
    text_width_0 = 0.0

    text_color = None
    best_ocr_conf = 0

    for i in range(n_boxes):
        if float(d['conf'][i]) >= 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            x1 = max(0, x-ocr_buffer)
            x2 = min(x+w+ocr_buffer, img.shape[1])
            y1 = max(0, y-ocr_buffer)
            y2 = min(y+h+ocr_buffer, img.shape[0])

            font_height_0 = get_min_max_height(h, *font_height_0)
            if w > text_width_0:
                text_width_0 = w
            if is_wiped:
                mask_0[y1:y2, x1:x2] = 1
            text_0 = text_0 + ' ' + d['text'][i]

            if float(d['conf'][i]) >= 50:
                font_height_50 = get_min_max_height(h, *font_height_50)
                if w > text_width_50:
                    text_width_50 = w
                text_50 = text_50 + ' ' + d['text'][i]
                if is_wiped:
                    mask_50[y1:y2, x1:x2] = 1

                    # Get the right text color
                    # Sometime tesseract returns wrong text position
                    if get_color and (y1 > 0 or x1 > 0):
                        text_img = img[y1:y2, x1:x2]
                        color = utils.get_text_color(text_img)
                        if text_color is None or float(d['conf'][i]) > best_ocr_conf:
                            text_color = color
                            best_ocr_conf = float(d['conf'][i])

    for i in range(2):
        if font_height_50[i] >= img.shape[0]:
            font_height_50[i] = 0.0
        if font_height_0[i] >= img.shape[0]:
            font_height_0[i] = 0.0
    if is_wiped:
        filled_50 = cv2.inpaint(img, mask_50, 3, cv2.INPAINT_NS)
        filled_0 = cv2.inpaint(img, mask_0, 3, cv2.INPAINT_NS)
    else:
        filled_50 = img
        filled_0 = img

    # Calculate the image feature vector of the img
    text_features_50 = string_embed.embedding(text_50)
    text_features_0 = string_embed.embedding(text_0)

    features_50 = {
        'text':            text_features_50,
        'min_font_height': np.array([font_height_50[0]/shape[1]]),
        'max_font_height': np.array([font_height_50[1]/shape[1]]),
        'text_width':      np.array([text_width_50/shape[0]]),
        'ocr_text': get_best_text(d),
        'text_color': text_color,
        }
    features_0 = {
        'text':            text_features_0,
        'min_font_height': np.array([font_height_0[0]/shape[1]]),
        'max_font_height': np.array([font_height_0[1]/shape[1]]),
        'text_width':      np.array([text_width_0/shape[0]]),
        }

    return filled_50, features_50 , filled_0, features_0


def get_text_feature_vector(img, shape, texts):
    text_50 = texts[2]
    text_0 = texts[0]

    mask_50 = np.zeros((img.shape[:2]), dtype=np.uint8)
    mask_0 = np.zeros((img.shape[:2]), dtype=np.uint8)

    font_height_50 = [0.0, img.shape[0]]
    text_width_50 = 0.0

    font_height_0 = [0.0, img.shape[0]]
    text_width_0 = 0.0

    for bound_0 in texts[1]:
        x1 = max(0, bound_0[0]-ocr_buffer)
        x2 = min(bound_0[2]+ocr_buffer, img.shape[1])
        y1 = max(0, bound_0[1]-ocr_buffer)
        y2 = min(bound_0[3]+ocr_buffer, img.shape[0])
        h = y2-y1
        w = x2-x1

        font_height_0 = get_min_max_height(h, *font_height_0)
        if w > text_width_0:
            text_width_0 = w
        mask_0[y1:y2, x1:x2] = 1

    for bound_50 in texts[3]:
        x1 = max(0, bound_50[0]-ocr_buffer)
        x2 = min(bound_50[2]+ocr_buffer, img.shape[1])
        y1 = max(0, bound_50[1]-ocr_buffer)
        y2 = min(bound_50[3]+ocr_buffer, img.shape[0])
        h = y2-y1
        w = x2-x1

        font_height_50 = get_min_max_height(h, *font_height_50)
        if w > text_width_50:
            text_width_50 = w
        mask_50[y1:y2, x1:x2] = 1

    for i in range(2):
        if font_height_50[i] >= img.shape[0]:
            font_height_50[i] = 0.0
        if font_height_0[i] >= img.shape[0]:
            font_height_0[i] = 0.0

    filled_50 = cv2.inpaint(img, mask_50, 3, cv2.INPAINT_NS)
    filled_0 = cv2.inpaint(img, mask_0, 3, cv2.INPAINT_NS)

    # Calculate the image feature vector of the img
    text_features_50 = string_embed.embedding(text_50)
    text_features_0 = string_embed.embedding(text_0)

    features_50 = {
        'text':            text_features_50,
        'min_font_height': np.array([font_height_50[0]/shape[1]]),
        'max_font_height': np.array([font_height_50[1]/shape[1]]),
        'text_width':      np.array([text_width_50/shape[0]]),
        'ocr_text': text_50,
        }
    features_0 = {
        'text':            text_features_0,
        'min_font_height': np.array([font_height_0[0]/shape[1]]),
        'max_font_height': np.array([font_height_0[1]/shape[1]]),
        'text_width':      np.array([text_width_0/shape[0]]),
        'ocr_text': text_0,
        }

    return filled_50, features_50 , filled_0, features_0


def non_zero_default(x):
    if x == 0:
        return 0.000001
    return x


def get_element_features(bounds, shape):
    w = non_zero_default((bounds[2]-bounds[0])/shape[0])
    h = non_zero_default((bounds[3]-bounds[1])/shape[1])
    x = non_zero_default(bounds[0]/shape[0])
    y = non_zero_default(bounds[1]/shape[1])

    element_features = {
        'element_width':  np.array([w]),
        'element_height': np.array([h]),
        'element_x':      np.array([x]),
        'element_y':      np.array([y])
        }

    return element_features


def err_handler(type, flag):
    print("Floating point error (%s), with flag %s" % (type, flag))
    return 1


np.seterrcall(err_handler)
np.seterr(all='call')


def validate_len(v, d):
    if len(v) == 0:
        raise Exception('No file found for dir: ' + d)
    if len(v) > 2:
        raise Exception('Multiple files found for dir: ' + d)


def write_dict(elems, output_file_name):
    for elem in elems:
        for k, v in elem['features'].items():
            print(v, type(v))
            elem['features'][k] = v.tolist()

    with open(output_file_name, 'w') as fp:
        json.dump(elems, fp)