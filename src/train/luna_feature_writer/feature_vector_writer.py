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

import string_embed

from lxml import etree

import pytesseract
from pytesseract import Output

from PIL import Image
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

border = 30
ocr_buffer = 2

# Loads the module
module = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4")

min_text_eval_len = 2


#################################################
# This function:
# Loads the mobilenet model in TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_image_feature_vectors(img):
    try:
        # Resizes the image to 224 x 224 x 3 shape tensor
        img = tf.image.resize_with_pad(img, 224, 224)
        # Converts the data type of uint8 to float32 by adding a new axis
        # img becomes 1 x 224 x 224 x 3 tensor with data type of float32
        # This is required for the mobilenet model we are using

        img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

        # Calculate the image feature vector of the img
        features = module(img)

        # Remove single-dimensional entries from the 'features' array
        feature_set = np.squeeze(features)

        return feature_set

    except Exception as e:
        print("Exception", e)
        return None


#################################################
# This function:
# Takes an img, which represents a cropped area of mobile
# screen, extracts the text from the img using tesseract,
# uses the loaded universal-sentence-encoder model from TF.HUB
# to create a text embedding into a 512 x 1 tensor of the
# extracted text. It returns the text feature embeddings
# and the cropped image with the text area cut out.
#################################################
def get_text_feature_vector(img, shape):
    try:
        gry1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d = pytesseract.image_to_data(gry1, config='--psm 12',
                                      output_type=Output.DICT)

        text = ""
        n_boxes = len(d['level'])

        mask = np.zeros((img.shape[:2]), dtype=np.uint8)

        max_font_height = 0.0
        min_font_height = shape[0]
        text_width = 0.0
        for i in range(n_boxes):
            if len(d['text'][i]) > 0:
                (x, y, w, h) = (d['left'][i], d['top'][i],
                                d['width'][i], d['height'][i])

                if h > max_font_height:
                    max_font_height = h
                if h < min_font_height:
                    min_font_height = h
                if w > text_width:
                    text_width = w

                x1 = max(0, x-ocr_buffer)
                x2 = min(x+w+ocr_buffer, img.shape[1])
                y1 = max(0, y-ocr_buffer)
                y2 = min(y+h+ocr_buffer, img.shape[0])
                mask[y1:y2, x1:x2] = 1
                text = text + ' ' + d['text'][i]

        if text != ' ':
            text = text.strip()

        if max_font_height == 0.0:
            max_font_height = 0.0

        if min_font_height == shape[0] or min_font_height == 0.0:
            min_font_height = 0.0

        filled = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

        # Calculate the image feature vector of the img
        if text == '':
            text = ' '
        text_features = string_embed.embedding(text)

        # features = [x * string_factor for x in features]
        # text_features = np.squeeze(features)
        features = {
            'text':            text_features,
            'min_font_height': np.array([min_font_height/shape[1]]),
            'max_font_height': np.array([max_font_height/shape[1]]),
            'text_width':      np.array([text_width/shape[0]]),
            }

        return filled, features

    except Exception as e:
        print("Exception", e)
        return None, None


def non_zero_default(x):
    if x == 0:
        return 0.000001
    return x


def get_element_features(bounds, shape):
    w = non_zero_default((bounds[3]-bounds[1])/shape[0])
    h = non_zero_default((bounds[2]-bounds[0])/shape[1])
    x = non_zero_default(bounds[1]/shape[0])
    y = non_zero_default(bounds[0]/shape[1])

    element_features = {
        'element_width':  np.array([w]),
        'element_height': np.array([h]),
        'element_x':      np.array([x]),
        'element_y':      np.array([y])
        }

    return element_features


def process_image(im, root):
    references = []
    ios = False

    # the position in the xml is different for Android and iOS
    if 'width' not in root.attrib:
        if 'width' not in root[0].attrib:
            raise Exception("width attribute not in root xml")
        else:
            ios = True
            screen_width = int(root[0].attrib['width'])
    else:
        screen_width = int(root.attrib['width'])

    if 'height' not in root.attrib:
        if 'height' not in root[0].attrib:
            raise Exception("height attribute not in root xml")
        else:
            screen_height = int(root[0].attrib['height'])
    else:
        screen_height = int(root.attrib['height'])

    # the iOS images xml width and height are different
    # than the image width and height, so we have to scale it.
    scale_x = 1
    scale_y = 1
    if ios:
        scale_x = im.shape[1] / screen_width
        scale_y = im.shape[0] / screen_height
    shape = [screen_width, screen_height]

    if ios:
        it = root[0]
    else:
        it = root

    for node in it.iter():

        path = etree.ElementTree(node).getpath(node)

        if 0 != len(node):
            continue

        if 'width' in node.attrib:
            if node.attrib['width'] == '0':
                continue
            else:
                width = int(node.attrib['width'])

        if 'height' in node.attrib:
            if node.attrib['height'] == '0':
                continue
            else:
                height = int(node.attrib['height'])

        if ios:
            y = int(scale_y*min(max(0, int(node.attrib['y'])), width))
            x = int(scale_x*min(max(0, int(node.attrib['x'])), height))
            bounds = [x, y,
                      int(x+scale_x*int(width)),
                      int(y+scale_y*int(height))]
        else:
            bounds = node.attrib['bounds']
            bounds = bounds.replace("][", ",")\
                           .replace("[", "")\
                           .replace("]", "")\
                           .split(",")
            bounds = [int(i) for i in bounds]

        # Crop the image to the area
        cropped_bounds = im[bounds[1]:bounds[3], bounds[0]:bounds[2]]

        # get the text features and with image with the text areas
        # wiped out
        wiped_img, text_features = get_text_feature_vector(
            cropped_bounds, shape)

        # get the image feature vector from the wiped image
        image_feature_set = get_image_feature_vectors(wiped_img)

        if image_feature_set is None:
            print("empty feature set")
            continue

        image_features = {'image': image_feature_set}

        element_features = get_element_features(bounds, shape)

        # concatenate the individual feature vectors into a dict.
        feature_dict = {
            **text_features,
            **element_features,
            **image_features
            }

        entry = {}
        entry['xpath'] = path
        entry['bounds'] = bounds
        entry['features'] = feature_dict

        references.append(entry)

    return references


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


def process_image_xml_pair(d, output_file_name):
    xmlfiles = glob.glob(str(d + '/screen-xml-*.xml'))
    validate_len(xmlfiles, d)
    xmlfile = xmlfiles[0]

    jpeg_name = xmlfile.replace('-xml', '').replace('.xml', '.jpeg')

    imgfiles = glob.glob(jpeg_name)
    imgfile = imgfiles[0]

    tree = etree.parse(xmlfile)
    root = tree.getroot()
    img = Image.open(imgfile)

    im = Image.new("RGB", img.size, (255, 255, 255))
    x, y = img.size
    im.paste(img, (0, 0, x, y))

    img = np.array(im)

    elems = process_image(img, root)
    write_dict(elems, output_file_name)


def main(argv):
    print(argv)
    if len(argv) != 2:
        print("Usage: python match_screens.py <directory> <output file>")
    process_image_xml_pair(argv[0], argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])
