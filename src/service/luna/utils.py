import numpy as np
from service import constants, xml_utils, utils as u
from service.luna import feature_vector_writer as fvw


def generate_feature_vector_element(shape, elements_image, elements_metadata, elements_text):
  dependencies_50 = []
  dependencies_0 = []

  images_50 = []
  images_0 = []

  for i, element_img in enumerate(elements_image):
    element_metadata = elements_metadata[i]
    element_xpath = element_metadata[0]
    element_text = elements_text[i]
    bounds = element_metadata[1:]
    is_blank = element_metadata[-1]
    element_classname = u.classname_from_xpath(element_xpath)
    if '.' in element_classname:
      element_classname = element_classname.split('.')[-1]
    wiped_img_50, text_features_50, wiped_img_0, text_features_0 = fvw.get_text_feature_vector(element_img, shape, element_text)
    text_features_50['text'] = text_features_50['text'].astype('float64')
    text_50 = text_features_50.pop('ocr_text', '')
    text_color = text_features_50.pop('text_color', None)

    text_features_0['text'] = text_features_0['text'].astype('float64')
    images_50.append(wiped_img_50)
    images_0.append(element_img)
    element_features = fvw.get_element_features(bounds, shape)
    feature_dict_50 = {
      **text_features_50,
      **element_features
    }
    feature_dict_0 = {
      **text_features_0,
      **element_features
    }
    entry_50 = {}
    entry_50['xpath'] = element_metadata[0]
    entry_50['bounds'] = bounds
    entry_50['ocr_text'] = text_50
    entry_50['text_color'] = text_color
    entry_50['features'] = feature_dict_50

    entry_0 = {}
    entry_0['xpath'] = element_metadata[0]
    entry_0['bounds'] = bounds
    entry_0['features'] = feature_dict_0

    dependencies_50.append(entry_50)
    dependencies_0.append(entry_0)

  image_feature_sets_50 = fvw.get_image_feature_vectors_nopad(images_50)
  image_feature_sets_0 = fvw.get_image_feature_vectors_pad(images_0)

  for i in range(len(dependencies_0)):
    dependencies_50[i]['features']['image'] = image_feature_sets_50[i].numpy()
    dependencies_0[i]['features']['image'] = image_feature_sets_0[i].numpy()
  return dependencies_50, dependencies_0


def generate_feature_vector_assertion_element(shape, elements_image, elements_metadata):
  dependencies_50 = []
  dependencies_0 = []

  images_50 = []
  images_0 = []

  for i, element_img in enumerate(elements_image):
    element_metadata = elements_metadata[i]
    bounds = element_metadata[1:]
    wiped_img_50, text_features_50, wiped_img_0, text_features_0 = fvw.get_text_feature_vector_assertion(
      element_img, shape)
    text_features_50['text'] = text_features_50['text'].astype('float64')
    text_50 = text_features_50.pop('ocr_text', '')
    text_color = text_features_50.pop('text_color', None)

    text_features_0['text'] = text_features_0['text'].astype('float64')
    images_50.append(wiped_img_50)
    images_0.append(element_img)

    element_features = fvw.get_element_features(bounds, shape)
    feature_dict_50 = {
      **text_features_50,
      **element_features
    }
    feature_dict_0 = {
      **text_features_0,
      **element_features
    }
    entry_50 = {}
    entry_50['xpath'] = element_metadata[0]
    entry_50['bounds'] = bounds
    entry_50['ocr_text'] = text_50
    entry_50['text_color'] = text_color
    entry_50['features'] = feature_dict_50

    entry_0 = {}
    entry_0['xpath'] = element_metadata[0]
    entry_0['bounds'] = bounds
    entry_0['features'] = feature_dict_0

    dependencies_50.append(entry_50)
    dependencies_0.append(entry_0)

  image_feature_sets_50 = fvw.get_image_feature_vectors_nopad(images_50)
  image_feature_sets_0 = fvw.get_image_feature_vectors_pad(images_0)

  for i in range(len(dependencies_0)):
    dependencies_50[i]['features']['image'] = image_feature_sets_50[i].numpy()
    dependencies_0[i]['features']['image'] = image_feature_sets_0[i].numpy()
  return dependencies_50, dependencies_0


def find_nearest_neighbors_indexs(index, elements_features):
  """
  Find the nearest left, top, right, bottom of an element. Current method takes the center of the input
  element, then go straight forward to the 4 directions.
  @param index: the element to find neighbors
  @param elements_features: list of all element
  @return: index of nearest left, top, right, bottom for the given element.
  """
  my_bound = elements_features[index]['bounds']
  center = u.get_center_point_of_bound(my_bound)
  bounds = [x['bounds'] for x in elements_features]
  left_cand, top_cand, right_cand, bottom_cand = [], [], [], []
  for index, bound in enumerate(bounds):
    if center[0] >= bound[2] and bound[1] <= center[1] <= bound[3]:
      left_cand.append(index)
    if center[1] >= bound[3] and bound[0] <= center[0] <= bound[2]:
      top_cand.append(index)
    if center[0] <= bound[0] and bound[1] <= center[1] <= bound[3]:
      right_cand.append(index)
    if center[1] <= bound[1] and bound[0] <= center[0] <= bound[2]:
      bottom_cand.append(index)
  left, top, right, bottom = [-1] * 4

  if left_cand:
    left = left_cand[np.argmax(np.array([bounds[index][2] for index in left_cand]))]
  if top_cand:
    top = top_cand[np.argmax(np.array([bounds[index][3] for index in top_cand]))]
  if right_cand:
    right = right_cand[np.argmin(np.array([bounds[index][0] for index in right_cand]))]
  if bottom_cand:
    bottom = bottom_cand[np.argmin(np.array([bounds[index][1] for index in bottom_cand]))]

  return left, top, right, bottom


# TODO: Reduce the dimension of output later
def concat_neighbors_feature(index, original_features, neighbors_indexs):
  """
  Return a single element features which is concatenated by 4 neighbor element features. It will concatenate a list of
  zeros with the same dimension in case of no neighbor element.
  """
  concatenated_result = []
  concatenated_result += original_features[index]
  feature_size = len(original_features[index])

  for item in neighbors_indexs:
    if item == -1:
      # No neighbor element
      feature_to_concat = [0.] * feature_size
    else:
      feature_to_concat = original_features[item]
    concatenated_result += feature_to_concat
  return concatenated_result
