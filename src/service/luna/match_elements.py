import pickle
import json
import scipy
from scipy.spatial import distance
import numpy as np
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from service.luna import utils
from service import constants, xml_utils, utils as u
from service.element import Element

weights2_50 = pickle.load(open('service/luna/weights/weight_2_50.pkl', 'rb'))
weights2_full = pickle.load(open('service/luna/weights/weight_2_full.pkl', 'rb'))
weights1_50 = pickle.load(open('service/luna/weights/weight_1_50.pkl', 'rb'))
weights1_full = pickle.load(open('service/luna/weights/weight_1_full.pkl', 'rb'))
cumulative_indices = pickle.load(open('service/luna/weights/idx.pkl', 'rb'))

MATCHING_THRESHOLD = json.load(open('service/luna/thresholds.json'))

feature_names = ['image', 'text', 'min_font_height', 'max_font_height', 'text_width',
                 'element_width', 'element_height', 'element_x', 'element_y']


def get_elements(session, element_type, include_webview_element=False):
  if element_type == constants.ElementType.TEXTUAL:
    elements = session.get_textual_elements()
  elif element_type == constants.ElementType.SCROLLABLE:
    elements = session.get_scrollable_elements()
  elif element_type == constants.ElementType.KEYBOARD:
    elements = session.get_keyboard_elements()
  else:
    elements = session.get_visual_elements(include_webview_element)
  return elements


def get_element_in_screen(session, elements):
  elements_img = []
  elements_metadata = []
  elements_text = []

  texts_0 = np.asarray(session.ocr_texts)
  text_bounds_0 = np.asarray(session.bounds)
  text_scores = np.asarray(session.ocr_scores)
  valid_idx = np.where((text_scores) >= 0.5)[0]

  texts_50 = texts_0[valid_idx]
  text_bounds_50 = text_bounds_0[valid_idx]

  for element in elements:
    elements_img.append(element.img)
    elements_metadata.append((element.xpath, *element.bound, element.is_blank()))
    text_0, bounds_0 = u.extract_ocr_text_by_bound_iou(element.bound, text_bounds_0, texts_0)
    text_50, bounds_50 = u.extract_ocr_text_by_bound_iou(element.bound, text_bounds_50, texts_50)
    elements_text.append((text_0, bounds_0, text_50, bounds_50))

  shape = [session.screen_img.shape[1], session.screen_img.shape[0]]
  if len(elements_img) > 0:
    return utils.generate_feature_vector_element(shape, elements_img, elements_metadata, elements_text)
  return [], []


def get_element_in_assertion_screen(session, elements):
  elements_img = []
  elements_metadata = []
  elements_text = []

  for i, element in enumerate(elements):
    elements_img.append(element.img)
    elements_metadata.append((element.xpath, *element.bound))

  shape = [session.screen_img.shape[1], session.screen_img.shape[0]]
  if len(elements_img) > 0:
    return utils.generate_feature_vector_assertion_element(shape, elements_img, elements_metadata)
  return [], []


def apply_weights(f, weights1, weights2):
  ref = []
  for i, w in enumerate(weights1):
    begin = cumulative_indices[i]
    end = cumulative_indices[i + 1]
    feature_name = feature_names[i]
    w2 = weights2[begin:end]
    tmp = np.multiply(f[feature_name], w2).tolist()
    ref.extend(tmp * w)
  return ref


def create_cost_matrix(left_ref, right_ref):
  cost_matrix = 1 - distance.cdist(left_ref, right_ref, 'cosine')
  cost_matrix = np.around(cost_matrix, decimals=5)
  return cost_matrix


def match_items(cost_matrix):
  return scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)


def get_diff(ratio):
  gaps = [max(np.median(x) - np.min(x), np.max(x) - np.median(x)) for x in ratio]
  return gaps


def get_markers_in_group(matrix, left_features, right_features, group_elements, markers,
                         markers_left_p, markers_right_p, element_type_name):
  # calculate marker threshold for each group
  max_similarity = np.max(matrix[[e[1] for e in group_elements], :])
  threshold_upper = max_similarity - constants.MARKER_PADDING
  threshold_lower_bound = max(threshold_upper - constants.MARKER_GAP,
                              MATCHING_THRESHOLD[element_type_name]['marker_lower'])
  threshold = (threshold_upper + threshold_lower_bound) / 2
  ambiguous_gap_threshold = MATCHING_THRESHOLD[element_type_name]['ambiguous_gap_threshold']
  marker_threshold_step = MATCHING_THRESHOLD[element_type_name]['marker_threshold_step']
  count = 0
  previous_count = 0

  right_candidates = [x for x in list(range(len(right_features))) if x not in markers[1]]
  sims = [np.max(matrix[e[1]]) for e in group_elements]
  idx = list(np.argsort(sims))
  while len(idx) > 0 and len(right_candidates) > 0 \
      and (previous_count < count or threshold >= threshold_lower_bound) \
      and count <= constants.NUMBER_MARKERS_NEED:
    previous_count = count
    is_ambiguous = []
    for i in reversed(idx):
      e = group_elements[i]
      left_index = e[1]

      similarities = matrix[left_index, right_candidates]
      matched_sim = np.max(similarities)
      right_index = right_candidates[np.argmax(similarities)]
      ambiguous = np.where(np.abs(similarities - matched_sim) <= ambiguous_gap_threshold)[0]
      is_ambiguous.append(len(ambiguous) > 1)
      if matched_sim > threshold and len(ambiguous) <= 1:
        left_cands = [group_elements[j][1] for j in idx]
        left_sims = matrix[left_cands, right_index]
        left_ambiguous = np.where(np.abs(left_sims - matched_sim) <= ambiguous_gap_threshold)[0]
        if len(left_ambiguous) == 1:
          bounds1 = left_features[left_index]['bounds']
          center1 = u.get_center_point_of_bound(bounds1)
          bounds2 = right_features[right_index]['bounds']
          center2 = u.get_center_point_of_bound(bounds2)
          markers[0].append(left_index)
          markers[1].append(right_index)
          markers_left_p.append(center1)
          markers_right_p.append(center2)
          count += 1
          idx.remove(i)
          # remove already matched right elements
          right_candidates.remove(right_index)
          if len(right_candidates) == 0:
            break
    # if there is no marker then decrease threshold
    if previous_count == count:
      if all(is_ambiguous) and ambiguous_gap_threshold > constants.MIN_AMBIGUOUS_THRESHOLD:
        if ambiguous_gap_threshold > 0.1:
          ambiguous_gap_threshold -= 0.05
        else:
          ambiguous_gap_threshold = constants.MIN_AMBIGUOUS_THRESHOLD
      else:
        threshold -= marker_threshold_step


def get_markers(matrix, left_features, right_features, left_xml_tree, right_xml_tree, element_type_name, left_xpath):
  markers_left_p = []
  markers_right_p = []
  markers = [[], []]
  groups_not_scrollable_elements = []
  priority_not_scrollable = True

  # left xpath, index
  left_elements = []
  for i, e in enumerate(left_features):
    if e['xpath'] == left_xpath:
      left_elements.insert(0, [e['xpath'], i])
    else:
      left_elements.append([e['xpath'], i])

  while len(left_elements) > 0:
    element = left_elements.pop(0)
    xpath = element[0]
    scrollable_parent_xpath = xml_utils.get_parent_group(xpath, left_xml_tree)

    if scrollable_parent_xpath is None:
      if element[0] == left_xpath:
        priority_not_scrollable = False
      groups_not_scrollable_elements.append(element)

    if not priority_not_scrollable and len(groups_not_scrollable_elements) > 0:
      get_markers_in_group(matrix, left_features, right_features, groups_not_scrollable_elements, markers,
                           markers_left_p, markers_right_p, element_type_name)
      groups_not_scrollable_elements.pop(-1)
      priority_not_scrollable = True

    if scrollable_parent_xpath:
      list_remain_xpaths = [e[0] for e in left_elements]
      idx = xml_utils.find_candidate_in_scroll(scrollable_parent_xpath, list_remain_xpaths, left_xml_tree)
      groups_scrollable_elements = [element]
      for i in reversed(idx):
        e = left_elements.pop(i)
        groups_scrollable_elements.append(e)
      if len(groups_scrollable_elements) > 0:
        get_markers_in_group(matrix, left_features, right_features, groups_scrollable_elements, markers,
                             markers_left_p, markers_right_p,
                             element_type_name)

  if priority_not_scrollable and len(groups_not_scrollable_elements) > 0:
    get_markers_in_group(matrix, left_features, right_features, groups_not_scrollable_elements, markers,
                         markers_left_p, markers_right_p, element_type_name)

  return markers, markers_left_p, markers_right_p


def second_pass(matrix, left_features, right_features,
                markers, markers_left_p, markers_right_p, left_xml_tree, right_xml_tree, element_type_name,
                candidate=None):
  markers_out_scroll = \
    np.where([xml_utils.get_parent_group(left_features[i]['xpath'], left_xml_tree) is None for i in markers[0]])[0]
  markers_xpath = [left_features[i]['xpath'] for i in markers[0]]
  markers_xpath_right = [right_features[i]['xpath'] for i in markers[1]]
  second_pass_conditions = copy.deepcopy(MATCHING_THRESHOLD[element_type_name]["second_pass"])
  half_number_possible_pairs = int(min(len(left_features), len(right_features)) / 2)
  for i in range(len(left_features)):
    if i not in markers[0]:
      bounds1 = left_features[i]['bounds']
      center1 = u.get_center_point_of_bound(bounds1)
      ambiguous_right_p = []
      ambiguous_left_p = [center1]
      not_matched_right_indices = np.array(list(set(np.arange(len(right_features))) - set(markers[1])))

      if not_matched_right_indices.shape[0] > 0:
        for idx in not_matched_right_indices:
          bounds2 = right_features[idx]['bounds']
          center2 = u.get_center_point_of_bound(bounds2)
          ambiguous_right_p.append(center2)

        marker_amb_l_dist = euclidean_distances(ambiguous_left_p, markers_left_p)
        marker_amb_r_dist = euclidean_distances(ambiguous_right_p, markers_right_p)

        marker_amb_l_cosine_dist = cosine_distances(ambiguous_left_p, markers_left_p)
        marker_amb_r_cosine_dist = cosine_distances(ambiguous_right_p, markers_right_p)

        scrollable_parent_xpath = xml_utils.get_parent_group(left_features[i]['xpath'], left_xml_tree)
        number_markers = len(markers[0])
        left_distance = marker_amb_l_dist[0, :]
        left_cosine_distance = marker_amb_l_cosine_dist[0, :]
        rights_distances = marker_amb_r_dist[:, :]
        rights_cosine_distances = marker_amb_r_cosine_dist[:, :]
        right_indices = np.array(list(np.arange(len(not_matched_right_indices))))
        ambiguous_in_scroll = []
        ambiguous_out_scroll = []
        if scrollable_parent_xpath:
          markers_in_scroll = xml_utils.find_candidate_in_scroll(scrollable_parent_xpath, markers_xpath,
                                                                 left_xml_tree)

          if len(markers_in_scroll) > 0:
            # get parent scrollable xpath in revisit
            scrollable_parent_xpath_right = list(
              set([xml_utils.get_parent_group(markers_xpath_right[i], right_xml_tree) for i in
                   markers_in_scroll]))
            not_matched_right_xpaths = [right_features[j]['xpath'] for j in not_matched_right_indices]
            if len(scrollable_parent_xpath_right) == 1 and scrollable_parent_xpath_right[0] is not None:
              markers_in_scroll_right = xml_utils.find_candidate_in_scroll(
                scrollable_parent_xpath_right[0], markers_xpath_right, right_xml_tree)
              if len(markers_in_scroll_right) > len(markers_in_scroll):
                markers_in_scroll = markers_in_scroll_right
              ambiguous_in_scroll = xml_utils.find_candidate_in_scroll(scrollable_parent_xpath_right[0],
                                                                       not_matched_right_xpaths,
                                                                       right_xml_tree)
            else:
              ambiguous_in_scroll = np.where(
                [xml_utils.get_parent_group(xpath, right_xml_tree) is not None for xpath in
                 not_matched_right_xpaths])[0]
            number_markers = len(markers_in_scroll)
            left_distance = marker_amb_l_dist[0, markers_in_scroll]
            left_cosine_distance = marker_amb_l_cosine_dist[0, markers_in_scroll]
            if len(ambiguous_in_scroll) > 0:
              rights_distances = marker_amb_r_dist[ambiguous_in_scroll, :][:,
                                 markers_in_scroll]
              rights_cosine_distances = marker_amb_r_cosine_dist[ambiguous_in_scroll, :][:,
                                        markers_in_scroll]
            else:
              rights_distances = marker_amb_r_dist[:, markers_in_scroll]
              rights_cosine_distances = marker_amb_r_cosine_dist[:, markers_in_scroll]

        elif len(markers_out_scroll) > 0:
          number_markers = len(markers_out_scroll)
          left_distance = marker_amb_l_dist[0, markers_out_scroll]
          left_cosine_distance = marker_amb_l_cosine_dist[0, markers_out_scroll]
          ambiguous_out_scroll = np.where(
            [xml_utils.get_parent_group(right_features[j]['xpath'], right_xml_tree) is None for j in
             not_matched_right_indices])[0]
          rights_distances = marker_amb_r_dist[ambiguous_out_scroll, :][:, markers_out_scroll]
          rights_cosine_distances = marker_amb_r_cosine_dist[ambiguous_out_scroll, :][:, markers_out_scroll]

        cand_no_ambiguous = -1
        sims = matrix[i, not_matched_right_indices]
        matched_sim = np.max(sims)
        diff = np.abs(sims - matched_sim)
        candidates_right_index = not_matched_right_indices[
          np.where(diff <= MATCHING_THRESHOLD[element_type_name]['second_pass_ambiguous'])[0]]
        if len(candidates_right_index) == 1 and matched_sim > MATCHING_THRESHOLD[element_type_name][
          'no_ambiguous_threshold']:
          right_cand_index = not_matched_right_indices[np.argmax(sims)]
          # check case if there are ambiguous on left screen
          not_match_left_indices = np.array(list(set(np.arange(len(left_features))) - set(markers[0])))
          left_sims = matrix[not_match_left_indices, right_cand_index]
          left_diff = np.abs(left_sims - matched_sim)
          candidates_left_index = not_match_left_indices[
            np.where(left_diff <= MATCHING_THRESHOLD[element_type_name]['second_pass_ambiguous_left'])[0]]
          if len(candidates_left_index) == 1:
            cand_no_ambiguous = right_cand_index

        cand_second_pass = -1
        if rights_distances is not None and len(rights_distances) > 0:
          zeros_idx = np.where(left_distance == 0)[0]
          if len(zeros_idx) > 0:
            ratio_diff = rights_distances[:, zeros_idx[0]]
          else:
            distance_ratios = rights_distances / left_distance
            cosine_ratios = rights_cosine_distances / left_cosine_distance
            distance_ratios_diff = get_diff(distance_ratios)
            cosine_ratio_diff = get_diff(cosine_ratios)
            ratio_diff = (np.array(distance_ratios_diff) + np.array(cosine_ratio_diff)) / 2
          if element_type_name != constants.ElementType.SCROLLABLE.name:
            min_gap_right_indices = [i for i in range(len(ratio_diff))]
            max_sim_second_pass = 0.
            gap_second_pass = 99  # Why 99? It's just a lucky number.
            min_gap_second_pass = [-1, -1]

            if scrollable_parent_xpath:
              if len(ambiguous_in_scroll) > 0:
                right_indices = ambiguous_in_scroll[min_gap_right_indices]
            else:
              if len(ambiguous_out_scroll) > 0:
                right_indices = ambiguous_out_scroll[min_gap_right_indices]
            similarities = matrix[i, not_matched_right_indices[right_indices]]
            for j, right_index in enumerate(right_indices):
              min_gap = ratio_diff[j]
              similarity = similarities[j]

              if min_gap_second_pass[0] == -1 or min_gap_second_pass[1] > min_gap:
                min_gap_second_pass = [not_matched_right_indices[right_index], min_gap]

              if similarity >= MATCHING_THRESHOLD[element_type_name]['no_ambiguous_threshold']:
                if (similarity - max_sim_second_pass) >= constants.MIN_AMBIGUOUS_THRESHOLD:
                  # If element have more than 10% percent (after recalculated) of similarity, then use it
                  max_sim_second_pass = similarity
                  gap_second_pass = min_gap
                  cand_second_pass = not_matched_right_indices[right_index]
                else:
                  # Select element with smaller "skeleton" shape difference in case similarity not dominant
                  if gap_second_pass > min_gap and similarity > max_sim_second_pass:
                    max_sim_second_pass = similarity
                    gap_second_pass = min_gap
                    cand_second_pass = not_matched_right_indices[right_index]

            if candidate and i == candidate[0] and number_markers > constants.NUMBER_MARKERS_NEED and \
                -1 < min_gap_second_pass[1] < constants.DEFAULT_GAP_THRESHOLD:
              candidate.append(min_gap_second_pass[0])
          else:
            min_gap = np.min(ratio_diff)
            min_gap_right_indices = [i for i, gap in enumerate(ratio_diff) if abs(gap - min_gap) < 0.001]
            if scrollable_parent_xpath:
              right_indices = ambiguous_in_scroll[min_gap_right_indices]
            else:
              right_indices = ambiguous_out_scroll[min_gap_right_indices]

            similarities = matrix[i, not_matched_right_indices[right_indices]]
            similarity = np.max(similarities)
            max_sim_idx = np.argmax(similarities)
            right_index = right_indices[max_sim_idx]
            min_gap = ratio_diff[min_gap_right_indices[max_sim_idx]]

            for cond_pair in second_pass_conditions:
              if min_gap <= cond_pair[0] and similarity >= cond_pair[1]:
                cand_second_pass = not_matched_right_indices[right_index]
                break

        final_candidate = -1
        if cand_second_pass > -1 and cand_no_ambiguous > -1:
          if number_markers > constants.NUMBER_MARKERS_NEED:
            final_candidate = cand_second_pass
          else:
            final_candidate = cand_no_ambiguous
        elif cand_second_pass > -1 and number_markers >= min(constants.NUMBER_MARKERS_NEED,
                                                             half_number_possible_pairs):
          final_candidate = cand_second_pass
        elif cand_no_ambiguous > -1:
          final_candidate = cand_no_ambiguous
        if final_candidate > -1:
          markers[0].append(i)
          markers[1].append(final_candidate)
  return markers


def create_final_cost_matrix(left_features, right_features):
  ref_left_50 = []
  for feature in left_features[0]:
    ref_left_50.append(apply_weights(feature['features'], weights1_50, weights2_50))

  ref_left_0 = []
  for feature in left_features[1]:
    ref_left_0.append(apply_weights(feature['features'], weights1_full, weights2_full))

  ref_right_50 = []
  for feature in right_features[0]:
    ref_right_50.append(apply_weights(feature['features'], weights1_50, weights2_50))

  ref_right_0 = []
  for feature in right_features[1]:
    ref_right_0.append(apply_weights(feature['features'], weights1_full, weights2_full))

  if len(ref_left_50) > 0 and len(ref_right_50) > 0:
    cost_matrix_50 = create_cost_matrix(ref_left_50, ref_right_50)
    cost_matrix_0 = create_cost_matrix(ref_left_0, ref_right_0)
    cost_matrix = cost_matrix_50 * 0.5 + cost_matrix_0 * 0.5
    return cost_matrix

  return None


def recalculate_similarity(left_features, right_features, mode='neighbor', anchor_size=0.3):
  """
  This function recalculate cosine similarity between elements by combined the feature information from elements around
  @param left_features: all manual features
  @param right_features: all revisit features
  @param mode: define mode to choose element to combine
          neighbor: closest left, right, top, bottom elements
          anchor: left, right, top, bottom elements in an anchor following a specific size
  @param anchor_size: use for anchor mode only
  @return: recalculated cosine similarity matrix
  """
  # Load weight to create features
  ref_left_50 = []
  for feature in left_features[0]:
    ref_left_50.append(apply_weights(feature['features'], weights1_50, weights2_50))

  ref_left_0 = []
  for feature in left_features[1]:
    ref_left_0.append(apply_weights(feature['features'], weights1_full, weights2_full))

  ref_right_50 = []
  for feature in right_features[0]:
    ref_right_50.append(apply_weights(feature['features'], weights1_50, weights2_50))

  ref_right_0 = []
  for feature in right_features[1]:
    ref_right_0.append(apply_weights(feature['features'], weights1_full, weights2_full))

  # "enhance" means feature concatenated with neighbor features
  enhance_ref_left_50 = []
  enhance_ref_left_0 = []
  enhance_ref_right_50 = []
  enhance_ref_right_0 = []

  if mode == 'neighbor':
    for index, element in enumerate(left_features[0]):
      element_neighbors = utils.find_nearest_neighbors_indexs(index, left_features[0])  # left, top, right, bottom
      enhance_ref_left_50.append(utils.concat_neighbors_feature(index, ref_left_50, element_neighbors))
      enhance_ref_left_0.append(utils.concat_neighbors_feature(index, ref_left_0, element_neighbors))

    for index, element in enumerate(right_features[0]):
      element_neighbors = utils.find_nearest_neighbors_indexs(index, right_features[0])  # left, top, right, bottom
      enhance_ref_right_50.append(utils.concat_neighbors_feature(index, ref_right_50, element_neighbors))
      enhance_ref_right_0.append(utils.concat_neighbors_feature(index, ref_right_0, element_neighbors))

  # TODO: implement anchor mode (find neighbor elements by using a specific size of anchor)
  if len(enhance_ref_left_50) > 0 and len(enhance_ref_right_50) > 0:
    cost_matrix_50 = create_cost_matrix(enhance_ref_left_50, enhance_ref_right_50)
    cost_matrix_0 = create_cost_matrix(enhance_ref_left_0, enhance_ref_right_0)
    cost_matrix = cost_matrix_50 * 0.5 + cost_matrix_0 * 0.5
    return cost_matrix

  return None


def find_best_matches(left_features, right_features, left_xml_tree, right_xml_tree,
                      element_type_name, candidate=None, left_xpath=None):
  cost_matrix_1 = create_final_cost_matrix(left_features, right_features)
  cost_matrix_2 = recalculate_similarity(left_features, right_features)  # using neighbor infor
  cost_matrix = (cost_matrix_1 + cost_matrix_2) / 2
  result = {}
  if cost_matrix is not None:
    markers, left_center_p, right_center_p = get_markers(cost_matrix, left_features[0], right_features[0],
                                                         left_xml_tree, right_xml_tree, element_type_name,
                                                         left_xpath)
    # Sort markers by y-axis
    idx = list(np.argsort([x[1] for x in left_center_p]))
    markers[0] = [markers[0][i] for i in idx]
    markers[1] = [markers[1][i] for i in idx]
    left_center_p = [left_center_p[i] for i in idx]
    right_center_p = [right_center_p[i] for i in idx]

    if len(left_center_p) > 0:
      best_matches = second_pass(cost_matrix, left_features[0], right_features[0], markers,
                                 left_center_p, right_center_p, left_xml_tree, right_xml_tree,
                                 element_type_name, candidate=candidate)

      for m in zip(*best_matches[:2]):
        result[left_features[0][m[0]]['xpath']] = {'match': right_features[0][m[1]]['xpath'],
                                                   'similarity': cost_matrix[m[0], m[1]],
                                                   'ocr_text': left_features[0][m[0]]['ocr_text'],
                                                   'match_ocr_text': right_features[0][m[1]]['ocr_text'],
                                                   'text_color': left_features[0][m[0]]['text_color'],
                                                   'match_text_color': right_features[0][m[1]]['text_color']}
  return result, cost_matrix


def get_parent_matching(elements, element_features, xpaths, matched_xpaths, element_type_name, session):
  leaf_unmatched = [element for element in elements if
                    not element.has_children(elements) and element.xpath not in xpaths]
  for i, element in enumerate(elements):
    if element.has_children(elements) and element.xpath not in xpaths \
        and not any([xml_utils.is_parent(element.xpath, child.xpath) for child in leaf_unmatched]):
      children = xml_utils.get_children(xpaths, element.xpath)
      if len(children) > 0:
        matched_children = [matched_xpaths[xpaths.index(k)] for k in children]
        common_parent = xml_utils.get_common_parent(matched_children)

        if common_parent is not None and common_parent not in matched_xpaths:
          cand_xpaths, cand_sims = find_best_match_from_parent(common_parent, (
            [element_features[0][i]], [element_features[1][i]]), None, session)
          if len(cand_sims) > 0 and cand_sims[-1] > MATCHING_THRESHOLD[element_type_name][
            'no_ambiguous_threshold']:
            matched_element = Element(cand_xpaths[-1], session)
            xpaths.append(element.xpath)
            matched_xpaths.append(matched_element.xpath)
            yield element, matched_element, cand_sims[-1]


def match_screen_pair(left_session, right_session, element_type=constants.ElementType.VISUAL):
  left_elements = get_elements(left_session, element_type)
  left_features = get_element_in_screen(left_session, left_elements)

  right_elements = get_elements(right_session, element_type)
  right_features = get_element_in_screen(right_session, right_elements)

  if len(left_features[0]) == 0 or len(right_features[0]) == 0:
    return {}

  result, cost_matrix = find_best_matches(left_features, right_features, left_session.xml_tree,
                                          right_session.xml_tree,
                                          element_type.name)
  # Find parent matching for text assertion
  if element_type == constants.ElementType.TEXTUAL:
    # Get element that contains others -> remove text color
    left_matched_elements = [e for e in left_elements if e.xpath in result]
    for e in left_matched_elements:
      if e.has_children(left_matched_elements):
        result[e.xpath]['text_color'] = None
        result[e.xpath]['match_text_color'] = None

    # Left elements
    xpaths = list(result.keys())
    matched_xpaths = [m['match'] for m in result.values()]
    leaf_unmatched = [element for element in right_elements if
                      not element.has_children(right_elements) and element.xpath not in matched_xpaths]
    for (element, matched_element, similarity) in get_parent_matching(left_elements, left_features, xpaths,
                                                                      matched_xpaths, element_type.name,
                                                                      right_session):
      if not any([xml_utils.is_parent(matched_element.xpath, child.xpath) for child in leaf_unmatched]):
        result[element.xpath] = {'match': matched_element.xpath,
                                 'similarity': similarity,
                                 'ocr_text': element.ocr_text,
                                 'match_ocr_text': matched_element.ocr_text}
    # Right elements
    xpaths = [m['match'] for m in result.values()]
    matched_xpaths = list(result.keys())
    leaf_unmatched = [element for element in left_elements if
                      not element.has_children(left_elements) and element.xpath not in matched_xpaths]
    for (element, matched_element, similarity) in get_parent_matching(right_elements, right_features, xpaths,
                                                                      matched_xpaths, element_type.name,
                                                                      left_session):
      if not any([xml_utils.is_parent(matched_element.xpath, child.xpath) for child in leaf_unmatched]):
        result[matched_element.xpath] = {'match': element.xpath,
                                         'similarity': similarity,
                                         'ocr_text': matched_element.ocr_text,
                                         'match_ocr_text': element.ocr_text}
  return result


def match_screen_pair_by_segmentation(left_session, right_session, element_type=constants.ElementType.VISUAL):
  left_elements = get_visible_elements(left_session, get_elements(left_session, element_type))
  left_features = get_element_in_assertion_screen(left_session, left_elements)

  right_elements = get_visible_elements(right_session, get_elements(right_session, element_type))
  right_features = get_element_in_assertion_screen(right_session, right_elements)

  if len(left_features[0]) == 0 or len(right_features[0]) == 0:
    return {}

  result, cost_matrix = find_best_matches(left_features, right_features, left_session.xml_tree,
                                          right_session.xml_tree,
                                          element_type.name)
  # Find parent matching for text assertion
  if element_type == constants.ElementType.TEXTUAL:
    # Left elements
    xpaths = list(result.keys())
    matched_xpaths = [m['match'] for m in result.values()]
    leaf_unmatched = [element for element in right_elements if
                      not element.has_children(right_elements) and element.xpath not in matched_xpaths]
    for (element, matched_element, similarity) in get_parent_matching(left_elements, left_features, xpaths,
                                                                      matched_xpaths, element_type, right_session):
      if not any([xml_utils.is_parent(matched_element.xpath, child.xpath) for child in leaf_unmatched]):
        result[element.xpath] = {'match': matched_element.xpath,
                                 'similarity': similarity,
                                 'ocr_text': element.ocr_text,
                                 'match_ocr_text': matched_element.ocr_text}

    # Right elements
    xpaths = [m['match'] for m in result.values()]
    matched_xpaths = list(result.keys())
    leaf_unmatched = [element for element in left_elements if
                      not element.has_children(left_elements) and element.xpath not in matched_xpaths]
    for (element, matched_element, similarity) in get_parent_matching(right_elements, right_features, xpaths,
                                                                      matched_xpaths, element_type, left_session):
      if not any([xml_utils.is_parent(matched_element.xpath, child.xpath) for child in leaf_unmatched]):
        result[matched_element.xpath] = {'match': element.xpath,
                                         'similarity': similarity,
                                         'ocr_text': matched_element.ocr_text,
                                         'match_ocr_text': element.ocr_text}

  return result


def find_best_match_from_parent(right_xpath, left_element, left_session, right_session):
  assertion = left_session is None
  if assertion:
    left_features = left_element
  else:
    left_features = get_element_in_screen(left_session, [left_element])

  right_xpath_token = right_xpath.split('/')
  xpath_depth = len(right_xpath_token)
  begin_idx = max(2, xpath_depth - 3)
  right_candidate_xpaths = []
  right_candidates = []
  for i in range(begin_idx, xpath_depth + 1):
    right_candidate_xpath = '/'.join(right_xpath_token[:i])
    right_candidate = Element(right_candidate_xpath, right_session)
    left, top, right, bottom = right_candidate.bound
    if right - left > 0 and bottom - top > 0:
      right_candidate_xpaths.append(right_candidate_xpath)
      right_candidates.append(right_candidate)
  if assertion:
    right_features = get_element_in_assertion_screen(right_session, right_candidates)
  else:
    right_features = get_element_in_screen(right_session, right_candidates)

  cost_matrix = create_final_cost_matrix(left_features, right_features)
  if cost_matrix is not None:
    cand_sims = cost_matrix[0]
    cand_xpaths = np.array(right_candidate_xpaths)
    sort_idx = np.argsort(cand_sims)
    cand_sims = list(cand_sims[sort_idx])
    cand_xpaths = list(cand_xpaths[sort_idx])
    return cand_xpaths, cand_sims
  return [], []


def find_scrollable_element(left_session, right_session, left_xpath, element_type):
  cand_xpaths = []
  cand_sims = []

  left_elements = get_elements(left_session, element_type)
  left_xpaths = [e.xpath for e in left_elements]
  if left_xpath in left_xpaths:
    left_idx = left_xpaths.index(left_xpath)
    left_features = get_element_in_screen(left_session, left_elements)
    right_elements = get_elements(right_session, element_type)
    right_features = get_element_in_screen(right_session, right_elements)

    if len(left_features[0]) == 0 or len(right_features[0]) == 0:
      return cand_xpaths, cand_sims, False

    cost_matrix = create_final_cost_matrix(left_features, right_features)
    cand_sims = cost_matrix[left_idx]
    cand_xpaths = np.array([e.xpath for e in right_elements])
    sort_idx = np.argsort(cand_sims)
    cand_sims = list(cand_sims[sort_idx])
    cand_xpaths = list(cand_xpaths[sort_idx])
  return cand_xpaths, cand_sims, False


def get_features(sessions):
  left_features, right_features = [], []
  number_thread = len(sessions)
  with ThreadPool(processes=number_thread) as pool:
    results = []
    for i in range(number_thread):
      results.append(pool.apply_async(get_element_in_screen, (sessions[i][0], sessions[i][1],)))
    left_features = results[0].get()
    right_features = results[1].get()
  return left_features, right_features


def find_element(left_session, right_session, left_xpath, element_type=constants.ElementType.VISUAL):
  if element_type == constants.ElementType.SCROLLABLE:
    return find_scrollable_element(left_session, right_session, left_xpath, element_type)

  # Luna flow doesn't cover HTML elements yet. So enable this flag in order to provide HTML elements
  include_webview_element = left_session.element.is_webview_element()

  left_elements = get_elements(left_session, element_type, include_webview_element)
  cand_xpaths = []
  cand_sims = []
  left_idx = []
  if len(left_elements) > 0:
    # visual : sort by iou with finding element
    ious = [u.get_iou(left_session.element.bound, e.bound) for e in left_elements]
    finding_element_area = u.get_area(*left_session.element.bound) * constants.AREA_PADDING
    iou_sort_idx = np.argsort(ious)
    # case overlap -> get all elements that have center in finding element
    for index in reversed(iou_sort_idx):
      number_childrens = 0
      for i, e in enumerate(left_elements):
        if i != index and u.is_bound1_inside_bound2(e.bound, left_elements[index].bound):
          number_childrens += 1

      if ious[index] > 0.5:
        candidate_center = u.get_center_point_of_bound(left_elements[index].bound)
        if (u.get_area(*left_elements[index].bound) <= finding_element_area and u.is_point_inside_bound(
            candidate_center, left_session.element.bound)) \
            or (number_childrens <= 1 and u.intersect(left_session.element.bound,
                                                      left_elements[index].bound) >= 0.5):
          left_idx.append(index)
      else:
        break
  left_xpaths = [e.xpath for e in left_elements]
  is_parent = [xml_utils.is_parent(left_xpath, e_xpath) for e_xpath in left_xpaths]
  can_find_left_element = (left_xpath in left_xpaths or any(is_parent))
  priority_luna = len(left_idx) > 0 and can_find_left_element and not left_session.is_dynamic_content_element()
  if len(left_idx) > 0:
    potential_candidate = [left_idx[0]]
    right_elements = get_elements(right_session, element_type, include_webview_element)
    left_features, right_features = get_features([[left_session, left_elements], [right_session, right_elements]])
    if len(left_features[0]) == 0 or len(right_features[0]) == 0:
      return cand_xpaths, cand_sims, False
    matching_result, cost_matrix = find_best_matches(left_features, right_features, left_session.xml_tree,
                                                     right_session.xml_tree, element_type.name,
                                                     candidate=potential_candidate, left_xpath=left_xpath)
    left_candidate_xpaths = []
    if left_xpath in matching_result:
      cand_xpaths.append(matching_result[left_xpath]['match'])
      cand_sims.append(matching_result[left_xpath]['similarity'])
    else:
      for left_cand_idx in left_idx:
        left_cand_xpath = left_elements[left_cand_idx].xpath
        if left_cand_xpath in matching_result:
          left_candidate_xpaths.append(left_cand_xpath)
          cand_xpaths.append(matching_result[left_cand_xpath]['match'])
          cand_sims.append(matching_result[left_cand_xpath]['similarity'])

      if len(cand_xpaths) > 1 or (
          len(left_candidate_xpaths) == 1 and xml_utils.is_parent(left_xpath, left_candidate_xpaths[0])):
        if len(cand_xpaths) > 1:
          left_parent_xpath = xml_utils.get_common_parent(left_candidate_xpaths)
        else:
          left_parent_xpath = left_xpath

        if left_parent_xpath:
          # parent
          parent_xpath = xml_utils.get_common_parent(cand_xpaths)
          if parent_xpath:
            cand_xpaths, cand_sims = find_best_match_from_parent(parent_xpath, left_session.element,
                                                                 left_session, right_session)
            return cand_xpaths, cand_sims, False

      if len(cand_xpaths) > 0:
        # overlap
        idx = np.argmax(cand_sims)
        cand_sims = [cand_sims[idx]]
        cand_xpaths = [cand_xpaths[idx]]
        return cand_xpaths, cand_sims, False

    # dynamic content case / similarity too small / all matching
    matching_elements = [v['match'] for k, v in matching_result.items()]
    if len(cand_xpaths) == 0 and can_find_left_element:
      if left_xpath in left_xpaths:
        xpath_idx = left_xpaths.index(left_xpath)
      else:
        xpath_idx = is_parent.index(True)

      if left_session.is_dynamic_content_element():
        # dynamic content -> get all the simmilarities
        right_candidates_indices = [i for i in range(len(right_elements))]
      else:
        # normal case -> get the similarities of not matching elements
        right_candidates_indices = [i for i, e in enumerate(right_elements) if e.xpath not in matching_elements]

      if len(right_candidates_indices) > 0:
        cand_sims = cost_matrix[xpath_idx, right_candidates_indices]
        cand_xpaths = np.array([e.xpath for e in np.array(right_elements)[right_candidates_indices]])
        sort_idx = np.argsort(cand_sims)
        cand_sims = list(cand_sims[sort_idx])
        cand_xpaths = list(cand_xpaths[sort_idx])
        priority_luna = False

      if not left_session.is_dynamic_content_element() and len(cand_xpaths) > 0:
        # push the potential to last
        if len(potential_candidate) == 2:
          xpath = right_elements[potential_candidate[1]].xpath
          if potential_candidate[1] in right_candidates_indices:
            idx = cand_xpaths.index(xpath)
            cand_xpaths.pop(idx)
            similarity = cand_sims.pop(idx)
          else:
            similarity = cost_matrix[potential_candidate[0], potential_candidate[1]]
          cand_xpaths.append(xpath)
          cand_sims.append(similarity)

  return cand_xpaths, cand_sims, priority_luna


def find_element_by_segmentaion(left_session, right_session, return_xpath=True):
  cand_sims = []
  cand_bounds = []

  left_elements = left_session.get_visual_elements_by_bound()
  right_elements = right_session.get_visual_elements_by_bound()
  left_features = get_element_in_screen(left_session, left_elements)
  right_features = get_element_in_screen(right_session, right_elements)

  if len(left_features[0]) == 0 or len(right_features[0]) == 0:
    return [], []

  left_finding_element_bound = left_session.element.bound
  # Matching
  cost_matrix = create_final_cost_matrix(left_features, right_features)
  element_type_name = constants.ElementType.VISUAL.name
  result = {}
  if cost_matrix is not None:
    markers, left_center_p, right_center_p = get_markers(cost_matrix, left_features[0], right_features[0],
                                                         None, None, element_type_name)
    if len(left_center_p) > 0:
      best_matches = second_pass(cost_matrix, left_features[0], right_features[0], markers,
                                 left_center_p, right_center_p, None, None, element_type_name)
      for m in zip(*best_matches[:2]):
        left_bound = left_features[0][m[0]]['bounds']
        if u.intersect(left_bound, left_finding_element_bound) > constants.INTERSECTION_THRESHOLD:
          cand_bounds.append(right_features[0][m[1]]['bounds'])
          cand_sims.append(cost_matrix[m[0], m[1]])

  if len(cand_bounds) > 0:
    common_bound = u.get_common_bound(cand_bounds)
    similarity = np.max(cand_sims)
    if return_xpath:
      revisit_elements = right_session.elements + right_session.webview_elements
      common_xpath = xml_utils.get_common_parent_by_bound(common_bound, revisit_elements)
      if common_xpath is not None:
        return [common_xpath], [similarity]
    else:
      return [common_bound], [similarity]
  return [], []


def get_visible_elements(session, elements):
  visible_bounds = [ele.bound for ele in session.get_visual_elements_by_bound()]
  bounds_by_xml = [ele.bound for ele in elements]
  filtered_elements = []

  for idx, bound in enumerate(bounds_by_xml):
    ious = []
    for visible_bound in visible_bounds:
      ious.append(u.get_iou(bound, visible_bound))

    if max(ious) != 0:
      max_index = np.argmax(ious)
      del visible_bounds[max_index]
      filtered_elements.append(elements[idx])
  return filtered_elements
