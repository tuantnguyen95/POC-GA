import os
import sys
import cv2
import copy
import traceback
import numpy as np
import xml.etree.ElementTree as ET
import service.constants as constants

SCROLLABLE_CLASS_NAMES_ANDROID = [
  'android.widget.ListView',
  'android.widget.GridView',
  'android.widget.ScrollView',
  'android.widget.HorizontalScrollView',
  'android.support.v7.widget.RecyclerView',
  'androidx.recyclerview.widget.RecyclerView',
  'android.support.v4.view.ViewPager',
  'androidx.viewpager.widget.ViewPager'
]

SCROLLABLE_CLASS_NAMES_IOS = [
  'XCUIElementTypeCollectionView',
  'XCUIElementTypeScrollView',
  'XCUIElementTypeTable'
]

WEB_VIEW_CLASS_NAMES_ANDROID = [
  'android.webkit.WebView'
]

WEB_VIEW_CLASS_NAMES_IOS = [
  'XCUIElementTypeWebView'
]

SCROLLABLE_MAIN_TYPES = [SCROLLABLE_CLASS_NAMES_ANDROID, SCROLLABLE_CLASS_NAMES_IOS, WEB_VIEW_CLASS_NAMES_ANDROID, WEB_VIEW_CLASS_NAMES_IOS]
SCROLLABLE_ALL_TYPES = [atype for mtype in SCROLLABLE_MAIN_TYPES for atype in mtype]
SCROLLABLE_VERTICAL = [stype for stype in SCROLLABLE_ALL_TYPES  if 'HorizontalScrollView' not in stype]
SCROLLABLE_HORIZONTAL = [stype for stype in SCROLLABLE_ALL_TYPES if stype not in SCROLLABLE_VERTICAL]

def match_position(cvim1, cvim2):
  gray1_origin = cv2.cvtColor(cvim1, cv2.COLOR_BGR2GRAY)
  gray2_origin = cv2.cvtColor(cvim2, cv2.COLOR_BGR2GRAY)

  feature_extractor = cv2.SIFT_create()

  kp1, des1 = feature_extractor.detectAndCompute(gray1_origin, None)
  kp2, des2 = feature_extractor.detectAndCompute(gray2_origin, None)

  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1 ,des2, k=2)

  good_match = []
  for m in matches:
    if m[0].distance/(m[1].distance + 1) < 0.5:
      good_match.append(m)
  good_match_arr = np.asarray(good_match)

  points1 = np.float32([ kp1[m.queryIdx].pt for m in good_match_arr[:,0] ]).reshape(-1,1,2)
  points2 = np.float32([ kp2[m.trainIdx].pt for m in good_match_arr[:,0] ]).reshape(-1,1,2)

  H, masked = cv2.findHomography(points2, points1, cv2.RANSAC , 5.0)
  first_keypoint_x = 10000000
  last_keypoint_x = 0
  for point in points2:
    if point[0][0] < first_keypoint_x:
      first_keypoint_x = point[0][0]
    if point[0][0] < first_keypoint_x:
      first_keypoint_x = point[0][0]
  first_keypoint_x = int(first_keypoint_x)
  matchx1 = round( np.matmul([1, 1, first_keypoint_x], H)[2] )
  matchx2 = round( np.matmul([1, 1, cvim2.shape[1]], H)[2] )
  return matchx1, matchx2, first_keypoint_x

def stitch_image(cvimg1, cvimg2, padding=0):
  cvim1 = cv2.rotate(cvimg1, cv2.ROTATE_90_COUNTERCLOCKWISE)
  if padding:
    cvimg2 = cvimg2[padding:cvimg2.shape[0], 0:cvimg2.shape[1]]

  cvim2 = cv2.rotate(cvimg2, cv2.ROTATE_90_COUNTERCLOCKWISE)

  im1_x1, im1_x2, im2_x1 = match_position(cvim1, cvim2)

  if im1_x2 < cvimg1.shape[1]:
    return cvimg1, im1_x1, im1_x2

  new_img = np.zeros((cvim1.shape[0], im1_x2, 3), dtype=np.uint8)
  new_img[:, 0:cvim1.shape[1]] = cvim1
  new_img[:, im1_x1:im1_x2] = cvim2[:, im2_x1:]
  new_img = cv2.rotate(new_img, cv2.ROTATE_90_CLOCKWISE)
  return new_img, im1_x1, im1_x2, im2_x1

def extract_fixed_region(previous_scrollable_img, current_scrollable_img):
  shortest_height = min(previous_scrollable_img.shape[0], current_scrollable_img.shape[0])
  previous_scrollable_img = cv2.cvtColor(previous_scrollable_img, cv2.COLOR_BGR2GRAY)
  current_scrollable_img = cv2.cvtColor(current_scrollable_img, cv2.COLOR_BGR2GRAY)

  previous_scrollable_img = previous_scrollable_img[:shortest_height, :].astype(np.int32)
  current_scrollable_img = current_scrollable_img[:shortest_height, :].astype(np.int32)
  result = np.abs(np.sum(previous_scrollable_img - current_scrollable_img, axis=1))
  binary_result = np.where(result > constants.STITCHING_PADDING_THRESHOLD, 1, 0)

  for i, value in enumerate(binary_result):
    if value != 0:
      return i
  return 0

def stitch_multiple_images(cvimgs, xml_trees):
  stitched_image = None
  stitched_xml = None
  num_success = 0
  l_img_pos = []

  scrollable_img_1, _, _ = get_scrollable_region(cvimgs[0], xml_trees[0])
  scrollable_img_2, _, _ = get_scrollable_region(cvimgs[1], xml_trees[1])

  padding = extract_fixed_region(scrollable_img_1, scrollable_img_2)

  for i, (cvimg, xml_tree)in enumerate(zip(cvimgs, xml_trees)):
    scrollable_img, scrollable_bb, scrollable_ele = get_scrollable_region(cvimg, xml_tree)

    if type(scrollable_img) == type(None):
      return stitched_image, num_success, l_img_pos
    if i == 0:
      stitched_image = scrollable_img
      l_img_pos.append((0, scrollable_img.shape[0]))
    else:
      stitched_image, im1x1, im1x2, im2x1 = stitch_image(stitched_image, scrollable_img, padding)
      if im1x1-im2x1 < 0 or im1x2 < stitched_image.shape[0]:
        return stitched_image, num_success, l_img_pos
      else:
        l_img_pos.append((im1x1-im2x1-padding, im1x2))

        num_success += 1
  return stitched_image, num_success, l_img_pos

def get_bound(element, scale=1):
  bounding_box = [0] * 4
  if 'bounds' in element.attrib:
    bounds = element.get('bounds')
    parts = bounds[1:-1].replace('][', ',').split(',')
    bounding_box = list(map(int, map(float, parts)))
  elif 'x' in element.attrib and 'y' in element.attrib and 'width' in element.attrib and 'height' in element.attrib:
    fields = ['x', 'y', 'width', 'height']
    left, top, width, height = [int(element.get(f)) for f in fields]
    bounding_box = [left, top, left+width, top+height]
  return bounding_box * int(scale)

def assign_bound(element, bound):
  if 'bounds' in element.attrib:
    element.attrib['bounds'] = "[{},{}][{},{}]".format(*bound)
  elif 'x' in element.attrib and 'y' in element.attrib \
      and 'width' in element.attrib and 'height' in element.attrib:
    element.attrib['x'] = bound[0]
    element.attrib['y'] = bound[1]
    element.attrib['width'] = bound[2] - bound[0]
    element.attrib['height'] = bound[3] - bound[1]

def get_scrollable_region(cvimg, xml_tree, horizontal=False):
  root = xml_tree.getroot()
  max_len = 0
  bb = None
  scrollable_img = None
  scrollable_ele = None
  scrollable_types = SCROLLABLE_HORIZONTAL if horizontal else SCROLLABLE_VERTICAL
  for s in scrollable_types:
    scrollable_regions = root.iter(s)
    for scroll in scrollable_regions:
      is_scrollable = scroll.attrib.get('scrollable')
      if is_scrollable == 'true':
        box = get_bound(scroll)
        if (box[3] - box[1]) > max_len:
          max_len = box[3] - box[1]
          bb = box
          scrollable_ele = scroll
  if bb:
    scrollable_img = cvimg[bb[1]:bb[3], bb[0]:bb[2]]
  return scrollable_img, bb, scrollable_ele

def remove_xml_node(xml_mapping, node):
  for ele in list(node):
    xml_mapping[node].append(ele)
    xml_mapping[ele] = xml_mapping[node]
  xml_mapping[node].remove(node)

def get_id_from_element(element):
  if 'instance-id' in element.attrib:
    e_id = element.tag + '-instance-id-' + element.attrib['instance-id']
  elif 'id' in element.attrib:
    e_id = element.tag + '-id-' + element.attrib['id']
  return e_id

def child_to_parent(element):
  xml_parent_mapping = {}
  for p in element.iter():
    for c in p:
      xml_parent_mapping[c] = p
  return xml_parent_mapping

def child_id_to_child(element, childparent):
  cid_2_c = {}
  for c in element.iter():
    element_id = get_id_from_element(c)
    parent_element_id = get_id_from_element(childparent[c])
    cid_2_c[element_id + '-' + parent_element_id] = c
  return cid_2_c

def merge_scrollable_element(list_scrollable_element, shape, root):
  final_scrollable = list_scrollable_element[0]

  assign_bound(final_scrollable, shape)

  final_child_parent = child_to_parent(root)

  final_child_id_child = child_id_to_child(final_scrollable, final_child_parent)

  # Stitch other xmls into first xml
  for i, ele in enumerate(list_scrollable_element[1:]):
    list_child = [c for c in ele.iter()]

    for child in list_child:
      if 'id' in child.attrib or 'instance-id' in child.attrib: # Not add if id exists
        child_id = get_id_from_element(child)
        child_id += '-' + get_id_from_element(final_child_parent[child])
        if child_id in final_child_id_child: # If exists, merge bounds from two xmls
          b1 = get_bound(child) # new bound
          b2 = get_bound(final_child_id_child[child_id]) # old bound
          new_bound = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
          assign_bound(final_child_id_child[child_id], new_bound)
        else: # Add if id not exists, add element to its parent
          while (1):
            parent = final_child_parent[child]
            parent_id = get_id_from_element(parent)
            parent_id += '-' + get_id_from_element(final_child_parent[parent])
            if parent_id in final_child_id_child:
              for subchild in list(child):
                if subchild == child:
                  continue
                child.remove(subchild)
              final_child_id_child[parent_id].append(child)
              final_child_id_child[child_id] = child
              break
            else:
              child = parent
              child_id = parent_id
  return final_scrollable

def get_full_image(rimgs, rxmls, l_img_pos, stitched_image, rpos1):
  final_screen = copy.deepcopy(rimgs[-1])
  final_xml = copy.deepcopy(rxmls[-1])
  final_root = final_xml.getroot()

  scrollable_img, scrollable_bb, scrollable_ele = get_scrollable_region(final_screen, final_xml)
  if rpos1 + scrollable_img.shape[0] > stitched_image.shape[0]:
    rpos1 -= rpos1 + scrollable_img.shape[0] - stitched_image.shape[0]
  rpos2 = rpos1 + scrollable_img.shape[0]
  final_screen[scrollable_bb[1]:scrollable_bb[3], scrollable_bb[0]:scrollable_bb[2]] = stitched_image[rpos1:rpos2,:]

  # Get revisit xmls that overlap with final revisit screen
  l_overlap_pos = []
  for i, img_pos in enumerate(l_img_pos):
    if (img_pos[0] <= rpos1 and rpos1 <= img_pos[1]) \
      or (img_pos[0] <= rpos2  and rpos2 <= img_pos[1]):
      overlapx1 = max(img_pos[0], rpos1)
      overlapx2 = min(img_pos[1], rpos2)
      x1 = overlapx1-img_pos[0]
      x2 = overlapx2-img_pos[0]
      l_overlap_pos.append((x1, x2))
    else:
      l_overlap_pos.append(None)

  final_xml_mapping = {c:p for p in final_xml.getroot().iter() for c in p}
  final_xml_mapping[scrollable_ele].remove(scrollable_ele)
  list_scrollables = []

  offset_pixel = None
  for i, pos in enumerate(l_overlap_pos):
    if pos != None:
      offset_pixel = pos[0] + l_img_pos[i][0]
      break

  # Start stitching xmls
  for i, pos in enumerate(l_overlap_pos):
    if pos == None:
      continue
    _, s_bb, s_ele = get_scrollable_region(rimgs[i], rxmls[i])

    xml_mapping = {c:p for p in s_ele.iter() for c in p}

    list_ele = [ele for ele in s_ele.iter()]
    for ele in list_ele:
      if ele == s_ele:
        continue
      box = get_bound(ele)

      if box[3] < s_bb[1] + pos[0] or box[1] > s_bb[1] + pos[1]:
        remove_xml_node(xml_mapping, ele)
        continue

      box[0] = int(box[0])
      box[2] = int(box[2])
      stitch_pos_x1 = box[1] - s_bb[1] - offset_pixel + l_img_pos[i][0]
      stitch_pos_x2 = box[3] - s_bb[1] - offset_pixel + l_img_pos[i][0]
      box[1] =  int(stitch_pos_x1 + scrollable_bb[1])
      box[3] =  int(stitch_pos_x2 + scrollable_bb[1])

      if box[1] < scrollable_bb[1]:
        box[1] = scrollable_bb[1]
      if box[3] > scrollable_bb[3]:
        box[3] = scrollable_bb[3]

      assign_bound(ele, box)

    list_scrollables.append(s_ele)
    final_xml_mapping[scrollable_ele].append(s_ele)

  final_scrollable = merge_scrollable_element(list_scrollables, scrollable_bb, final_root)

  for element in list_scrollables:
    final_xml_mapping[scrollable_ele].remove(element)
  final_xml_mapping[scrollable_ele].append(final_scrollable)

  return final_screen, final_xml

def get_new_revisit_image(pimg, pxml, rimgs, rxmls, stitched_img, l_img_pos):
  scroll_img, _, _ = get_scrollable_region(pimg, pxml)
  cvim1 = cv2.rotate(stitched_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
  cvim2 = cv2.rotate(scroll_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
  if type(scroll_img) == type(None):
    return None, None

  # Get match position of prime screen in stitched screen to extract revisit screen
  im1x1, _, _ = match_position(cvim1, cvim2)
  if im1x1 < 0: # Cannot find prime screen on revisit screen
    return None, None

  # Get final revisit screen and xml that matches the most with prime screen
  new_rimg, new_rxml = get_full_image(rimgs, rxmls, l_img_pos, stitched_img, im1x1)
  return new_rimg, new_rxml

