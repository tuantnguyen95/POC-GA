from lxml import etree
import numpy as np
from service import constants, utils
from service.element import Element


def get_text_attribute(element):
  attribs = ['text', 'placeholder']
  text = ''
  attr_ind = 0
  while text == '' and attr_ind < len(attribs):
    attr = attribs[attr_ind]
    text = element.get(attr, '').strip()
    attr_ind += 1
  return text


def textual_elements(xml_tree, session):
  """
  Return xpaths of all textual element on the screen
  """
  gray_screen_img = utils.grayscale(session.screen_img)
  if session.platform == constants.Platform.IOS:
    selectors = ["//" + key + "[number(@y) > 20][not(*)]\
                               [not(ancestor-or-self::XCUIElementTypeAlert)]\
                               [not(contains(@name, 'scroll bar'))]"
                 for key in constants.IOS_TEXT_ATTRS if key != "XCUIElementTypeOther"]
    selector = " | ".join(selectors)
    ele_finder = etree.XPath(selector)
  else:
    ele_finder = etree.XPath("//*[not(*)][string-length(@text)>0 or string-length(@content-desc)>0][not(ancestor-or-self::*[\
        substring(@package, string-length(@package) - string-length('packageinstaller') + 1) = 'packageinstaller' or\
        substring(@package, string-length(@package) - string-length('permissioncontroller') + 1) = 'permissioncontroller' or\
        substring(@package, string-length(@package) - string-length('gms') + 1) = 'gms'])]"
                             "[not(ancestor-or-self::android.webkit.WebView)]\
                              [not(ancestor-or-self::android.widget.FrameLayout/@package='com.android.systemui')]")

  candidates = ele_finder(xml_tree)
  if session.platform == constants.Platform.IOS:
    other_finder = etree.XPath("//XCUIElementTypeOther[number(@y) > 20][not(*)]\
                              [not(ancestor-or-self::XCUIElementTypeAlert)]\
                              [not(contains(@name, 'scroll bar'))]")

    others_candidates = other_finder(xml_tree)
    screen_shape = (session.screen_img.shape[1], session.screen_img.shape[0])
    xpaths = []
    bbs = []
    for cand in candidates:
      bb = utils.decode_bound(cand, session.platform, pixel_map=session.pixel_map, screen_shape=screen_shape)
      xpath = xml_tree.getpath(cand)
      img = utils.crop_img(session.screen_img, bb)
      bg_color = utils.get_bg_color_beside_element(bb, gray_screen_img)
      if utils.is_visual_box(img, session.screen_img.shape, xpath, bg_color=bg_color):
        bbs.append(bb)
        xpaths.append(xpath)
    # Add non overlap XCUIElementTypeOther
    other_candidate_bbs = []
    other_candidate_areas = []
    for other_cand in others_candidates:
      bb = utils.decode_bound(other_cand, session.platform, pixel_map=session.pixel_map, screen_shape=screen_shape)
      area = (bb[2]-bb[0]) * (bb[3]-bb[1])
      other_candidate_bbs.append(bb)
      other_candidate_areas.append(area)

    idx = np.argsort(other_candidate_areas)
    for i in idx:
      other_cand = others_candidates[i]
      bb = other_candidate_bbs[i]
      xpath = xml_tree.getpath(other_cand)
      img = utils.crop_img(session.screen_img, bb)
      bg_color = utils.get_bg_color_beside_element(bb, gray_screen_img)
      if utils.is_visual_box(img, session.screen_img.shape, xpath, bg_color=bg_color) and not utils.is_overlap_within(bb, bbs):
        bbs.append(bb)
        xpaths.append(xpath)
  else:
    xpaths = [xml_tree.getpath(cand) for cand in candidates]
  return xpaths


def webview_textual_elements(html_tree):
  ele_finder = etree.XPath("//*[@bounds]"
                           "[string-length(@text)>0 or self::a or string-length(text())>0]")
  candidates = ele_finder(html_tree)
  final_candidates = [cand for cand in candidates if len(cand.getchildren()) == 0]
  final_xpaths = [html_tree.getpath(cand) for cand in final_candidates]

  for cand in candidates:
    children = cand.getchildren()
    if len(children) > 0:
      xpath = html_tree.getpath(cand)
      bound = cand.get('bounds')
      text = get_text_attribute(cand)

      same_as_child = False
      for i, candidate_xpath in enumerate(final_xpaths):
        if (xpath + '/') in candidate_xpath:
          candidate_bound = final_candidates[i].get('bounds')
          candidate_text = get_text_attribute(final_candidates[i])
          if (bound == candidate_bound) or (text == candidate_text and text != ''):
            same_as_child = True
            break
      if not same_as_child:
        final_candidates.append(cand)
        final_xpaths.append(xpath)

  return final_xpaths

def get_scrollable_element_xpath(xpath, scrollable_classes):
  parents = xpath.split('/')[:-1]

  for name in scrollable_classes:
    for i in reversed(range(len(parents))):
      item = parents[i]
      if name in item:
        return '/'.join(parents[:i+1])
  return None


def get_elements(session, scrollable=False):
  candidates = None
  elements = []
  image_shape = session.screen_img.shape[:2]
  if session.platform == constants.Platform.ANDROID:
    if scrollable:
      ele_finder = etree.XPath("//*[@bounds]")
    else:
      ele_finder = etree.XPath("//*[@bounds][not(ancestor-or-self::html)]")
  else:
    if scrollable:
      ele_finder = etree.XPath("//*[@width or (@bounds and @scrollable='true')]")
    else:
      ele_finder = etree.XPath("//*[@width]")

  candidates = ele_finder(session.xml_tree)
  if candidates:
    xpaths = [session.xml_tree.getpath(candidate) for candidate in candidates]
    elements = [Element(xpath, session) for xpath in xpaths]
    elements = [element for element in elements if element.img is not None]
    if scrollable:
      elements = [element for element in elements if element.is_scrollable() and utils.is_visual_box(element.img, image_shape, element.xpath)]
  return elements


def get_leaf_visual_verification_elements(session, include_webview_element=False):
  front_element_bounds = []
  front_element_xpaths = []
  image_shape = session.screen_img.shape[:2]
  gray_screen_img = utils.grayscale(session.screen_img)
  dialog_elements = []
  elements = []

  if session.platform == constants.Platform.IOS:
    ele_finder = etree.XPath("//*[@width or @bounds]"
                             "[not(ancestor-or-self::XCUIElementTypeStatusBar)]"
                             "[not(ancestor-or-self::XCUIElementTypeKeyboard)]"
                             "[count(./*)=0][not(contains(@name, 'scroll bar'))]")

    frame_finder = etree.XPath("//XCUIElementTypeWindow[@visible='true']")
    frame_elements = frame_finder(session.xml_tree)
    front_element_xpaths.extend([session.xml_tree.getpath(frame_element) for frame_element in frame_elements])
  else:
    ele_finder = etree.XPath("//*[@bounds][count(./*)=0][not(ancestor-or-self::android.widget.FrameLayout/@package='com.android.systemui')]")

    frame_finder = etree.XPath("//hierarchy/android.widget.FrameLayout[not(@package='com.android.systemui')]")
    frame_elements = frame_finder(session.xml_tree)
    while len(frame_elements) == 1 and len(frame_elements[0].getchildren()) > 0:
      frame_elements = frame_elements[0].getchildren()

    # dialogs
    dialog_finder = etree.XPath("//android.app.Dialog")
    dialog_elements = dialog_finder(session.xml_tree)

    # get the front frame first
    for frame in reversed(frame_elements):
      xpath = session.xml_tree.getpath(frame)
      element = Element(xpath, session)
      if element.img is not None and \
      ('/html' in xpath or len(front_element_bounds) == 0 or not utils.is_overlap_within(element.bound, front_element_bounds, threshold=0.5)):
        bg_color = utils.get_bg_color_beside_element(element.bound, gray_screen_img)
        if len(frame.getchildren()) > 0:
          front_element_bounds.append(element.bound)
          front_element_xpaths.append(element.xpath)
        elif utils.is_visual_box(element.img, image_shape, element.xpath, bg_color=bg_color):
          elements.append(element)

  # Finding elements in HTML (not XML)
  if include_webview_element:
    ele_finder = etree.XPath("//*[not(self::script or self::style or self::noscript)]"
                             "[@bounds]"
                             "[self::svg or self::button or count(./*)<=1]")

  candidates = ele_finder(session.xml_tree)
  element_xpaths = [e.xpath for e in elements]
  # check dialog elements
  dialogs_bounds = []
  dialogs_xpath = [session.xml_tree.getpath(dialog) for dialog in dialog_elements]
  for candidate in candidates:
    xpath = session.xml_tree.getpath(candidate)
    if xpath not in element_xpaths:
      element = Element(xpath, session)
      bg_color = utils.get_bg_color_beside_element(element.bound, gray_screen_img)
      for dialog_xpath in dialogs_xpath:
        if is_parent(dialog_xpath, xpath):
          dialog_element = Element(dialog_xpath, session)
          if dialog_element.img is not None and utils.is_visual_box(element.img, image_shape, element.xpath, bg_color=bg_color):
            elements.append(element)
            element_xpaths.append(xpath)
            dialogs_bounds.append(dialog_element.bound)
            break

  final_cands = []
  for candidate in candidates:
    xpath = session.xml_tree.getpath(candidate)
    if xpath not in element_xpaths:
      element = Element(xpath, session)
      bg_color = utils.get_bg_color_beside_element(element.bound, gray_screen_img)
      if utils.is_visual_box(element.img, image_shape, element.xpath, bg_color=bg_color) and \
        not utils.is_overlap_within(element.bound, dialogs_bounds, threshold=0.):
        final_cands.append(element)
  removed_overlap = [element for element in final_cands if any([x in element.xpath for x in front_element_xpaths])]
  if removed_overlap:
    return elements + removed_overlap
  return elements + final_cands


def get_parent_group(xpath, xml_tree):
  scrollable_classes = list(constants.ANDROID_SCROLLABLE_CLASSES) + list(constants.IOS_SCROLLABLE_CLASSES) + list(constants.WEBVIEW_UNSCROLLABLE_CLASSES)
  parents = xpath.split('/')[:-1]
  for i in reversed(range(1, len(parents))):
    if '.' in parents[i]:
      parent_class_name = parents[i].split('.')[-1]
    else:
      parent_class_name = parents[i]
    idx = parent_class_name.find('[')
    if idx > -1:
      parent_class_name = parent_class_name[:idx]
    parent_xpath = '/'.join(parents[:i+1])
    finder = etree.XPath(parent_xpath)
    parent_element = finder(xml_tree)[0]
    if parent_element.attrib.get('scrollable', 'false') == 'true' or parent_class_name in scrollable_classes:
      return parent_xpath + '/'
  return None


def find_candidate_in_scroll(scroll_xpath, xpaths, xml_tree, child=False):
  if child:
    cands = []
    for i, xpath in enumerate(xpaths):
      parent_scrollable_xpath = get_parent_group(xpath, xml_tree)
      if scroll_xpath in xpath and (parent_scrollable_xpath == scroll_xpath or is_parent(scroll_xpath[:-1], parent_scrollable_xpath)):
        cands.append(i)
    return np.array(cands)
  else:
    return np.where([scroll_xpath in xpath and get_parent_group(xpath, xml_tree) == scroll_xpath for xpath in xpaths])[0]


def is_parent(parent, child):
  return child.startswith(parent + '/')


def get_common_parent(xpaths):
  children = [xpath.split('/') for xpath in xpaths]
  max_child = children[np.argmin([len(child) for child in children])]
  if len(max_child) > 2:
    for i in reversed(range(2, len(max_child)+1)):
      parent = '/'.join(max_child[:i])
      if all([parent == child or is_parent(parent, child) for child in xpaths]):
        return parent
  return None


def get_children(xpaths, parent_xpath):
  children = []
  for xpath in xpaths:
    if parent_xpath + '/' in xpath:
      children.append(xpath)
  return children


def get_common_parent_by_bound(common_bound, elements):
  max_iou_candidate_areas = []
  max_iou_candidate_indies = []
  candidates_contain_common_bound_ious = []
  candidates_contain_common_bound_indices = []
  ious = []

  for idx, ele in enumerate(elements):
    bound = ele.bound
    iou = utils.get_iou(bound, common_bound)

    if iou > constants.MAX_IOU_THRESHOLD:
      max_iou_candidate_indies.append(idx)
      max_iou_candidate_areas.append(utils.get_area(*bound))

    elif utils.is_bound1_inside_bound2(common_bound, bound):
      candidates_contain_common_bound_indices.append(idx)
      candidates_contain_common_bound_ious.append(iou)

    ious.append(iou)

  if len(max_iou_candidate_indies) != 0:
    min_area_candidate_index = max_iou_candidate_indies[np.argmin(max_iou_candidate_areas)]
    return elements[min_area_candidate_index].xpath

  elif len(candidates_contain_common_bound_indices) != 0:
    max_iou_candidate_index = candidates_contain_common_bound_indices[np.argmax(candidates_contain_common_bound_ious)]
    return elements[max_iou_candidate_index].xpath

  max_iou_candidate_index = np.argmax(ious)
  return elements[max_iou_candidate_index].xpath


def is_toolbar(xpath):
  for toolbar_class in constants.TOOLBAR:
    if '/' + toolbar_class in xpath:
      return True
  return False


def get_keyboard_elements(session):
  image_shape = session.screen_img.shape[:2]
  if session.platform == constants.Platform.ANDROID:
    selectors = ["//*[@bounds][count(./*)=0][ancestor-or-self::android.widget.FrameLayout/@package='%s']" % (package) for package in constants.KEYBOARD_PACKAGE]
    selector = " | ".join(selectors)
    ele_finder = etree.XPath(selector)
  else:
    ele_finder = etree.XPath("//*[@width or @bounds][ancestor-or-self::XCUIElementTypeKeyboard]")
  candidates = ele_finder(session.xml_tree)
  elements = []
  for candidate in candidates:
    xpath = session.xml_tree.getpath(candidate)
    element = Element(xpath, session)
    if utils.is_visual_box(element.img, image_shape, element.xpath):
      elements.append(element)
  return elements
