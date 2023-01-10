from typing import Iterable
from lxml import etree

import service.constants
from service import utils as u, constants


class Element:
  def __init__(self, xpath, session, bound=None):
    self.sess = session
    self.xpath = self.get_xpath(xpath)
    if not u.is_empty_string(xpath):
      self._element = self.get_element()
      self.depth_level = len(self.xpath.split('/')) - 1
      self.bound = self.get_bound()
    elif bound is not None:
      self.bound = bound
      self._element = None

    self.ocr_text = self.get_ocr_text()
    self.log_obj = self.get_log_obj()

  def get_bound(self):
    screen_height, screen_width = self.sess.screen_img.shape[:2]
    return u.decode_bound(self._element, self.sess.platform, self.sess.pixel_map, screen_shape=(screen_width, screen_height))

  @property
  def locating_id(self):
    return u.get_locating_id(self._element, self.sess.platform)

  @property
  def img(self):
    return u.crop_img(self.sess.screen_img, self.bound)

  @property
  def tf_img(self):
    return u.cut_element(self.sess.screen_img, self.bound)

  @property
  def vertical_tf_img(self):
    return u.get_vertical_cut(self.sess.screen_img, self.bound)

  @property
  def vertical_small_tf_img(self):
    return u.get_vertical_small_cut(self.sess.screen_img, self.bound)

  @property
  def horizontal_tf_img(self):
    return u.get_horizontal_cut(self.sess.screen_img, self.bound)

  @property
  def horizontal_small_tf_img(self):
    return u.get_horizontal_small_cut(self.sess.screen_img, self.bound)

  @property
  def text(self):
    if self.is_webview_element():
      return ''.join(self._element.itertext())
    return u.get_element_text(self._element, self.sess.platform)

  def get_ocr_text(self):
    text, _ = u.extract_ocr_text_by_bound_iou(self.bound, self.sess.bounds, self.sess.ocr_texts)
    text = u.right_strip_text(text.lower())
    if text:
      text = u.ascii_normalize(text)
    return text

  @property
  def recursive_text(self):
    return u.get_recursive_text(self._element, self.sess.platform)

  @property
  def classname(self):
    return u.classname_from_xpath(self.xpath)

  @property
  def attrib(self):
    return self._element.attrib

  def get_log_obj(self):
    return {'xpath': self.xpath, 'ocr': self.ocr_text, 'bounds': self.bound}

  def is_scrollable(self, scrollable_classes: Iterable[str] = None):
    if u.is_empty_string(self.xpath):
      return False
    if scrollable_classes is None:
      scrollable_classes = constants.get_scrollable_classes(self.sess.platform)

    return u.is_scrollable_view(self.xpath, self.sess.xml_tree, self.sess.platform,
                                scrollable_classes)

  def get_xpath(self, xpath):
    # Convert html to hybrid (XML + HTML) xpath
    if xpath.startswith('/html'):
      return self.sess.webview_xpath + xpath
    return xpath

  def is_webview_element(self):
    return '/html' in self.xpath

  def get_element(self):
    if self.is_webview_element():
      idx = self.xpath.find('/html')
      xpath = self.xpath[idx:]
      return u.get_ele_by_xpath(self.sess.html_tree, xpath)
    return u.get_ele_by_xpath(self.sess.xml_tree, self.xpath)

  def is_visual(self):
    if not self.is_overlapped_by_keyboard():
      bg_color = u.get_bg_color_beside_element(self.bound, u.grayscale(self.sess.screen_img))
      return u.is_visual_box(self.img, self.sess.screen_img.shape, self.xpath, bg_color=bg_color)
    return False

  def is_blank(self):
    bg_color = u.get_bg_color_beside_element(self.bound, u.grayscale(self.sess.screen_img))
    return u.is_blank_img(self.img, bg_color)

  def is_overlapped_by_keyboard(self):
    element_bound = u.decode_bound(element=self._element, platform=self.sess.platform)
    element_bottom = element_bound[3]
    if self.sess.platform == constants.Platform.IOS:
      keyboard = u.get_ios_keyboard(self.sess.xml_tree)
      if keyboard is None:
        return False

      keyboard_bound = u.decode_bound(element=keyboard, platform=self.sess.platform)
      keyboard_top = keyboard_bound[1]
      return element_bottom > keyboard_top
    else:
      # There are cases that container of keyboard cover almost the screen
      # So we will find the most top element of keyboard which is small enough
      selectors = ["//*[@package='%s']" % package for package in constants.KEYBOARD_PACKAGE]
      selector = " | ".join(selectors)
      ele_finder = etree.XPath(selector)
      keyboard_elements = ele_finder(self.sess.xml_tree)
      if not keyboard_elements:
        return False

      screen_height, screen_width = self.sess.screen_img.shape[:2]
      max_height = screen_height / 3

      def filter_big_element_func(e):
        e_bound = u.decode_bound(element=e, platform=self.sess.platform)
        return e_bound[3] - e_bound[1] <= max_height

      keyboard_elements = list(filter(filter_big_element_func, keyboard_elements))
      if not keyboard_elements:
        return False

      def get_top_of_element(e):
        e_bound = u.decode_bound(element=e, platform=self.sess.platform)
        return e_bound[1]

      keyboard_top = min(map(get_top_of_element, keyboard_elements))
      return element_bottom > keyboard_top

  def has_children(self, eles):
    for e in eles:
      if e.bound != self.bound:
        if u.is_bound1_inside_bound2(e.bound, self.bound):
          return True
    return False

  def is_keyboard(self, platform):
    if platform == service.constants.Platform.ANDROID:
      package = ''
      if 'package' in self.attrib:
        package = self.attrib['package']
      return package in constants.KEYBOARD_PACKAGE
    else:
      return 'XCUIElementTypeKeyboard' in self.xpath.split('/')
