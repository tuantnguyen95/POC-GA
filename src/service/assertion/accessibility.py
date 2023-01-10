from abc import ABC, abstractmethod
from service.constants import Platform, AccessibilityAssertionType, AccessibilityAssertionCategory
from service.constants import ACCESSIBILITY_ASSERTION_MESSAGE_DICT
import json
from service.screen_density_utils import convert_font_height_px2dp
from service.sessions import Session
from service.element import Element
import service.utils as u
from service.visual_verification import VisualVerificationElement

ACCESSIBILITY = json.load(open('service/assertion/constants.json'))

class AccessibilitySession(Session):
  def __init__(self, req):
    super().__init__(req)
    self.visual_elements = self.get_visual_elements()
    self.image_with, self.image_height = u.get_image_size_in_xml(self.xml_tree)

  @property
  def device(self):
    return ''

  @property
  def platform(self):
    if self.req.platform_name.upper() == Platform.ANDROID.name:
      return Platform.ANDROID
    if self.req.platform_name.upper() == Platform.IOS.name:
      return Platform.IOS
    return None

  def get_xml(self):
    return self.req.xml

  def get_screen_img(self):
    return u.decode_img_bytes(self.req.screen)

  @property
  def density(self):
    if self.req.screen_density == 0:
      return 1
    return self.req.screen_density

  @property
  def pixel_map(self):
    """
    Pixel map is only for iOS devices.
    """
    if self.platform == Platform.IOS:
      return self.req.pixel_map
    return None

  def verified_elements(self):
    elements = []
    for element in self.visual_elements:
      if self.is_valid_bound_element(element) and not element.is_overlapped_by_keyboard():
        touch_target_element = TouchTargetElement(element, self).get_accessibility_assertion_dict()
        if touch_target_element:
          elements.append(touch_target_element)

        if self.is_valid_text_element(element):
          low_contrast_element = LowContrastTextElement(element, self).get_accessibility_assertion_dict()
          if low_contrast_element:
            elements.append(low_contrast_element)
    return elements

  def is_valid_text_element(self, element):
    if self.platform == Platform.IOS:
      is_valid = (element.classname == 'XCUIElementTypeStaticText')
    else:
      is_valid = element.attrib.get('text') != '' and \
        (element.classname == 'android.widget.TextView' or \
          element.classname == 'android.widget.EditText') \
        or (element.attrib.get('content-desc') != '' and \
        element.classname == 'android.view.View')
    return is_valid

  def is_valid_bound_element(self, element):
    if self.platform == Platform.IOS:
      horizontal_scale = 1 if self.pixel_map.horizontal_scale is None else \
                                      self.pixel_map.horizontal_scale
      is_valid_horizontal_bound = element.bound[2] <= int(self.image_with) * horizontal_scale
      vertical_scale = 1 if self.pixel_map.vertical_scale is None else \
                                            self.pixel_map.vertical_scale
      is_valid_vertical_bound = element.bound[3]  <= int(self.image_height) * vertical_scale
    else:
      is_valid_horizontal_bound = element.bound[2] <= int(self.image_with)
      is_valid_vertical_bound = element.bound[3] <= int(self.image_height)
    return is_valid_horizontal_bound and is_valid_vertical_bound

  def print_pixel_map(self, pixel_map):
    return {
      'horizontal_pre_scale': pixel_map.horizontal_pre_scale,
      'vertical_pre_scale': pixel_map.vertical_pre_scale,
      'horizontal_scale': pixel_map.horizontal_scale,
      'vertical_scale': pixel_map.vertical_scale,
      'x_offset': pixel_map.x_offset,
      'y_offset': pixel_map.y_offset
    }
 

  @property
  def instance_log(self):
    log_info = {
      'platform': self.platform.name,
      'density': self.density,
      'elements': [e.log_obj for e in self.visual_elements]
    }

    if self.platform == Platform.IOS:
      log_info['pixel_map'] = self.print_pixel_map(self.pixel_map)

    return log_info


class AccessibilityElement(ABC):
  def __init__(self, element: Element, session: AccessibilitySession):
    self.xpath = element.xpath
    self.tag_name = element.classname
    self.attributes = element.attrib
    self.bounds = element.bound
    if session.platform == Platform.IOS \
        or session.density == 0:  # for dC device
      self.logical_screen_density = 1
    else:
      self.logical_screen_density = session.density/160
    self.screen_height, self.screen_width = session.screen_img.shape[:2]

  @property
  @abstractmethod
  def category(self):
    pass

  @property
  @abstractmethod
  def meta_data(self):
    pass

  @property
  @abstractmethod
  def is_assertive(self):
    pass

  @property
  def type(self):
    return AccessibilityAssertionType.WARNING.name if self.is_assertive else ''

  @property
  def message(self):
    return ACCESSIBILITY_ASSERTION_MESSAGE_DICT[self.category] if self.is_assertive else ''

  @property
  def rect(self):
    return {'_x': self.bounds[0],
            '_y': self.bounds[1],
            '_width': self.bounds[2] - self.bounds[0],
            '_height': self.bounds[3] - self.bounds[1]}

  @property
  def element(self):
    attrib_dict = u.get_attributes_dict(self.attributes)
    return {'tag_name': self.tag_name,
            'rect': self.rect,
            'attributes': {
              'xpath': self.xpath,
              **attrib_dict}}

  def get_accessibility_assertion_dict(self):
    if self.is_assertive:
      return {'category': self.category,
              'type': self.type,
              'message': self.message,
              'meta_data': self.meta_data,
              'element': self.element}
    return None


class TouchTargetElement(AccessibilityElement):
  def __init__(self, element: Element, session: AccessibilitySession):
    super().__init__(element, session)
    self._session = session
    self._width = self.rect['_width']
    self._height = self.rect['_height']

    if session.platform == Platform.IOS:
      self._actual_width = round(int(self.attributes['width']) * self._session.pixel_map.horizontal_scale)
      self._actual_height = round(int(self.attributes['height']) * self._session.pixel_map.vertical_scale)
    else:
      self._actual_width = round(self.rect['_width'] / self.logical_screen_density)
      self._actual_height = round(self.rect['_height'] / self.logical_screen_density)

    if session.platform == Platform.IOS:
      self._required_width = ACCESSIBILITY["iOS"]["TOUCH_TARGET_MIN_WIDTH"]
      self._required_height = ACCESSIBILITY["iOS"]["TOUCH_TARGET_MIN_HEIGHT"]
    elif self.is_against_side():
      self._required_width = ACCESSIBILITY["Android"]["TOUCH_TARGET_MIN_WIDTH_ON_EDGE"]
      self._required_height = ACCESSIBILITY["Android"]["TOUCH_TARGET_MIN_HEIGHT_ON_EDGE"]
    else:
      self._required_width = ACCESSIBILITY["Android"]["TOUCH_TARGET_MIN_WIDTH"]
      self._required_height = ACCESSIBILITY["Android"]["TOUCH_TARGET_MIN_HEIGHT"]

  @property
  def category(self):
    return AccessibilityAssertionCategory.TOUCH_TARGET_SIZE.name if self.is_assertive else ''

  @property
  def message(self):
    if self.is_assertive:
      category = None
      if self._actual_width < self._required_width and self._actual_height < self._required_height:
        category = AccessibilityAssertionCategory.TOUCH_TARGET_SIZE.name
      elif self._actual_width < self._required_width:
        category = AccessibilityAssertionCategory.TOUCH_TARGET_WIDTH.name
      elif self._actual_height < self._required_height:
        category = AccessibilityAssertionCategory.TOUCH_TARGET_HEIGHT.name
      return ACCESSIBILITY_ASSERTION_MESSAGE_DICT[category] if category else ''
    return ''

  @property
  def meta_data(self):
    if self.is_assertive:
      return {'width': self._actual_width,
              'height': self._actual_height,
              'required_width': self._required_width,
              'required_height': self._required_height}
    return {}

  @property
  def is_assertive(self):
    if self._session.platform == Platform.IOS:
      is_displayed = u.is_truth_attribute(self.attributes, 'visible')
      clickable_vals = [u.is_truth_attribute(self.attributes, k) \
        for k in ['enabled', 'accessible']]
    else:
      is_displayed = u.is_truth_attribute(self.attributes, 'displayed') \
        or u.is_truth_attribute(self.attributes, 'enabled')
      clickable_vals = [u.is_truth_attribute(self.attributes, k) \
        for k in ['clickable', 'long-clickable']]
    return is_displayed and any(clickable_vals) and \
           (self._actual_width < self._required_width or \
            self._actual_height < self._required_height)

  def is_against_side(self):
    return self.bounds[0] == 0 or self.bounds[2] == self.screen_width

  def is_against_top_bottom(self):
    return self.bounds[1] == 0 or self.bounds[3] == self.screen_height


class LowContrastTextElement(AccessibilityElement):
  def __init__(self, element: Element, session: AccessibilitySession):
    super().__init__(element, session)
    self._element = element
    self._session = session
    self._background_color = u.get_background_color(element.img)
    _, self._foreground_color = u.get_text_color(element.img)
    self._contrast_ratio, self._threshold = self.get_contrast_info()

  @property
  def category(self):
    return AccessibilityAssertionCategory.LOW_CONTRAST.name if self.is_assertive else ''

  @property
  def meta_data(self):
    if self.is_assertive:
      return {'threshold': self._threshold,
              'value': self._contrast_ratio,
              'foreground_color': u.rgb2hex(self._foreground_color.tolist()),
              'background_color': u.rgb2hex(self._background_color.tolist())}
    return {}

  @property
  def is_assertive(self):
    if self._session.platform == Platform.IOS:
      visible_vals = [u.is_truth_attribute(self.attributes, k) for k in ['enabled', 'visible']]
    else:
      visible_vals = [u.is_truth_attribute(self.attributes, k) for k in ['enabled']]
    return all(visible_vals) and self._threshold

  def is_large_text(self):
    is_large_text = False
    vve = VisualVerificationElement(self.xpath, self._session)
    font_height_px = vve.get_font_height()
    if font_height_px:
      text_dp_size = convert_font_height_px2dp(font_height_px, self._session.density)
      is_large_text = text_dp_size >= ACCESSIBILITY["Android"]["WCAG_LARGE_REGULAR_TEXT_MIN_SIZE"]
    return is_large_text

  def get_contrast_info(self):
    contrast_ratio = None
    required_contrast_ratio = None
    threshold = None
    fg_color = self._foreground_color
    bg_color = self._background_color

    if u.is_different_color(fg_color, bg_color):
      contrast_ratio = u.calculate_contrast_ratio(fg_color, bg_color)
    if contrast_ratio and contrast_ratio != 1.0:
      if self._session.platform == Platform.IOS:
        required_contrast_ratio = ACCESSIBILITY["iOS"]["CONTRAST_RATIO_TEXT"]
      else:
        required_contrast_ratio = ACCESSIBILITY["Android"]["CONTRAST_RATIO_WCAG_LARGE_TEXT"] if self.is_large_text() else \
                                ACCESSIBILITY["Android"]["CONTRAST_RATIO_WCAG_NORMAL_TEXT"]
    if required_contrast_ratio and \
      required_contrast_ratio - contrast_ratio > ACCESSIBILITY["CONTRAST_TOLERANCE"]:
      threshold = required_contrast_ratio

    return contrast_ratio, threshold
