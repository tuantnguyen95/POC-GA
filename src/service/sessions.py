import base64
import service.kobiton_ocr as kobiton_ocr_service
from abc import ABC, abstractmethod
from lxml import etree
from service import utils as u
from service import xml_utils
from service import constants
from service.element import Element
from service.constants import Platform
from service.utils import logger
from flask import request


class Session(ABC):

  def __init__(self, req):
    self.req = req
    self.logger = logger
    self.screen_img = self.get_screen_img()
    self.xml = self.get_xml()
    if not u.is_empty_string(self.xml):
      self.xml_tree = u.parse_xml_tree(self.xml)
      self.html = u.get_html(self.xml)
      self.html_tree = u.parse_html_tree(self.html) if self.html else None
      self.webview_xpath = self.get_webview_contain_html_xpath()
    action_id = request.args.get('action_id', '0')
    session_id = request.args.get('session_id', '0')
    request_id = request.args.get('request_id', '0')

    if u.kobiton_ocr_service_info != (None, None):
      self.ocr_texts, self.bounds, self.ocr_scores = self.__extract_ocr_texts_in_imgs([self.screen_img], session_id, action_id, request_id)[0]
    else:
      self.ocr_texts, self.bounds, self.ocr_scores = u.detect_texts(self.screen_img)

  def __extract_ocr_texts_in_imgs(self, imgs, session_id=0, action_id=0, request_id=0):
    self.logger.info("Extracting OCR texts using Kobiton OCR service")
    host, port = u.kobiton_ocr_service_info
    
    kobiton_ocr = kobiton_ocr_service.KobitonOCR(self.logger, {constants.KOBITON_OCR_SERVICE: '%s:%s' % (host, port)} )
    logger_params_sota = {
      'log': "Error on getting OCR text",
      'extras': {
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id
      }
    }
    
    results = u.retry_call_n_times(kobiton_ocr.detect_texts_in_images, self.logger, logger_params_sota, session_id, action_id, request_id, imgs)
    return results

  @property
  @abstractmethod
  def platform(self):
    pass

  @property
  @abstractmethod
  def device(self):
    pass

  @property
  @abstractmethod
  def density(self):
    pass

  @property
  @abstractmethod
  def pixel_map(self):
    pass

  @property
  @abstractmethod
  def instance_log(self):
    pass

  @abstractmethod
  def get_xml(self):
    pass

  @abstractmethod
  def get_screen_img(self):
    pass

  def get_webview_contain_html_xpath(self):
    webview_element = u.get_webview_contain_html(self.xml_tree, self.platform)
    if webview_element is not None:
      return self.xml_tree.getpath(webview_element)
    return ''

  def get_visual_elements(self, include_webview_element=False):
    elements = xml_utils.get_leaf_visual_verification_elements(self, include_webview_element)
    elements = u.remove_large_elements(self, elements)
    elements = u.remove_overlap_elements(elements)
    # sort element from top to toe
    elements = sorted(elements, key=lambda e: e.bound[1])
    return elements

  def get_scrollable_elements(self):
    return xml_utils.get_elements(self, scrollable=True)

  def get_textual_elements(self):
    element_xpaths = xml_utils.textual_elements(self.xml_tree, self)
    if self.html_tree is not None:
      element_xpaths += xml_utils.webview_textual_elements(self.html_tree)
    elements = [Element(xpath, self) for xpath in element_xpaths]
    elements = [element for element in elements if element.is_visual()]

    results = []
    for element in elements:
      if not u.partially_beyond_viewport(element.bound, self.screen_img,
                                         element.ocr_text, element.text, self.platform):
        is_in = False
        for i, e in enumerate(results):
          if u.intersect(element.bound, e.bound) == 1 and u.intersect(e.bound, element.bound) > 0.7:
            is_in = True
            break
          elif u.intersect(e.bound, element.bound) == 1 and u.intersect(element.bound, e.bound) > 0.7:
            results[i] = element
            is_in = True
            break
        if not is_in:
          results.append(element)
    results = u.remove_large_elements(self, results)
    return results

  def set_visual_elements_by_bound(self, bounds):
    self.segment_elements = [Element('', self, bound=bound) for bound in bounds]

  def get_visual_elements_by_bound(self):
    return self.segment_elements

  def get_keyboard_elements(self):
    return xml_utils.get_keyboard_elements(self)

class PrimeSession(Session):
  def __init__(self, req):
    super(PrimeSession, self).__init__(req)


  @property
  def platform(self):
    if self.req.prime_platform_name.upper() == Platform.ANDROID.name:
      return Platform.ANDROID
    elif self.req.prime_platform_name.upper() == Platform.IOS.name:
      return Platform.IOS
    return None

  @property
  def density(self):
    return self.req.prime_screen_density

  @property
  def device(self):
    return self.req.prime_device_name

  @property
  def element(self):
    if self.req.prime_xpath and self.req.prime_xml:
      return Element(self.req.prime_xpath, self)
    else:
      prime_bound = [self.req.prime_bound.x1, self.req.prime_bound.y1, self.req.prime_bound.x2, self.req.prime_bound.y2]
      return Element('', self, bound=prime_bound)

  @property
  def pixel_map(self):
    """
    Pixel map is only for iOS devices.
    """
    if self.platform == constants.Platform.IOS:
      return self.req.prime_pixel_map
    return None

  @property
  def instance_log(self):
    return {
      'prime_device': self.device,
      'prime_platform': self.platform.name,
      'is_scrollable': self.element.is_scrollable(),
      'prime_xpath': self.element.xpath,
      'prime_horizontal_scale': self.pixel_map.horizontal_scale if self.pixel_map else None,
      'prime_vertical_scale': self.pixel_map.vertical_scale if self.pixel_map else None,
      'prime_horizontal_pre_scale': self.pixel_map.horizontal_pre_scale if self.pixel_map else None,
      'prime_vertical_pre_scale': self.pixel_map.vertical_pre_scale if self.pixel_map else None,
      'prime_x_offset': self.pixel_map.x_offset if self.pixel_map else None,
      'prime_y_offset': self.pixel_map.y_offset if self.pixel_map else None
    }

  def get_xml(self):
    return self.req.prime_xml

  def get_screen_img(self):
    return u.decode_img_bytes(self.req.prime_screen)

  def is_dynamic_content_element(self):
    if self.req.is_dynamic_content_element:
      return True
    return False


class RevisitSession(Session):
  def __init__(self, req):
    super(RevisitSession, self).__init__(req)

  @property
  def platform(self):
    if self.req.revisit_platform_name.upper() == Platform.ANDROID.name:
      return Platform.ANDROID
    return Platform.IOS

  @property
  def density(self):
    return self.req.revisit_screen_density

  @property
  def device(self):
    return self.req.revisit_device_name

  @property
  def pixel_map(self):
    """
    Pixel map is only for iOS devices.
    """
    if self.platform == constants.Platform.IOS:
      return self.req.revisit_pixel_map
    return None

  @property
  def instance_log(self):
    return {
      'revisit_device': self.device,
      'revisit_platform': self.platform.name,
      'revisit_horizontal_pre_scale': self.pixel_map.horizontal_pre_scale if self.pixel_map else None,
      'revisit_vertical_pre_scale': self.pixel_map.vertical_pre_scale if self.pixel_map else None,
      'revisit_horizontal_scale': self.pixel_map.horizontal_scale if self.pixel_map else None,
      'revisit_vertical_scale': self.pixel_map.vertical_scale if self.pixel_map else None,
      'revisit_x_offset': self.pixel_map.x_offset if self.pixel_map else None,
      'revisit_y_offset': self.pixel_map.y_offset if self.pixel_map else None
    }

  @property
  def elements(self):
    return xml_utils.get_elements(self)

  @property
  def webview_elements(self):
    candidates = None
    if self.html is not None:
      ele_finder = etree.XPath("//*[not(self::script or self::style or self::noscript)]"
                               "[@bounds]"
                               "[self::svg or self::button or count(./*)<=2]"
                               )
      candidates = ele_finder(self.html_tree)
    if candidates:
      xpaths = [self.html_tree.getpath(candidate) for candidate in candidates]
      return [Element(xpath, self) for xpath in xpaths]
    return []

  def get_xml(self):
    return self.req.revisit_xml

  def get_screen_img(self):
    return u.decode_img_bytes(self.req.revisit_screen)


class ScreenshotSession(Session):
  def __init__(self, req):
    super(ScreenshotSession, self).__init__(req)

  @property
  def platform(self):
    if self.req.platform_name.upper() == Platform.ANDROID.name:
      return Platform.ANDROID
    elif self.req.platform_name.upper() == Platform.IOS.name:
      return Platform.IOS
    return None

  @property
  def density(self):
    pass

  @property
  def device(self):
    pass

  @property
  def elements(self):
    return xml_utils.get_elements(self)

  @property
  def pixel_map(self):
    """
    Pixel map is only for iOS devices.
    """
    if self.platform == constants.Platform.IOS:
      return self.req.pixel_map
    return None

  @property
  def instance_log(self):
    return {
      'platform': self.platform.name,
      'horizontal_scale': self.pixel_map.horizontal_scale if self.pixel_map else None,
      'vertical_scale': self.pixel_map.vertical_scale if self.pixel_map else None,
      'horizontal_pre_scale': self.pixel_map.horizontal_pre_scale if self.pixel_map else None,
      'vertical_pre_scale': self.pixel_map.vertical_pre_scale if self.pixel_map else None,
      'x_offset': self.pixel_map.x_offset if self.pixel_map else None,
      'y_offset': self.pixel_map.y_offset if self.pixel_map else None
    }

  def point(self):
    if self.req.point.x is None or self.req.point.y is None:
      return [None, None]
    return [self.req.point.x, self.req.point.y]

  def get_xml(self):
    return self.req.xml

  def get_screen_img(self):
    return u.decode_img_bytes(self.req.screen)

  @property
  def webview_elements(self):
    candidates = None
    if self.html is not None:
      ele_finder = etree.XPath("//*[not(self::script or self::style or self::noscript)]"
                               "[@bounds]"
                               )
      candidates = ele_finder(self.html_tree)
    if candidates:
      xpaths = [self.html_tree.getpath(candidate) for candidate in candidates]
      return [Element(xpath, self) for xpath in xpaths]
    return []


class FindingByImageSession(Session):
  def __init__(self, req):
    super(FindingByImageSession, self).__init__(req)

  @property
  def platform(self):
    if self.req.platform_name.upper() == Platform.ANDROID.name:
      return Platform.ANDROID
    elif self.req.platform_name.upper() == Platform.IOS.name:
      return Platform.IOS
    return None

  @property
  def elements(self):
    return xml_utils.get_elements(self)

  @property
  def pixel_map(self):
    """
    Pixel map is only for iOS devices.
    """
    if self.platform == constants.Platform.IOS:
      return self.req.pixel_map
    return None

  @property
  def instance_log(self):
    return {
      'platform': self.platform.name,
      'horizontal_scale': self.pixel_map.horizontal_scale if self.pixel_map else None,
      'vertical_scale': self.pixel_map.vertical_scale if self.pixel_map else None,
      'horizontal_pre_scale': self.pixel_map.horizontal_pre_scale if self.pixel_map else None,
      'vertical_pre_scale': self.pixel_map.vertical_pre_scale if self.pixel_map else None,
      'x_offset': self.pixel_map.x_offset if self.pixel_map else None,
      'y_offset': self.pixel_map.y_offset if self.pixel_map else None
    }

  @property
  def webview_elements(self):
    candidates = None
    if self.html is not None:
      ele_finder = etree.XPath("//*[not(self::script or self::style or self::noscript)]"
                               "[@bounds]"
                               )
      candidates = ele_finder(self.html_tree)
    if candidates:
      xpaths = [self.html_tree.getpath(candidate) for candidate in candidates]
      return [Element(xpath, self) for xpath in xpaths]
    return []

  @property
  def density(self):
    pass

  @property
  def device(self):
    pass

  def get_xml(self):
    return self.req.xml

  def get_screen_img(self):
    return u.decode_img_bytes(self.req.screenshot)
  
  def get_query_img(self):
    img_bytes = base64.b64decode(self.req.base64_encoded_query_image)
    return u.decode_img_bytes(img_bytes)

  def get_threshold(self):
    return self.req.threshold

  def get_method(self):
    return self.req.method