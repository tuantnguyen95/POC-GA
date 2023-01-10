import service.constants as constants
from service.ai_components import AIComponent
from service.sessions import PrimeSession, RevisitSession
from service.element import Element
from service import utils as u

from service.utils import is_in_scroll_view, \
  get_element_margin, is_bound1_inside_bound2, calculate_padding, \
  index_of, detect_texts, filter_text
from service.screen_density_utils import calculate_ppi_of_screen, inch_to_cm, convert_pixels2inches, \
  convert_font_height_px2dp, get_screen_density, convert_font_height_px2pt
from service.luna.match_elements import match_screen_pair, match_screen_pair_by_segmentation
import numpy as np

import service.image_stitching as imgstc

import xml.etree.ElementTree as ET

import cv2
import traceback
import base64

class VisualVerifyPrimeSession(PrimeSession):
  def __init__(self, req, logger):
    super().__init__(req)
    self.display_screen_size = req.prime_display_screen_size

  def get_xml(self):
    if len(self.req.prime_data) == 1:
      return self.req.prime_data[0].prime_xml

  def get_screen_img(self):
    if len(self.req.prime_data) == 1:
      return u.decode_img_bytes(self.req.prime_data[0].prime_screen)

  @property
  def density(self):
    return get_screen_density(self.xml_tree, self.screen_img, self.platform,\
                                self.pixel_map, self.display_screen_size, self.device)


class VisualVerifyRevisitSession(RevisitSession):
  def __init__(self, req, logger):
    self.id_revisit_screen = 0
    for i, screen_data in enumerate(req.revisit_data):
      if screen_data.is_navigation_screen == True:
        self.id_revisit_screen = i
        break
    self.logger = logger
    self.final_screen = u.decode_img_bytes(req.revisit_data[self.id_revisit_screen].revisit_screen)
    self.final_xml = req.revisit_data[self.id_revisit_screen].revisit_xml
    self.is_stitching = False
    super().__init__(req)

    self.display_screen_size = req.revisit_display_screen_size

  # This function should be called after get_screen_img()
  def get_xml(self):
    return self.final_xml

  def get_screen_img(self):
    if len(self.req.revisit_data) > 1:
      try:
        target_revisit_screen, target_revisit_xml = self.get_target_revisit_screen()
      except Exception:
        self.logger.error(traceback.format_exc())
        target_revisit_screen = target_revisit_xml = None

      if target_revisit_screen is not None and target_revisit_xml:
        self.logger.info("Using screen from stitched image.")
        self.final_xml = ET.tostring(target_revisit_xml.getroot(), encoding='utf-8')
        self.final_screen = target_revisit_screen
        self.is_stitching = True

    return self.final_screen

  def get_target_revisit_screen(self):
    rxmls = [ET.ElementTree(ET.fromstring(f.revisit_xml)) for f in self.req.revisit_data]
    rimgs = [cv2.cvtColor(u.decode_img_bytes(f.revisit_screen), cv2.COLOR_RGB2BGR) for f in self.req.revisit_data]
    pimg = cv2.cvtColor(u.decode_img_bytes(self.req.prime_data[0].prime_screen), cv2.COLOR_RGB2BGR)
    pxml = ET.ElementTree(ET.fromstring(self.req.prime_data[0].prime_xml))

    stitched_image, num_success, l_img_pos = imgstc.stitch_multiple_images(rimgs, rxmls)
    if stitched_image is not None:
      if num_success != len(rimgs) - 1:
        self.logger.error('Cannot stitch all revisit images.')
        return None, None
      new_rimg, new_rxml = imgstc.get_new_revisit_image(pimg, pxml, rimgs, rxmls, stitched_image, l_img_pos)
      if new_rimg is None:
        return None, None
      return new_rimg, new_rxml
    else:
      self.logger.error('Cannot get stitched image.')
      return None, None
    
  @property
  def density(self):
    return get_screen_density(self.xml_tree, self.screen_img, self.platform,\
                                self.pixel_map, self.display_screen_size, self.device)

class VisualVerificationElement(Element):
  def __init__(self, xpath, session):
    super().__init__(xpath, session)
    self.cm_bound = self.get_cm_bound()
    self.cm_margin, self.parent = self.get_cm_margin()
    self.ocr_bounds = self.get_ocr_bounds()

    self.text_block_bound = self.get_element_text_block_bound()
    self.inch_text_block_padding = self.get_inch_text_block_padding()
    self.font_height = self.get_font_height()

  def get_cm_bound(self):
    cm_bound = []
    for coordinate in self.bound:
      inch_coord = convert_pixels2inches(coordinate, self.sess.density)
      cm_coord = inch_to_cm(inch_coord)
      cm_bound.append(cm_coord)
    return cm_bound

  def get_cm_margin(self):
    cm_margin = []
    parent_bound, parent = get_element_margin(self._element, self.sess.platform, self.sess.pixel_map)
    for coordinate in parent_bound:
      inch_coord = convert_pixels2inches(coordinate, self.sess.density)
      cm_coord = inch_to_cm(inch_coord)
      cm_margin.append(cm_coord)
    return cm_margin, parent

  @property
  def is_in_scroll_view(self):
    return is_in_scroll_view(self.xpath, self.sess.xml_tree, self.sess.platform)

  def get_ocr_bounds(self):
    return [bound for bound in self.sess.bounds if is_bound1_inside_bound2(bound, self.bound)]

  # IN DEVELOPING BUT NOT RELEASE YET
  def get_element_text_block_bound(self):
    min_left = None
    min_top = None
    max_right = None
    max_bottom = None
    for bound in self.ocr_bounds:
      (left, top, right, bottom) = bound
      if min_left is None or left < min_left:
        min_left = left
      if min_top is None or top < min_top:
        min_top = top
      if max_right is None or right > max_right:
        max_right = right
      if max_bottom is None or bottom > max_bottom:
        max_bottom = bottom

    if min_left and min_top and max_right and max_bottom:
      return [min_left, min_top, max_right, max_bottom]
    return None

  # IN DEVELOPING BUT NOT RELEASE YET
  def get_inch_text_block_padding(self):
    if self.text_block_bound:
      text_block_padding = calculate_padding(element_bound=self.text_block_bound,
                                             parent_bound=self.bound)
      inch_text_block_padding = []
      for coordinate in text_block_padding:
        inch_text_block_padding.append(convert_pixels2inches(coordinate, self.sess.density))
      return inch_text_block_padding

  def get_font_height(self):
    texts_origin, bounds_origin, scores_origin = detect_texts(self.img)
    thresholds = [90, 50, 0]
    for threshold in thresholds:
      texts, bounds, _ = filter_text(texts_origin, bounds_origin, scores_origin, threshold)
      if len(texts) > 0:
        break
    if len(bounds) == 0:
      return None
    heights = [bound[3] - bound[1] for bound in bounds]
    return np.max(heights)

  def get_parent_xpath(self):
    if self.parent is not None:
      if self.parent in self.sess.xml_tree.iter():
        return self.sess.xml_tree.getpath(self.parent)
      elif self.parent in self.sess.html_tree.iter():
        return self.sess.webview_xpath + self.sess.html_tree.getpath(self.parent)
    return None


class VisualVerification(AIComponent):
  def __init__(self, services, logger):
    super().__init__(services, logger=logger)
    self.func = self._try_get_visual_bounds_by_segmentation

  def __call__(self, prime: VisualVerifyPrimeSession, revisit: VisualVerifyRevisitSession,
               session_id: int = 0, action_id: int = 0, request_id: int = 0):

    solver = VisualVerificationSolver(prime, revisit, self.logger, self._try_get_visual_bounds_by_segmentation,
                                      session_id=session_id,
                                      action_id=action_id,
                                      request_id=request_id)
    return solver.get_response()


class VisualVerificationSolver:
  def __init__(self, prime: VisualVerifyPrimeSession, revisit: VisualVerifyRevisitSession,
               logger, func,
               session_id: int = 0, action_id: int = 0, request_id: int = 0):
    
    self.prime = prime
    self.revisit = revisit
    self.logger = logger
    self.session_id = session_id
    self.action_id = action_id
    self.request_id = request_id
    self.struct_subs = self.prime.req.structure_submissions
    self.prime_struct_sub_xpaths = [submission.prime_xpath
                                    for submission in self.struct_subs]
    self.revisit_struct_sub_xpaths = [submission.revisit_xpath
                                      for submission in self.struct_subs]
    self.layout_subs = self.prime.req.layout_submissions
    self.layout_sub_xpaths = [submission.prime_xpath
                              for submission in self.layout_subs]
    self.func = func

  def get_response(self):
    
    matched_pairs = self.match_elements()
    self.log(f'There are {len(matched_pairs)} matched pairs', xpaths=matched_pairs)
    unmatched_pairs = self.get_unmatched_elements(matched_pairs)
    self.log(f'There are {len(unmatched_pairs)} un-matched pairs', xpaths=unmatched_pairs)
    pairs = matched_pairs + unmatched_pairs
    elements = []

    for info in self.extract_pair_infos(pairs):
      prime_element = info['prime_element']
      revisit_element = info['revisit_element']
      struct_level = info['struct_level']
      layout_level = info['layout_level']

      if prime_element is not None and revisit_element is not None:
        struct_status = self.structure_assertion(struct_level)
      else:
        if prime_element is not None and prime_element.is_blank():
          self.log(f'Skip {prime_element.xpath} because it\'s blank')
          continue
        elif revisit_element is not None and revisit_element.is_blank():
          self.log(f'Skip {revisit_element.xpath} because it\'s blank')
          continue
        struct_status = self.structure_assertion(struct_level, matched=False)

      layout_status = self.layout_assertion(prime_element, revisit_element, layout_level)
      fontsize_status = self.fontsize_assertion(prime_element, revisit_element)
      elements.append(self.extract_pair_result(prime_element, revisit_element,
                                               struct_status, layout_status, fontsize_status))

    response = {}
    response['elements'] = elements

    # Return xml, screen of stitched image if possibble
    if self.revisit.is_stitching:
      retval, buf = cv2.imencode('.jpg', self.revisit.final_screen)
      revisit_screen_str = base64.b64encode(buf).decode('utf-8')
      revisit_xml_str = self.revisit.final_xml.decode('utf-8')
      response['stitchingScreenData'] = {'source': revisit_xml_str, 'screenshot': revisit_screen_str }

    self.log('Visual verification finished a request', result=response['elements'])
    return response

  
  def match_elements(self, using_segment=False):
    if using_segment:
      sessions_elements = self.func([self.prime.screen_img, self.revisit.screen_img])
      prime_elements = sessions_elements[0]
      self.logger.debug('Elements returned by segmentation for prime', extras={'elements': prime_elements})
      self.prime.set_visual_elements_by_bound(prime_elements)
      revisit_elements = sessions_elements[1]
      self.revisit.set_visual_elements_by_bound(revisit_elements)
      self.logger.debug('Elements returned by segmentation for revisit', extras={'elements': revisit_elements})

      matched_pairs = match_screen_pair_by_segmentation(self.prime, self.revisit)
    else:
      matched_pairs = match_screen_pair(self.prime, self.revisit)
    return [(prime_xpath, matched_pairs[prime_xpath]['match']) for prime_xpath in matched_pairs]

  def extract_pair_infos(self, pairs):
    for prime_xpath, revisit_xpath in pairs:
      struct_level = constants.StructureAssertion.STRICT.name
      struct_idx = max(index_of(prime_xpath, self.prime_struct_sub_xpaths),
                       index_of(revisit_xpath, self.revisit_struct_sub_xpaths))
      if struct_idx != -1:
        struct_level = self.struct_subs[struct_idx].level.upper()

      layout_level = constants.LayoutAssertion.SKIP.name \
        if struct_level == constants.StructureAssertion.SKIP.name \
        else constants.LayoutAssertion.STRICT.name
      layout_idx = index_of(prime_xpath, self.layout_sub_xpaths)
      if layout_idx != -1:
        layout_level = self.layout_subs[layout_idx].level.upper()

      yield {
        'prime_element': VisualVerificationElement(prime_xpath, self.prime) if prime_xpath else None,
        'revisit_element': VisualVerificationElement(revisit_xpath, self.revisit) if revisit_xpath else None,
        'struct_level': struct_level,
        'layout_level': layout_level
      }

  def get_unmatched_elements(self, matched_pairs):
    prime_xpaths = set([element.xpath for element in self.prime.get_visual_elements()])
    revisit_xpaths = set([element.xpath for element in self.revisit.get_visual_elements()])
    matched_prime_xpaths = set([pair[0] for pair in matched_pairs])
    matched_revisit_xpaths = set([pair[1] for pair in matched_pairs])

    unmatched_prime_xpaths = list(prime_xpaths.difference(matched_prime_xpaths))
    unmatched_revisit_xpaths = list(revisit_xpaths.difference(matched_revisit_xpaths))

    unmatched_prime_xpaths = [(prime_xpath, None) for prime_xpath in unmatched_prime_xpaths]
    unmatched_revisit_xpaths = [(None, revisit_xpath) for revisit_xpath in unmatched_revisit_xpaths]
    return unmatched_prime_xpaths + unmatched_revisit_xpaths

  def structure_assertion(self, level, matched=True):
    if level == constants.StructureAssertion.STRICT.name:
      if matched:
        status = constants.StructureAssertionStatus.PASSED.name
      else:
        status = constants.StructureAssertionStatus.FAILED.name
    else:
      status = constants.StructureAssertionStatus.SKIP.name
    return {
      'level': level,
      'status': status
    }

  def layout_assertion(self, element_1: VisualVerificationElement, element_2: VisualVerificationElement,
                            level):
    threshold = None
    if level == constants.LayoutAssertion.STRICT.name:
      threshold = 0.1
    elif level == constants.LayoutAssertion.RELAXED_PLACEMENT.name:
      threshold = 0.3

    if level == constants.LayoutAssertion.SKIP.name or element_1 is None or element_2 is None:
      status = constants.LayoutAssertionStatus.SKIP.name
    else:
      status = constants.LayoutAssertionStatus.PASSED.name
      status = self.__coord_assertion(element_1, element_2, status, threshold)

      status = self.__margin_assertion(element_1, element_2, status, threshold)
    return {'status': status,
            'level': level,
            'threshold': threshold,
            'prime_bound': element_1.cm_bound if element_1 else None,
            'revisit_bound': element_2.cm_bound if element_2 else None,
            'prime_margin': element_1.cm_margin if element_1 else None,
            'revisit_margin': element_2.cm_margin if element_2 else None
            }

  def fontsize_assertion(self, element_1: VisualVerificationElement, element_2: VisualVerificationElement):
    status = constants.FontSizeAssertionStatus.SKIP.name
    fontsize1 = None
    fontsize2 = None

    if element_1 and element_2 and element_1.font_height and element_2.font_height:
      if element_1.sess.platform == constants.Platform.ANDROID:
        fontsize1 = convert_font_height_px2dp(element_1.font_height, element_1.sess.density)
      else:
        fontsize1 = convert_font_height_px2pt(element_1.font_height, element_1.sess.density)
      if element_2.sess.platform == constants.Platform.ANDROID:
        fontsize2 = convert_font_height_px2dp(element_2.font_height, element_2.sess.density)
      else:
        fontsize2 = convert_font_height_px2pt(element_2.font_height, element_2.sess.density)
      threshold = 1.
      if abs(fontsize1 - fontsize2) <= threshold:
        status = constants.FontSizeAssertionStatus.PASSED.name
      else:
       status = constants.FontSizeAssertionStatus.FAILED.name

    return {'status': status,
            'prime_font_size': fontsize1,
            'revisit_font_size': fontsize2
            }


  def __coord_assertion(self, element_1: VisualVerificationElement, element_2: VisualVerificationElement,
                        prev_status,
                        threshold):
    status = prev_status
    bound_distance = [abs(coord_1 - coord_2)
                      for coord_1, coord_2 in zip(element_1.cm_bound, element_2.cm_bound)]

    if not element_1.is_in_scroll_view:
      for coord_distance in bound_distance:
        if coord_distance > threshold:
          status = constants.LayoutAssertionStatus.FAILED.name
          break
    else:
      width_distance = abs(bound_distance[2] - bound_distance[0])
      high_distance = abs(bound_distance[3] - bound_distance[1])
      if width_distance > threshold or high_distance > threshold:
        status = constants.LayoutAssertionStatus.FAILED.name
    return status

  def __margin_assertion(self, element_1: VisualVerificationElement, element_2: VisualVerificationElement,
                         prev_status,
                         threshold):
    status = prev_status

    margin_distance = [abs(coord_1 - coord_2)
                       for coord_1, coord_2 in zip(element_1.cm_margin, element_2.cm_margin)]
    if not element_1.is_in_scroll_view:
      for padding_coord_distance in margin_distance:
        if padding_coord_distance > threshold:
          status = constants.LayoutAssertionStatus.FAILED.name
          break
    return status

  @staticmethod
  def extract_pair_result(prime_element, revisit_element, structure_status, layout_status, fontsize_status):
    pair_result = {
      "prime_xpath": prime_element.xpath if prime_element else None,
      "prime_text": prime_element.text if prime_element else None,
      "prime_bound": prime_element.bound if prime_element else None,
      "prime_parent": prime_element.get_parent_xpath() if prime_element else None,
      "revisit_xpath": revisit_element.xpath if revisit_element else None,
      "revisit_text": revisit_element.text if revisit_element else None,
      "revisit_bound": revisit_element.bound if revisit_element else None,
      "revisit_parent": revisit_element.get_parent_xpath() if revisit_element else None,
      "assertion": {
        "structure": structure_status,
        "layout": layout_status,
        "fontsize": fontsize_status
      }
    }
    return pair_result

  def log(self, content, **kwargs):
    self.logger.info(content,
                     extras={'session_id': self.session_id,
                             'action_id': self.action_id,
                             'request_id': self.request_id,
                             **kwargs})
