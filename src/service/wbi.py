import service.constants as constants
import numpy as np
import pickle

from service.ai_components import AIComponent
from service.sessions import PrimeSession, RevisitSession
from service.element import Element
from service.utils import detect_texts, filter_text
from service.screen_density_utils import convert_font_height_px2mm, calculate_ppi_of_screen, get_screen_density, \
  convert_pixel2mm, convert_font_height_px2pt, convert_font_height_px2dp
from service.luna.match_elements import match_screen_pair
from service.constants import WBI_MODEL_PATH

model = pickle.load(open(WBI_MODEL_PATH, 'rb'))


class WBIPrimeSession(PrimeSession):
  def __init__(self, req):
    super().__init__(req)
    self.display_screen_size = req.prime_display_screen_size
    self.screen_width, self.screen_height = self.screen_img.shape[:2]
    self.ppi = calculate_ppi_of_screen(self.screen_width, self.screen_height, self.display_screen_size)

  @property
  def density(self):
    return get_screen_density(self.xml_tree, self.screen_img, self.platform, self.pixel_map,
                              self.display_screen_size, self.device)


class WBIRevisitSession(RevisitSession):
  def __init__(self, req):
    super().__init__(req)
    self.display_screen_size = req.revisit_display_screen_size
    self.screen_width, self.screen_height = self.screen_img.shape[:2]
    self.ppi = calculate_ppi_of_screen(self.screen_width, self.screen_height, self.display_screen_size)

  @property
  def density(self):
    return get_screen_density(self.xml_tree, self.screen_img, self.platform, self.pixel_map,
                              self.display_screen_size, self.device)


class WBIElement(Element):
  def __init__(self, xpath, session):
    super().__init__(xpath, session)
    self.font_height = self.get_font_height()

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


class WBI(AIComponent):
  def __init__(self, services, logger):
    super().__init__(services, logger=logger)

  def __call__(self, prime: WBIPrimeSession, revisit: WBIRevisitSession,
               session_id: int = 0, action_id: int = 0, request_id: int = 0):
    solver = WBISolver(prime, revisit, self.logger, session_id=session_id,
                       action_id=action_id, request_id=request_id)
    return solver.get_response()


class WBISolver:
  def __init__(self, prime: WBIPrimeSession, revisit: WBIRevisitSession,
               logger, session_id: int = 0, action_id: int = 0, request_id: int = 0):
    self.prime = prime
    self.revisit = revisit
    self.logger = logger
    self.session_id = session_id
    self.action_id = action_id
    self.request_id = request_id

  def get_response(self):
    matched_pairs = self.match_elements()
    response = []
    if len(matched_pairs) == 0:
      self.log('Could not find any matched pair', result=response)
    else:
      for info in self.extract_pair_infos(matched_pairs):
        prime_element = info['prime_element']
        revisit_element = info['revisit_element']
        prime_ocr_text = info['prime_ocr_text']
        revisit_ocr_text = info['revisit_ocr_text']
        fontsize_status = self.font_size_assertion(prime_element, revisit_element)
        pair_result = self.extract_pair_result(prime_element, revisit_element,
                                               prime_ocr_text, revisit_ocr_text, fontsize_status)

        if pair_result['assertion']['fontsize']['prime_font_size'] and \
                pair_result['assertion']['fontsize']['revisit_font_size'] and \
                ((pair_result['prime_text'] == pair_result['revisit_text'] and pair_result['prime_text'] != '' and
                  pair_result['revisit_text'] != '') or
                 (pair_result['prime_ocr_text'] == pair_result['revisit_ocr_text'] and
                  pair_result['prime_ocr_text'] != '' and pair_result['revisit_ocr_text'] != '')):
          response.append(self.check_font_size_is_normal(self.prime, self.revisit, pair_result))
      self.log('Finished checking normal/abnormal', result=response)
    return response

  def match_elements(self):
    matched_pairs = match_screen_pair(self.prime, self.revisit)
    pairs = []
    for prime_xpath in matched_pairs:
      pairs.append((prime_xpath, matched_pairs[prime_xpath]['ocr_text'],
                    matched_pairs[prime_xpath]['match'], matched_pairs[prime_xpath]['match_ocr_text']))
    return pairs

  def extract_pair_infos(self, pairs):
    for prime_xpath, prime_ocr_text, revisit_xpath, revisit_ocr_text in pairs:
      yield {
        'prime_element': WBIElement(prime_xpath, self.prime) if prime_xpath else None,
        'revisit_element': WBIElement(revisit_xpath, self.revisit) if revisit_xpath else None,
        'prime_ocr_text': prime_ocr_text,
        'revisit_ocr_text': revisit_ocr_text
      }

  def font_size_assertion(self, element_1: WBIElement, element_2: WBIElement):
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

    return {'status': status, 'prime_font_size': fontsize1, 'revisit_font_size': fontsize2}

  def check_font_size_is_normal(self, prime_session, revisit_session, pair_result):
    prime_bound = pair_result["prime_bound"]
    prime_ppi = prime_session.ppi
    prime_screen_width = convert_pixel2mm(prime_session.screen_width, prime_ppi)
    prime_screen_height = convert_pixel2mm(prime_session.screen_height, prime_ppi)
    prime_font_size = convert_font_height_px2mm(pair_result["assertion"]["fontsize"]["prime_font_size"])

    revisit_bound = pair_result["revisit_bound"]
    revisit_ppi = revisit_session.ppi
    revisit_screen_width = convert_pixel2mm(revisit_session.screen_width, revisit_ppi)
    revisit_screen_height = convert_pixel2mm(revisit_session.screen_height, revisit_ppi)
    revisit_font_size = convert_font_height_px2mm(pair_result["assertion"]["fontsize"]["revisit_font_size"])

    sample = np.array([prime_font_size, prime_screen_width, prime_screen_height,
                       prime_ppi, prime_bound[0], prime_bound[1],
                       revisit_font_size, revisit_screen_width, revisit_screen_height,
                       revisit_ppi, revisit_bound[0], revisit_bound[1]])

    sample = np.reshape(sample, (-1, 12))
    prediction = model.predict(sample)
    status = 'normal'
    if prediction != [0]:
      status = 'abnormal'
    result = {
      "prime_xpath": pair_result["prime_xpath"],
      "prime_text": pair_result["prime_text"] if pair_result["prime_text"] != '' else pair_result['prime_ocr_text'],
      "prime_font_size": pair_result["assertion"]["fontsize"]["prime_font_size"],
      "prime_font_size_mm": prime_font_size,
      "revisit_xpath": pair_result["revisit_xpath"],
      "revisit_text": pair_result["revisit_text"] if pair_result["revisit_text"] != '' else pair_result['revisit_ocr_text'],
      "revisit_font_size": pair_result["assertion"]["fontsize"]["revisit_font_size"],
      "revisit_font_size_mm": revisit_font_size,
      "status": status
    }
    return result


  @staticmethod
  def extract_pair_result(prime_element, revisit_element, prime_ocr_text, revisit_ocr_text, fontsize_status):
    pair_result = {
      "prime_xpath": prime_element.xpath if prime_element else None,
      "prime_text": prime_element.text if prime_element.text != '' else '',
      "prime_ocr_text": prime_ocr_text if prime_ocr_text != '' else '',
      "prime_bound": prime_element.bound if prime_element else None,
      "revisit_xpath": revisit_element.xpath if revisit_element else None,
      "revisit_text": revisit_element.text if revisit_element.text != '' else '',
      "revisit_ocr_text": revisit_ocr_text if revisit_ocr_text != '' else '',
      "revisit_bound": revisit_element.bound if revisit_element else None,
      "assertion": {
        "fontsize": fontsize_status
      }
    }
    return pair_result

  def log(self, content, **kwargs):
    self.logger.info(content, extras={'session_id': self.session_id, 'action_id': self.action_id,
                                      'request_id': self.request_id, **kwargs})