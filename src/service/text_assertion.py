import time
import service.constants as constants
from service.ai_components import AIComponent
from service.element import Element
from service.element_finder import ElementFinder
from service.google_ocr import GoogleOCR
from service.kobiton_ocr import KobitonOCR
from service.sessions import PrimeSession, RevisitSession
from service.luna import match_elements as luna

from service.utils import ascii_normalize, remove_icon_chars, get_text_color, \
  is_in_scroll_view, is_google_vision_enabled, rgb2hex, is_empty_string, retry_call_n_times, right_strip_text, \
  random_color, draw_debug_info, text_similarity, compare_colors_in_lab, compare_colors_in_rgb, \
  save_screen_debug, partially_beyond_viewport, extract_ocr_text_by_bound_iou


class TextAssertionElement(Element):
  def __init__(self, xpath, session):
    super().__init__(xpath, session)
    self.ocr_text = self.get_ocr_text()
    self.lab_text_color, self.text_color = get_text_color(self.img)

  def set_ocr_text(self, text):
    if self.expect_str is None:
      text = ascii_normalize(text.strip())
      if self.text != '':
        text = remove_icon_chars(text, self.text)
      self.ocr_text = text

  def get_ocr_text(self):
    if self.expect_str is not None:
      return self.expect_str
    ocr_text, _ = extract_ocr_text_by_bound_iou(self.bound, self.sess.bounds, self.sess.ocr_texts)
    if ocr_text:
      ocr_text = ascii_normalize(ocr_text.strip())
      ocr_text = remove_icon_chars(ocr_text, self.text)
    return ocr_text

  def set_text_color(self, text_color):
    self.lab_text_color, self.text_color = text_color

  @property
  def is_in_scroll_view(self):
    return is_in_scroll_view(self.xpath, self.sess.xml_tree, self.sess.platform)

  @property
  def expect_str(self):
    if type(self.sess) is TextAssertPrimeSession:
      for submission in self.sess.req.expect_text_submissions:
        if self.xpath == submission.prime_xpath:
          return submission.expect_str
    return None


class TextAssertPrimeSession(PrimeSession):
  def __init__(self, req):
    super().__init__(req)
    self.threshold = req.threshold
    self.text_submission_xpaths = [submission.prime_xpath
                                   for submission in req.submissions]
    self.text_color_submission_xpaths = [submission.prime_xpath
                                         for submission in req.color_submissions]
    self.expect_text_submission_xpaths = [submission.prime_xpath
                                          for submission in req.expect_text_submissions]
    self.element = None

  @property
  def text_assert_level(self):
    upper_level = self.req.level.upper()
    if self.element:
      if self.element.xpath in self.text_submission_xpaths:
        idx = self.text_submission_xpaths.index(self.element.xpath)
        upper_level = self.req.submissions[idx].level.upper()
        self.threshold = self.req.submissions[idx].threshold \
          if self.req.submissions[idx].threshold != 0.0 \
          else self.threshold
      elif self.element.xpath in self.expect_text_submission_xpaths:
        idx = self.expect_text_submission_xpaths.index(self.element.xpath)
        upper_level = self.req.expect_text_submissions[idx].level.upper()
        self.threshold = self.req.expect_text_submissions[idx].threshold \
          if self.req.expect_text_submissions[idx].threshold != 0.0 \
          else self.threshold

    if upper_level == constants.TextAssertion.SKIP.name:
      return constants.TextAssertion.SKIP
    if upper_level == constants.TextAssertion.EXACT.name:
      return constants.TextAssertion.EXACT
    if upper_level == constants.TextAssertion.BEGINNING_OF_TEXT.name:
      return constants.TextAssertion.BEGINNING_OF_TEXT
    if upper_level == constants.TextAssertion.CONTAIN.name:
      return constants.TextAssertion.CONTAIN
    return constants.TextAssertion.RELAXED_PLACEMENT

  @property
  def color_assert_level(self):
    upper_level = self.req.color_level.upper()
    if self.element and self.element.xpath in self.text_color_submission_xpaths:
      idx = self.text_color_submission_xpaths.index(self.element.xpath)
      upper_level = self.req.color_submissions[idx].level.upper()

    if upper_level == constants.ColorTextAssertion.SKIP.name:
      return constants.ColorTextAssertion.SKIP
    if upper_level == constants.ColorTextAssertion.STRICT.name:
      return constants.ColorTextAssertion.STRICT
    return constants.ColorTextAssertion.RELAXED

  @property
  def textual_elements(self):
    results = []
    if self.req.prime_element_xpaths:
      element_xpaths = self.req.prime_element_xpaths
      elements = [TextAssertionElement(xpath, self) for xpath in element_xpaths]
      elements = [element for element in elements if element.is_visual()]

      for element in elements:
        if not partially_beyond_viewport(element.bound, self.screen_img,
                                         element.ocr_text, element.text, self.platform):
          results.append(element)
    else:
      results = [TextAssertionElement(e.xpath, self) for e in self.get_textual_elements()]
    return results

  @property
  def element(self):
    return self._element

  @element.setter
  def element(self, ele: Element):
    self._element = ele

class PostProcessText:
  """
  Post-process for prime and revisit text element before assertion
  ...

  Attributes
  ----------
  logger : LoggingWrapper

  Methods
  -------
  is_qualify_text(prime_text, revisit_text, min_len)
    Return is text element have less than min_len characters

  is_text_in_keyboard(prime, revisit_element)
    Return either prime or revisit element is in keyboard 

  is_text_in_image(prime_element, revisit_element)
    Return either prime or revisit element is in an image

  modify_low_confident_text(prime_text, revisit_text, prime_element, revisit_element)
    Return prime-text and revisit-text when CTC confident don't meet threshold

  run(prime_text, revisit_text, prime_element, revisit_element)
    Return post processed texts
  
  """
  def __init__(self, logger):
    self.logger = logger

  def is_qualify_text(self, prime_text, revisit_text, min_len = 2, session_id = 1, action_id = 0, request_id = 1,):
    """ 
    Return is text element have less than min_len characters
    Args:
      prime_text (str): ocr text from prime
      revisit_text (str):  ocr text from revisit

    Returns:
      bool
    """
    if len(prime_text) > min_len and len(revisit_text) > min_len:
      return True
    else:
      self.logger.info("Text element has {} characters or less, skipping text assertion".format(min_len),
                    extras={
                            'session_id': session_id, 'action_id': action_id,
                            'request_id': request_id})
      return False

  def is_text_in_keyboard(self, prime, revisit_element, session_id, action_id, request_id):
    """
    Return either prime or revisit element is in keyboard 
    Args:
        prime (TextAssertPrimeSession): prime session
        revisit_element (Element): revisit element

    Returns:
        bool
    """
    if prime.element.is_keyboard(prime.platform) and revisit_element.is_keyboard(prime.platform):
      self.logger.info("Text element is in a keyboard, skipping text assertion",
                    extras={
                            'session_id': session_id, 'action_id': action_id,
                            'request_id': request_id})
      return True
    else:
      return False  

  def is_text_in_image(self, prime_element, revisit_element, session_id, action_id, request_id):
    """
    Return either prime or revisit element is in an image
    Args:
        prime_element (Element): prime element
        revisit_element (Element): revisit element

    Returns:
        bool
    """
    if is_empty_string(prime_element.text) or is_empty_string(revisit_element.text):
      self.logger.info("Text element is an image, skipping text assertion",
                    extras={
                            'session_id': session_id, 'action_id': action_id,
                            'request_id': request_id})
      return True
    else:
      return False  
      
  def modify_low_confident_text(self, prime_text, revisit_text, prime_element, revisit_element): 
    """
    Use Tessaract as a fall-back option when CTC confidence is inaccurate and fail to meet threshold
    Args:
        prime_text (str): ocr text of Prime extracted by ocr-service or ggvision that is None due to low confident filter
        revisit_text (str): ocr text of Revisit extracted by ocr-service or ggvision that is None due to low confident filter
        prime_element (Element): prime element
        revisit_element (Element): revisit element

    Returns:
        str, str
    """
    if is_empty_string(prime_text) and not is_empty_string(prime_element.ocr_text):
      prime_text = prime_element.ocr_text # ocr_text is Tessaract response
    if is_empty_string(revisit_text) and not is_empty_string(revisit_element.ocr_text):
      revisit_text = revisit_element.ocr_text # ocr_text is Tessaract response
    return prime_text, revisit_text

  def run(self, prime_text, revisit_text, prime_element, revisit_element):
    """
    Apply post processes
    Args:
        prime_text (str):  ocr text from prime
        revisit_text (str):  ocr text from revisit
        prime_element (Element): prime element
        revisit_element (Element): revisit element

    Returns:
        str, str
    """
    prime_text, revisit_text = self.modify_low_confident_text(prime_text, revisit_text, prime_element, revisit_element)

    prime_text = ascii_normalize(prime_text)
    revisit_text = ascii_normalize(revisit_text)
    
    prime_text = remove_icon_chars(prime_text, prime_element.text)
    revisit_text = remove_icon_chars(revisit_text, revisit_element.text)

    prime_text = prime_text.lower()
    revisit_text = revisit_text.lower()
    
    return prime_text, revisit_text


class TextAssertion(AIComponent):
  def __init__(self, services, logger, debug=False):
    super().__init__(services, logger=logger)
    self.finder = ElementFinder(services, logger=logger)
    self.debug = debug
    self.services = services
    if services[constants.KOBITON_OCR_SERVICE] != 'None:None':
      self.use_kobiton_ocr = True
    else:
      self.use_kobiton_ocr = False
    self.post_process = PostProcessText(self.logger)


  def __call__(self, prime: TextAssertPrimeSession, revisit: RevisitSession,
               session_id: int = 0, action_id: int = 0, request_id: int = 0):
    asserted_elements = []
    start = time.time()
    self.log_start_assertion(prime,
                             session_id=session_id, action_id=action_id, request_id=request_id)

    text_element_pairs = self.find_pairs_of_text_elements(prime, revisit,
                                                          session_id=session_id,
                                                          action_id=action_id,
                                                          request_id=request_id,
                                                          using_segment=False)

    if text_element_pairs:
      self.log_use_in_assertion(text_element_pairs, session_id=session_id, action_id=action_id, request_id=request_id)
      self.text_assertion_extractions(prime, revisit, asserted_elements, text_element_pairs, session_id=session_id,
                                      action_id=action_id, request_id=request_id)
    if self.debug:
      self.debug_mode(prime, revisit, asserted_elements, request_id)

    self.logger.info(
      'Finish a text assertion request', extras={
        'session_id': session_id, 'action_id': action_id,
        'request_id': request_id, 'text_elements': asserted_elements, 'processed_time': time.time() - start})
        
    return asserted_elements

  def text_assertion_extractions(self, prime, revisit, text_elements, text_element_pairs,
                             session_id=0, action_id=0, request_id=0):
    assert_elements = sum([[item[0], item[1]] for item in text_element_pairs], [])

    if self.use_kobiton_ocr:
      ocr_texts, ocr_bounds = [prime.ocr_texts, revisit.ocr_texts], [prime.bounds, revisit.bounds]
    elif is_google_vision_enabled():
      imgs = [prime.screen_img, revisit.screen_img]
      ocr_texts, ocr_bounds, _ = self.__extract_google_texts_in_imgs(imgs, session_id=session_id,
                                                                      action_id=action_id,
                                                                      request_id=request_id)

    for i in range(0, len(assert_elements), 2):
      prime_element = assert_elements[i]
      prime.element = prime_element
      revisit_element = assert_elements[i + 1]
      find_score = text_element_pairs[int(i / 2)][2]
      if not self.post_process.is_text_in_keyboard(prime, revisit_element, session_id=session_id, action_id=action_id, request_id=request_id) and not self.post_process.is_text_in_image(prime_element,revisit_element, session_id=session_id, action_id=action_id, request_id=request_id):
        if is_google_vision_enabled():
          prime_text, _ = extract_ocr_text_by_bound_iou(prime_element.bound, ocr_bounds[0], ocr_texts[0], 0.8)
          revisit_text, _ = extract_ocr_text_by_bound_iou(revisit_element.bound, ocr_bounds[1], ocr_texts[1], 0.8)
        elif self.use_kobiton_ocr:
          prime_text, _ = extract_ocr_text_by_bound_iou(prime_element.bound, ocr_bounds[0], ocr_texts[0], 0.65)
          revisit_text, _ = extract_ocr_text_by_bound_iou(revisit_element.bound, ocr_bounds[1], ocr_texts[1], 0.65)
        else: # Using Tessaract ocr 
          prime_text = prime_element.ocr_text
          revisit_text = revisit_element.ocr_text
        if is_empty_string(prime_text) and is_empty_string(revisit_text):
          continue

        prime_text, revisit_text = self.post_process.run(prime_text, revisit_text, prime_element, revisit_element)

        if self.post_process.is_qualify_text(prime_text, revisit_text, session_id=session_id, action_id=action_id, request_id=request_id):
          if prime_element.expect_str is not None:
            prime_text = prime_element.expect_str

          text_assertion = self.__textual_assertions(prime_text,
                                                    revisit_text,
                                                    prime.text_assert_level,
                                                    threshold=prime.threshold)

          color_assert_level = prime.color_assert_level if revisit_text else constants.ColorTextAssertion.SKIP
          color_assertion = self.__textual_color_assertion((prime_element.lab_text_color, prime_element.text_color),
                                                          (revisit_element.lab_text_color, revisit_element.text_color),
                                                          color_assert_level)
          element = {
            'debug_idx': i,
            'prime_xpath': prime_element.xpath,
            'prime_ocr': prime_text,
            'prime_text': prime_element.text,
            'is_in_scrollable': prime_element.is_in_scroll_view,
            'prime_color': rgb2hex(prime_element.text_color) if prime_element.text_color is not None else '',
            'revisit_xpath': revisit_element.xpath,
            'revisit_ocr': revisit_text,
            'revisit_text': revisit_element.text,
            'revisit_color': rgb2hex(revisit_element.text_color) if revisit_element.text_color is not None else '',
            'visual_matching_score': find_score,
            'threshold': prime.threshold,
            'assertion': {
              'ocr': text_assertion,
              'color': color_assertion
            }
          }
          text_elements.append(element)

  def find_element(self, prime: TextAssertPrimeSession, revisit: RevisitSession, threshold=0.5, session_id=0,
                   action_id=0, request_id=0):
    element_xpaths, sims = self.finder(prime, revisit,
                                       session_id=session_id,
                                       action_id=action_id,
                                       request_id=request_id)

    if sims and len(sims) > 0 and sims[-1] > threshold:
      element = TextAssertionElement(element_xpaths[-1], revisit)
      if not element.is_overlapped_by_keyboard():
        return element, sims[-1]
      self.log_skip_overlapped_keyboard(prime.element.xpath, element_xpaths[-1],
                                        session_id=session_id,
                                        action_id=action_id,
                                        request_id=request_id)
    return None, None

  def find_pairs_of_text_elements(self, prime: TextAssertPrimeSession, revisit: RevisitSession,
                                  session_id=0, action_id=0, request_id=0, using_segment=False):
    if using_segment:
      sessions_elements = self._try_get_visual_bounds_by_segmentation([prime.screen_img, revisit.screen_img])
      prime_elements = sessions_elements[0]
      prime_elements = [value[1] for value in prime_elements]
      prime.set_visual_elements_by_bound(prime_elements)
      revisit_elements = sessions_elements[1]
      revisit_elements = [value[1] for value in revisit_elements]
      revisit.set_visual_elements_by_bound(revisit_elements)
      matching_pairs = luna.match_screen_pair_by_segmentation(prime, revisit,
                                                              element_type=constants.ElementType.TEXTUAL)
    else:
      matching_pairs = luna.match_screen_pair(prime, revisit, element_type=constants.ElementType.TEXTUAL)

    element_pairs = []
    for prime_element_xpath, match_item in matching_pairs.items():
      prime_text_element = TextAssertionElement(prime_element_xpath, prime)
      prime.element = prime_text_element
      revisit_element = TextAssertionElement(match_item['match'], revisit)
      find_score = match_item['similarity']

      if match_item['ocr_text'] != prime_text_element.ocr_text:
        prime_text_element.set_ocr_text(match_item['ocr_text'])
      if 'text_color' in match_item and match_item['text_color'] is not None:
        prime_text_element.set_text_color(match_item['text_color'])
      else:
        prime_text_element.set_text_color((None, None))
      if match_item['match_ocr_text'] != revisit_element.ocr_text:
        revisit_element.set_ocr_text(match_item['match_ocr_text'])
      if 'match_text_color' in match_item and match_item['match_text_color'] is not None:
        revisit_element.set_text_color(match_item['match_text_color'])
      else:
        revisit_element.set_text_color((None, None))

      if partially_beyond_viewport(revisit_element.bound, revisit.screen_img,
                                   revisit_element.ocr_text, revisit_element.text,
                                   prime.platform):
        self.log_skip_beyond_view_element(prime, revisit_element, idx,
                                          session_id=session_id,
                                          action_id=action_id,
                                          request_id=request_id)
        continue

      if (not is_google_vision_enabled() or not self.use_kobiton_ocr) and is_empty_string(prime_text_element.ocr_text) and is_empty_string(
              revisit_element.ocr_text):
        continue

      element_pairs.append([prime_text_element, revisit_element, find_score])
    return element_pairs

  def __extract_google_texts_in_imgs(self, imgs, session_id=0, action_id=0, request_id=0):
    self.logger.info("Extracting OCR texts using Google OCR")
    google_ocr = GoogleOCR()
    logger_params_gg = {
      'log': "Error on getting OCR text",
      'extras': {
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id
      }
    }
    texts = retry_call_n_times(google_ocr.detect_texts_in_images, self.logger, logger_params_gg, imgs)
    return texts

  @staticmethod
  def __textual_assertions(prime_ocr, revisit_ocr, level, threshold=0.8):
    passed = constants.TextAssertionStatus.PASSED.name
    failed = constants.TextAssertionStatus.FAILED.name
    status = passed

    prime_ocr = right_strip_text(prime_ocr)
    revisit_ocr = right_strip_text(revisit_ocr)

    def get_response(**args):
      return {
        "level": level.name,
        "status": status,
        **args
      }

    if level == constants.TextAssertion.SKIP:
      status = passed
      return get_response()
    if level == constants.TextAssertion.EXACT:
      if prime_ocr != revisit_ocr:
        status = failed
      return get_response()
    if level == constants.TextAssertion.CONTAIN:
      if prime_ocr not in revisit_ocr:
        status = failed
      return get_response()
    if level == constants.TextAssertion.BEGINNING_OF_TEXT:
      if (prime_ocr and not revisit_ocr) or (not prime_ocr and revisit_ocr):
        status = failed
      if not prime_ocr.startswith(revisit_ocr) and not revisit_ocr.startswith(prime_ocr):
        status = failed
      return get_response()
    if level == constants.TextAssertion.RELAXED_PLACEMENT:
      sim = text_similarity(prime_ocr, revisit_ocr)
      if sim < threshold:
        status = failed
      if status == failed and \
              prime_ocr and revisit_ocr and \
              (prime_ocr.startswith(revisit_ocr) or revisit_ocr.startswith(prime_ocr)):
        status = passed
      return get_response(similarity=sim)

  @staticmethod
  def __textual_color_assertion(prime_text_color, rev_text_color, mode):
    color_distance = 0.
    if prime_text_color[0] is not None and rev_text_color[0] is not None:
      color_distance = min(compare_colors_in_lab(prime_text_color[0], rev_text_color[0]),
                           compare_colors_in_rgb(prime_text_color[1], rev_text_color[1]))

    status = constants.TextAssertionStatus.PASSED.name

    def get_color_response():
      return {
        'mode': mode.name.upper(),
        'color_distance': color_distance,
        'status': status
      }

    if mode == constants.ColorTextAssertion.SKIP:
      return get_color_response()
    if mode == constants.ColorTextAssertion.STRICT:
      if color_distance > constants.ColorTextAssertion.STRICT.value:
        status = constants.TextAssertionStatus.FAILED.name
    elif mode == constants.ColorTextAssertion.RELAXED:
      if color_distance > constants.ColorTextAssertion.RELAXED.value:
        status = constants.TextAssertionStatus.FAILED.name
    return get_color_response()

  def log_start_assertion(self, prime: TextAssertPrimeSession, session_id: int = 0,
                          action_id: int = 0, request_id: int = 0):
    self.logger.info('Start text_assert', extras={'session_id': session_id,'action_id': action_id,
                                                  'request_id': request_id,
                                                  'level': prime.text_assert_level.name})

  def log_warn_no_candidate(self, prime: TextAssertPrimeSession, idx, session_id: int = 0,
                            action_id: int = 0, request_id: int = 0):
    self.logger.warning(
      'Cannot found a text element for assertion', extras={
        'debug_idx': idx,
        'prime_text_xpath': prime.element.xpath,
        'session_id': session_id, 'action_id': action_id,
        'request_id': request_id})

  def log_skip_beyond_view_element(self, prime: TextAssertPrimeSession, revisit_element: TextAssertionElement, idx,
                                   session_id: int = 0, action_id: int = 0, request_id: int = 0):
    self.logger.warning(
      'Ignore text-assertion for this prime element due '
      'to the revisit element has a part going beyond the screen',
      extras={
        'debug_idx': idx,
        'prime_text_xpath': prime.element.xpath,
        'revisit_text_xpath': revisit_element.xpath,
        'session_id': session_id, 'action_id': action_id,
        'request_id': request_id})

  def log_use_in_assertion(self, element_pairs, session_id: int = 0, action_id: int = 0, request_id: int = 0):
    prime_xpaths = []
    for (prime_element, revisit_element, find_score) in element_pairs:
      prime_xpaths.append(prime_element.xpath)

    self.logger.warning(
      "Try to extract OCR text of prime and revisit elements", extras={
        'prime_xpaths': prime_xpaths,
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
      })

  def log_skip_overlapped_keyboard(self, prime_xpath, revisit_xpath, session_id: int = 0,
                                   action_id: int = 0, request_id: int = 0):
    self.logger.warning(
      "Ignore text-assertion for this pair due to revisit element is overlapped by keyboard",
      extras={
        'prime_xpath': prime_xpath,
        'revisit_xpath': revisit_xpath,
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id})

  @staticmethod
  def debug_mode(prime: TextAssertPrimeSession, revisit: RevisitSession, text_elements, request_id: int = 0):
    prime_debug_screen = prime.screen_img.copy()
    revisit_debug_screen = revisit.screen_img.copy()
    for idx, pair in enumerate(text_elements):
      draw_color = random_color()
      prime_element = TextAssertionElement(pair['prime_xpath'], prime)
      revisit_element = TextAssertionElement(pair['revisit_xpath'], revisit)
      draw_debug_info(prime_element.bound, prime_debug_screen, '', draw_color, idx)
      draw_debug_info(revisit_element.bound, revisit_debug_screen, str(pair['assertion']), draw_color, idx)
    save_screen_debug(prime_debug_screen, revisit_debug_screen, request_id, prime.text_assert_level)
