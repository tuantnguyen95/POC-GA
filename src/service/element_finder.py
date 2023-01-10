import time
import traceback
from typing import List
import numpy as np

from service import utils as u
from service import xml_utils, constants
from service.sessions import PrimeSession, RevisitSession
from service.element import Element
from service.ai_components import AIComponent
from service.luna import match_elements as luna

class ElementFinder(AIComponent):
  """
  Find an element of an app in 2 different devices
  based on Machine Learning method.
  """

  def __init__(self, services, logger):
    super(ElementFinder, self).__init__(services, logger=logger)

  def __call__(self, prime: PrimeSession, revisit: RevisitSession,
               session_id: int = 0, action_id: int = 0, request_id: int = 0, using_segment: bool = False):
    if using_segment:
      solver = ElementFinderBySegmentation(prime, revisit, self.logger,
                                   self._try_get_visual_bounds_by_segmentation,
                                   session_id=session_id,
                                   action_id=action_id,
                                   request_id=request_id)
    else:
      solver = ElementFinderByXML(prime, revisit, self.logger,
                                 self._try_get_visual_embeddings,
                                 session_id=session_id,
                                 action_id=action_id,
                                 request_id=request_id,
                                 classifier_model=self.sk_cls)
    return solver.get_response()
class ElementFinderSolver:
  def __init__(self, prime: PrimeSession, revisit: RevisitSession, logger, func,
               session_id: int = 0, action_id: int = 0, request_id: int = 0):
    self.prime = prime
    self.revisit = revisit
    self.logger = logger
    self.session_id = session_id
    self.action_id = action_id
    self.request_id = request_id
    self.func = func

  def get_response(self):
    start = time.time()
    self.logger.debug("[get_revisit_element] sessions info", extras={
      'request_id': self.request_id,
      'session_id': self.session_id,
      'action_id': self.action_id,
      **self.prime.instance_log,
      **self.revisit.instance_log
    })

    self.logger.debug('[get_revisit_element] prime element info', extras={
      'is_scrollable': self.prime.element.is_scrollable(),
      'session_id': self.session_id, 'action_id': self.action_id, 'request_id': self.request_id,
      'prime_xpath': self.prime.element.xpath, 'ocr_text': self.prime.element.ocr_text,
      'prime_ele_bound': self.prime.element.bound, 'prime_text': self.prime.element.text,
      'prime_recur_text': self.prime.element.recursive_text})

    cand_xpaths, cand_sims = self.find_element()
    self.logger.debug('Candidates returned by AI service %s' % (self.__class__.__name__),
                    extras={'candidates': [x for x in zip(cand_xpaths, cand_sims)],
                            'session_id': self.session_id, 'action_id': self.action_id,
                            'request_id': self.request_id, 'processed_time': time.time() - start})
    return cand_xpaths, cand_sims

  def find_element(self):
    pass



"""
Finding element using FaceNet and Luna models (these approaches require XML file)
"""
class ElementFinderByXML(ElementFinderSolver):
  def __init__(self, prime: PrimeSession, revisit: RevisitSession, logger, func,
               classname_threshold=0.71, xpath_threshold=0.3, depth_xpath_threshold=10,
               session_id: int = 0, action_id: int = 0, request_id: int = 0,
               get_visual_embeddings_func=None, classifier_model=None):
    super(ElementFinderByXML, self).__init__(prime, revisit, logger, func, session_id, action_id, request_id)
    self.classname_threshold = classname_threshold
    self.xpath_threshold = xpath_threshold
    self.depth_xpath_threshold = depth_xpath_threshold
    self.classifier_model = classifier_model

  def find_element(self):
    candidate_xpaths_luna, candidate_confidence_luna, priority_luna = self.find_element_by_luna()
    candidate_xpaths_facenet, candidate_confidence_facenet = [], []
    if not priority_luna or (len(candidate_confidence_luna) > 0 and
                             candidate_confidence_luna[-1] < constants.LUNA_FINDING_ELEMENT_CONFIDENCE_THRESHOLD):
      candidates_facenet = self._filter_revisit_candidates()
      if len(candidates_facenet) == 0:
        candidates_facenet = [Element(candidate_xpath, self.revisit) for candidate_xpath in candidate_xpaths_luna]
      candidate_xpaths_facenet, candidate_confidence_facenet = self.find_element_by_facenet(candidates=candidates_facenet)

    if self.prime.element.is_scrollable():
      return self.merge_predictions_scrollable(candidate_xpaths_facenet, candidate_confidence_facenet, candidate_xpaths_luna, candidate_confidence_luna)

    if self.prime.element.is_keyboard(self.prime.platform):
      prime_keyboard_elements = xml_utils.get_keyboard_elements(self.prime)
      revisit_keyboard_elements = xml_utils.get_keyboard_elements(self.revisit)
      line_index_of_prime_element = u.get_lines_in_keyboard(prime_keyboard_elements).index(self.prime.element.bound[3])

      if line_index_of_prime_element <= len(u.get_lines_in_keyboard(revisit_keyboard_elements)):
        candidates_facenet = u.get_elements_in_line_of_keyboard(line_index_of_prime_element, revisit_keyboard_elements)
        candidate_xpaths_facenet, candidate_confidence_facenet = self.find_element_by_facenet(candidates=candidates_facenet)
        index_of_revisit_element_in_line = next((i for i, item in enumerate(candidates_facenet) if item.xpath == candidate_xpaths_facenet[-1]))
        line_of_prime_element = u.get_elements_in_line_of_keyboard(line_index_of_prime_element, prime_keyboard_elements)
        if candidate_confidence_facenet[-1] < constants.FINDING_ELEMENT_CONFIDENCE_THRESHOLD \
          and len(line_of_prime_element) == len(candidates_facenet) \
          and u.get_index_of_elements_in_line(self.prime.element, line_of_prime_element) == \
                index_of_revisit_element_in_line:
            candidate_confidence_facenet[-1] = constants.FINDING_ELEMENT_CONFIDENCE_THRESHOLD
        return candidate_xpaths_facenet[-5:], candidate_confidence_facenet[-5:]

    candidate_xpaths, candidate_confidence = self.merge_predictions(candidate_xpaths_facenet,
                                                                    candidate_confidence_facenet, candidate_xpaths_luna,
                                                                    candidate_confidence_luna, priority_luna)
    return candidate_xpaths[-5:], candidate_confidence[-5:]

  def merge_predictions(self, candidate_xpaths_facenet, candidate_confidence_facenet, candidate_xpaths_luna, candidate_confidence_luna, priority_luna = True):
    if priority_luna and candidate_confidence_luna and candidate_confidence_luna[-1] >= constants.LUNA_FINDING_ELEMENT_CONFIDENCE_THRESHOLD:
      candidate_confidence_luna[-1] = max(constants.FINDING_ELEMENT_CONFIDENCE_THRESHOLD_SCROLL, candidate_confidence_luna[-1])
      return candidate_xpaths_luna, candidate_confidence_luna

    # Merge
    # Case similarity too small - use Facenet to confirm
    if candidate_xpaths_luna and candidate_xpaths_facenet and \
        (candidate_confidence_facenet[-1] >= constants.FACENET_FINDING_ELEMENT_CONFIDENCE_THRESHOLD or priority_luna):
      element_luna = Element(candidate_xpaths_luna[-1], self.revisit)
      center_luna = u.get_center_point_of_bound(element_luna.bound)
      element_facenet = Element(candidate_xpaths_facenet[-1], self.revisit)
      center_facenet = u.get_center_point_of_bound(element_facenet.bound)
      if candidate_xpaths_luna[-1] == candidate_xpaths_facenet[-1] or \
        (u.is_point_inside_bound(center_luna, element_facenet.bound) and \
        u.is_point_inside_bound(center_facenet, element_luna.bound)):
        return [candidate_xpaths_facenet[-1]], \
          [np.max([constants.FINDING_ELEMENT_CONFIDENCE_THRESHOLD, candidate_confidence_luna[-1], candidate_confidence_facenet[-1]])]

    # Case Luna fails to find the element - Use Facenet to find
    if not priority_luna:
      candidate_xpaths = candidate_xpaths_facenet
      candidate_confidence = candidate_confidence_facenet
      for xpath, sim in zip(candidate_xpaths_luna, candidate_confidence_luna):
        if xpath in candidate_xpaths:
          ind = candidate_xpaths.index(xpath)
          candidate_confidence[ind] = max(candidate_confidence[ind], sim)
        else:
          candidate_xpaths.append(xpath)
          candidate_confidence.append(sim)
      sort_idx = np.argsort(candidate_confidence)
      return np.array(candidate_xpaths)[sort_idx].tolist(), np.array(candidate_confidence)[sort_idx].tolist()

    # Case not match is right
    return candidate_xpaths_luna[-1:], candidate_confidence_luna[-1:]

  def merge_predictions_scrollable(self, candidate_xpaths_facenet, candidate_confidence_facenet, candidate_xpaths_luna, candidate_confidence_luna):
    candidate_xpaths = candidate_xpaths_facenet
    candidate_confidence = candidate_confidence_facenet
    for xpath, sim in zip(candidate_xpaths_luna, candidate_confidence_luna):
      if xpath in candidate_xpaths:
        ind = candidate_xpaths.index(xpath)
        candidate_confidence[ind] = max(candidate_confidence[ind], sim)
      else:
        candidate_xpaths.append(xpath)
        candidate_confidence.append(sim)
    sort_idx = np.argsort(candidate_confidence)[-5:]
    return np.array(candidate_xpaths)[sort_idx].tolist(), np.array(candidate_confidence)[sort_idx].tolist()

  def find_element_by_facenet(self, candidates=None):
    start = time.time()
    if candidates is None:
      candidates = self._filter_revisit_candidates()
    if len(candidates) == 0:
      self.logger.debug('No candidate found after filter',
                        extras={'is_scrollable': self.prime.element.is_scrollable(),
                                'session_id': self.session_id,
                                'action_id': self.action_id,
                                'request_id': self.request_id})
      return [], []

    self.logger.debug('There are %d candidates after filter' % (len(candidates),), extras={
      'is_scrollable': self.prime.element.is_scrollable(),
      'session_id': self.session_id,
      'action_id': self.action_id,
      'request_id': self.request_id})

    # Calculate similarities between prime and candidates
    similarities = self._extract_similarities(candidates)

    # Run multi-modal classifier
    res_confidence, res_xpaths = self._classify(candidates, similarities)
    self.logger.debug('Candidates returned by Facenet for prime xpath %s' % (self.prime.element.xpath),
                    extras={'candidates': [x for x in zip(res_xpaths, res_confidence)],
                            'session_id': self.session_id, 'action_id': self.action_id,
                            'request_id': self.request_id, 'processed_time': time.time() - start})
    return res_xpaths, res_confidence

  def _classify(self, candidates, similarities):
    start = time.time()
    res_xpaths, res_confidence = [], []
    for idx, (visual_sim,
              horizontal_sim, horizontal_small_sim,
              vertical_sim, vertical_small_sim,
              ocr_sim, xpath_sim,
              classname_sim, id_sim, text_sim, recur_text_sim) in enumerate(zip(*similarities)):
      sk_input = [visual_sim, horizontal_sim, horizontal_small_sim,
                  vertical_sim, vertical_small_sim, ocr_sim, xpath_sim,
                  classname_sim, id_sim, text_sim, recur_text_sim]
      if not self.prime.element.is_scrollable():
        sk_input = np.array(sk_input).reshape(1, -1)
        pre_ = self.classifier_model.predict(sk_input)[0]

      madeup_conf = u.overall_similarity(
        visual_sim,
        horizontal_sim, horizontal_small_sim,
        vertical_sim, vertical_small_sim,
        ocr_sim, xpath_sim,
        classname_sim, id_sim,
        text_sim, recur_text_sim, self.prime.platform, self.revisit.platform,
        scrollable_element=self.prime.element.is_scrollable())

      if self.prime.element.is_scrollable():
        res_confidence.append(madeup_conf)
        pre_ = -2
      elif pre_ == 1:
        res_confidence.append((1 + madeup_conf) / 2)
      else:
        res_confidence.append(madeup_conf)
      res_xpaths.append(candidates[idx].xpath)
      candidates[idx].log_obj.update(
        {'visual_sim': visual_sim, 'horizontal_sim': horizontal_sim,
         'horizontal_small_sim': horizontal_small_sim,
         'vertical_small_sim': vertical_small_sim,
         'vertical_sim': vertical_sim, 'ocr_sim': ocr_sim,
         'xpath_sim': xpath_sim, 'classname_sim': classname_sim,
         'recur_text_sim': recur_text_sim, 'text_sim': text_sim,
         'overall': res_confidence[-1], 'classified': int(pre_)})

    self.logger.debug('Finish finding element',
                      extras={'is_scrollable': self.prime.element.is_scrollable(),
                              'session_id': self.session_id,
                              'action_id': self.action_id,
                              'request_id': self.request_id,
                              'prime_xpath': self.prime.element.xpath,
                              'cand_len': len(res_xpaths),
                              'candidates': [candidate.log_obj for candidate in candidates]})
    self.logger.debug(
      "Considering %s candidates took %ss" % (len(res_confidence), time.time() - start),
      extras={
        'session_id': self.session_id, 'action_id': self.action_id,
        'request_id': self.request_id, 'prime_xpath': self.prime.element.xpath})
    res_confidence, res_xpaths = zip(*sorted(zip(res_confidence, res_xpaths)))
    return list(res_confidence), list(res_xpaths)

  def _extract_similarities(self, candidates: List[Element]):

    element_imgs = [self.prime.element.tf_img] + [candidate.tf_img for candidate in candidates]
    v_ele_imgs = [self.prime.element.vertical_tf_img] + [candidate.vertical_tf_img for candidate in candidates]
    vs_ele_imgs = [self.prime.element.vertical_small_tf_img] + \
                  [candidate.vertical_small_tf_img for candidate in candidates]
    h_ele_imgs = [self.prime.element.horizontal_tf_img] + \
                 [candidate.horizontal_tf_img for candidate in candidates]
    hs_ele_imgs = [self.prime.element.horizontal_small_tf_img] + \
                  [candidate.horizontal_small_tf_img for candidate in candidates]

    if self.prime.element.is_scrollable():
      zeros = [0 for _ in element_imgs[:-1]]
      hs_visual_sims = zeros
      vs_visual_sims = zeros
      ocr_sims = zeros
      id_sims = zeros
      text_sims = zeros
      recur_text_sims = zeros
      start_embedding = time.time()
      imgs = element_imgs + h_ele_imgs + v_ele_imgs
      imgs = np.stack(imgs)
      ele_visual_embs = self.func(imgs,
                                session_id=self.session_id,
                                action_id=self.action_id,
                                request_id=self.request_id)
      if ele_visual_embs is None:
        self.logger.error(
          "Error in getting visual embeddings",
          extras={
            'session_id': self.session_id,
            'action_id': self.action_id,
            'request_id': self.request_id})
        raise ValueError('Error connecting to the TF serving servers')

      ele_visual_embs = np.array_split(ele_visual_embs, 3)
      ele_visual_embs, ele_h_visual_embs, ele_v_visual_embs = ele_visual_embs
      visual_prime_emb = ele_visual_embs[0]
      visual_h_prime_emb = ele_h_visual_embs[0]
      visual_v_prime_emb = ele_v_visual_embs[0]
      visual_sims = u.visual_cosine_sim(visual_prime_emb, ele_visual_embs[1:])
      h_visual_sims = u.visual_cosine_sim(visual_h_prime_emb, ele_h_visual_embs[1:])
      v_visual_sims = u.visual_cosine_sim(visual_v_prime_emb, ele_v_visual_embs[1:])
      total_time = time.time() - start_embedding
    else:
      start_embedding = time.time()
      imgs = element_imgs + h_ele_imgs + hs_ele_imgs + v_ele_imgs + vs_ele_imgs
      imgs = np.stack(imgs)
      visual_embs = self.func(imgs,
                            session_id=self.session_id,
                            action_id=self.action_id,
                            request_id=self.request_id)
      if visual_embs is None:
        self.logger.error(
          "Error connecting to the TF serving servers",
          extras={
            'session_id': self.session_id,
            'action_id': self.action_id,
            'request_id': self.request_id})
        raise ValueError('Error connecting to the TF serving servers')

      visual_embs = np.array_split(visual_embs, 5)
      ele_visual_embs, ele_h_visual_embs, ele_hs_visual_embs, ele_v_visual_embs, ele_vs_visual_embs \
        = visual_embs
      visual_prime_emb = ele_visual_embs[0]
      visual_h_prime_emb = ele_h_visual_embs[0]
      h_prime_small_emb = ele_hs_visual_embs[0]
      visual_v_prime_emb = ele_v_visual_embs[0]
      v_prime_small_emb = ele_vs_visual_embs[0]
      visual_sims = u.visual_cosine_sim(visual_prime_emb, ele_visual_embs[1:])
      h_visual_sims = u.visual_cosine_sim(visual_h_prime_emb, ele_h_visual_embs[1:])
      hs_visual_sims = u.visual_cosine_sim(h_prime_small_emb, ele_hs_visual_embs[1:])
      v_visual_sims = u.visual_cosine_sim(visual_v_prime_emb, ele_v_visual_embs[1:])
      vs_visual_sims = u.visual_cosine_sim(v_prime_small_emb, ele_vs_visual_embs[1:])
      ocr_sims = u.text_similarities(self.prime.element.ocr_text,
                                     [candidate.ocr_text for candidate in candidates])
      text_sims = u.text_similarities(self.prime.element.text,
                                      [candidate.text for candidate in candidates])
      recur_text_sims = u.text_similarities(self.prime.element.recursive_text,
                                            [candidate.recursive_text for candidate in candidates])
      id_sims = u.text_similarities(self.prime.element.locating_id,
                                    [candidate.locating_id for candidate in candidates])
      total_time = time.time() - start_embedding
    self.logger.debug(
      "Getting visual embeddings for %s scrollable candidates took %ss" % (len(visual_sims), total_time),
      extras={
        'session_id': self.session_id, 'action_id': self.action_id, 'request_id': self.request_id,
        'prime_xpath': self.prime.element.xpath})

    xpath_sims = u.xpath_similarity(self.prime.element.xpath,
                                    [candidate.xpath for candidate in candidates])
    class_sims = u.text_similarities(self.prime.element.classname,
                                     [candidate.classname for candidate in candidates])

    self.logger.debug(
      '[get_revisit_element] all similarities', extras={
        'is_scrollable': self.prime.element.is_scrollable(),
        'session_id': self.session_id, 'action_id': self.action_id, 'request_id': self.request_id,
        'visual_sims': visual_sims, 'h_visual_sims': h_visual_sims,
        'h_visual_small_sims': hs_visual_sims,
        'v_visual_sims': v_visual_sims, 'v_visual_small_sims': vs_visual_sims,
        'ocr_sims': ocr_sims, 'id_sims': id_sims,
        'xpath_sims': xpath_sims, 'cand_class_sims': class_sims,
        'text_sims': text_sims, 'recur_text_sims': recur_text_sims})
    return visual_sims, h_visual_sims, hs_visual_sims, \
           v_visual_sims, vs_visual_sims, ocr_sims, xpath_sims, \
           class_sims, id_sims, text_sims, recur_text_sims

  def _filter_revisit_candidates(self):

    """
    Filter the element in Revisit before evaluating each of them.
    :return: [FinderElementsInfo]
    """
    if self.prime.element.is_webview_element():
      revisit_elements = self.revisit.webview_elements
    else:
      revisit_elements = self.revisit.elements
    self.logger.debug('There are %d candidates found' % len(revisit_elements),
                      extras={'session_id': self.session_id,
                              'action_id': self.action_id,
                              'request_id': self.request_id})

    elements = []

    skipped_scrollable_xpaths = []
    skipped_depth_xpaths = []
    skipped_length_xpaths = []
    skipped_xpath_xpaths = []
    skipped_classname_xpaths = []
    skipped_image_xpaths = []
    for candidate in revisit_elements:
      try:
        # Filter by scrollable
        if self.prime.element.is_scrollable() and not candidate.is_scrollable():
          skipped_scrollable_xpaths.append(candidate.xpath)
          continue

        # Filter by textual features
        # Depth level
        if abs(self.prime.element.depth_level - candidate.depth_level) > self.depth_xpath_threshold:
          skipped_depth_xpaths.append(candidate.xpath)
          continue

        if self.prime.platform == self.revisit.platform:
          # Xpath
          if not self.xpath_threshold < len(candidate.xpath) / len(self.prime.element.xpath) < 1 / self.xpath_threshold:
            skipped_length_xpaths.append(candidate.xpath)
            continue
          candidate_xpath_sim = u.text_similarity(candidate.xpath, self.prime.element.xpath)
          if candidate_xpath_sim < self.xpath_threshold:
            skipped_xpath_xpaths.append(candidate.xpath)
            continue
          # Classname
          candidate_class_sim = u.text_similarity(candidate.classname, self.prime.element.classname)
          if candidate_class_sim < self.classname_threshold:
            skipped_classname_xpaths.append(candidate.xpath)
            continue
        # Filter by valid image
        if candidate.tf_img is None:
          skipped_image_xpaths.append(candidate.xpath)
          continue
      except Exception:
        internal_traceback = traceback.format_exc()
        self.logger.error("Error in extracting candidate props", extras={
          'session_id': self.session_id,
          'action_id': self.action_id,
          'request_id': self.request_id,
          'error': internal_traceback,
          'prime_xpath': self.prime.element.xpath,
          'cand': dict(candidate.attrib)})
      else:
        elements.append(candidate)
    self.log_skipped(skipped_scrollable_xpaths, skipped_depth_xpaths,
                     skipped_length_xpaths, skipped_xpath_xpaths,
                     skipped_classname_xpaths, skipped_image_xpaths)
    return elements

  def find_element_by_luna(self):
    start = time.time()
    if self.prime.element.is_scrollable():
      cand_xpaths, cand_sims, priority_luna = luna.find_element(self.prime, self.revisit, self.prime.element.xpath, element_type=constants.ElementType.SCROLLABLE)
    elif self.prime.element.is_keyboard(self.prime.platform):
      cand_xpaths, cand_sims, priority_luna = luna.find_element(self.prime, self.revisit, self.prime.element.xpath, element_type=constants.ElementType.KEYBOARD)
    else:
      cand_xpaths, cand_sims, priority_luna = luna.find_element(self.prime, self.revisit, self.prime.element.xpath)

    self.logger.debug('Candidates returned by Luna for prime xpath %s' % (self.prime.element.xpath),
                    extras={'candidates': [x for x in zip(cand_xpaths, cand_sims)],
                            'session_id': self.session_id, 'action_id': self.action_id, 'request_id': self.request_id, 'processed_time':time.time() - start})
    return cand_xpaths, cand_sims, priority_luna

  def log_skipped(self, skipped_scrollable_xpaths, skipped_depth_xpaths, skipped_length_xpaths,
                  skipped_xpath_xpaths, skipped_classname_xpaths,
                  skipped_image_xpaths):
    def log(obj, substr):
      if obj:
        self.logger.debug(f'Skipped %s elements as {substr}'
                          % len(obj),
                          extras={'session_id': self.session_id,
                                  'action_id': self.action_id,
                                  'request_id': self.request_id,
                                  'xpaths': str(obj)})
    log(skipped_scrollable_xpaths, 'They are not scrollable view')
    log(skipped_depth_xpaths, 'Their depth level are too different with prime')
    log(skipped_length_xpaths, 'Their xpaths having too different length with prime')
    log(skipped_xpath_xpaths, 'Their xpath sim are too different with prime')
    log(skipped_classname_xpaths, 'Their classname are too different with prime')
    log(skipped_image_xpaths, 'Their image are broken')


"""
Finding element using Instance Segmentation model (this approach doesn't require XML file)
"""
class ElementFinderBySegmentation(ElementFinderSolver):
  def __init__(self, prime: PrimeSession, revisit: RevisitSession, logger, func,
               session_id: int = 0, action_id: int = 0, request_id: int = 0):
    super(ElementFinderBySegmentation, self).__init__(prime, revisit, logger, func, session_id, action_id, request_id)

  def find_element(self):
    start = time.time()
    sessions_elements = self.func([self.prime.screen_img, self.revisit.screen_img])
    prime_elements = sessions_elements[0]
    self.prime.set_visual_elements_by_bound(prime_elements)
    self.logger.debug('Elements returned by segmentation for prime', extras={'elements':prime_elements})
    revisit_elements = sessions_elements[1]
    self.revisit.set_visual_elements_by_bound(revisit_elements)
    self.logger.debug('Elements returned by segmentation for revisit', extras={'elements':revisit_elements})
    cand_xpaths, cand_sims = luna.find_element_by_segmentaion(self.prime, self.revisit,
                    return_xpath= not (self.revisit.xml is None or self.prime.xml is None
                    or u.is_empty_string(self.prime.element.xpath)))
    return cand_xpaths, cand_sims
