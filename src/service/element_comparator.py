import traceback

import numpy as np

from service.ai_components import AIComponent
from service.sessions import PrimeSession, RevisitSession
from service.element import Element
from service import utils as u


class CompareRevisitSession(RevisitSession):
  def __init__(self, req):
    super(CompareRevisitSession, self).__init__(req)

  @property
  def element(self):
    return Element(self.req.revisit_xpath, self)


class ElementComparator(AIComponent):
  def __init__(self, services, logger):
    super(ElementComparator, self).__init__(services, logger=logger)

  def __call__(self, prime: PrimeSession, revisit: CompareRevisitSession,
               session_id=0, action_id=0, request_id=0):
    self.logger.debug(
      '[compare_element] request info',
      extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'prime_xpath': prime.element.xpath,
        **prime.instance_log,
        **revisit.instance_log
      })

    self.logger.debug(
      '[compare_element] info of the prime element',
      extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'prime_xpath': prime.element.xpath,
        'prime_ele_ocr': prime.element.ocr_text,
        'prime_text': prime.element.text,
        'prime_recur_text': prime.element.recursive_text,
        'prime_ele_bound': prime.element.bound})

    self.logger.debug(
      '[compare_element] info of the revisit element',
      extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'revisit_xpath': revisit.element.xpath,
        'revisit_ele_ocr': revisit.element.ocr_text,
        'revisit_text': revisit.element.text,
        'revisit_recur_text': revisit.element.recursive_text,
        'revisit_ele_bound': revisit.element.bound})

    if prime.element.tf_img is None or revisit.element.tf_img is None:
      return 0.

    # get visual embeddings
    prime_visuals = np.stack([prime.element.tf_img,
                              prime.element.horizontal_tf_img, prime.element.horizontal_small_tf_img,
                              prime.element.vertical_tf_img, prime.element.vertical_small_tf_img])
    revisit_visuals = np.stack([revisit.element.tf_img,
                                revisit.element.horizontal_tf_img,
                                revisit.element.horizontal_small_tf_img,
                                revisit.element.vertical_tf_img,
                                revisit.element.vertical_small_tf_img])
    prime_embs = self._try_get_visual_embeddings(
      prime_visuals, session_id, action_id, request_id)
    revisit_embs = self._try_get_visual_embeddings(
      revisit_visuals, session_id, action_id, request_id)
    if prime_embs is None or revisit_embs is None:
      self.logger.error(
        "Error in getting visual embeddings",
        extras={
          'session_id': session_id,
          'action_id': action_id,
          'request_id': request_id,
          'error': traceback.format_exc()})
      raise ValueError('Error on getting visual embeddings')

    # calculate all the similarities
    visual_sims = [u.emb_cosine_sim(p, r) for (p, r) in zip(prime_embs, revisit_embs)]
    ocr_sims = u.text_similarities(prime.element.ocr_text, [revisit.element.ocr_text])
    text_sims = u.text_similarities(prime.element.text, [revisit.element.text])
    recur_text_sims = u.text_similarities(prime.element.recursive_text,
                                          [revisit.element.recursive_text])
    id_sims = u.text_similarities(prime.element.locating_id, [revisit.element.locating_id])
    classname_sims = u.text_similarities(prime.element.classname, [revisit.element.classname])
    xpath_sims = u.xpath_similarity(prime.element.xpath, [revisit.element.xpath])

    sk_input = [*visual_sims, ocr_sims[0], xpath_sims[0], classname_sims[0],
                id_sims[0], text_sims[0], recur_text_sims[0]]
    sk_input = np.array(sk_input).reshape(1, -1)
    pre_ = self.sk_cls.predict(sk_input)[0]
    madeup_conf = u.overall_similarity(
      *visual_sims, ocr_sims[0], xpath_sims[0], classname_sims[0],
      id_sims[0], text_sims[0], recur_text_sims[0],
      prime.platform, revisit.platform, scrollable_element=False)
    if pre_ == 1:
      madeup_conf = (2 + madeup_conf) / 3
    else:
      madeup_conf = (0.6 + 2 * madeup_conf) / 3

    self.logger.debug(
      'compare_element calculate all the similarities',
      extras={
        'visual_sims': visual_sims,
        'ocr_sims': ocr_sims,
        'text_sims': text_sims,
        'recur_text_sims': recur_text_sims,
        'classname_sims': classname_sims,
        'id_sims': id_sims,
        'xpath_sims': xpath_sims,
        'action_id': action_id,
        'session_id': session_id,
        'request_id': request_id,
        'madeup_conf': madeup_conf,
        'pre_': int(pre_)})
    return madeup_conf
