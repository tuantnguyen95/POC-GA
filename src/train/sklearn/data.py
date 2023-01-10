import os
import csv
import base64
import traceback
import numpy as np
import pandas as pd
import sk_utils as utils

class DataGenerator(object):
  def __init__(
      self, csv_path, visual_gen=None, raw_path=None,
      use_visual_sim=True, use_xpath_sim=True,
      use_ocr_sim=True, use_classname_sim=True,
      use_id=True, use_text=True, use_recur_text=True):
    self.is_debugging = True
    self.use_visual_sim = use_visual_sim
    self.use_xpath_sim = use_xpath_sim
    self.use_ocr_sim = use_ocr_sim
    self.use_classname_sim = use_classname_sim
    self.use_id = use_id
    self.use_text = use_text
    self.use_recur_text = use_recur_text
    self.data = None
    self.target = None
    self.sims = None
    self.csv_path = csv_path
    self.raw_path = raw_path
    self.ocr_cache = {}
    self.ocr_file = 'ocr_cache.txt'
    self.visual_gen = visual_gen
    self.restore_ocr_cache()
    self.empty_text_sim = 0
    self.read_data()


  def read_data(self):
    self.data = []
    self.target = []
    self.sims = []
    data_frame = pd.read_csv(
      self.csv_path, delimiter=',',
      quoting=csv.QUOTE_ALL)
    for idx, tup in enumerate(data_frame.itertuples()):
      try:
        features, compare_sims = [], []
        prime_tree = utils.tree_from_file(self.raw_path, tup.prime_screen_name)
        revisit_tree = utils.tree_from_file(self.raw_path, tup.revisit_screen_name)
        prime_element = utils.get_ele_by_xpath(prime_tree, tup.prime_xpath)
        revisit_element = utils.get_ele_by_xpath(revisit_tree, tup.revisit_xpath)

        if self.use_visual_sim:
          visual_sim = self.visual_sim(tup, prime_element, revisit_element)
          h_visual_sim = self.horizontal_visual_sim(tup, prime_element, revisit_element)
          h_visual_small_sim = self.horizontal_visual_small_sim(tup, prime_element, revisit_element)
          v_visual_sim = self.vertical_visual_sim(tup, prime_element, revisit_element)
          v_visual_small_sim = self.vertical_visual_small_sim(tup, prime_element, revisit_element)

          if visual_sim is None:
            continue
          features.extend(
            [visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim, v_visual_small_sim])
          compare_sims.extend(
            [visual_sim, h_visual_sim, h_visual_small_sim, v_visual_sim, v_visual_small_sim])

        if self.use_ocr_sim:
          prime_ocr_exist, revisit_ocr_exist, ocr_sim = self.ocr_sim(
            tup, prime_element, revisit_element)
        else:
          prime_ocr_exist, revisit_ocr_exist, ocr_sim = 0, 0, 0
        features.append(ocr_sim)
        compare_sims.extend([prime_ocr_exist, ocr_sim, revisit_ocr_exist])

        if self.use_xpath_sim:
          xpath_sim = utils.lev_sim(tup.prime_xpath, tup.revisit_xpath)
          features.append(xpath_sim)
          compare_sims.append(xpath_sim)

        if self.use_classname_sim:
          classname_sim = utils.lev_sim(
            utils.class_name_from_xpath(tup.prime_xpath),
            utils.class_name_from_xpath(tup.revisit_xpath))
          features.append(classname_sim)
          compare_sims.append(classname_sim)

        if self.use_id:
          prime_id_exist, revisit_id_exist, id_sim = self.id_sim(
            tup, prime_element, revisit_element)
          features.append(id_sim)
          compare_sims.extend([prime_id_exist, id_sim, revisit_id_exist])

        if self.use_text:
          prime_txt_exist, revisit_txt_exist, text_sim = self.text_sim(
            tup, prime_element, revisit_element, False)
          features.append(text_sim)
          compare_sims.extend([prime_txt_exist, text_sim, revisit_txt_exist])

        if self.use_recur_text:
          prime_recur_text_exist, revisit_recur_text_exist, recur_text_sim = self.text_sim(
            tup, prime_element, revisit_element, True)
          features.append(recur_text_sim)
          compare_sims.extend([prime_recur_text_exist, recur_text_sim, revisit_recur_text_exist])

        if str(tup.matched) == '1':
          self.target.append(1)
        else:
          self.target.append(-1)
        if self.is_debugging:
          print('idx: ', idx)
          print('features: ', features)
          print('class: %s, visual_sim: %f, h_visual_sim: %f, v_visual_sim: %f,'
                'h_visual_small_sim: %s, v_visual_small_sim: %s '
                'ocr_sim: %f, xpath_sim: %f, class_sim: %f, '
                'id_sim: %f, text_sim: %f, recur_sim: %f, overall: %f\n\n\n\n'
                % (tup.matched, visual_sim, h_visual_sim, v_visual_sim,
                   h_visual_small_sim, v_visual_small_sim, ocr_sim, xpath_sim,
                   classname_sim, id_sim, text_sim, recur_text_sim,
                   utils.overall_similarity(*compare_sims)))
        self.data.append(np.array([features]))
        self.sims.append(utils.overall_similarity(*compare_sims))
      except Exception as e:
        trace = traceback.format_exc()
        print(trace)
        print('index: ', idx)
        print('error data: ', tup)
    print("number of features: ", self.data[0].shape)
    self.target = np.array(self.target)
    self.data = np.vstack(self.data)
    if self.visual_gen:
      self.visual_gen.store_cache()
    self.store_ocr_cache()
    utils.calculate_thresholds(self.sims, self.target)


  def visual_sim(self, tup, prime_element, revisit_element):
    '''
    Calculate similarity for the exact cut of each element
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    if self.visual_gen.has_key(tup.prime_element_name):
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name, None)
    else:
      prime_bound = utils.decode_bound(prime_element, tup.prime_platform, None)
      prime_img = utils.exact_cut(tup.prime_screen_name, self.raw_path, prime_bound)
      if prime_img is None:
        return None
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name, prime_img)

    if self.visual_gen.has_key(tup.revisit_element_name):
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name, None)
    else:
      revisit_bound = utils.decode_bound(revisit_element, tup.prime_platform, None)
      revisit_img = utils.exact_cut(tup.revisit_screen_name, self.raw_path, revisit_bound)
      if revisit_img is None:
        return None
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name, revisit_img)

    if prime_emb is not None and revisit_emb is not None:
      return utils.cosine(prime_emb, revisit_emb)
    return None


  def horizontal_visual_sim(self, tup, prime_element, revisit_element):
    '''
    Calculate similarity for the fully horizontally extended cuts.
    The cut is expanded to the screen edges
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    if self.visual_gen.has_key(tup.prime_element_name + 'horizontal_full'):
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'horizontal_full', None)
    else:
      prime_bound = utils.decode_bound(prime_element, tup.prime_platform, None)
      prime_img = utils.horizonal_cut(tup.prime_screen_name, self.raw_path, prime_bound)
      if prime_img is None:
        print('Error in geting horizontal_visual_sim')
        return 0
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'horizontal_full', prime_img)

    if self.visual_gen.has_key(tup.revisit_element_name + 'horizontal_full'):
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name + 'horizontal_full', None)
    else:
      revisit_bound = utils.decode_bound(revisit_element, tup.prime_platform, None)
      revisit_img = utils.horizonal_cut(tup.revisit_screen_name, self.raw_path, revisit_bound)
      if revisit_img is None:
        print('Eror in getting horizontal_visual_sim')
        return 0
      revisit_emb = self.visual_gen.get_emb(
        tup.revisit_element_name + 'horizontal_full', revisit_img)

    if prime_emb is not None and revisit_emb is not None:
      return utils.cosine(prime_emb, revisit_emb)
    return 0


  def horizontal_visual_small_sim(self, tup, prime_element, revisit_element):
    '''
    Calculate similarity for the slightly horizontally extended cuts.
    The width of the cut is 3 times the width of the element.
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    if self.visual_gen.has_key(tup.prime_element_name + 'horizontal_small'):
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'horizontal_small', None)
    else:
      prime_bound = utils.decode_bound(prime_element, tup.prime_platform, None)
      prime_img = utils.horizonal_cut_small(tup.prime_screen_name, self.raw_path, prime_bound)
      if prime_img is None:
        print('Error in getting horizontal_visual_small_sim')
        return 0
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'horizontal_small', prime_img)

    if self.visual_gen.has_key(tup.revisit_element_name + 'horizontal_small'):
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name + 'horizontal_small', None)
    else:
      revisit_bound = utils.decode_bound(revisit_element, tup.prime_platform, None)
      revisit_img = utils.horizonal_cut_small(tup.revisit_screen_name, self.raw_path, revisit_bound)
      if revisit_img is None:
        print('Error in getting horizontal_visual_small_sim')
        return 0
      revisit_emb = self.visual_gen.get_emb(
        tup.revisit_element_name + 'horizontal_small', revisit_img)

    if prime_emb is not None and revisit_emb is not None:
      return utils.cosine(prime_emb, revisit_emb)
    return 0


  def vertical_visual_sim(self, tup, prime_element, revisit_element):
    '''
    Calculate similarity for the fully vertically extended cuts.
    The cut is expanded to the screen edges
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    if self.visual_gen.has_key(tup.prime_element_name + 'vertical_full'):
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'vertical_full', None)
    else:
      prime_bound = utils.decode_bound(prime_element, tup.prime_platform, None)
      prime_img = utils.vertical_cut(tup.prime_screen_name, self.raw_path, prime_bound)
      if prime_img is None:
        print('Error in getting vertical_visual_sim')
        return 0
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'vertical_full', prime_img)

    if self.visual_gen.has_key(tup.revisit_element_name + 'vertical_full'):
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name + 'vertical_full', None)
    else:
      revisit_bound = utils.decode_bound(revisit_element, tup.prime_platform, None)
      revisit_img = utils.vertical_cut(tup.revisit_screen_name, self.raw_path, revisit_bound)
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name + 'vertical_full', revisit_img)
      if revisit_img is None:
        print('Error in getting vertical_visual_sim')
        return 0

    if prime_emb is not None and revisit_emb is not None:
      return utils.cosine(prime_emb, revisit_emb)
    return 0


  def vertical_visual_small_sim(self, tup, prime_element, revisit_element):
    '''
    Calculate similarity for the slightly vertically extended cuts.
    The width of the cut is 3 times the width of the element.
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    if self.visual_gen.has_key(tup.prime_element_name + 'vertical_small'):
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'vertical_small', None)
    else:
      prime_bound = utils.decode_bound(prime_element, tup.prime_platform, None)
      prime_img = utils.vertical_cut_small(tup.prime_screen_name, self.raw_path, prime_bound)
      if prime_img is None:
        print('Error in getting vertical_visual_small_sim')
        return 0
      prime_emb = self.visual_gen.get_emb(tup.prime_element_name + 'vertical_small', prime_img)

    if self.visual_gen.has_key(tup.revisit_element_name + 'vertical_small'):
      revisit_emb = self.visual_gen.get_emb(tup.revisit_element_name + 'vertical_small', None)
    else:
      revisit_bound = utils.decode_bound(revisit_element, tup.prime_platform, None)
      revisit_img = utils.vertical_cut_small(tup.revisit_screen_name, self.raw_path, revisit_bound)
      revisit_emb = self.visual_gen.get_emb(
        tup.revisit_element_name + 'vertical_small', revisit_img)
      if revisit_img is None:
        print('Error in getting vertical_visual_small_sim')
        return 0

    if prime_emb is not None and revisit_emb is not None:
      return utils.cosine(prime_emb, revisit_emb)
    return 0


  def ocr_sim(self, tup, prime_element, revisit_element):
    '''
    Calculate similarity for extracted ocr text.
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    prime_ocr_exist = 1
    revisit_ocr_exist = 1
    if tup.prime_element_name in self.ocr_cache:
      prime_txt = self.ocr_cache[tup.prime_element_name]
    else:
      prime_bound = utils.decode_bound(prime_element, tup.prime_platform, None)
      prime_img = utils.ocr_cut(tup.prime_screen_name, self.raw_path, prime_bound)
      prime_txt = utils.ocr_img_text(prime_img)
      self.ocr_cache[tup.prime_element_name] = prime_txt
    if utils.is_empty_string(prime_txt):
      prime_ocr_exist = 0

    if tup.revisit_element_name in self.ocr_cache:
      revisit_txt = self.ocr_cache[tup.revisit_element_name]
    else:
      revisit_bound = utils.decode_bound(revisit_element, tup.prime_platform, None)
      revisit_img = utils.ocr_cut(tup.revisit_screen_name, self.raw_dir, revisit_bound)
      revisit_txt = utils.ocr_img_text(revisit_img)
      self.ocr_cache[tup.revisit_element_name] = revisit_txt
    if utils.is_empty_string(revisit_txt):
      revisit_ocr_exist = 0

    if utils.is_empty_string(prime_txt) or utils.is_empty_string(revisit_txt):
      return prime_ocr_exist, revisit_ocr_exist, 0
    return prime_ocr_exist, revisit_ocr_exist, 1 - utils.lev_distance(
      prime_txt, revisit_txt)/max(len(prime_txt), len(revisit_txt))


  def id_sim(self, tup, prime_element, revisit_element):
    '''
    Similarity of ids from the two elements
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    prime_id = utils.get_id(prime_element, tup.prime_platform)
    revisit_id = utils.get_id(revisit_element, tup.revisit_platform)
    prime_id_exist = 1
    revisit_id_exist = 1
    if utils.is_empty_string(prime_id):
      prime_id_exist = 0
    if utils.is_empty_string(revisit_id):
      revisit_id_exist = 0
    if utils.is_empty_string(prime_id) or utils.is_empty_string(revisit_id):
      return prime_id_exist, revisit_id_exist, 0
    return prime_id_exist, revisit_id_exist, 1 - utils.lev_distance(
      prime_id, revisit_id)/max(len(prime_id), len(revisit_id))


  def text_sim(self, tup, prime_element, revisit_element, is_recursive=False):
    '''
    Similarity of text from the two elements
    :param namedtuple tup: each row of the DF
    :param xmlNode prime_element:
    :param xmlNode revisit_element:
    '''
    if is_recursive:
      prime_txt = utils.get_recursive_text(prime_element, tup.prime_platform)
      revisit_txt = utils.get_recursive_text(revisit_element, tup.revisit_platform)
    else:
      prime_txt = utils.get_element_text(prime_element, tup.prime_platform)
      revisit_txt = utils.get_element_text(revisit_element, tup.revisit_platform)
    prime_txt_exist = 1
    revisit_txt_exist = 1
    if utils.is_empty_string(prime_txt):
      prime_txt_exist = 0
    if utils.is_empty_string(revisit_txt):
      revisit_txt_exist = 0
    if utils.is_empty_string(prime_txt) and utils.is_empty_string(revisit_txt):
      return 0, 0, self.empty_text_sim
    if utils.is_empty_string(prime_txt) or utils.is_empty_string(revisit_txt):
      return prime_txt_exist, revisit_txt_exist, 0
    return prime_txt_exist, revisit_txt_exist, 1 - utils.lev_distance(
      prime_txt, revisit_txt)/max(len(prime_txt), len(revisit_txt))


  def restore_ocr_cache(self):
    if os.path.exists(self.ocr_file):
      with open(self.ocr_file, 'r') as f:
        for line in f:
          try:
            parts = line.strip().split('\t')
            if len(parts) > 1 and len(parts[1]) > 0:
              self.ocr_cache[parts[0]] = base64.b64decode(parts[1]).decode('utf-8')
            else:
              self.ocr_cache[parts[0]] = ''
          except Exception as e:
            print('Error: ', e, '\t at line: ', line)


  def store_ocr_cache(self):
    with open(self.ocr_file, 'w') as f:
      for key, value in self.ocr_cache.items():
        if not utils.is_empty_string(value):
          encoded = base64.b64encode(bytes(value, 'utf-8'))
          f.write('%s\t%s\n'%(key, encoded.decode("utf-8")))
        else:
          f.write('%s\t%s\n'%(key, value))
