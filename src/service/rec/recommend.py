from service.visual_verification import VisualVerificationElement
from service.sessions import Session
from service.utils import decode_img_bytes, get_ele_by_xpath, decode_bound, intersect
from service.xml_utils import get_scrollable_element_xpath
from keras.models import model_from_json
from service.screen_density_utils import get_screen_density, convert_font_height_px2dp, \
  convert_font_height_px2pt, calculate_ppi_of_screen
from service.constants import Platform, FontSizeStatus, LocationInScreen, get_scrollable_classes, \
  DENSITY_BUCKET_DICT, DENSITY_BUCKETS, ELEMENT_TYPES, SCREEN_PARTS
import pytesseract
import tensorflow as tf
import cv2
import numpy as np

# load model
json_file = open("./service/rec/fontsize_model.json", 'r')
loaded_model_json = json_file.read()
fontsize_model = model_from_json(loaded_model_json)
fontsize_model.load_weights("./service/rec/fontsize_model_weights.h5")


class RecommendSession(Session):
  def __init__(self, req):
    super().__init__(req)
    self.display_screen_size = req.display_screen_size
    self.verified_element_xpaths = req.element_xpaths

  @property
  def platform(self):
    if self.req.platform_name.upper() == Platform.ANDROID.name:
      return Platform.ANDROID
    elif self.req.platform_name.upper() == Platform.IOS.name:
      return Platform.IOS
    return None

  @property
  def device(self):
    return self.req.device_name

  def get_xml(self):
    return self.req.xml

  def get_screen_img(self):
    return decode_img_bytes(self.req.screen)

  @property
  def number_elements(self):
    visual_elements = [e for e in self.get_visual_elements()]
    return len(visual_elements)

  @property
  def screen_resolution(self):
    return self.screen_img.shape[:2]

  @property
  def density(self):
    return get_screen_density(self.xml_tree, self.screen_img, self.platform,
                              self.pixel_map, self.display_screen_size, self.device)

  @property
  def pixel_map(self):
    """
    Pixel map is only for iOS devices.
    """
    if self.platform == Platform.IOS:
      return self.req.pixel_map
    return None

  def verified_elements(self):
    return [RecommendElement(xpath, self) for xpath in self.verified_element_xpaths]

  @property
  def instance_log(self):
    return {
      'device': self.device,
      'platform': self.platform.name,
      'horizontal_pre_scale': self.pixel_map.horizontal_pre_scale if self.pixel_map else None,
      'vertical_pre_scale': self.pixel_map.vertical_pre_scale if self.pixel_map else None,
      'horizontal_scale': self.pixel_map.horizontal_scale if self.pixel_map else None,
      'vertical_scale': self.pixel_map.vertical_scale if self.pixel_map else None,
      'x_offset': self.pixel_map.x_offset if self.pixel_map else None,
      'y_offset': self.pixel_map.y_offset if self.pixel_map else None
    }


class RecommendElement(VisualVerificationElement):
  def __init__(self, xpath, session):
    super().__init__(xpath, session)
    self.location_in_screen = self.get_location_in_screen()
    self.fontsize_recommend = ''
    self.ppi = calculate_ppi_of_screen(self.sess.screen_resolution[0], self.sess.screen_resolution[1],
                                       self.sess.display_screen_size)
    self.width_ratio, self.height_ratio = self.get_width_height_ratio()
    self.font_height = self.get_font_height()
    self.char_width = self.get_character_width(self.sess.get_screen_img())
    self.font_dp = self.get_font_dp(self.font_height, self.ppi)
    self.density_bucket = self.get_density_bucket(self.ppi)
    self.element_type = self.get_element_type(self.xpath)
    self.screen_part = str(self.get_location_in_screen()).split('.')[-1].lower()

  @property
  def fontsize(self):
    if self.font_height is None:
      return None
    if self.sess.platform == Platform.ANDROID:
      return convert_font_height_px2dp(self.font_height, self.sess.density)
    else:
      return convert_font_height_px2pt(self.font_height, self.sess.density)

  @property
  def fontsize_status(self):
    sample = {
      'ppi':                    float(self.ppi),
      'width_ratio':            float(self.width_ratio),
      'height_ratio':           float(self.height_ratio),
      'display_screen_size':    float(self.sess.display_screen_size),
      'font_height':            float(self.font_height),
      'char_width':             float(self.char_width),
      'font_dp':                float(self.font_dp),
      'density_bucket_0':       0.0,
      'density_bucket_1':       0.0,
      'density_bucket_2':       0.0,
      'density_bucket_3':       0.0,
      'density_bucket_4':       0.0,
      'element_type_0':         0.0,
      'element_type_1':         0.0,
      'element_type_2':         0.0,
      'element_type_3':         0.0,
      'element_type_4':         0.0,
      'element_type_5':         0.0,
      'element_type_6':         0.0,
      'element_type_7':         0.0,
      'element_type_8':         0.0,
      'element_type_9':         0.0,
      'element_type_10':        0.0,
      'element_type_11':        0.0,
      'element_type_12':        0.0,
      'element_type_13':        0.0,
      'element_type_14':        0.0,
      'element_type_15':        0.0,
      'element_type_16':        0.0,
      'element_type_17':        0.0,
      'element_type_18':        0.0,
      'element_type_19':        0.0,
      'element_type_20':        0.0,
      'element_type_21':        0.0,
      'element_type_22':        0.0,
      'screen_part_0':          0.0,
      'screen_part_1':          0.0,
      'screen_part_2':          0.0,
      'screen_part_3':          0.0
    }

    # set value for the sample
    index_density_bucket = DENSITY_BUCKETS.index(self.density_bucket)
    index_element_type = ELEMENT_TYPES.index(self.element_type)
    index_screen_part = SCREEN_PARTS.index(self.screen_part)

    density_bucket_name = 'density_bucket_' + str(index_density_bucket)
    element_type_name = 'element_type_' + str(index_element_type)
    screen_part_name = 'screen_part_' + str(index_screen_part)

    sample[density_bucket_name] = 1.0
    sample[element_type_name] = 1.0
    sample[screen_part_name] = 1.0

    sample_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = fontsize_model.predict(sample_dict)
    status = FontSizeStatus.SKIP
    if np.argmax(predictions) == 0:
      status = FontSizeStatus.LARGE
    if np.argmax(predictions) == 1:
      status = FontSizeStatus.GOOD
    if np.argmax(predictions) == 2:
      status = FontSizeStatus.SMALL
    return status.name

  @property
  def is_in_scroll_view(self):
    parent = '/'.join(self.xpath.split('/')[:-1])
    scrollable_classes = get_scrollable_classes(self.sess.platform)
    for name in scrollable_classes:
      if name in parent:
        return True
    return False

  def get_location_in_screen(self):
    if self.is_in_scroll_view:
      return LocationInScreen.SCROLL

    screen_height, screen_width = self.sess.screen_resolution
    overlaps = []
    begin = 0

    part_height = int(screen_height / 5) + 1
    bb = self.bound
    for i in range(3):
      if i == 1:
        end = min(begin + part_height * 3, screen_height)
      else:
        end = min(begin + part_height, screen_height)
      screen_part = [0, begin, screen_width, end]
      overlaps.append(intersect(bb, screen_part))
      begin = end
    return LocationInScreen(np.argmax(overlaps))

  def get_character_width(self, image, threshold=50):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    d = pytesseract.image_to_data(img, config='--psm 12', output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    character_width = 0.0
    max_conf = 0
    for i in range(n_boxes):
      text = d['text'][i]
      conf = d['conf'][i]
      if len(text) > 0 and text.isalpha() and conf >= threshold:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if max_conf < conf:
          character_width = float(w) / len(text)
          max_conf = conf
    return character_width

  def get_width_height_ratio(self):
    element_width = self.bound[2] - self.bound[0]
    element_height = self.bound[3] - self.bound[1]
    if self.location_in_screen == LocationInScreen.SCROLL:
      scrollable_classes = get_scrollable_classes(self.sess.platform)
      scrollable_xpath = get_scrollable_element_xpath(self.xpath, scrollable_classes)

      scrollable_element = get_ele_by_xpath(self.sess.xml_tree, scrollable_xpath)
      scrollable_bound = decode_bound(scrollable_element, self.sess.platform, self.sess.pixel_map)
      width = scrollable_bound[2] - scrollable_bound[0]
      height = scrollable_bound[3] - scrollable_bound[1]
    else:
      height, width = self.sess.screen_resolution
    width_ratio = element_width / width
    height_ratio = element_height / height
    return width_ratio, height_ratio

  def get_font_dp(self, font_height, ppi):
    return round(font_height / (ppi / 160), 1)

  def get_density_bucket(self, ppi):
    min_diff = -1
    db = None
    for k, v in DENSITY_BUCKET_DICT.items():
      diff = abs(ppi - v['dpi'])
      if min_diff == -1 or min_diff > diff:
        min_diff = diff
        db = k
    return db

  def get_element_type(self, xpath):
    # get element type
    element_type = xpath.split('/')[-1]
    if '[' in element_type:
      begin = element_type.find('[')
      end = element_type.find(']') + 1
      element_type = element_type[:begin] + element_type[end:]
    return element_type

  def get_fontsize_recommend_dict(self):
    return {'status': self.fontsize_status,
            'xpath': self.xpath,
            'reference_fontsize': self.fontsize_recommend,
            'ocr_text': self.ocr_text,
            'fontsize': self.fontsize}
