import time
import pickle
import lightgbm as lgb
import numpy as np
from collections import defaultdict
from service import utils as u, xml_utils
from service.ai_components import AIComponent
from service.sessions import ScreenshotSession
from service.constants import REMOVE_LARGE_ELEMENT_ON_SCREEN_THRESHOLD
from service.constants import ELEMENT_SELECTION_MODEL_PATH

# Load lightgbm model 
element_selection_model = lgb.Booster(model_file=ELEMENT_SELECTION_MODEL_PATH)  # init model

def distance(pointA, pointB):
  return np.sqrt( (pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2 )

def normalized_distance(distance, im_shape):
  return distance / np.sqrt(im_shape[0]**2 + im_shape[1]**2)

def area(box):
  return (box[2]-box[0]) * (box[3]-box[1])

def center(box):
  return (int((box[2]+box[0]) / 2), int((box[3]+box[1]) / 2))

def convert_box(box):
  box = [box['x1'], box['y1'], box['x2'], box['y2']]
  return [int(b) for b in box]

def is_box_inside(boxA, boxB):
  if boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and boxA[2] <= boxB[2] and boxA[3] <= boxB[3]:
    return True
  return False

def is_box_outside(boxA, boxB):
  if boxA[0] < boxB[0] and boxA[1] < boxB[1] and boxA[2] > boxB[2] and boxA[3] > boxB[3]:
    return True
  return False

def inter_point(boxA, boxB):
  xmin = max(boxA[0], boxB[0])
  ymin = max(boxA[1], boxB[1])
  xmax = min(boxA[2], boxB[2])
  ymax = min(boxA[3], boxB[3])
  interPoint = [xmin, ymin, xmax, ymax]
  return interPoint

def inter_area(boxA, boxB):
  interPoint = inter_point(boxA, boxB)
  interArea = max(0, interPoint[2] - interPoint[0]) * max(0, interPoint[3] - interPoint[1])
  return interArea

def iou(boxA, boxB):
  interArea = inter_area(boxA, boxB)
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou

def generate_features(imshape, target_xml, xmls, segs, touchpoint):
  """
  All the formulas can be found here:
    - Observations: https://docs.google.com/document/d/1E6qGa60mOYq-Bui28cRhvV-Dt17mIIr94VxIbSggIfc/
    - Other features: https://docs.google.com/document/d/1y2pCKBIwVigoa708VYTviMFUo90oXZ14J_c3yzanNV0/
  """

  # Xml features
  im_area = imshape[0] * imshape[1]
  xml_area = area(target_xml)
  xml_ratio = xml_area/im_area
  xml_x_ratio = (target_xml[2] - target_xml[0]) / imshape[1]
  xml_y_ratio = (target_xml[3] - target_xml[1]) / imshape[0]

  # Xml vs touchpoint
  xml_touchpoint_distance = normalized_distance(distance(center(target_xml), touchpoint), imshape)

  # Xmls features
  n_inside, n_outside, n_overlap = 0, 0, 0
  for xml in xmls:
    if is_box_inside(xml, target_xml):
      n_inside += 1
    elif is_box_outside(xml, target_xml):
      n_outside += 1
    else:
      n_overlap += 1

  # Xml vs segmentations
  n_segments = 0
  best_seg, best_iou, best_cls = None, 0, ''
  for cls_seg, box_seg in segs:
    iou_score = iou(target_xml, box_seg)
    if iou_score > 0.1:
      n_segments += 1
      if iou_score > best_iou:
        best_seg = box_seg
        best_cls = cls_seg
        best_iou = iou_score
        
  # Observation 1
  if best_seg:
    xml_seg_distance = normalized_distance(distance(center(target_xml), center(best_seg)), imshape)
  else:
    xml_seg_distance = 0

  if best_cls == 'Text':
    # Observation 2 and 3
    interPoint = inter_point(target_xml, best_seg)
    new_target_xml = [target_xml[0], target_xml[1], interPoint[2], target_xml[3]]
    interArea = inter_area(new_target_xml, best_seg) / area(new_target_xml)
  elif best_cls != '':
    interArea = inter_area(target_xml, best_seg) / area(target_xml)
  else:
    interArea = 0

  features = {}
  if best_seg:
    features['best_seg'] = best_seg
  else:
    features['best_seg'] = ''
  features['xml_ratio'] = xml_ratio
  features['xml_x_ratio'] = xml_x_ratio
  features['xml_y_ratio'] = xml_y_ratio
  features['n_inside'] = n_inside
  features['n_outside'] = n_outside
  features['n_overlap'] = n_overlap
  features['n_segments'] = n_segments
  features['best_iou'] = best_iou
  features['xml_seg_distance'] = xml_seg_distance
  features['xml_touchpoint_distance'] = xml_touchpoint_distance
  features['interArea'] = interArea

  return features

def remove_duplicate_xml_boxes(boxes):
  nodup_boxes = []
  final_boxes = []

  # Remove duplicate box
  for i, e in enumerate(boxes):
    if e['bound'] not in nodup_boxes:
      nodup_boxes.append(e['bound'])
      final_boxes.append(e)

  return final_boxes

class ElementFinderAtPoint(AIComponent):

  def __init__(self, services, logger):
    super(ElementFinderAtPoint, self).__init__(services, logger=logger)

  def __call__(self, session: ScreenshotSession, session_id: int = 0, action_id: int = 0, request_id: int = 0):
    self.session = session
    self.session_id = session_id
    self.action_id = action_id
    self.request_id = request_id
    self.func = self._try_get_visual_bounds_by_segmentation

    return self.get_response()

  def get_response(self):
    start = time.time()
    self.logger.debug("[get_element_xpath_from_point] sessions info", extras={
      'request_id': self.request_id,
      'session_id': self.session_id,
      'action_id': self.action_id,
      **self.session.instance_log
    })
    xpath_result = self.find_element_at_point()
    self.logger.debug('Xpath returned by AI service %s' % self.__class__.__name__,
                      extras={'point': str(self.session.point()), 'xpath': xpath_result,
                              'session_id': self.session_id, 'action_id': self.action_id,
                              'request_id': self.request_id, 'processed_time': time.time() - start})
    return xpath_result

  def find_element_at_point(self):
    start = time.time()
    point = self.session.point()

    # Get xmls in touchpoint
    xmls = list(set(self.session.elements + self.session.webview_elements))
    xmls_contain_point = []
    screen_shape = self.session.screen_img.shape
    screen_area = screen_shape[0] * screen_shape[1]

    for xml in xmls:
      if u.is_point_inside_bound(point, xml.bound):
        xml_info = {'xpath': xml.xpath, 'bound': xml.bound}
        xmls_contain_point.append(xml_info)

    # Remove duplicate boxes
    xmls_contain_point = remove_duplicate_xml_boxes(xmls_contain_point)

    # Return xml if there is only 1 xml at touchpoint
    if len(xmls_contain_point) == 1:
      return [{'xpath': xmls_contain_point[0]['xpath'], 'confidence': 1}]

    result = []
    xmls_without_large_boxes = []

    # Compute score for large boxes
    for xml in xmls_contain_point:
      if u.get_area(*xml['bound'])/screen_area <= REMOVE_LARGE_ELEMENT_ON_SCREEN_THRESHOLD:
        xmls_without_large_boxes.append(xml)
      else:
        xml['confidence'] = 0
        xml.pop('bound', None)
        result.append(xml)

    # Compute score for small boxes
    if xmls_without_large_boxes:

      segmentation_bounds = self.func([self.session.screen_img])[0]
      if not segmentation_bounds:
        self.logger.info('No segmentation boxes were found.')

      # get xml features
      xmls_features = []
      for target_xml in xmls_without_large_boxes:
        other_xmls = [xml['bound'] for xml in xmls_without_large_boxes if xml != target_xml]

        features = generate_features(screen_shape, target_xml['bound'], other_xmls, segmentation_bounds, point)
        l_features = []
        for k, v in features.items():
          if k != 'best_seg':
            l_features.append(v)
        xmls_features.append(l_features)

      np_xmls_features = np.array(xmls_features)

      predictions = element_selection_model.predict(np_xmls_features, num_iteration=element_selection_model.best_iteration)

      for pred, xml in zip(predictions, xmls_without_large_boxes):
        result.append({'xpath': xml['xpath'], 'confidence': round(pred, 3)})
  
    self.logger.debug("Finding element at point by segmentation took %f seconds" % (time.time() - start),
                      extras={'session_id': self.session_id, 'action_id': self.action_id,
                              'request_id': self.request_id})
    return result


