import cv2
import numpy as np

import time
from service.ai_components import AIComponent
from service.sessions import FindingByImageSession
class ElementFinderByImage(AIComponent):

  def __init__(self, services, logger):
    super(ElementFinderByImage, self).__init__(services, logger=logger)

  def __call__(self, session: FindingByImageSession, session_id: int = 0, action_id: int = 0, request_id: int = 0):
    self.session = session
    self.session_id = session_id
    self.action_id = action_id
    self.request_id = request_id
    return self.get_response()

  def get_response(self):
    start = time.time()
    self.logger.debug("[get_element_xpath_from_image] sessions info", extras={
      'request_id': self.request_id,
      'session_id': self.session_id,
      'action_id': self.action_id,
      **self.session.instance_log
    })
    results = self.find_element_by_image()

    self.logger.debug("Finding element by image took %f seconds" % (time.time() - start),
                  extras={'session_id': self.session_id, 'action_id': self.action_id,
                          'request_id': self.request_id})
    return results

  def find_element_by_image(self):
    start = time.time()
    screen = self.session.get_screen_img()
    query_img = self.session.get_query_img()
    threshold = self.session.get_threshold()
    method = self.session.get_method()
    elements = list(set(self.session.elements + self.session.webview_elements))
    
    matched_ele_boxes, match_ele_confs = self.find_element(query_img, screen, method=method, top=100)
    matched_xpaths = [self.get_best_element(box, elements) for box in matched_ele_boxes]
    matched_xpaths = self.remove_duplicates_xpaths(matched_xpaths, match_ele_confs)
    matched_xpaths = [{'xpath': key, 'confidence': round(value, 3)} for key, value in matched_xpaths.items() if value >= threshold]
    return matched_xpaths

  def get_image_position(self, template_position, ele_shape):
    top_left = template_position[::-1]
    bottom_right = (top_left[0] + ele_shape[1], top_left[1] + ele_shape[0])
    return [*top_left, *bottom_right]

  def find_element(self, query_img, screen, method=cv2.TM_CCORR_NORMED, top=100):
    """
    Function to find image using template matching from OpenCV
    ...

    Attributes
    ----------
    query_img: numpy.ndarray
      Query image.
    screen: numpy.ndarray
      Screenshot.
    method: int
      Matching template method, from 0->5. Below are the methods: 
        - TM_CCOEFF = 0
        - TM_CCOEFF_NORMED = 1
        - TM_CCORR = 2
        - TM_CCORR_NORMED = 3
        - TM_SQDIFF = 4
        - TM_SQDIFF_NORMED = 5
      To have a range from [0, 1], we need to choose method 1, 3, 5.
    top: int
      Top N regions that match with Query image.
    """

    # Get template matching result
    res = cv2.matchTemplate(screen, query_img, method)
    
    # Get top max values and top min values of of template matching result 
    idx = np.squeeze( np.dstack(np.unravel_index(np.argsort(res.ravel()), res.shape )) )
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
      min_locs, min_vals = idx[:top], [res[ pos[0] ][ pos[1] ] for pos in idx[:top]]
      matched_elements = [self.get_image_position(location, query_img.shape) for location in min_locs]
      confidences = np.float64(min_vals)
    else:
      max_locs, max_vals = idx[-top:], [res[ pos[0] ][ pos[1] ] for pos in idx[-top:]]
      matched_elements = [self.get_image_position(location, query_img.shape) for location in max_locs]
      confidences = np.float64(max_vals)
    
    return matched_elements, confidences

  def bb_intersection_over_union(self, boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
      
  def get_best_element(self, match_box, elements):
    max_iou, max_ele = 0, None
    for ele in elements:
      iou = self.bb_intersection_over_union(match_box, ele.bound)
      if iou > max_iou:
        max_iou = iou
        max_ele = ele
    if max_ele is None:
      return None
    
    match_xpath = max_ele.xpath
    return match_xpath

  def remove_duplicates_xpaths(self, xpaths, confs):
    final_xpaths = {}
    for xpath, conf in zip(xpaths, confs):
      if xpath == None:
        continue
      elif xpath not in final_xpaths:
        final_xpaths[xpath] = conf
      elif conf > final_xpaths[xpath]:
        final_xpaths[xpath] = conf
    return final_xpaths