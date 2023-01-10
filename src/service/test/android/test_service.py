
import os
import sys
import cv2
import time
import base64
import requests
import unittest
import traceback
from glob import glob
from lxml import etree
import service.schema.booster.ai.service_request_pb2 as request_pb2
import service.constants as constants

def read_file_bytes(filename):
  with open(filename, 'rb') as data:
    byte_data = data.read()
    return byte_data

def get_platform_scale(tree, img):
  platform = 'android'
  hscale, wscale = 1.0, 1.0
  app_finder = etree.XPath("//XCUIElementTypeApplication")
  app = app_finder(tree)
  if len(app) > 0:
    platform = 'ios'
    h = int(app[0].get('height'))
    hscale = img.shape[0]/h
    w = int(app[0].get('width'))
    wscale = img.shape[1]/w
  return platform, hscale, wscale

def call_finding_service(endpoint, prime_xpath, prime_xml_binary, prime_screen_binary,
                         revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.ElementFindingData()
  req.prime_xpath = prime_xpath
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'android'
  req.prime_pixel_map.horizontal_scale = 2.0
  req.prime_pixel_map.vertical_scale = 2.0
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'android'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  req.revisit_pixel_map.horizontal_scale = 2.0
  req.revisit_pixel_map.vertical_scale = 2.0
  req.revisit_pixel_map.x_offset = 0
  req.revisit_pixel_map.y_offset = 0
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    return requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())

  except Exception as err:
    print('Exception:', err)
  return None

def call_finding_element_by_image_service(endpoint, screen, xml, query_image, scale_x=1, scale_y=1,
                                  platform='android', threshold=0, method=3 ):
  req = request_pb2.ElementFindingByImage()

  req.xml = xml
  req.screenshot = screen
  req.base64_encoded_query_image = query_image
  req.platform_name = platform
  req.pixel_map.horizontal_scale = scale_x
  req.pixel_map.vertical_scale = scale_y
  req.pixel_map.x_offset = 0
  req.pixel_map.y_offset = 0 
  req.method = method
  req.threshold = threshold
  
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    return requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())

  except Exception as err:
    print('Exception:', err)
  return None


def call_comparison_service(endpoint, xpath, prime_xml_binary, prime_screen_binary,
                            revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.ElementCompareData()
  req.prime_xpath = xpath
  req.revisit_xpath = xpath
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'android'
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'android'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
  return res


def call_text_assertion(
    endpoint, prime_screen_binary, prime_xml_binary,
    revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.TextAssertions()
  req.level = 'RELAXED_PLACEMENT'
  req.threshold = 0.9
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'android'
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'android'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  req.color_level = 'relaxed'
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
  return res


def call_screen_size_verification(endpoint):
  req = request_pb2.SameDeviceScreenSizeVerificationRequest()
  req.device_names.append('Galaxy S6')
  req.target_device_name = 'Galaxy S6'
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
  return res


def call_visual_assertion(endpoint, prime_screen, prime_xml_binary, revisit_screen, revisit_xml_binary):
  req = request_pb2.VisualVerification()
  req.prime_screen = prime_screen
  req.prime_xml = prime_xml_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_display_screen_size = 5.5
  req.prime_platform_name = 'android'
  req.revisit_display_screen_size = 5.5
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'android'
  req.revisit_screen = revisit_screen
  req.revisit_xml = revisit_xml_binary
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
  return res


def call_wbi_verification(endpoint, prime_screen, prime_xml_binary, revisit_screen, revisit_xml_binary,
                   prime_scale=1., revisit_scale=1., platform='android'):
  req = request_pb2.FontSize()
  req.prime_screen = prime_screen
  req.prime_xml = prime_xml_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_display_screen_size = 5.5
  req.prime_platform_name = platform
  req.prime_pixel_map.horizontal_scale = prime_scale
  req.prime_pixel_map.vertical_scale = prime_scale
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_display_screen_size = 5.5
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = platform
  req.revisit_screen = revisit_screen
  req.revisit_xml = revisit_xml_binary
  req.revisit_pixel_map.horizontal_scale = revisit_scale
  req.revisit_pixel_map.vertical_scale = revisit_scale
  req.revisit_pixel_map.x_offset = 0
  req.revisit_pixel_map.y_offset = 0
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
  return res


def call_visual_assertion_w_submission(endpoint, prime_screen, prime_xml_binary,
                                       revisit_screen, revisit_xml_binary):
  req = request_pb2.VisualVerification()

  structure_submission = req.structure_submissions.add()
  structure_submission.level = 'SKIP'
  structure_submission.prime_xpath = '/hierarchy/android.widget.FrameLayout/' \
                                     'android.widget.LinearLayout/android.widget.FrameLayout/' \
                                     'android.widget.FrameLayout/android.widget.LinearLayout/' \
                                     'android.widget.RelativeLayout/android.widget.LinearLayout/' \
                                     'android.widget.LinearLayout/android.widget.ImageView'

  layout_submission = req.layout_submissions.add()
  layout_submission.prime_xpath = '/hierarchy/android.widget.FrameLayout' \
                                  '/android.widget.LinearLayout/android.widget.FrameLayout' \
                                  '/android.widget.FrameLayout/android.widget.LinearLayout' \
                                  '/android.widget.FrameLayout/android.widget.LinearLayout' \
                                  '/android.support.v4.view.ViewPager/android.widget.ScrollView' \
                                  '/android.widget.FrameLayout/android.widget.RelativeLayout' \
                                  '/android.widget.Spinner/android.widget.LinearLayout' \
                                  '/android.widget.Button'
  layout_submission.level = 'SKIP'

  req.prime_screen = prime_screen
  req.prime_xml = prime_xml_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_display_screen_size = 5.5
  req.prime_platform_name = 'android'
  req.revisit_display_screen_size = 5.5
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'android'
  req.revisit_screen = revisit_screen
  req.revisit_xml = revisit_xml_binary
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
  return res


def call_text_assertion_w_prime_element_xpaths(endpoint, prime_screen_binary,
                                               prime_xml_binary, revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.TextAssertions()
  req.prime_element_xpaths.append('/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.widget.FrameLayout/android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.support.v4.view.ViewPager/'
                                  'android.widget.ScrollView/android.widget.FrameLayout/android.widget.RelativeLayout/'
                                  'android.widget.Spinner/android.widget.RelativeLayout/android.widget.TextView')
  req.prime_element_xpaths.append('/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.widget.FrameLayout/android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.support.v4.view.ViewPager/android.widget.ScrollView/'
                                  'android.widget.FrameLayout/android.widget.RelativeLayout/'
                                  'android.widget.Spinner/android.widget.EditText[1]')

  req.level = 'exact'
  req.color_level = 'relaxed'
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'android'
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'android'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
    return None
  return res

class AIServiceTestCase(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(AIServiceTestCase, self).__init__(*args, **kwargs)
    self.host = '0.0.0.0'
    self.port = '5000'
      
  def test_finding_element_by_image(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    manual_img_path = glob( os.path.join(current_dir, 'find_by_image/manual/*.jpg') )[0] or glob( os.path.join(current_dir, 'find_by_image/manual/*.jpeg') )[0]
    manual_query_img_path = glob( os.path.join(current_dir, 'find_by_image/*.jpg') )[0] or glob( os.path.join(current_dir, 'find_by_image/*.jpeg') )[0]
    manual_xml_path = glob(os.path.join(current_dir, "find_by_image/manual/*.xml"))[0]
    
    screen_binary = read_file_bytes(manual_img_path)
    xml_binary = read_file_bytes(manual_xml_path)
    query_img = cv2.imread(manual_query_img_path)
    retval, buffer = cv2.imencode('.jpg', query_img)
    string_query_image = base64.b64encode( buffer ) 
    correct_xpath = '/hierarchy/android.widget.FrameLayout[5]/android.widget.FrameLayout[1]/android.widget.LinearLayout'
    
    tree = etree.parse(manual_xml_path)
    root = tree.getroot()
    img = cv2.imread(manual_img_path)

    platform, scale_y, scale_x = get_platform_scale(root, img)

    url = f'http://{self.host}:{self.port}{constants.ELEMENT_FINDING_BY_IMAGE_URL}'
    url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
    start = time.time()
    try:
      res = call_finding_element_by_image_service(
        url, screen_binary, xml_binary, string_query_image,
        scale_x, scale_y, platform,
        method=3, threshold=0)
      print('Test finding element by image service...')
      print('==> Run time:', time.time() - start)
      print('==> Result: ', res.json())
      print('==> Correct xpath:', correct_xpath)
      print('==> Extracted xpath:', res.json()[-1]['xpath'])
      
      # Test result
      self.assertEqual(res.json()[-1]['xpath'], correct_xpath, "Correct xpath and found xpath are not the same.")
      self.assertGreater(res.json()[-1]['confidence'], 0.8, "Confidence of best element are lower than 0.8.")
    except Exception:
      print('Error when finding element by image.')
      print('Response:', res)
      print(traceback.format_exc())

def main_finding_element():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_tree = etree.fromstring(prime_xml_binary).getroottree()
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')

  ele_finder = etree.XPath("//*[@bounds][not(ancestor::android.webkit.WebView)]")
  candidates = ele_finder(prime_tree)
  count = 0
  for i, cand in enumerate(candidates):
    cand_xpath = prime_tree.getpath(cand)
    url = sys.argv[2]
    url += '?session_id={}&action_id={}&request_id={}'.format(1, i, time.time())
    res = call_finding_service(
      url, cand_xpath, prime_xml_binary, prime_screen_binary,
      prime_screen_binary, prime_xml_binary)
    try:
      print('\n\nIdx: {}'.format(i))
      print('Prime:', cand_xpath)
      found_element_xpath = res.json()['elements'][-1]['xpath']
      confidence = res.json()['elements'][-1]['confidence']
      print('Revis:', found_element_xpath)
      print('Confidence', confidence)
      if cand_xpath == found_element_xpath:
        count += 1
    except Exception as e:
      print('Error at', i, cand_xpath)
      print(e)
      print('Response:', res.json())

  print('Summary:')
  print('Found {}/{} elements'.format(count, len(candidates)))


def main_finding_key_in_keyboard():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime_keyboard.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime_keyboard.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit_keyboard.xml')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit_keyboard.jpg')

  cand_xpath = "/hierarchy/android.widget.FrameLayout[4]/android.view.View/bkn_x0024_a[20]"
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_finding_service(
    url, cand_xpath, prime_xml_binary, prime_screen_binary,
    revisit_screen_binary, revisit_xml_binary)
  try:
    print('call_finding_key_in_keyboard: ')
    print('Prime: ', cand_xpath)
    found_element_xpath = res.json()['elements'][-1]['xpath']
    confidence = res.json()['elements'][-1]['confidence']
    print('Revisit: ', found_element_xpath)
    print('Confidence: ', confidence)
  except Exception as e:
    print('Error when finding key in keyboard')
    print(e)
    print('Response:', res.json())

def main_compare_element():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_tree = etree.fromstring(prime_xml_binary).getroottree()
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')

  ele_finder = etree.XPath("//*[@bounds]")
  candidates = ele_finder(prime_tree)
  count = 0
  for i, cand in enumerate(candidates):
    cand_xpath = prime_tree.getpath(cand)
    url = sys.argv[2]
    url += '?session_id={}&action_id={}&request_id={}'.format(1, i, time.time())
    res = call_comparison_service(
      url, cand_xpath, prime_xml_binary, prime_screen_binary,
      revisit_screen_binary, revisit_xml_binary)
    try:
      score = res.json()['elements']
      print('\n\nIdx: {}'.format(i))
      print('cand: ', cand_xpath)
      print('comparing score: ', score)
      if score > 0.9:
        count += 1
    except Exception as e:
      print('Error at', i, cand_xpath)
      print(e)
      print('Response:', res.json())
  print('Summary:')
  print('Found {}/{} elements'.format(count, len(candidates)))


def main_text_assertion():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_text_assertion(
    url, prime_screen_binary, prime_xml_binary,
    revisit_screen_binary, revisit_xml_binary)
  print('call_text_assertion: ', res.json())


def main_text_assertion_w_prime_element_xpaths():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_text_assertion_w_prime_element_xpaths(url, prime_screen_binary, prime_xml_binary,
                                                   revisit_screen_binary, revisit_xml_binary)
  print('call_text_assertion: ', res.json())


def main_visual_assertion():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_visual_assertion(url, prime_screen_binary, prime_xml_binary,
                              revisit_screen_binary, revisit_xml_binary)
  print('call_visual_assertion: ', res.json())


def main_wbi_verification():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_wbi_verification(url, prime_screen_binary, prime_xml_binary,
                              revisit_screen_binary, revisit_xml_binary)
  print('call_wbi_verification: ', res.json())


def main_visual_assertion_w_submission():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_visual_assertion_w_submission(url, prime_screen_binary, prime_xml_binary,
                                           revisit_screen_binary, revisit_xml_binary)
  print('call_visual_assertion: ', res.json())


def main_screen_size_verification():
  url = sys.argv[2]
  res = call_screen_size_verification(url)
  print('call_screen_size_verification: ', res.json())


def main_fontsize_recommendation():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  xml_binary = read_file_bytes(current_dir + '/prime.xml')
  screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  url = sys.argv[2]

  req = request_pb2.RecommendScreenRequest()
  req.screen = screen_binary
  req.xml = xml_binary
  req.device_name = 'prime_device_name'
  req.display_screen_size = 5.5
  req.platform_name = 'android'
  req.element_xpaths.append('/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.widget.FrameLayout/android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.widget.FrameLayout/android.widget.LinearLayout/'
                                  'android.support.v4.view.ViewPager/android.widget.ScrollView/'
                                  'android.widget.FrameLayout/android.widget.RelativeLayout/'
                                  'android.widget.Spinner/android.widget.FrameLayout/android.widget.TextView[1]')
  req.element_xpaths.append('/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/'
                                   'android.widget.FrameLayout/android.widget.FrameLayout/android.widget.LinearLayout/'
                                   'android.widget.FrameLayout/android.widget.LinearLayout/'
                                   'android.support.v4.view.ViewPager/android.widget.ScrollView/'
                                   'android.widget.FrameLayout/android.widget.RelativeLayout/'
                                   'android.widget.Spinner/android.widget.TextView[2]')
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      url, headers=headers,
      data=req.SerializeToString())
    print('call_recommendation_fontsize: ', res.json())
  except Exception as err:
    print('Exception:', err)

def main_accessibility_assertion():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  accessibility_test_dir = current_dir + '/accessibility'
  number = '01'
  session_type = 'prime'
  accessibility_test_file = accessibility_test_dir + '/%s/%s' % (number, session_type)
  xml_binary = read_file_bytes(accessibility_test_file + '.xml')
  screen_binary = read_file_bytes(accessibility_test_file + '.jpg')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  req = request_pb2.AccessibilityAssertionRequest()
  req.platform_name = 'android'
  req.xml = xml_binary
  req.screen = screen_binary
  req.screen_density = 480

  print("check accessibility session: ", accessibility_test_file)
  headers = {
    "Accept": "application/x-protobuf",
    "Content-type": "application/x-protobuf"}
  try:
    res = requests.post(
      url, headers=headers,
      data=req.SerializeToString())
    print('call_accessibility_assertion: ', res.json())
  except Exception as err:
    print('Exception:', err)

if __name__ == '__main__':
  unittest.main()
  if sys.argv[1] == 'text_assertion':
    main_text_assertion()
  elif sys.argv[1] == 'text_assertion_w_prime_element_xpaths':
    main_text_assertion_w_prime_element_xpaths()
  elif sys.argv[1] == 'compare':
    main_compare_element()
  elif sys.argv[1] == 'find':
    main_finding_element()
  elif sys.argv[1] == 'visual_assert':
    main_visual_assertion()
  elif sys.argv[1] == 'wbi':
    main_wbi_verification()
  elif sys.argv[1] == 'visual_assert_w_submission':
    main_visual_assertion_w_submission()
  elif sys.argv[1] == 'same_screen':
    main_screen_size_verification()
  elif sys.argv[1] == 'rec_fontsize':
    main_fontsize_recommendation()
  elif sys.argv[1] == 'accessibility_assert':
    main_accessibility_assertion()
  elif sys.argv[1] == 'find_key_in_keyboard':
    main_finding_key_in_keyboard()
