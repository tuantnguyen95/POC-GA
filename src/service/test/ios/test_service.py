import sys
import os
import time
import requests
from lxml import etree
import service.schema.booster.ai.service_request_pb2 as request_pb2


def read_file_bytes(filename):
  with open(filename, 'rb') as f:
    byte_data = f.read()
    return byte_data


def call_finding_service(endpoint, prime_xpath, prime_screen_binary, prime_xml_binary,
                         revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.ElementFindingData()
  req.prime_xpath = prime_xpath
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'ios'
  req.prime_pixel_map.horizontal_scale = 2.0
  req.prime_pixel_map.vertical_scale = 2.0
  req.prime_pixel_map.horizontal_pre_scale = 1
  req.prime_pixel_map.vertical_pre_scale = 1
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'ios'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  req.revisit_pixel_map.horizontal_scale = 2.0
  req.revisit_pixel_map.vertical_scale = 2.0
  req.revisit_pixel_map.horizontal_pre_scale = 1
  req.revisit_pixel_map.vertical_pre_scale = 1
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
    return 300
  return res


def call_comparison_service(endpoint, xpath, prime_screen_binary,
                            prime_xml_binary, revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.ElementCompareData()
  req.prime_xpath = xpath
  req.revisit_xpath = xpath
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'ios'
  req.prime_pixel_map.horizontal_scale = 2.0
  req.prime_pixel_map.vertical_scale = 2.0
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'ios'
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
    res = requests.post(
      endpoint, headers=headers,
      data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
    return 300
  return res


def call_visual_assertion(endpoint, prime_screen, prime_xml_binary, revisit_screen, revisit_xml_binary):
  req = request_pb2.VisualVerification()
  req.prime_screen = prime_screen
  req.prime_xml = prime_xml_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_display_screen_size = 6.1
  req.prime_platform_name = 'ios'
  req.prime_pixel_map.horizontal_scale = 2.0
  req.prime_pixel_map.vertical_scale = 2.0
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_display_screen_size = 6.1
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'ios'
  req.revisit_screen = revisit_screen
  req.revisit_xml = revisit_xml_binary
  req.revisit_pixel_map.horizontal_scale = 3.0
  req.revisit_pixel_map.vertical_scale = 3.0
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


def call_text_assertion(endpoint, prime_screen_binary,
                        prime_xml_binary, revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.TextAssertions()

  req.level = 'exact'
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'ios'
  req.prime_pixel_map.horizontal_scale = 2.0
  req.prime_pixel_map.vertical_scale = 2.0
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'ios'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  req.revisit_pixel_map.horizontal_scale = 2.0
  req.revisit_pixel_map.vertical_scale = 2.0
  req.revisit_pixel_map.x_offset = 0
  req.revisit_pixel_map.y_offset = 0
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
    return None
  return res


def call_text_assertion_w_submission(endpoint, prime_screen_binary,
                                     prime_xml_binary, revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.TextAssertions()

  submission = req.submissions.add()
  submission.level = 'beginning_of_text'
  submission.prime_xpath = '/AppiumAUT/XCUIElementTypeApplication/XCUIElementTypeWindow[1]' \
                           '/XCUIElementTypeOther/XCUIElementTypeOther' \
                           '/XCUIElementTypeOther/XCUIElementTypeStaticText'
  submission.threshold = 0.8

  color_submission = req.color_submissions.add()
  color_submission.level = 'strict'
  color_submission.prime_xpath = '/AppiumAUT/XCUIElementTypeApplication/XCUIElementTypeWindow[1]' \
                                 '/XCUIElementTypeOther/XCUIElementTypeOther' \
                                 '/XCUIElementTypeOther/XCUIElementTypeStaticText'

  req.level = 'exact'
  req.color_level = 'relaxed'
  req.prime_xml = prime_xml_binary
  req.prime_screen = prime_screen_binary
  req.prime_device_name = 'prime_device_name'
  req.prime_screen_density = 100
  req.prime_platform_name = 'ios'
  req.prime_pixel_map.horizontal_scale = 3.0
  req.prime_pixel_map.vertical_scale = 3.0
  req.prime_pixel_map.x_offset = 0
  req.prime_pixel_map.y_offset = 0
  req.revisit_screen_density = 200
  req.revisit_device_name = 'revisit_device_name'
  req.revisit_platform_name = 'ios'
  req.revisit_screen = revisit_screen_binary
  req.revisit_xml = revisit_xml_binary
  req.revisit_pixel_map.horizontal_scale = 3.0
  req.revisit_pixel_map.vertical_scale = 3.0
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
    return None
  return res


def main_finding_element():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  prime_tree = etree.fromstring(prime_xml_binary).getroottree()
  ele_finder = etree.XPath("//*[@width][@visible='true'][not(ancestor::XCUIElementTypeWebView)]")
  candidates = ele_finder(prime_tree)
  count = 0
  for i, cand in enumerate(candidates):
    cand_xpath = prime_tree.getpath(cand)
    url = sys.argv[2]
    url += '?session_id={}&action_id={}&request_id={}'.format(1, i, time.time())
    res = call_finding_service(url, cand_xpath, prime_screen_binary, prime_xml_binary,
                               revisit_screen_binary, revisit_xml_binary)
    print('i: ', i)
    print('Prime:', cand_xpath)
    try:
      if 'elements' in res.json() and len(res.json()['elements']) > 0:
        found_element_xpath = res.json()['elements'][-1]['xpath']
        print('Revis:', res.json()['elements'][-1]['confidence'], ' ', found_element_xpath, '\n\n')
        if cand_xpath == found_element_xpath:
          count += 1
      else:
        print('Not found\n\n')
    except Exception as e:
      print('Error at', i, cand_xpath)
      print(e)
      print('Response:', res.json())
  print('Summary:')
  print('Found {}/{} elements'.format(count, len(candidates)))


def main_finding_element_webview():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/screen-webview.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/screen-webview.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/screen-webview.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/screen-webview.xml')

  cand_xpath = "/AppiumAUT/XCUIElementTypeApplication/XCUIElementTypeWindow[1]/XCUIElementTypeOther/XCUIElementTypeOther[2]/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther/XCUIElementTypeOther[2]/XCUIElementTypeWebView/XCUIElementTypeWebView"
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_finding_service(url, cand_xpath, prime_screen_binary, prime_xml_binary,
                               revisit_screen_binary, revisit_xml_binary)
  print('Prime:', cand_xpath)
  try:
    if 'elements' in res.json() and len(res.json()['elements']) > 0:
      found_element_xpath = res.json()['elements'][-1]['xpath']
      print('Revis:', res.json()['elements'][-1]['confidence'], ' ', found_element_xpath, '\n\n')
      if cand_xpath != found_element_xpath:
        print('Not found\n\n')
  except Exception as e:
    print('Error at', cand_xpath)
    print(e)
    print('Response:', res.json())


def main_comparison_element():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_tree = etree.fromstring(prime_xml_binary).getroottree()
  ele_finder = etree.XPath("//*[@width][@visible='true']")
  candidates = ele_finder(prime_tree)
  count = 0
  for i, cand in enumerate(candidates):
    cand_xpath = prime_tree.getpath(cand)
    print('\n\nIdx: {}'.format(i))
    print('cand_xpath: ' + cand_xpath)
    url = sys.argv[2]
    url += '?session_id={}&action_id={}&request_id={}'.format(1, i, time.time())
    res = call_comparison_service(
      url, cand_xpath, prime_screen_binary, prime_xml_binary,
      prime_screen_binary, prime_xml_binary)
    try:
      score = res.json()['elements']
      print('comparing score: ', score)
      count += 1
    except Exception as e:
      print('Error at', i, cand_xpath)
      print(e)
      print('Response:', res.json())
      time.sleep(5)
  print('Summary:')
  print('Found {}/{} elements'.format(count, len(candidates)))


def main_text_assertion():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  rev_screen_binary = read_file_bytes(current_dir + '/revisit.jpg')
  rev_xml_binary = read_file_bytes(current_dir + '/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_text_assertion(
    url, prime_screen_binary, prime_xml_binary,
    rev_screen_binary, rev_xml_binary)
  print('call_text_assertion: ', res.json())


def main_text_assertion_w_submission():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_screen_binary = read_file_bytes(current_dir + '/submission_case/prime.jpeg')
  prime_xml_binary = read_file_bytes(current_dir + '/submission_case/prime.xml')
  rev_screen_binary = read_file_bytes(current_dir + '/submission_case/revisit.jpeg')
  rev_xml_binary = read_file_bytes(current_dir + '/submission_case/revisit.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_text_assertion_w_submission(url, prime_screen_binary, prime_xml_binary,
                                         rev_screen_binary, rev_xml_binary)
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

def main_accessibility_assertion():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  accessibility_test_dir = current_dir + '/accessibility'
  number = '02-ipad'
  session_type = 'revisit'
  accessibility_test_file = accessibility_test_dir + '/%s/%s' % (number, session_type)
  xml_binary = read_file_bytes(accessibility_test_file + '.xml')
  screen_binary = read_file_bytes(accessibility_test_file + '.jpeg')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  req = request_pb2.AccessibilityAssertionRequest()
  req.platform_name = 'ios'
  req.xml = xml_binary
  req.screen = screen_binary
  req.screen_density = 0
  req.pixel_map.horizontal_scale = 2.0
  req.pixel_map.vertical_scale = 2.0
  req.pixel_map.x_offset = 0
  req.pixel_map.y_offset = 0

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
  if sys.argv[1] == 'text_assertion':
    main_text_assertion()
  elif sys.argv[1] == 'text_assertion_w_submission':
    main_text_assertion_w_submission()
  elif sys.argv[1] == 'compare':
    main_comparison_element()
  elif sys.argv[1] == 'find':
    main_finding_element()
  elif sys.argv[1] == 'find_webview':
    main_finding_element_webview()
  elif sys.argv[1] == 'accessibility_assert':
      main_accessibility_assertion()
  else:
    main_visual_assertion()
