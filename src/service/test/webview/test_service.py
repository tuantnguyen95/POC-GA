import sys
import os
import time
import requests
from lxml import html, etree
import service.schema.booster.ai.service_request_pb2 as request_pb2
import service.utils as u


def read_file_bytes(filename):
  with open(filename, 'rb') as data:
    byte_data = data.read()
    return byte_data


def call_finding_service(endpoint, prime_xpath, prime_xml_binary, prime_screen_binary,
                         revisit_screen_binary, revisit_xml_binary):
  req = request_pb2.ElementFindingData()
  req.prime_xpath = prime_xpath
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


def call_text_assertion(endpoint, prime_screen_binary, prime_xml_binary,
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


def main_finding_element():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_tree = etree.fromstring(prime_xml_binary).getroottree()
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  ele_finder = etree.XPath("//*[@bounds][not(ancestor-or-self::android.webkit.WebView)]")
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


def main_finding_html_element():
  webview_xpath = '/hierarchy/android.widget.FrameLayout/android.widget.FrameLayout[1]/android.webkit.WebView'
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_html_binary = u.get_html(prime_xml_binary)
  prime_html_tree = html.fromstring(prime_html_binary).getroottree()
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  ele_finder = etree.XPath("//*[not(self::script or self::style or self::noscript or self::path or self::hr)]"
                           "[@bounds]"
                           "[self::svg or count(./*)=0]")
  candidates = ele_finder(prime_html_tree)
  count = 0
  for i, cand in enumerate(candidates):
    cand_xpath = webview_xpath + prime_html_tree.getpath(cand)
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


def main_compare_element():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  prime_xml_binary = read_file_bytes(current_dir + '/prime.xml')
  prime_tree = html.fromstring(prime_xml_binary).getroottree()
  prime_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/prime.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/prime.xml')

  ele_finder = etree.XPath("//svg[@bounds]")
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
  prime_xml_binary = read_file_bytes(current_dir + '/text/prime.xml')
  prime_screen_binary = read_file_bytes(current_dir + '/text/prime.jpg')
  revisit_screen_binary = read_file_bytes(current_dir + '/text/prime.jpg')
  revisit_xml_binary = read_file_bytes(current_dir + '/text/prime.xml')
  url = sys.argv[2]
  url += '?session_id={}&action_id={}&request_id={}'.format(1, 0, time.time())
  res = call_text_assertion(
    url, prime_screen_binary, prime_xml_binary,
    revisit_screen_binary, revisit_xml_binary)
  print('call_text_assertion: ', res.json())


if __name__ == '__main__':
  if sys.argv[1] == 'compare':
    main_compare_element()
  elif sys.argv[1] == 'find':
    main_finding_element()
  elif sys.argv[1] == 'find_html':
    main_finding_html_element()
  elif sys.argv[1] == 'text_assertion':
    main_text_assertion()
