import sys
import time
from multiprocessing import Pool
import requests
import numpy as np
import service.schema.booster.ai.service_request_pb2 as request_pb2


def read_file_bytes(filename):
  with open(filename, 'rb') as f:
    byte_data = f.read()
    return byte_data


def get_response_time(endpoint):
  req = request_pb2.ElementFindingData()
  req.prime_xpath = '''/hierarchy/android.widget.FrameLayout/android.widget.LinearLayout/
                       android.widget.FrameLayout/android.widget.FrameLayout/
                       android.widget.LinearLayout/android.widget.RelativeLayout/
                       android.widget.LinearLayout/android.widget.TextView'''
  prime_screen_binary = read_file_bytes('service/benchmark/prime.jpg')
  prime_xml_binary = read_file_bytes('service/benchmark/prime.xml')
  revisit_screen_binary = read_file_bytes('service/benchmark/revisit.jpg')
  revisit_xml_binary = read_file_bytes('service/benchmark/revisit.xml')
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
  start = time.time()
  headers = {
      "Accept": "application/x-protobuf",
      "Content-type": "application/x-protobuf"}
  try:
    requests.post(
        endpoint, headers=headers,
        data=req.SerializeToString())
  except Exception as err:
    print('Exception:', err)
    return 300
  return time.time() - start


def main():
  if len(sys.argv) < 3:
    print('Usage: host port num_req')
    sys.exit()
  url = sys.argv[1]
  pools = Pool(200)
  n_id = int(sys.argv[2])
  urls = [url + '&action_id=%s'%(i,) for i in range(n_id)]
  ex_times = pools.map(get_response_time, urls)
  print('average time: ', np.average(ex_times))


if __name__ == '__main__':
  main()
