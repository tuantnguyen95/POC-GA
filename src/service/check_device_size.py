import os
from pandas import read_csv

PHONE_DATA_PATH = os.environ.get('PHONE_DATA_PATH', './service/static/screen_sizes.csv')
PHONE_DATA = read_csv(PHONE_DATA_PATH, sep=";")


def get_phone_size(device_name):
  values = PHONE_DATA[PHONE_DATA['name'].str.lower() == device_name.lower()]['size'].values
  return values[0] if len(values) > 0 else None


def get_same_size_devices(device_name, candidate_names):
  result = []
  device_size = get_phone_size(device_name)
  if device_size is not None:
    for cand in candidate_names:
      cand_size = get_phone_size(cand)
      if cand_size == device_size:
        result.append(cand)
  return {'screenSize': device_size, 'deviceNames': result}
