import numpy as np
import sys
import re
import os
import csv
from collections import defaultdict
import random

csvFile = sys.argv[1]
outFile = csvFile + '.filterred'
DUPLICATE_THRES = 30
element_name_neg, element_name_pos = defaultdict(list), defaultdict(list)
neg_locating_eles_maps = defaultdict(lambda : defaultdict(int))
pos_locating_eles_maps = defaultdict(lambda : defaultdict(int))

pos_locating_counts = defaultdict(int)
neg_locating_counts = defaultdict(int)
skipping_lines = {}
count = -1
with open(csvFile, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
      count += 1
      (matched, screen_matching, package, prime_deivce_name,
       prime_device_api_level, prime_screen_size, prime_device_density,
       prime_model, prime_pixel_ratio, prime_stat_bar_height,
       prime_screen_name, revisit_screen_name, prime_activity_name,
       prime_session_id, prime_element_name, prime_locating_id, prime_xpath,
       prime_left, prime_top, prime_right, prime_bottom, prime_clickable,
       revisit_device_name, revisit_api_level, revisit_screen_size,
       revisit_screen_density, revisit_model, revisit_pixel_ratio,
       revisit_stat_bar_height, revisit_activity_name, revisit_session_id,
       revisit_element_name, revisit_locating_id,
       revisit_xpath, revisit_left, revisit_top, revisit_right,
       revisit_bottom, revisit_clickable) = row
      if (screen_matching == '1' or not prime_locating_id or
         not prime_locating_id.isspace()):
        skipping_lines[count] = 1
        continue

      if matched == '-1':
        element_name_neg[prime_element_name].append(count)
        neg_locating_counts[package + prime_locating_id] += 1
        neg_locating_eles_maps[package + prime_locating_id][prime_element_name] += 1
      else:
        element_name_pos[prime_element_name].append(count)
        pos_locating_counts[package + prime_locating_id] += 1
        pos_locating_eles_maps[package + prime_locating_id][prime_element_name] += 1

for prime_locating_id in pos_locating_counts.keys():
  locating_count = pos_locating_counts[prime_locating_id]
  while (locating_count > DUPLICATE_THRES and
          len(pos_locating_eles_maps[prime_locating_id]) > 1):
    removing_prime_ele = random.choice(
                        list(pos_locating_eles_maps[prime_locating_id].keys()))
    locating_count -= pos_locating_eles_maps[
                        prime_locating_id][removing_prime_ele]
    for line_pos in element_name_pos[removing_prime_ele]:
      skipping_lines[line_pos] = 1
    del pos_locating_eles_maps[prime_locating_id][removing_prime_ele]

for prime_locating_id in neg_locating_counts.keys():
  locating_count = neg_locating_counts[prime_locating_id]
  while (locating_count > 3*DUPLICATE_THRES and
         len(neg_locating_eles_maps[prime_locating_id]) > 1):
    removing_prime_ele = random.choice(
                      list(neg_locating_eles_maps[prime_locating_id].keys()))
    locating_count -= neg_locating_eles_maps[
                            prime_locating_id][removing_prime_ele]
    for line_pos in element_name_neg[removing_prime_ele]:
      skipping_lines[line_pos] = 1
    del neg_locating_eles_maps[prime_locating_id][removing_prime_ele]
    
count = -1
with open(csvFile, 'r') as f, open(outFile, 'w') as w:
  next(f)
  for line in f:
    count += 1
    if count in skipping_lines:
      continue
    w.write (line)
      