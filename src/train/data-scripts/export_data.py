import re
import sqlite3
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
import argparse


def get_base_tables(db_path):
  db = sqlite3.connect(db_path)
  match_df = pd.read_sql_query("select * from matches;", db)
  element_df = pd.read_sql_query("select * from elements;", db)
  device_df = pd.read_sql_query("select * from devices;", db)
  screen_df = pd.read_sql_query("select * from screens;", db)
  db.close()
  return match_df, element_df, device_df, screen_df


def preprocess_match_df(match_df, drop_cols=('id', 'matched_by', 'user_id', 'annotated_time')):
  match_df.dropna(subset=['user_id'], inplace=True)
  match_df.drop(list(drop_cols), axis=1, inplace=True)
  return match_df


def preprocess_element_df(element_df, drop_cols=('file_path', 'inserted', 'auto_id')):
  element_df['scrollable'] = -1
  element_df.drop(list(drop_cols), axis=1, inplace=True)
  return element_df


def preprocess_screen_df(screen_df, drop_cols=('insert_time', 'file_path',
                                               'duplicated', 'not_found')):
  screen_df.drop(list(drop_cols), axis=1, inplace=True)
  return screen_df


def preprocess_device_df(device_df, drop_cols=()):
  return device_df


def merge_df(df, element_df, screen_df, device_df, drop_screen_name_first=False):
  if drop_screen_name_first:
    element_df = element_df.drop(['screen_name'], axis=1)
  merged_df = pd.merge(left=df, right=element_df.add_prefix('prime_'),
                       left_on='prime_element_name',
                       right_on='prime_element_name')
  merged_df = pd.merge(left=merged_df, right=element_df.add_prefix('revisit_'),
                       left_on='revisit_element_name',
                       right_on='revisit_element_name')
  merged_df = pd.merge(left=merged_df, right=screen_df.add_prefix('prime_'),
                       left_on='prime_screen_name',
                       right_on='prime_name')
  merged_df.drop(['prime_name'], axis=1, inplace=True)
  merged_df = pd.merge(left=merged_df, right=screen_df.add_prefix('revisit_'),
                       left_on='revisit_screen_name',
                       right_on='revisit_name')
  merged_df.drop(['revisit_name'], axis=1, inplace=True)
  merged_df = pd.merge(left=merged_df, right=device_df.add_prefix('prime_'),
                       left_on='prime_device_id',
                       right_on='prime_device_id')
  merged_df.rename({'prime_name': 'prime_device_name'}, axis=1, inplace=True)
  merged_df = pd.merge(left=merged_df, right=device_df.add_prefix('revisit_'),
                       left_on='revisit_device_id',
                       right_on='revisit_device_id')
  merged_df.rename({'revisit_name': 'revisit_device_name'}, axis=1, inplace=True)
  return merged_df


def remove_duplicated_pairs(df):
  temp_df = pd.DataFrame(np.sort(df[['prime_element_name', 'revisit_element_name']], axis=1))
  df = df[~temp_df.duplicated()].reset_index(drop=True)
  return df


def get_positive_df(df):
  positive_df = df[(df['matched'] == 1) & (df['is_good_match'] == 1) &
                   (df['not_found'] != 1)].reset_index(drop=True)
  return positive_df


def calculate_iou(ele_1, ele_2, element_df):
  temp_element_df = element_df.set_index('element_name')
  try:
    left_1 = temp_element_df.loc[ele_1, 'left']
    left_2 = temp_element_df.loc[ele_2, 'left']
    right_1 = temp_element_df.loc[ele_1, 'right']
    right_2 = temp_element_df.loc[ele_2, 'right']
    top_1 = temp_element_df.loc[ele_1, 'top']
    top_2 = temp_element_df.loc[ele_2, 'top']
    bot_1 = temp_element_df.loc[ele_1, 'bottom']
    bot_2 = temp_element_df.loc[ele_2, 'bottom']
  except KeyError as e:
    print(e)
    return np.NaN
  # determine the (x, y)-coordinates of the intersection rectangle
  x_a = max(left_1, left_2)
  y_a = max(top_1, top_2)
  x_b = min(right_1, right_2)
  y_b = min(bot_1, bot_2)
  # compute the area of intersection rectangle
  inter_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
  if inter_area == 0:
    return 0
  box_a_area = abs((right_1 - left_1) * (bot_1 - top_1))
  box_b_area = abs((right_2 - left_2) * (bot_2 - top_2))
  iou = inter_area / float(box_a_area + box_b_area - inter_area)
  return iou


def filter_overlaps_with_matched_element(match_ele, unmatched_element_ls, element_df):
  clone_unmatched_elements = unmatched_element_ls.copy()
  for unmatched_ele in clone_unmatched_elements:
    if calculate_iou(match_ele, unmatched_ele, element_df) != 0:
      print('Dropped {} due to overlap with {}'.format(unmatched_ele, match_ele))
      unmatched_element_ls.remove(unmatched_ele)
  return unmatched_element_ls


def generate_positive_df(positive_df):
  matching_groups = positive_df.groupby(['prime_element_name'])['revisit_element_name'].apply(
    list).reset_index(name='revisit_elements')['revisit_elements'].tolist()
  pairs = []
  for group in matching_groups:
    comb = list(combinations(group, 2))
    for pair in comb:
      pairs.append(pair)
  generated_positive_df = pd.DataFrame(pairs, columns=['prime_element_name',
                                                       'revisit_element_name'])
  generated_positive_df['matched'] = 1
  generated_positive_df['not_found'] = 0
  generated_positive_df['is_visually_same'] = 1
  generated_positive_df['is_good_match'] = 1
  print('Generated positive df is {}'.format(generated_positive_df.shape[0]))
  return generated_positive_df


def generate_negative_df(positive_df, element_df):
  temp_element_df = element_df.set_index('element_name')
  generated_neg_ls = []
  for _, row in positive_df.iterrows():
    prime_name = row['prime_element_name']
    revisit_name = row['revisit_element_name']
    try:
      prime_screen = temp_element_df.loc[prime_name]['screen_name']
      revisit_screen = temp_element_df.loc[revisit_name]['screen_name']
    except KeyError as e:
      print(e)
      continue
    unmatched_elements = element_df[(element_df['element_name'] != revisit_name) &
                                    (element_df['screen_name'] == revisit_screen)][
      'element_name'].tolist()
    unmatched_elements = filter_overlaps_with_matched_element(revisit_name, unmatched_elements, element_df)
    for unmatched_element in unmatched_elements:
      new_sample = (prime_screen, revisit_screen, prime_name, unmatched_element)
      generated_neg_ls.append(new_sample)
  generated_neg_df = pd.DataFrame(generated_neg_ls,
                                  columns=['prime_screen_name', 'revisit_screen_name',
                                           'prime_element_name', 'revisit_element_name'])
  generated_neg_df['matched'] = -1
  generated_neg_df['not_found'] = 0
  generated_neg_df['is_visually_same'] = 0
  generated_neg_df['is_good_match'] = 1
  print('Generated negative df is {}'.format(generated_neg_df.shape[0]))
  return generated_neg_df


def lev_distance(s1, s2):
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  distances = range(len(s1) + 1)
  for idx_2, char_2 in enumerate(s2):
    distances_ = [idx_2 + 1]
    for idx_1, char_1 in enumerate(s1):
      if char_1 == char_2:
        distances_.append(distances[idx_1])
      else:
        distances_.append(
          1 + min((distances[idx_1],
                   distances[idx_1 + 1], distances_[-1])))
    distances = distances_
  return distances[-1]


def xpath_similarity(row):
  return 1 - lev_distance(row.prime_xpath, row.revisit_xpath) / \
         max(len(row.prime_xpath), len(row.revisit_xpath))


def parse_xpath(xpath, id_equal=True):
  """
  Get a xpath and return a list of Android classes and their ids (if available) in that xpath.
  """
  regex = r"\/([\w+\.?]+)|(\[\d+\])"
  patterns = re.findall(regex, xpath)
  tokens = [''.join(pattern) for pattern in patterns]
  tokens = [token.lower() for token in tokens]  # Convert string to lowercase
  if id_equal:
    # Convert all ids into <idx> to treat different id equally
    for idx, token in enumerate(tokens):
      if token.startswith('['):
        tokens[idx] = '<idx>'
  return tokens


def count_vectorize_xpath(xpaths):
  vocabulary = {'<idx>': 1,
                'appiumaut': 2,
                'xcuielementtypealert': 3,
                'xcuielementtypeapplication': 4,
                'xcuielementtypebutton': 5,
                'xcuielementtypecell': 6,
                'xcuielementtypecollectionview': 7,
                'xcuielementtypeimage': 8,
                'xcuielementtypekeyboard': 9,
                'xcuielementtypelink': 10,
                'xcuielementtypenavigationbar': 11,
                'xcuielementtypeother': 12,
                'xcuielementtypepageindicator': 13,
                'xcuielementtypescrollview': 14,
                'xcuielementtypestatictext': 15,
                'xcuielementtypetabbar': 16,
                'xcuielementtypetable': 17,
                'xcuielementtypetextfield': 18,
                'xcuielementtypetextview': 19,
                'xcuielementtypewebview': 20,
                'xcuielementtypewindow': 21,
                'UNKNOWN': 0
                }
  transformer = CountVectorizer(tokenizer=parse_xpath, vocabulary=vocabulary)
  vector = transformer.transform(xpaths)
  return list(vector.toarray())


def cosine(row):
  return np.dot(row['prime_vector'], row['revisit_vector'].T) / \
         (norm(row['prime_vector']) * norm(row['revisit_vector']))


def filter_negative_df_cosine(df):
  """
  Due to so many negative samples we generate, we will just pick the samples
  which has the highest cosine similarity.
  :param df: df that contains only negative samples.
  :return: df with top 2 cosine similarity negative samples.
  """
  df['prime_vector'] = count_vectorize_xpath(df['prime_xpath'])
  df['revisit_vector'] = count_vectorize_xpath(df['revisit_xpath'])
  df['cos_sim'] = df.apply(cosine, axis=1)
  df.sort_values(['cos_sim'], ascending=False, inplace=True)
  df = df.groupby(['prime_element_name', 'revisit_screen_name']) \
    .head(3).reset_index(drop=True)
  df.drop(['cos_sim', 'prime_vector', 'revisit_vector'], axis=1, inplace=True)
  return df


def filter_negative_df_lev(df):
  """
  Due to so many negative samples we generate, we will just pick the samples which
  has the highest Levenshtein similarity.
  :param df: df that contains only negative samples.
  """
  print('Warning: Levenshtein metric requires very long time to run and is unstable')
  df['xpath_sim'] = df.apply(xpath_similarity, axis=1)
  df.sort_values(['xpath_sim'], ascending=False, inplace=True)
  df = df.groupby(['prime_element_name']).head(2).reset_index(drop=True)
  df.drop(['xpath_sim'], axis=1, inplace=True)
  return df


def filter_negative_df(df, sim_metric='cosine'):
  if sim_metric == 'cosine':
    df = filter_negative_df_cosine(df)
  elif sim_metric == 'lev':
    df = filter_negative_df_lev(df)
  else:
    raise Exception('Only cosine and lev are available metrics')
  print('Finish filtering negative')
  print('Negative generated samples is {}'.format(df.shape[0]))
  return df


def remove_wrong_logic_coordinate(df):
  """
  Drop rows which contain coordinate value < 0 and too small size
  :param df:
  :return: filtered df
  """
  previous_len = len(df)
  df.drop(df[(df['prime_right'] < 0) | (df['prime_left'] < 0) |
             (df['prime_bottom'] < 0) | (df['prime_top'] < 0)].index, inplace=True)
  df.drop(df[(df['revisit_right'] < 0) | (df['revisit_left'] < 0) |
             (df['revisit_bottom'] < 0) | (df['revisit_top'] < 0)].index, inplace=True)
  df.drop(df[(df['prime_right'] - df['prime_left'] < 10) |
             (df['prime_bottom'] - df['prime_top'] < 10)].index, inplace=True)
  df.drop(df[(df['revisit_right'] - df['revisit_left'] < 10) |
             (df['revisit_bottom'] - df['revisit_top'] < 10)].index, inplace=True)
  after_len = len(df)
  print("Number of samples was dropped:", previous_len - after_len)
  return df


def rearrange_to_standard_columns(df):
  standard_cols = ['matched', 'package', 'prime_device_name', 'prime_device_api_level',
                   'prime_screen_size',
                   'prime_device_density', 'prime_model',
                   'prime_pixel_ratio', 'prime_stat_bar_height',
                   'prime_screen_name',
                   'revisit_screen_name', 'prime_activity_name',
                   'prime_session_id', 'prime_element_name',
                   'prime_locating_id',
                   'prime_xpath', 'prime_left', 'prime_top',
                   'prime_right', 'prime_bottom', 'prime_clickable',
                   'revisit_device_name',
                   'revisit_api_level', 'revisit_screen_size',
                   'revisit_screen_density', 'revisit_model',
                   'revisit_pixel_ratio',
                   'revisit_stat_bar_height', 'revisit_activity_name',
                   'revisit_session_id', 'revisit_element_name',
                   'revisit_locating_id', 'revisit_xpath',
                   'revisit_left', 'revisit_top', 'revisit_right',
                   'revisit_bottom',
                   'revisit_clickable', 'prime_txt', 'prime_recur_text',
                   'revisit_txt', 'revisit_recur_text']
  df.rename({'prime_app_package': 'package',
             'prime_api_level': 'prime_device_api_level',
             'prime_screen_density': 'prime_device_density'
             }, axis=1, inplace=True)
  df = df[standard_cols]
  return df


def postprocess(df):
  df = rearrange_to_standard_columns(df)
  df = remove_wrong_logic_coordinate(df)
  return df


def split_data_by_quantity(df, rate=0.4):
  """
  Split data by picking apps for test data that assure the ratio
  number of elements in test apps by train apps equal to the set value.
  :param df: a DataFrame contains all of our data
  :param rate: The ratio between test data and train data
  :return: train data & test data
  """
  num_samples = len(df)
  app_df = df.groupby('package').count().sort_values(['prime_screen_name'])
  if len(app_df) > 0:
    for i in range(len(app_df)):
      num_test_sample = np.sum(app_df['prime_screen_name'].values[:(i + 1)])
      if num_test_sample / num_samples >= rate:
        break
    test_apps = app_df.index[:i + 1]
    train_apps = app_df.index[i + 1:]
    train_df = df[df['package'].isin(train_apps)]
    test_df = df[df['package'].isin(test_apps)]
  return train_df, test_df


def split_data_manually(df):
  """
  Choose list of apps as test set
  :param df: a DataFrame contains all of our data
  :return: train data and test data
  """
  try:
    from .test_apps import test_app_ls
  except ImportError:
    raise ImportError('Please config again the list of test apps in test_apps.py following your data.')
  train_df = df[~df['package'].isin(test_app_ls)]
  test_df = df[df['package'].isin(test_app_ls)]
  return train_df, test_df


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--db-path', type=str,
    default='booster.sqlite',
    help='Path to database')
  parser.add_argument(
    '--split-method', type=str,
    default='Q',
    help="Method to split data into train and test data. (Q=Quantity/ M=Manual)"
  )
  parser.add_argument(
    '--full-csv', type=str,
    default='cos_full.csv',
    help='File path to store full data')
  parser.add_argument(
    '--train-csv', type=str,
    default='cos_train.csv',
    help='File path to store train data')
  parser.add_argument(
    '--test-csv',
    default='cos_test.csv', type=str,
    help='File path to store train data')

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  match_df, element_df, device_df, screen_df = get_base_tables(args.db_path)
  match_df = preprocess_match_df(match_df)
  element_df = preprocess_element_df(element_df)
  device_df = preprocess_device_df(device_df)
  screen_df = preprocess_screen_df(screen_df)
  merged_df = merge_df(match_df, element_df,
                       screen_df, device_df,
                       drop_screen_name_first=True)
  # Generate positive samples
  positive_df = get_positive_df(match_df)
  generated_positive_df = generate_positive_df(positive_df)
  merged_positive_df = merge_df(positive_df, element_df,
                                screen_df, device_df,
                                drop_screen_name_first=True)
  merged_generated_positive_df = merge_df(generated_positive_df, element_df, screen_df, device_df)
  positive_df = pd.concat([merged_positive_df, merged_generated_positive_df],
                          ignore_index=True, sort=False)
  print('Total positive cases:', positive_df.shape[0])
  # Generate negative samples
  generated_negative_df = generate_negative_df(positive_df, element_df)
  merged_negative_df = merge_df(generated_negative_df, element_df,
                                screen_df, device_df,
                                drop_screen_name_first=True)
  # Filter negative dataframe
  merged_negative_df = filter_negative_df(merged_negative_df)
  # Concat pos and neg
  generated_df = pd.concat([positive_df, merged_negative_df], ignore_index=True, sort=False)
  # Concat all to get full dataframe
  full_df = pd.concat([generated_df, merged_df], ignore_index=True, sort=False)
  full_df = remove_duplicated_pairs(full_df)
  full_df = postprocess(full_df)
  if args.split_method == 'Q':
    train_df, test_df = split_data_by_quantity(full_df)
  elif args.split_method == 'M':
    train_df, test_df = split_data_manually(full_df)
  full_df.to_csv(args.full_csv, index=None)
  train_df.to_csv(args.train_csv, index=None)
  test_df.to_csv(args.test_csv, index=None)
  print('Export data successful!!!')
  print('Train Dataset:', train_df.shape)
  print('Test Dataset:', test_df.shape)


if __name__ == '__main__':
  main()
