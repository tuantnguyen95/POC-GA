import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import pickle
import gzip
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.optimize import dual_annealing
from sklearn.metrics import log_loss
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import logging
import argparse
import json
import os
from s3 import DataS3
import glob

logging.getLogger().setLevel(logging.INFO)

# nohup python3 -m train --weight-version 1 --data-path data/ --weight-path weights/ --batch-size 1024 --epochs 100 --compare-weight-version 0 --data-version-min 0 --data-version-max 1 > log3 &

train_data_1 = None
weight2_to_train_w1 = None
weight_path = 'weights/'
weight_1_file_name = 'weight_1.pkl'

weight_types = ['1', '2']
feature_names_0 = ['image', 'text', 'min_font_height', 'max_font_height', 'text_width',
                   'element_width', 'element_height', 'element_x', 'element_y']
feature_names_50 = ['image_50', 'text_50', 'min_font_height_50', 'max_font_height_50', 'text_width_50',
                    'element_width', 'element_height', 'element_x', 'element_y']
feature_names_full = ['image_full', 'text', 'min_font_height', 'max_font_height', 'text_width',
                      'element_width', 'element_height', 'element_x', 'element_y']
data_types = ['50', 'full']
feature_names = [feature_names_50, feature_names_full]
number_features = len(feature_names_0)
dist_negative = 20


def get_features(data, data_type_idx):
  final_data = []
  for item in data:
    sample = []
    # Feature 1
    for feature_name in feature_names[data_type_idx]:
      sample.append(item[6][feature_name])
    # Feature 2
    for feature_name in feature_names[data_type_idx]:
      sample.append(item[7][feature_name])
    # Label
    sample.append(item[-2])
    final_data.append(sample)
  return np.array(final_data)


def load_data(s3, data_version_min, data_version_max, data_folder):
  train_data_names = []
  test_data_names = []
  # get train version as demand
  for data_version in range(data_version_min, data_version_max + 1):
    train_object_name_prefix = 'luna_data_v%s/features_train_' % data_version
    train_file_names = s3.download_files(train_object_name_prefix, data_folder)
    # train_file_names = glob.glob(os.path.join(dir_name, "**", train_object_name_prefix + '*.pklz'), recursive=True)
    train_data_names.extend(train_file_names)

  # get all test data
  for data_version in range(data_version_min, data_version_max + 1):
    test_object_name_prefix = 'luna_data_v%s/features_test_' % data_version
    test_file_names = s3.download_files(test_object_name_prefix, data_folder)
    # test_file_names = glob.glob(os.path.join(dir_name, "**", test_object_name_prefix + '*.pklz'))
    test_data_names.extend(test_file_names)
  return train_data_names, test_data_names


def load_weight(s3, weight_version, weight_folder, data_type):
  weights = []
  for weight_type in weight_types:
    object_name = '%d/weight_%s_%s.pkl' % (weight_version, weight_type, data_type)
    file_name = weight_folder + object_name
    if not os.path.exists(file_name):
      s3.download_file('weights/' + object_name, weight_folder + object_name)
    weights.append(pickle.load(open(file_name, 'rb')))
  return weights


def loss_1(weight_1):
  n = len(train_data_1)
  y_hat = np.zeros(n)
  feast1 = train_data_1[:, :number_features]
  feast2 = train_data_1[:, number_features:-1]
  y = train_data_1[:, -1].astype('int')

  we = [int(round(w, 0)) for w in weight_1]
  logging.info(we)
  loss = 0
  for i in range(n):
    f1 = [feast1[i, 0] * weight2_to_train_w1[0]]
    f2 = [feast2[i, 0] * weight2_to_train_w1[0]]
    for j, w in enumerate(we):
      f1 += [feast1[i, j + 1] * weight2_to_train_w1[j + 1]] * w
      f2 += [feast2[i, j + 1] * weight2_to_train_w1[j + 1]] * w
    fe1 = np.concatenate(f1)
    fe2 = np.concatenate(f2)
    sim = cosine_similarity([fe1], [fe2])
    y_hat[i] = sim[0][0]
  #     dist = euclidean_distances([fe1], [fe2])[0][0]
  #     y_hat[i] = dist
  #     if y[i] == 1:
  #       loss += dist
  #     else:
  #       if dist > dist_negative:
  #         loss += 0
  #       else:
  #         loss += dist_negative-dist

  logging.info('len features: %d' % (fe1.shape[0]))
  loss = log_loss(y, y_hat)
  logging.info('Loss %f' % (loss))
  return loss


def loss_2(X, y, final_weight):
  binary_loss = tf.keras.losses.BinaryCrossentropy()
  #   mse = tf.keras.losses.MeanSquaredError()
  feature_1, feature_2 = X

  f1 = tf.multiply(feature_1, final_weight)
  f2 = tf.multiply(feature_2, final_weight)

  normalize_a = tf.nn.l2_normalize(f1, 1)
  normalize_b = tf.nn.l2_normalize(f2, 1)

  y_hat = tf.math.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
  loss = binary_loss(y, y_hat)
  #   y_hat = tf.norm(f1-f2, ord='euclidean')
  #   loss = mse(y, y_hat)
  return loss


def grad(X, y, final_weight):
  with tf.GradientTape() as tape:
    loss_value = loss_2(X, y, final_weight)
  return loss_value, tape.gradient(loss_value, final_weight)


def generate_batch(data_names, indices, weight_1, final_weight, data_type_idx, batch_size=128, balance=True):
  data_name_idx = 0
  data = None
  from_index = 0
  idx = []
  np.random.shuffle(data_names)
  number_elements = 0
  for i, w in enumerate(weight_1):
    begin = indices[i]
    end = indices[i + 1]
    number_elements += (end - begin) * w

  while True:
    if data is None or from_index >= len(idx):
      if from_index >= len(idx) and len(idx) > 0:
        data_name_idx += 1

      if data_name_idx == len(data_names) - 1:
        np.random.shuffle(data_names)
        data_name_idx = 0

      from_index = 0
      data_raw = np.concatenate([pickle.load(gzip.open(data_names[data_name_idx], 'rb')),
                                 pickle.load(gzip.open(data_names[data_name_idx + 1], 'rb'))])
      data = get_features(data_raw, data_type_idx)
      np.random.shuffle(data)
      y = data[:, -1].astype(np.float32)
      feast1 = data[:, :number_features]
      feast2 = data[:, number_features:-1]
      idx = np.arange(len(y))

      if balance:
        positive_idx = np.where(y == 1.)[0]
        negative_idx = np.where(y == 0.)[0]
        min_len = min(positive_idx.shape[0], negative_idx.shape[0])
        idx = []
        for i in range(min_len):
          idx.append(positive_idx[i])
          idx.append(negative_idx[i])

    to_index = from_index + batch_size
    if to_index > len(idx):
      to_index = len(idx)

    batch_idx = idx[from_index:to_index]
    np.random.shuffle(batch_idx)
    len_batch = to_index - from_index
    fe1 = np.zeros((len_batch, number_elements))
    fe2 = np.zeros((len_batch, number_elements))
    y_b = np.zeros(len_batch)
    for i, k in enumerate(batch_idx):
      fe1[i] = np.concatenate([np.concatenate([feast1[k, j]] * w) for j, w in enumerate(weight_1)])
      fe2[i] = np.concatenate([np.concatenate([feast2[k, j]] * w) for j, w in enumerate(weight_1)])
      y_b[i] = y[k]
    #       if y[k] == 0:
    #         dist = euclidean_distances([fe1[i] * final_weight], [fe2[i] * final_weight])[0][0]
    #         if dist >= dist_negative:
    #           y_b[i] = dist
    #         else:
    #           y_b[i] = dist_negative
    #       else:
    #         y_b[i] = 0
    from_index += batch_size
    yield (fe1, fe2), y_b


def call_back_save_weight_1(x, f, context):
  if context == 0:
    with open('%s%s' % (weight_path, weight_1_file_name), 'wb') as fi:
      weight_1 = [1] + [int(round(w, 0)) for w in x]
      pickle.dump(weight_1, fi)
  elif context != 1:
    return True


def train_weight_1(train_data_names, data_type_idx, w0=None):
  global train_data_1
  # Hyper params
  lw = [5.] + [10.] * 7
  up = [27.] + [200.] * 7
  if w0 is not None:
    init_weight = w0[1:]
  else:
    init_weight = w0
  weight_1 = None

  np.random.shuffle(train_data_names)
  for i in range(len(train_data_names) - 1):
    train_data = np.concatenate([pickle.load(gzip.open(train_data_names[i], 'rb')),
                                 pickle.load(gzip.open(train_data_names[i + 1], 'rb'))])
    train_data_1 = get_features(train_data, data_type_idx)
    if init_weight is not None:
      result = dual_annealing(loss_1, bounds=list(zip(lw, up)), maxiter=10, callback=call_back_save_weight_1,
                              x0=init_weight)
    else:
      result = dual_annealing(loss_1, bounds=list(zip(lw, up)), maxiter=10, callback=call_back_save_weight_1)
    init_weight = result.x
    weight_1 = [1] + [int(round(w, 0)) for w in result.x]
  return weight_1


def train_weight_2(indices, train_data_names, test_data_names, batch_size, epochs, weight_path, weight_file_name,
                   data_type_idx, weight_1=None, weight_2_init=None):
  # Init weights
  if weight_1 is None:
    weight_1 = [1] * 9

  number_weight_2 = indices[-1]
  if weight_2_init is None:
    weight_2 = [tf.Variable(1., name='weight%d' % (i)) for i in range(number_weight_2)]
  else:
    weight_2 = [tf.Variable(weight_2_init[i], name='weight%d' % (i)) for i in range(number_weight_2)]
  final_weights = []
  for i, w in enumerate(weight_1):
    begin = indices[i]
    end = indices[i + 1]
    final_weights.extend(weight_2[begin:end] * w)

  # Params
  train_steps = 0
  for train_data_name in train_data_names:
    train_data = pickle.load(gzip.open(train_data_name, 'rb'))
    train_steps += int(len(train_data) / batch_size)

  val_steps = 0
  for test_data_name in test_data_names:
    test_data = pickle.load(gzip.open(test_data_name, 'rb'))
    val_steps += int(len(test_data) / batch_size)

  epoch = 0
  step = 0
  train_loss = 0
  min_val = -1
  tolerance = 0

  # Optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
  lr_decrease_step = [30, 40, 45]
  # Training
  inter_weights = [w.numpy() for w in final_weights]

  for X_train, y_train in generate_batch(train_data_names, indices, weight_1, inter_weights, data_type_idx,
                                         batch_size=batch_size, balance=True):
    loss_value, grads = grad(X_train, y_train, final_weights)
    opt.apply_gradients(zip(grads, final_weights))
    train_loss += loss_value
    step += 1
    logging.info("Iteration: {}, Loss: {}".format(step, loss_value))
    if step == train_steps:
      val_loss = 0
      val_step = 0
      for X_val, y_val in generate_batch(test_data_names, indices, weight_1, inter_weights, data_type_idx,
                                         batch_size=batch_size):
        val_loss += loss_2(X_val, y_val, final_weights).numpy()
        val_step += 1
        if val_step == val_steps:
          break
      epoch += 1
      vloss = val_loss / val_steps
      logging.info(
        "Epoch: {}, Loss: {}, Val Loss: {}, Learning Rate: {}".format(epoch, train_loss / train_steps, vloss,
                                                                      opt.lr))
      # decrease learning rate if val loss increase for 3 epochs
      if min_val == -1 or min_val > vloss:
        min_val = vloss
        tolerance = 0
        # save best
        saved_weights = [w.numpy() for w in final_weights]
        pickle.dump(saved_weights, open('%s%s' % (weight_path, weight_file_name), 'wb'))
      else:
        tolerance += 1

      if epoch in lr_decrease_step:
        opt.lr = opt.lr * 0.1
      step = 0
      train_loss = 0

      if epoch == epochs or tolerance == 20:
        break
    inter_weights = [w.numpy() for w in final_weights]
  return saved_weights


def predict(x, y, weights):
  fe1, fe2 = x
  f1 = np.multiply(fe1, weights)
  f2 = np.multiply(fe2, weights)

  sim_pred = []
  for i in range(len(y)):
    sim_pred.append(cosine_similarity([f1[i]], [f2[i]])[0][0])
  #     sim_pred.append(euclidean_distances([f1[i]],[f2[i]])[0][0])
  return sim_pred


def evaluate(weight_1, weight_2, data_names, indices, batch_size, data_type_idx, data_type, weight_path,
             weight_version):
  y_hat = []
  y_total = []
  number_elements = len(weight_2)

  for data_name in data_names:
    data = get_features(pickle.load(gzip.open(data_name, 'rb')), data_type_idx)
    length = len(data)
    fe1 = np.zeros((length, number_elements))
    fe2 = np.zeros((length, number_elements))
    y_b = data[:, -1].astype(np.float32)

    feast1 = data[:, :number_features]
    feast2 = data[:, number_features:-1]

    for i in range(length):
      fe1[i] = np.concatenate([np.concatenate([feast1[i, j]] * w) for j, w in enumerate(weight_1)])
      fe2[i] = np.concatenate([np.concatenate([feast2[i, j]] * w) for j, w in enumerate(weight_1)])

    sim_pred = predict((fe1, fe2), y_b, weight_2)
    y_total.extend(y_b)
    y_hat.extend(sim_pred)

  threshold = np.arange(0, 1, 0.01)
  #   threshold = np.arange(1, dist_negative, 1.)
  f1 = []
  for thres in threshold:
    y_pred = np.array(y_hat) >= thres
    f1.append(f1_score(y_total, y_pred))
  fpr, tpr, _ = roc_curve(y_total, y_hat)
  roc_auc = auc(fpr, tpr)
  np.save(weight_path + '%d/fpr_%s.npy' % (weight_version, data_type), fpr)
  np.save(weight_path + '%d/tpr_%s.npy' % (weight_version, data_type), tpr)

  # Visualize ROC Curve
  plt.figure()
  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
  plt.legend(loc='lower right')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  logging.info('Saving: ' + '%d/ROC_%s.png' % (weight_version, data_type))
  plt.savefig(weight_path + '%d/ROC_%s.png' % (weight_version, data_type), bbox_inches='tight')
  # plt.show()
  best_thres = threshold[np.argmax(f1)]
  logging.info('Threshold for max F1: %f, F1 %f' % (best_thres, np.max(f1)))

  # Visualize threshold distribution
  visualize(y_hat, y_total, best_thres, weight_path + '%d/theta_dist_%s.png' % (weight_version, data_type))

  y_predict = np.array(y_hat) >= best_thres
  precision = precision_score(y_total, y_predict)
  recall = recall_score(y_total, y_predict)
  logging.info('Precision %f, Recall %f' % (precision, recall))
  logging.info(' * AUC-ROC %f' % roc_auc)
  return best_thres, precision, recall


def visualize(y_predict, y_label, threshold, save_path):
  plt.figure()
  ones = []
  zeros = []

  for i, label in enumerate(y_label):
    if label == 1:
      ones.append(y_predict[i])
    else:
      zeros.append(y_predict[i])

  bins = np.linspace(0, 1, 100)

  plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
  plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

  mu_0 = np.mean(zeros)
  sigma_0 = np.std(zeros)
  y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
  plt.plot(bins, y_0, 'r--')
  mu_1 = np.mean(ones)
  sigma_1 = np.std(ones)
  y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
  plt.plot(bins, y_1, 'b--')
  plt.xlabel('theta')
  plt.ylabel('theta j Distribution')
  plt.title(
    r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

  print('threshold: ' + str(threshold))
  print('mu_0: ' + str(mu_0))
  print('sigma_0: ' + str(sigma_0))
  print('mu_1: ' + str(mu_1))
  print('sigma_1: ' + str(sigma_1))

  plt.legend(loc='upper right')
  plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
  plt.savefig(save_path)
  # plt.show()


def create_weight_folder(weight_path, weight_version):
  if weight_version > -1 and not os.path.exists('%s/%d' % (weight_path, weight_version)):
    os.makedirs(os.path.join('%s/%d' % (weight_path, weight_version)), exist_ok=True)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--weight-path', type=str, default='weights/')
  parser.add_argument('--compare-weight-version', type=int, default=-1)
  parser.add_argument('--weight-version', type=int, default=-1)
  parser.add_argument('--weight-init', type=int, default=-1)
  parser.add_argument('--data-path', type=str, default='data/')
  parser.add_argument('--data-version-min', type=int, default=0)
  parser.add_argument('--data-version-max', type=int, default=-1)
  parser.add_argument('--config-file', type=str, default='s3.json', help='Path to AWS config file')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  # Get parameters
  args = parse_args()
  config = {}
  with open(args.config_file) as f:
    config = json.load(f)
  s3 = DataS3(config)
  compare_weight_version = args.compare_weight_version
  weight_version = args.weight_version
  weight_initial_version = args.weight_init
  weight_path = args.weight_path
  s3.download_file('idx.pkl', weight_path + 'idx.pkl')
  indices = pickle.load(open(weight_path + 'idx.pkl', 'rb'))

  # Load data
  data_version_max = args.data_version_max
  if data_version_max == -1:
    data_version_max = s3.get_max_data_version()
  train_data_names, test_data_names = load_data(s3, args.data_version_min, data_version_max, args.data_path)

  # Create weight versions folder
  create_weight_folder(weight_path, weight_initial_version)
  create_weight_folder(weight_path, compare_weight_version)
  create_weight_folder(weight_path, weight_version)

  # Train & Save
  for data_type_idx, data_type in enumerate(data_types):
    logging.info('Current training %s' % data_type)
    weights = []
    # Get init weights
    weights_init = [None] * 2
    if weight_initial_version > -1:
      weights_init = load_weight(s3, weight_initial_version, weight_path, data_type)

    weight_1_file_name = '%d/weight_1_%s.pkl' % (weight_version, data_type)
    weight_2_file_name = '%d/weight_2_%s.pkl' % (weight_version, data_type)
    # Train weight 1 first
    wei2 = np.array(pickle.load(open(weight_path + weight_2_file_name, 'rb')))
    # wei2 = [1.] * 1927
    weight2_to_train_w1 = []
    for i in range(9):
      begin = indices[i]
      end = indices[i + 1]
      weight2_to_train_w1.append(np.array(wei2[begin:end]))

    weight_1 = train_weight_1(train_data_names, data_type_idx, w0=weights_init[0])
    with open(weight_path + weight_1_file_name, 'wb') as fi:
      pickle.dump(weight_1, fi)
    s3.upload_file(weight_path + weight_1_file_name, object_name='weights/%s' % (weight_1_file_name))
    # weight_1 = pickle.load(open(weight_path + weight_1_file_name, 'rb'))

    # Train weight 2
    weight_2 = train_weight_2(indices, train_data_names, test_data_names, args.batch_size, args.epochs, weight_path,
                              weight_2_file_name, data_type_idx, weight_1=weight_1, weight_2_init=weights_init[1])
    s3.upload_file(weight_path + weight_2_file_name, object_name='weights/%s' % (weight_2_file_name))

    weights.append(weight_1)
    weight_2 = pickle.load(open(weight_path + weight_2_file_name, 'rb'))
    if len(weight_2) > 1927:
      w2 = []
      w_begin = 0
      for i, w1 in enumerate(weight_1):
        begin = indices[i]
        end = indices[i + 1]
        length = end - begin
        w2.extend(weight_2[w_begin:w_begin + length])
        w_begin += length * w1
      with open(weight_path + weight_2_file_name, 'wb') as fi:
        pickle.dump(w2, fi)
      weights.append(weight_2)
    else:
      final_weights = []
      for i, w in enumerate(weight_1):
        begin = indices[i]
        end = indices[i + 1]
        final_weights.extend(weight_2[begin:end] * w)
      weights.append(final_weights)

    # Evaluate
    logging.info('For data type %s' % data_type)
    no_weights_1 = [1] * 9
    no_weights_2 = [1] * indices[-1]
    logging.info('No weights metrics')
    theshold, precision, recall = evaluate(no_weights_1, no_weights_2, test_data_names, indices, args.batch_size,
                                           data_type_idx, data_type, weight_path, weight_version)

    # Compare with previous best
    if compare_weight_version > -1:
      logging.info('Previous best weight metrics, weight version %d' % compare_weight_version)
      compare_weights = load_weight(s3, compare_weight_version, weight_path, data_type)
      pre_theshold, pre_precision, pre_recall = evaluate(compare_weights[0], compare_weights[1], test_data_names,
                                                         indices, args.batch_size, data_type_idx, data_type,
                                                         weight_path, weight_version)

    logging.info('Current metrics, weight version %d' % weight_version)
    theshold, precision, recall = evaluate(weights[0], weights[1], test_data_names, indices, args.batch_size,
                                           data_type_idx, data_type, weight_path, weight_version)
