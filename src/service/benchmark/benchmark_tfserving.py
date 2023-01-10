import sys
import traceback
import time
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import grpc
import imageio
from PIL import Image
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def get_prediction_stub(host=None, port=None):
  channel = grpc.insecure_channel(
    '{host}:{port}'.format(host=host, port=port))
  return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def prewhiten(tensor):
  return (tensor-127.5)*0.0078125


def read_and_resize(im_pth, output_size):
  ''' resize an image to a squared 2d array
  '''
  img = imageio.imread(im_pth, as_gray=False, pilmode="RGB")
  if img.shape[0] < 30 or img.shape[1] < 30:
    return None
  img = Image.fromarray(img)
  img = img.resize((output_size, output_size))
  prewhitened = prewhiten(np.array(img))
  return prewhitened


def get_visual_embeddings(images, pred_stub, session_id=0, model_name="element_embeddings"):
  ''' Get embedding for a nonscrollable element image
    images: [batch_size, w, h, c]
  '''
  prepare_time, request_time = 0, 0
  try:
    start = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    request.inputs['input'].CopyFrom(
      make_tensor_proto(images, shape=images.shape, dtype=tf.float32))
    prepare_time += time.time() - start
    start = time.time()
    result = pred_stub.Predict(request, 160.0)
    request_time += time.time() - start
    outputs_tensor_proto = result.outputs["embeddings"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    print('Response tensor shape:', shape)
  except Exception as err:
    trace = traceback.format_exc()
    print(trace)
  return prepare_time, request_time


def call_tf_serving(_):
  img_path = "./service/benchmark/prime.jpg"
  imgs = read_and_resize(img_path, 160)
  imgs = np.expand_dims(imgs, 0)
  imgs = [imgs for i in range(40)]
  imgs = np.concatenate(imgs)
  emb_host = sys.argv[1]
  emb_port = sys.argv[2]
  pred_stub = get_prediction_stub(emb_host, emb_port)
  return get_visual_embeddings(imgs, pred_stub)


def main():
  if len(sys.argv) < 4:
    print('Usage: host port num_req')
    sys.exit()
  pools = Pool(200)
  num = int(sys.argv[3])
  ex_times = pools.map(call_tf_serving, [None for _ in range(num)])
  request_times = [x[1] for x in ex_times]
  print('A request took average time of: ', np.average(request_times))


if __name__ == "__main__":
  main()
