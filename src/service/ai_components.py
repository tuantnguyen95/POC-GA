import os
import grpc

import numpy as np
import joblib
import tensorflow as tf
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import service.schema.booster.ai.segmentation_pb2 as segmentation_pb2
import service.schema.booster.ai.segmentation_pb2_grpc as segmentation_pb2_grpc
from service.logger import LoggingWrapper
from service.utils import retry_call_n_times, convert_image_numpy2bytes
import service.constants as constants


class AIComponent:
  def __init__(self, services, logger):
    # Config logger
    self.logger = logger
    # Config tf-serving server
    self.services = services
    self.visual_model_name = 'element_embeddings'
    # Config ML models
    self.sk_model_path = 'service/sk_models/merged/with_ocr/svm.joblib.pkl'
    self.sk_cls = joblib.load(self.sk_model_path)

  def __get_channel(self, service):
    channel = grpc.insecure_channel(service)
    self.logger.debug('channel_ready')
    return channel

  def __get_visual_embeddings(self, images, batch_size=16):
    """
    The purpose of this function is to feed elements images to FaceNet serving service. Instead of feeding 
    all images to serving, this function splits images into many packs by a specific number of images for each pack.
    @param images: All elements images
    @param batch_size: number of images in each pack. Default is 16 to limit computing resources.
    @return: FaceNet image embedding features
    """
    image_embeddings = np.zeros((images.shape[0], 128))
    channel = self.__get_channel(self.services[constants.TENSORFLOW_SERVING_SERVICE])
    pred_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    for i in range(0, images.shape[0], batch_size):
      begin = i
      end = min(begin + batch_size, images.shape[0])
      imgs = images[begin:end]

      request = predict_pb2.PredictRequest()
      request.model_spec.name = self.visual_model_name
      request.model_spec.signature_name = 'serving_default'

      request.inputs['input'].CopyFrom(
        make_tensor_proto(imgs, shape=imgs.shape, dtype=tf.float32))

      result = pred_stub.Predict(request, 160.0)  # timeout
      outputs_tensor_proto = result.outputs["embeddings"]
      shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
      image_embeddings[begin:end] = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())
    return image_embeddings

  def __get_visual_bounds_by_segmentation(self, images):
    channel = self.__get_channel(self.services[constants.SEGMENTATION_SERVICE])
    stub = segmentation_pb2_grpc.SegmentationStub(channel)
    segment_request = segmentation_pb2.SegRequest()
    for image in images:
      req = segment_request.images.add()
      req.image_data = convert_image_numpy2bytes(image)

    response = stub.predict(segment_request)
    results = response.images
    elements_bounds = [[(e.class_name, [e.bound.x1, e.bound.y1, e.bound.x2, e.bound.y2]) for e in image.regions] for image in results]
    return elements_bounds

  def _try_get_visual_embeddings(self, images, session_id=0, action_id=0, request_id=0):
    """
    Get visual embeddings for element images
    images: [batch_size, w, h, c]
    """
    logger_params = {
      'log': "Error on getting visual embeddings",
      'extras': {
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id
      }
    }
    result = retry_call_n_times(self.__get_visual_embeddings, self.logger, logger_params, images)
    return result

  def _try_get_visual_bounds_by_segmentation(self, images, is_text=False, session_id=0, action_id=0, request_id=0):
    logger_params = {
      'log': "Error on getting visual bounds by segmentation",
      'extras': {
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id
      }
    }
    result = retry_call_n_times(self.__get_visual_bounds_by_segmentation, self.logger, logger_params, images)
    if result is None:
      return [[]] * len(images)
    return result
