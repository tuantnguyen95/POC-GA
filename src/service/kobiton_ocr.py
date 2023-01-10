import service.utils as u
import grpc
import os
import service.constants as constants
import service.schema.booster.ai.ocr_pb2 as ocr_pb2
import service.schema.booster.ai.ocr_pb2_grpc as ocr_pb2_grpc
import ast


class KobitonOCR:
  def __init__(self, logger, services):
    self.services = services
    self.channel = grpc.insecure_channel(
        services[constants.KOBITON_OCR_SERVICE])
    self.logger = logger

  def detect_texts_in_images(self, session_id, action_id, request_id, imgs):
    res_ocr = []
    
    for img in imgs:
      stub = ocr_pb2_grpc.OCRStub(self.channel)
      req = ocr_pb2.OCRRequest()
      req.image_data = u.convert_image_numpy2bytes(img)
      req.session_id = session_id
      req.action_id = action_id
      req.request_id = request_id
      response = stub.predict(req)
      results = ast.literal_eval(response.result)
      if len(results) > 0:
        text, bound, confidence = self.extract_texts_and_bounds_from_sota_response(results)
        res_ocr.append([text, bound, confidence])
      else:
        self.logger.error("Request OCR Failed")
        res_ocr.append([])
    return res_ocr
  
  def get_texts_of_images_from_sota_response(self, imgs, res):
    sota_text, sota_bound = self.extract_texts_and_bounds_from_sota_response(res)
    texts = u.get_texts_of_images_from_ocr_response(sota_text, sota_bound, imgs, constants.SOTA_IMAGE_PADDING)
    return texts

  @staticmethod
  def extract_texts_and_bounds_from_sota_response(res):
    texts = []
    bounds = []
    confidences = []
    for item in res:
      text = item[1]
      conf = item[2]
      top_left = item[0][0]
      right_bottom = item[0][2]
      bounds.append([int(x) for x in top_left + right_bottom])
      texts.append(text)
      confidences.append(conf)
      
    return texts, bounds, confidences
