from google.cloud import vision
from google.cloud.vision import enums, types
import service.utils as u
from service.constants import GOOGLE_VISION_IMAGE_PADDING

class GoogleOCR:
  def __init__(self):
    self._ocr = vision.ImageAnnotatorClient()

  def recognize_text(self, img):
    if type(img) is not bytes:
      img = u.convert_image_numpy2bytes(img)
    img = vision.types.Image(content=img)
    response = self._ocr.text_detection(image=img)
    if response.error.message:
      raise ConnectionError(response.error.message)
    if len(response.text_annotations) > 0:
      return response.text_annotations[0].description
    return ''

  def detect_texts_in_images(self, element_imgs):
    client = vision.ImageAnnotatorClient()
    features = [types.Feature(type=enums.Feature.Type.TEXT_DETECTION)]
    requests = []
    bounds = []
    texts = []

    for image in element_imgs:
      img = u.convert_image_numpy2bytes(image)
      img = types.Image(content=img)
      request = types.AnnotateImageRequest(image=img, features=features)
      requests.append(request)

    response = client.batch_annotate_images(requests)

    for annotation_response in response.responses:
      if annotation_response.error.message:
        raise ConnectionError(annotation_response.error.message)
      if not annotation_response.text_annotations:
        bounds.append([])
        texts.append([])
      else:
        text, bound = self.extract_texts_and_bounds_from_google_response(annotation_response)
        bounds.append(bound)
        texts.append(text)
    return texts, bounds

  def get_texts_of_images_from_google_response(self, imgs, gg_response):
    gg_text, gg_bound = self.extract_texts_and_bounds_from_google_response(gg_response)
    texts = u.get_texts_of_images_from_ocr_response(gg_text, gg_bound, imgs, GOOGLE_VISION_IMAGE_PADDING)
    return texts

  @staticmethod
  def extract_texts_and_bounds_from_google_response(gg_response):
    res = gg_response.text_annotations[1:]
    texts = [item.description for item in res]
    bounds = []
    for item in res:
      vertices = item.bounding_poly.vertices
      top_left = [vertices[0].x, vertices[0].y]
      right_bottom = [vertices[2].x, vertices[2].y]
      bounds.append(top_left + right_bottom)
    return texts, bounds
