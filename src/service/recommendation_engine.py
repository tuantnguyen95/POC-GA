from flask import Blueprint, request, jsonify
import service.schema.booster.ai.service_request_pb2 as request_pb2
import http.client
import traceback

from service import constants
from service.rec.recommend import RecommendSession
from service.logger import LoggingWrapper

rec_engine = Blueprint('recommendation_engine', __name__)
logger = LoggingWrapper()


def error(message, error_code, trace=None):
  if trace:
    extras = {'error': trace}

  logger.error(message, extras)
  return '{"message":%s}' % (message), error_code


@rec_engine.route('/')
def index():
  return 'Recommendation engine service'


@rec_engine.route(constants.FONTSIZE_RECOMMENDATION_URL, methods=['POST'])
def get_fontsize_recommendation():
  session = None
  try:
    recommend_req = request_pb2.RecommendScreenRequest()
    recommend_req.ParseFromString(request.data)

    session = RecommendSession(recommend_req)
    response = [element.get_fontsize_recommend_dict() for element in session.verified_elements()]
    return jsonify(response), http.client.OK
  except:
    if not session:
      return error('Error on parsing data for fontsize recommendation', http.client.BAD_REQUEST,
                   trace=traceback.format_exc())
    else:
      return error('Error on getting fontsize recommendation', http.client.INTERNAL_SERVER_ERROR,
                   trace=traceback.format_exc())