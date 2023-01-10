from flask import Blueprint, request, jsonify
from service import constants, xml_utils
from service.sessions import PrimeSession, RevisitSession
import service.schema.booster.ai.service_request_pb2 as request_pb2
import service.metrics as metrics
import time
import traceback
import http.client
from service.logger import LoggingWrapper

from service.luna import match_elements

luna_engine = Blueprint('luna_engine', __name__)
logger = LoggingWrapper()


@luna_engine.route('/')
def index():
  return 'Serving for Luna engine'


def info(prime_sess, revisit_sess):
  action_id = request.args.get('action_id', 0)
  session_id = request.args.get('session_id', 0)
  request_id = request.args.get('request_id', 0)
  logger.info('Received an request',
              extras={
                'session_id': session_id,
                'action_id': action_id,
                'request_id': request_id,
                'prime_xpath': prime_sess.element.xpath,
                'request_prime_device_name': prime_sess.device,
                'request_prime_platform_version': prime_sess.platform.name,
                'request_prime_screen_density': prime_sess.density,
                'request_revisit_device_name': revisit_sess.device,
                'request_revisit_platform_version': revisit_sess.platform.name,
                'request_revisit_screen_density': revisit_sess.density
              })


def error(message, error_code, trace=None):
  action_id = request.args.get('action_id', 0)
  session_id = request.args.get('session_id', 0)
  request_id = request.args.get('request_id', 0)
  extras={"session_id": session_id,
          "action_id": action_id,
          "request_id": request_id}
  if trace:
    extras['error'] = trace

  logger.error(message, extras)
  return '{"message":%s}' % (message), error_code


@luna_engine.route(constants.ELEMENT_FINDING_URL, methods=['POST'])
def find_element():
  start_time = time.time()
  metrics.AI_SERVICE_ELEMENT_FINDING_TOTAL.inc()
  finding_req = request_pb2.ElementFindingData()

  try:
    finding_req.ParseFromString(request.data)
    prime_sess = PrimeSession(finding_req)
    revisit_sess = RevisitSession(finding_req)
    info(prime_sess, revisit_sess)
  except Exception:
    return error('Cannot parse request data', http.client.BAD_REQUEST, trace=traceback.format_exc())

  if not prime_sess.element.xpath or not prime_sess.xml or not revisit_sess.xml:
    return error('Missing request data', http.client.BAD_REQUEST)

  try:
    if prime_sess.element.is_scrollable():
      cand_xpaths, cand_sims, _ = match_elements.find_element(prime_sess, revisit_sess, prime_sess.element.xpath, element_type = constants.ElementType.SCROLLABLE)
    else:
      cand_xpaths, cand_sims, _ = match_elements.find_element(prime_sess, revisit_sess, prime_sess.element.xpath)
    response = {'elements': [{"xpath": cand[0], "confidence": cand[1]} for cand in zip(cand_xpaths[-5:], cand_sims[-5:])]}
  except Exception:
    return error('Error in finding element', http.client.INTERNAL_SERVER_ERROR, trace=traceback.format_exc())

  metrics.AI_SERVICE_ELEMENT_FINDING_SUCCESSFUL_TOTAL.inc()
  metrics.AI_SERVICE_ELEMENT_FINDING_LATENCY_SECONDS.observe(time.time() - start_time)
  return jsonify(response), http.client.OK
