import http.client
import traceback
from flask import Blueprint, request, jsonify
from flask.globals import session
import service.schema.booster.ai.service_request_pb2 as request_pb2

from service.assertion.accessibility import AccessibilitySession
from service.logger import LoggingWrapper


accessibility_assertion = Blueprint('accessibility_assertion', __name__)
logger = LoggingWrapper()


def info(sess: AccessibilitySession):
  action_id = request.args.get('action_id', 0)
  session_id = request.args.get('session_id', 0)
  request_id = request.args.get('request_id', 0)
  logger.info('Received an request',
              extras={
                'session_id': session_id,
                'action_id': action_id,
                'request_id': request_id,
                **sess.instance_log})


def error(message, error_code, trace=None):
  action_id = request.args.get('action_id', 0)
  session_id = request.args.get('session_id', 0)
  request_id = request.args.get('request_id', 0)
  extras = {'session_id': session_id,
            'action_id': action_id,
            'request_id': request_id}

  if trace:
    extras['error'] = trace

  logger.error(message, extras)
  return '{"message":%s}' % message, error_code


@accessibility_assertion.route('', methods=['POST'])
def get_accessibility_assertion():
  assert_req = request_pb2.AccessibilityAssertionRequest()

  try:
    assert_req.ParseFromString(request.data)
    sess = AccessibilitySession(assert_req)
    info(sess)
  except Exception:
    return error('Error on parsing data for accessibility assertion', http.client.BAD_REQUEST, \
                 trace=traceback.format_exc())

  try:
    response = sess.verified_elements()
  except Exception:
    return error('Error on getting accessibility assertion', http.client.INTERNAL_SERVER_ERROR, \
                 trace=traceback.format_exc())

  return jsonify(response), http.client.OK
