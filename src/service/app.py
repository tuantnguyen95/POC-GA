import traceback
import http.client
import os
import time
from flask import Flask, request, jsonify
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from prometheus_client import make_wsgi_app
import consul
import json

from service import constants

from service.element_finder import ElementFinder
from service.text_assertion import TextAssertPrimeSession, TextAssertion
from service.element_comparator import ElementComparator, CompareRevisitSession
from service.visual_verification import VisualVerification, VisualVerifyPrimeSession, VisualVerifyRevisitSession
from service.wbi import WBI, WBIPrimeSession, WBIRevisitSession
from service.sessions import PrimeSession, RevisitSession, ScreenshotSession, FindingByImageSession
import service.schema.booster.ai.service_request_pb2 as request_pb2
import service.metrics as metrics
from service.check_device_size import get_same_size_devices
from service.luna_engine import luna_engine
from service.facenet_engine import facenet_engine
from service.recommendation_engine import rec_engine
from service.accessibility_assertion import accessibility_assertion
from service.element_finder_at_point import ElementFinderAtPoint
from service.element_finder_by_image import ElementFinderByImage

from service.utils import config
from service.utils import logger

def get_segmentation_service_info(logger):
  try:
    consul_host = os.environ.get('KOBITON_CONSUL_HOST')
    consul_port = os.environ.get('KOBITON_CONSUL_REST_API_PORT')
    segmentation_id = os.getenv('KOBITON_AI_SEGMENTATION_ID', 'ita-segmentation-service')
    segmentation_url_key = 'grpc_service_private_url'
    if consul_host is not None and consul_port is not None:
      c = consul.Consul(host=consul_host, port=consul_port)
      _, metadata = c.catalog.service(segmentation_id)
      if metadata:
        segmentation_metadata = metadata[0]['ServiceMeta']
        if segmentation_url_key in segmentation_metadata:
          url = segmentation_metadata[segmentation_url_key].split(':')
          segment_host = url[0]
          segment_port = url[1]
          return segment_host, segment_port
  except Exception:
    trace = traceback.format_exc()
    logger.error(
      "Cannot get Segmentation service info from Consul",
      extras={
        "error": trace,
      },
    )
  return None, None

def get_kobiton_ocr_service_info(logger):
  try:
    consul_host = os.environ.get('KOBITON_CONSUL_HOST')
    consul_port = os.environ.get('KOBITON_CONSUL_REST_API_PORT')
    ocr_service_id = os.getenv('KOBITON_AI_OCR_SERVICE_ID', 'ita-ocr-service')
    ocr_url_key = 'grpc_service_private_url'
    if consul_host is not None and consul_port is not None:
      c = consul.Consul(host=consul_host, port=consul_port)
      _, metadata = c.catalog.service(ocr_service_id)
      if metadata:
        ocr_metadata = metadata[0]['ServiceMeta']
        if ocr_url_key in ocr_metadata:
          url = ocr_metadata[ocr_url_key].split(':')
          ocr_host = url[0]
          ocr_port = url[1]
          return ocr_host, ocr_port
  except Exception:
    trace = traceback.format_exc()
    logger.error(
      "Cannot get OCR service info from Consul",
      extras={
        "error": trace,
      },
    )
  return None, None

def element_finding_view_func(logger, finder: ElementFinder):
  # Build view function (controller) for element_finding based on config
  def element_finding():
    start_time = time.time()
    metrics.AI_SERVICE_ELEMENT_FINDING_TOTAL.inc()
    finding_req = request_pb2.ElementFindingData()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      finding_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      prime_sess = PrimeSession(finding_req)
      revisit_sess = RevisitSession(finding_req)

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
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    if not prime_sess.element.xpath or prime_sess.element.img is None or not prime_sess.xml or not revisit_sess.xml:
      logger.error(
        "Missing request data",
        extras={"session_id": session_id,
                "action_id": action_id,
                "prime_sess_element_xpath": prime_sess.element.xpath,
                "prime_sess_element_image": prime_sess.element.img is None,
                "not_prime_sess_xml": not prime_sess.xml,
                "not_revisit_sess_xml": not revisit_sess.xml
                },
      )
      return '{"message":"Missing request data"}', http.client.BAD_REQUEST

    try:
      revisit_xpaths, confidences = finder(
        prime_sess, revisit_sess,
        session_id=session_id, action_id=action_id, request_id=request_id)
      response = {'elements': []}
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in finding elements", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in finding element"}', http.client.INTERNAL_SERVER_ERROR

    if revisit_xpaths and len(revisit_xpaths) > 0:
      elements = [
        {"xpath": revisit_xpaths[i], "confidence": confidences[i]}
        for i in range(len(revisit_xpaths))
      ]
      response["elements"] = elements
    metrics.AI_SERVICE_ELEMENT_FINDING_SUCCESSFUL_TOTAL.inc()
    metrics.AI_SERVICE_ELEMENT_FINDING_LATENCY_SECONDS.observe(
      time.time() - start_time
    )
    return jsonify(response), http.client.OK

  return element_finding

def element_finding_at_point_view_func(logger, element_finder_at_point: ElementFinderAtPoint):
  def element_finding_at_point():
    start_time = time.time()
    metrics.AI_SERVICE_ELEMENT_FINDING_TOTAL.inc()
    finding_req = request_pb2.ElementFindingAtPoint()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      finding_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      session = ScreenshotSession(finding_req)

      logger.info('Received an request',
                  extras={
                    'session_id': session_id,
                    'action_id': action_id,
                    'request_id': request_id,
                    'request_platform_version': session.platform.name,
                    'point': str(session.point())
                  })
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    if session.screen_img is None or not session.xml:
      logger.error(
        "Missing request data",
        extras={"session_id": session_id, "action_id": action_id},
      )
      return '{"message":"Missing request data"}', http.client.BAD_REQUEST

    if session.point()[0] < 0 or session.point()[1] < 0 \
            or session.point()[0] > session.screen_img.shape[1] or session.point()[1] > session.screen_img.shape[0]:
      logger.error(
        "The coordinate of the point is invalid.",
        extras={"session_id": session_id, "action_id": action_id},
      )
      return '{"message":"The coordinate of the point is invalid."}', http.client.BAD_REQUEST

    try:
      response = element_finder_at_point(
        session, session_id=session_id, action_id=action_id, request_id=request_id)

    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in finding element at point:", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in finding element at point:"}', http.client.INTERNAL_SERVER_ERROR

    metrics.AI_SERVICE_ELEMENT_FINDING_SUCCESSFUL_TOTAL.inc()
    metrics.AI_SERVICE_ELEMENT_FINDING_LATENCY_SECONDS.observe(
      time.time() - start_time
    )
    return jsonify(response), http.client.OK

  return element_finding_at_point

def element_finding_by_image_view_func(logger, element_finder_by_image: ElementFinderByImage):
  def element_finding_by_image():
    start_time = time.time()
    finding_req = request_pb2.ElementFindingByImage()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      finding_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      session = FindingByImageSession(finding_req)

      logger.info('Received an request',
                  extras={
                    'session_id': session_id,
                    'action_id': action_id,
                    'request_id': request_id,
                    'request_platform_version': session.platform.name
                  })
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    if session.screen_img is None or session.get_query_img() is None or not session.xml:
      logger.error(
        "Missing request data",
        extras={"session_id": session_id, "action_id": action_id},
      )
      return '{"message":"Missing request data."}', http.client.BAD_REQUEST

    try:
      response = element_finder_by_image(
        session, session_id=session_id, action_id=action_id, request_id=request_id)

    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in finding element by image:", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in finding element by image:"}', http.client.INTERNAL_SERVER_ERROR
    return jsonify(response), http.client.OK
  
  return element_finding_by_image


def element_comparison_view_func(logger, comparator):
  # Build view function (controller) for element_comparison based on config
  def element_comparison():
    start_time = time.time()
    metrics.AI_SERVICE_ELEMENT_COMPARISON_TOTAL.inc()
    compare_req = request_pb2.ElementCompareData()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      compare_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error('Element comparison: cannot parse request data',
                   extras={
                     'session_id': session_id,
                     'action_id': action_id,
                     'request_id': request_id,
                     'error': trace})
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      prime_sess = PrimeSession(compare_req)
      revisit_sess = CompareRevisitSession(compare_req)

      logger.info('Element comparison: Received an request',
                  extras={
                    'session_id': session_id,
                    'action_id': action_id,
                    'request_id': request_id,
                    'prime_xpath': prime_sess.element.xpath,
                    'revisit_xpath': revisit_sess.element.xpath,
                    'request_prime_device_name': prime_sess.device,
                    'request_prime_platform_version': prime_sess.platform.name,
                    'request_prime_screen_density': prime_sess.density,
                    'request_revisit_device_name': revisit_sess.device,
                    'request_revisit_platform_version': revisit_sess.platform.name,
                    'request_revisit_screen_density': revisit_sess.density
                  })
    except Exception:
      trace = traceback.format_exc()
      logger.error('Element comparison: Cannot parse request data',
                   extras={
                     'session_id': session_id,
                     'action_id': action_id,
                     'request_id': request_id,
                     'error': trace})
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    if not prime_sess.element.xpath or\
       not compare_req.revisit_xpath or\
       not prime_sess.xml or not revisit_sess.xml:
      logger.error('Element comparison: Missing request data', extras={
        'session_id': session_id, 'action_id': action_id, 'request_id': request_id})
      return '{"message":"Missing request data"}', http.client.BAD_REQUEST

    try:
      similarity = comparator(
        prime_sess, revisit_sess,
        session_id=session_id, action_id=action_id, request_id=request_id)
      response = {'elements': similarity}
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in comparing two elements", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in comparing two elements"}', http.client.INTERNAL_SERVER_ERROR

    metrics.AI_SERVICE_ELEMENT_COMPARISON_SUCCESSFUL_TOTAL.inc()
    metrics.AI_SERVICE_ELEMENT_COMPARISON_LATENCY_SECONDS.observe(
      time.time() - start_time
    )
    return jsonify(response), http.client.OK

  return element_comparison


def text_assertion_view_func(logger, text_assertion: TextAssertion):
  def main_text_assertion_func():
    assert_req = request_pb2.TextAssertions()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      assert_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error('Text comparison: cannot parse request data',
                   extras={
                     'session_id': session_id,
                     'action_id': action_id,
                     'request_id': request_id,
                     'error': trace})
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      prime_sess = TextAssertPrimeSession(assert_req)
      revisit_sess = CompareRevisitSession(assert_req)
      results = text_assertion(prime_sess, revisit_sess,
                               session_id=session_id, action_id=action_id, request_id=request_id)
      response = {'elements': results, 'level': assert_req.level.upper()}
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in text assertion", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in text assertion"}', http.client.INTERNAL_SERVER_ERROR
    return jsonify(response), http.client.OK

  return main_text_assertion_func


def visual_verification_view_func(logger, visual_verification: VisualVerification):
  def main_visual_verification_func():
    assert_req = request_pb2.VisualVerification()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      assert_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error('Layout comparison: cannot parse request data',
                   extras={
                     'session_id': session_id,
                     'action_id': action_id,
                     'request_id': request_id,
                     'error': trace})
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      prime_sess = VisualVerifyPrimeSession(assert_req, logger)
      revisit_sess = VisualVerifyRevisitSession(assert_req, logger)
      logger.info('Visual verification starts processing a request',
                  extras={
                    'session_id': session_id,
                    'action_id': action_id,
                    'request_id': request_id,
                    'request_prime_device_name': prime_sess.device,
                    'request_prime_display_size': prime_sess.display_screen_size,
                    'request_prime_screen_density': prime_sess.density,
                    'request_revisit_device_name': revisit_sess.device,
                    'request_revisit_display_size': revisit_sess.display_screen_size,
                    'request_revisit_screen_density': revisit_sess.density
                  })
      results = visual_verification(prime_sess, revisit_sess,
                                    session_id=session_id,
                                    action_id=action_id,
                                    request_id=request_id)
      response = results
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in visual verification", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Visual verification got error"}', \
             http.client.INTERNAL_SERVER_ERROR

    return jsonify(response), http.client.OK

  return main_visual_verification_func

def wbi_view_func(logger, wbi: WBI):
  def main_wbi_func():
    assert_req = request_pb2.FontSize()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      assert_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error('Font size comparison: cannot parse request data',
                   extras={
                     'session_id': session_id,
                     'action_id': action_id,
                     'request_id': request_id,
                     'error': trace})
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      prime_sess = WBIPrimeSession(assert_req)
      revisit_sess = WBIRevisitSession(assert_req)
      logger.info('Font size starts processing a request',
                  extras={
                    'session_id': session_id,
                    'action_id': action_id,
                    'request_id': request_id,
                    'request_prime_device_name': prime_sess.device,
                    'request_prime_display_size': prime_sess.display_screen_size,
                    'request_prime_screen_density': prime_sess.density,
                    'request_revisit_device_name': revisit_sess.device,
                    'request_revisit_display_size': revisit_sess.display_screen_size,
                    'request_revisit_screen_density': revisit_sess.density
                  })
      results = wbi(prime_sess, revisit_sess, session_id=session_id, action_id=action_id, request_id=request_id)
      response = results
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in font size session", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Font size session got error"}', \
             http.client.INTERNAL_SERVER_ERROR

    return jsonify(response), http.client.OK
  return main_wbi_func


def check_service_availability_view_func():
  def check_service_availability():
    response = {'status': 'OK'}
    return jsonify(response), http.client.OK

  return check_service_availability


def same_size_verification_view_func(logger):
  def main_same_size_verification_func():
    screen_size_req = request_pb2.SameDeviceScreenSizeVerificationRequest()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      screen_size_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error('Layout comparison: cannot parse request data',
                   extras={
                     'session_id': session_id,
                     'action_id': action_id,
                     'request_id': request_id,
                     'error': trace})
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      response = get_same_size_devices(screen_size_req.target_device_name, screen_size_req.device_names)
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in visual verification", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in visual verification"}', \
             http.client.INTERNAL_SERVER_ERROR
    return jsonify(response), http.client.OK

  return main_same_size_verification_func


def element_finding_by_segmentation_view_func(logger, finder: ElementFinder):
  def element_finding_by_segmentation():
    start_time = time.time()
    metrics.AI_SERVICE_ELEMENT_FINDING_TOTAL.inc()
    finding_req = request_pb2.ElementFindingData()
    action_id = request.args.get('action_id', 0)
    session_id = request.args.get('session_id', 0)
    request_id = request.args.get('request_id', 0)
    try:
      finding_req.ParseFromString(request.data)
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    try:
      prime_sess = PrimeSession(finding_req)
      revisit_sess = RevisitSession(finding_req)

      logger.info('Received an request, finding element using segmentation',
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
    except Exception:
      trace = traceback.format_exc()
      logger.error(
        "Cannot parse request data",
        extras={
          "session_id": session_id,
          "action_id": action_id,
          "error": trace,
        },
      )
      return '{"message":"Cannot parse request data"}', http.client.BAD_REQUEST

    if prime_sess.element.img is None or prime_sess.screen_img is None \
        or revisit_sess.screen_img is None:
      logger.error(
        "Missing request data",
        extras={"session_id": session_id, "action_id": action_id},
      )
      return '{"message":"Missing request data"}', http.client.BAD_REQUEST

    try:
      revisit_xpaths, confidences = finder(
        prime_sess, revisit_sess,
        session_id=session_id, action_id=action_id, request_id=request_id, using_segment=True)
      response = {'elements': []}
    except Exception:
      trace = traceback.format_exc()
      logger.error("Error in finding elements", extras={
        'session_id': session_id,
        'action_id': action_id,
        'request_id': request_id,
        'error': trace})
      return '{"message":"Error in finding element"}', http.client.INTERNAL_SERVER_ERROR

    if revisit_xpaths and len(revisit_xpaths) > 0:
      elements = [
        {"xpath": revisit_xpaths[i], "confidence": confidences[i]}
        for i in range(len(revisit_xpaths))
      ]
      response["elements"] = elements
    metrics.AI_SERVICE_ELEMENT_FINDING_SUCCESSFUL_TOTAL.inc()
    metrics.AI_SERVICE_ELEMENT_FINDING_LATENCY_SECONDS.observe(
      time.time() - start_time
    )
    return jsonify(response), http.client.OK
  return element_finding_by_segmentation


def main():
  segmentation_info = get_segmentation_service_info(logger)
  ocr_service_info = get_kobiton_ocr_service_info(logger)
  app = Flask(constants.APP_NAME)

  services = {constants.TENSORFLOW_SERVING_SERVICE: '%s:%s' % (config.serving_host, config.serving_port),
            constants.SEGMENTATION_SERVICE: '%s:%s' % (segmentation_info[0], segmentation_info[1]),
            constants.KOBITON_OCR_SERVICE: '%s:%s' % (ocr_service_info[0], ocr_service_info[1])}

  finder = ElementFinder(services, logger=logger)
  finder_at_point = ElementFinderAtPoint(services, logger=logger)
  finder_by_image = ElementFinderByImage(services, logger=logger)
  comparator = ElementComparator(services, logger=logger)
  text_comparator = TextAssertion(services, logger=logger)
  visual_comparator = VisualVerification(services, logger=logger)
  
  wbi_comparator = WBI(services, logger=logger)

  # Add prometheus wsgi middleware to route /metrics requests
  combined_server = DispatcherMiddleware(app, {"/metrics": make_wsgi_app()})

  element_finding = element_finding_view_func(logger, finder)
  element_finding_by_segmentation = element_finding_by_segmentation_view_func(logger, finder)
  element_finding_at_point = element_finding_at_point_view_func(logger, finder_at_point)
  element_finding_by_image = element_finding_by_image_view_func(logger, finder_by_image)
  element_comparison = element_comparison_view_func(logger, comparator)
  text_assertion_func = text_assertion_view_func(logger, text_comparator)
  visual_verification_func = visual_verification_view_func(logger, visual_comparator)
  wbi_func = wbi_view_func(logger, wbi_comparator)
  same_size_verification_func = same_size_verification_view_func(logger)
  check_service_availability_func = check_service_availability_view_func()

  # Luna Blueprint
  app.register_blueprint(luna_engine, url_prefix=constants.LUNA_ENGINE)

  # Facenet Blueprint
  app.register_blueprint(facenet_engine, url_prefix=constants.FACENET_ENGINE)

  # Recommendation
  app.register_blueprint(rec_engine, url_prefix=constants.RECOMMENDATION_ENGINE)

  # Accessibility
  app.register_blueprint(accessibility_assertion, url_prefix=constants.ACCESSIBILITY_ASSERTION_URL)

  # Define API
  app.add_url_rule(
    constants.ELEMENT_FINDING_URL, methods=["POST"], view_func=element_finding)

  app.add_url_rule(
    constants.ELEMENT_FINDING_AT_POINT_URL, methods=["POST"], view_func=element_finding_at_point)

  app.add_url_rule(
    constants.ELEMENT_FINDING_BY_IMAGE_URL, methods=["POST"], view_func=element_finding_by_image)

  app.add_url_rule(
    constants.ELEMENT_FINDING_SEGMENTATION_URL, methods=["POST"], view_func=element_finding_by_segmentation)

  app.add_url_rule(
    constants.ELEMENT_COMPARISON_URL, methods=["POST"], view_func=element_comparison)

  app.add_url_rule(
    constants.TEXT_ASSERTION_URL, methods=["POST"], view_func=text_assertion_func)

  app.add_url_rule(
    constants.WBI_URL, methods=["POST"], view_func=wbi_func)

  app.add_url_rule(
    constants.VISUAL_VERIFICATION_URL, methods=["POST"], view_func=visual_verification_func)

  app.add_url_rule(
    constants.HEALTH_CHECK_URL, methods=["GET"], view_func=check_service_availability_func)

  app.add_url_rule(constants.DEVICE_SAME_SIZE_URL, methods=["POST"], view_func=same_size_verification_func)

  # Logging
  print('args.deploy_host: ', config.deploy_host)
  print('args.deploy_port: ', config.deploy_port)
  print('args.serving_host: ', config.serving_host)
  print('args.serving_port: ', config.serving_port)
  print('args.segmentation_host: ', segmentation_info[0])
  print('args.segmentation_port: ', segmentation_info[1])
  print('args.logstash_host: ', config.logstash_host)
  print('AI service start running...')
  run_simple(
    config.deploy_host,
    int(config.deploy_port),
    combined_server,
    # Turn 2 following flags to true for developmental mode
    use_reloader=False,
    use_debugger=False,
    threaded=True,
  )


if __name__ == "__main__":
  main()
