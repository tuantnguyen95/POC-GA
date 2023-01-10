import traceback
import sys
import socket
import logging
import time
from queue import Queue
from threading import Thread
import json
import requests


class LoggingWrapper():

  def __init__(self, logstash_host='',
               environment='test', component='ai-service', namespace=''):
    self.component = component
    self.namespace = namespace
    self.logger = logging.getLogger(self.namespace)
    if len(self.logger.handlers) == 0:
      handler = logging.StreamHandler(sys.stdout)
      formatter = logging.Formatter('%(asctime)s %(name)s %(message)s')
      handler.setFormatter(formatter)
      self.logger.addHandler(handler)
      self.logger.setLevel(logging.NOTSET)
      logging.root.setLevel(logging.NOTSET)
    self.environment = environment
    self.logstash_host = logstash_host
    self.preset = {}
    self.fill_preset()
    self.log_queue = Queue(maxsize=0)
    if not self.is_empty_string(self.logstash_host):
      self.push_worker = self.LogstashWorker(self.log_queue, logstash_host)
      self.push_worker.start()
  
  def is_empty_string(self, st):
    return not st or not st.strip()

  class LogstashWorker(Thread):

    def __init__(self, queue, host, *args, **kwargs):
      self.queue = queue
      self.host = host
      super().__init__(*args, **kwargs)

    def run(self):
      while True:
        log = self.queue.get()
        try:
          headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
          requests.put(self.host, data=json.dumps({'logs':[log]}), headers=headers, timeout=5)
        except Exception:
          trace = traceback.format_exc()
          logging.error('Cannot push %s to logstash due to %s.', log, trace)


  def put_logstash_queue(self, message, log_type='DEBUG'):
    if not self.is_empty_string(self.logstash_host):
      log_obj = self._create_logstash_obj(message, log_type)
      self.log_queue.put(log_obj)


  def fill_preset(self):
    self.preset['component'] = self.component
    self.preset['namespace'] = self.namespace
    self.preset['environment'] = self.environment
    hostname = socket.gethostname()
    self.preset['ipMachine'] = socket.gethostbyname(hostname)


  def _create_logstash_obj(self, message, log_type):
    all_data = {}
    all_data.update(self.preset)
    all_data['message'] = message
    all_data['type'] = log_type
    all_data['timestamp'] = int(time.time() * 1000)
    return all_data


  def _merge_with_preset(self, data):
    all_data = {}
    all_data.update(self.preset)
    all_data.update(data)
    return all_data


  def _format_log_message(self, message, extras=None):
    if not extras:
      return message
    return "%s %s" % (message, json.dumps(extras))


  def info(self, message, extras=None):
    message = self._format_log_message(message, extras=extras)
    self.logger.info(message)
    self.put_logstash_queue(message, 'INFO')


  def warning(self, message, extras=None):
    message = self._format_log_message(message, extras=extras)
    self.logger.warning(message)
    self.put_logstash_queue(message, 'WARNING')


  def error(self, message, extras=None):
    message = self._format_log_message(message, extras=extras)
    self.logger.error(message)
    self.put_logstash_queue(message, 'ERROR')


  def debug(self, message, extras=None):
    message = self._format_log_message(message, extras=extras)
    self.logger.debug(message)
    self.put_logstash_queue(message, 'DEBUG')


  def critical(self, message, extras=None):
    message = self._format_log_message(message, extras=extras)
    self.logger.critical(message)
    self.put_logstash_queue(message, 'CRITICAL')
