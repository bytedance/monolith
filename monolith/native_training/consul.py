# Copyright 2022 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consul client from bytedance pylib.
"""
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from typing import Dict

from six.moves.http_client import HTTPConnection


class ConsulException(Exception):
  pass


class UnixHTTPConnection(HTTPConnection):

  def __init__(self, path, **kwargs):
    kwargs["host"] = "localhost"
    HTTPConnection.__init__(self, **kwargs)
    self.path = path

  def connect(self):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(self.path)
    self.sock = sock


class Client:

  def __init__(self):
    self._lock = threading.Lock()
    self._cache = {}
    self._consul_sock = "/opt/tmp/sock/consul.sock"
    self._consul_host = os.environ.get("CONSUL_HTTP_HOST") or os.environ.get(
        "TCE_HOST_IP")
    if not self._consul_host:
      if os.path.isfile(self._consul_sock):
        self._consul_host = self._consul_sock
      else:
        self._consul_host = "127.0.0.1"
    self._consul_port = int(os.environ.get("CONSUL_HTTP_PORT") or 2280)

  def lookup(self, name, timeout=3, cachetime=0):
    now = time.time()
    if cachetime > 0:
      cache = self._cache.get(name)
      if cache and now - cache["cachetime"] <= cachetime:
        return cache["ret"]
      timeout = timeout if cache else 30
      with self._lock:
        ret = self.lookup(name, timeout)
    else:
      ret = self._lookup(name, timeout)
    with self._lock:
      self._cache[name] = {
          "ret": ret,
          "cachetime": now,
      }
    return ret

  def _lookup(self, name, timeout):
    if self._consul_host.startswith("/"):
      conn = UnixHTTPConnection(self._consul_host)
    else:
      conn = HTTPConnection(self._consul_host,
                            self._consul_port,
                            timeout=timeout)
    conn.request("GET", "/v1/lookup/name?name=" + name)
    response = conn.getresponse()
    status = response.status
    data = response.read()
    conn.close()
    if status != 200:
      logging.error("consul: %s %s", status, data.decode("utf8"))
      return []
    return json.loads(data.decode("utf8"))

  def register(self, name, port, tags=None, check_script=None, host=None):
    d = {
        "id": "%s-%s" % (name, port),
        "name": name,
        "port": int(port),
        "check": {
            "ttl": "60s",
        }
    }
    if tags is not None:
      d["tags"] = ["%s:%s" % (k, v) for k, v in tags.items()]
    if check_script:
      d["check"] = {"interval": "30s", "script": check_script}
    if not host:
      host = self._consul_host
    conn = HTTPConnection(host, self._consul_port, timeout=15)
    conn.request("PUT", "/v1/agent/service/register", json.dumps(d))
    response = conn.getresponse()
    status = response.status
    data = response.read()
    if status != 200:
      raise ConsulException(data.decode("utf8"))

    def _health_check():
      while True:
        now = time.time()
        try:
          conn.request("GET", f"/v1/agent/check/pass/service:{name}-{port}")
          conn.getresponse().read()
        except socket.error:
          print(traceback.format_exc(), file=sys.stderr)
          time.sleep(2)
          # Immediately retry
          now -= 30
        time.sleep(max(30 + now - time.time(), 0))

    th = threading.Thread(name=f"ConsulHealthCheck-{name}-{port}",
                          target=_health_check,
                          daemon=True)
    th.start()
    # Maybe in the future, we want to garbage collect threads.

  def deregister(self, name, port, host=None):
    host = host or self._consul_host
    conn = HTTPConnection(host, self._consul_port, timeout=15)
    conn.request("PUT", "/v1/agent/service/deregister/%s-%s" % (name, port))
    response = conn.getresponse()
    status = response.status
    data = response.read()
    if status != 200:
      raise ConsulException(data.decode("utf8"))
    conn.close()
