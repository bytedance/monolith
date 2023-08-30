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

import os
from absl import logging
from datetime import datetime, timedelta
from kazoo.client import KazooClient
from monolith.native_training.env_utils import get_zk_auth_data
import socket

_PORT = 2181


_HOSTS = ['10.226.91.73', '10.226.86.70', '10.224.126.131', '10.224.109.135']
_HOSTS_IPV6 = [
    'fdbd:dc02:ff:2:1:226:91:73', 'fdbd:dc02:ff:2:1:226:86:70',
    'fdbd:dc01:ff:1:1:224:126:131', 'fdbd:dc01:ff:1:1:224:109:135'
]


def is_ipv6_only():
  if "MY_HOST_IP" in os.environ or "MY_POD_IP" in os.environ or "MY_HOST_IPV6" in os.environ:
    # in tce/byterec environment
    ipv4_addr = os.environ.get("MY_HOST_IP", os.environ.get("MY_POD_IP", None))
    logging.info(f"in tce env, ipv4 address is {ipv4_addr}")
  else:
    try:
      ipv4_addr = socket.gethostbyname(socket.gethostname())
    except:
      ipv4_addr = None
    logging.info(f"not in tce env, ipv4 address is {ipv4_addr}")
  ipv6_only = not ipv4_addr
  logging.info(f"is_ipv6_only is {ipv6_only}")
  return ipv6_only
_HOSTS = []
_HOSTS_IPV6 = []
def default_zk_servers(use_ipv6: bool = False):
  if use_ipv6 or is_ipv6_only():
    return ','.join(
        ['[{ip}]:{port}'.format(ip=ip, port=_PORT) for ip in _HOSTS_IPV6])
  return ','.join(['{ip}:{port}'.format(ip=ip, port=_PORT) for ip in _HOSTS])


class MonolithKazooClient(KazooClient):

  def __init__(self, *args, **kwargs):
    if "auth_data" not in kwargs:
      kwargs["auth_data"] = get_zk_auth_data()
    super().__init__(*args, **kwargs)


def clear_zk_path(zk_server: str, job_name: str, force_clear_zk_path: bool):
  """Try to clear old path (no modification since 9 weeks ago), Clear ZK Path of current job."""

  zk_client = MonolithKazooClient(zk_server or default_zk_servers())
  base_path = '/monolith'
  delta = timedelta(weeks=9)  # two months

  try:
    zk_client.start()
    # 1) try to delete very old nodes, just like TTL
    zk_client.ensure_path(base_path)
    children = zk_client.get_children(base_path)
    for child in children:
      path = '{}/{}'.format(base_path, child)
      _, stat = zk_client.get_children(path, include_data=True)
      if datetime.fromtimestamp(stat.mtime // 1000) + delta < datetime.now():
        try:
          zk_client.delete(path, recursive=True)
        except:
          # in case error in parallel condition
          pass

    # 2) try to delete job_name
    job_path = '{}/{}'.format(base_path, job_name)
    state = zk_client.exists(job_path)
    if state is not None:
      if force_clear_zk_path:
        zk_client.delete(job_path, recursive=True)
      else:
        children = zk_client.get_children('/monolith')
        raise ValueError('there are [{}] in monolith zk path'.format(
            ','.join(children)))
  finally:
    zk_client.stop()
