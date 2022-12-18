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

from datetime import datetime, timedelta
from kazoo.client import KazooClient
from monolith.native_training.env_utils import get_zk_auth_data

_PORT = 2181
_HOSTS = []
_HOSTS_IPV6 = []
def default_zk_servers(use_ipv6: bool = False):
  if use_ipv6:
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
