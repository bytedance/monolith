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

import contextlib
import hashlib
import os
import subprocess
import socket

from absl import logging
def setup_hdfs_env():
  pass
def get_zk_auth_data():
  ZK_AUTH = os.getenv('ZK_AUTH', None)
  if ZK_AUTH:
    print("ZK_AUTH", ZK_AUTH)
    return [("digest", ZK_AUTH)]
  return None
