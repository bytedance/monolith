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

import atexit
import sys
import signal

from monolith.native_training import utils
from monolith.native_training import native_task_context
from monolith.native_training.metric import cli

sig_no = None


def sig_handler(signo, frame):
  global sig_no
  sig_no = signo
  sys.exit(signo)


signal.signal(signal.SIGHUP, sig_handler)
signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


@atexit.register
def exit_hook():
  ctx = native_task_context.get()
  mcli = cli.get_cli(utils.get_metric_prefix())
  index = ctx.worker_index if ctx.server_type == 'worker' else ctx.ps_index
  tags = {
      'server_type': ctx.server_type,
      'index': str(index),
      'sig': str(sig_no),
  }
  if sig_no is not None:
    mcli.emit_counter("exit_hook", 1, tags)
