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

import traceback
import signal
import time


def print_stack_trace(sig, frame):
  for line in traceback.format_stack(frame):
    print(line.strip())


def add_siguser1_handler():
  ret = signal.getsignal(signal.SIGUSR1)

  def handler(sig, frame):
    if callable(ret):
      ret(sig, frame)
    print_stack_trace(sig, frame)

  signal.signal(signal.SIGUSR1, handler)


# Adds default handler
add_siguser1_handler()
