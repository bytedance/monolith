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

from monolith import utils


def enable_tcmalloc():
  libs = os.environ.get("LD_PRELOAD", "").split(":")
  libs.append(
      utils.get_libops_path("../gperftools/libtcmalloc/lib/libtcmalloc.so"))
  os.environ["LD_PRELOAD"] = ":".join(libs)


def setup_heap_profile(heap_profile_inuse_interval=104857600,
                       heap_profile_allocation_interval=1073741824,
                       heap_profile_time_interval=0,
                       sample_ratio=1.0,
                       heap_profile_mmap=False):
  """See https://gperftools.github.io/gperftools/heapprofile.html for the meaning of each
  parameters meaning.

  Args:
    sample_ratio: ratio of new we tracked in the heap profiler. Since the full profiler is
    very slow, usually can be set something like 1/64.
  """
  enable_tcmalloc()
  os.environ["HEAPPROFILE"] = os.path.join(utils.find_main(), "hprof")
  os.environ["HEAP_PROFILE_INUSE_INTERVAL"] = str(
      int(heap_profile_inuse_interval / sample_ratio))
  os.environ["HEAP_PROFILE_ALLOCATION_INTERVAL"] = str(
      int(heap_profile_allocation_interval / sample_ratio))
  os.environ["HEAP_PROFILE_SAMPLE_RATIO"] = str(sample_ratio)
  os.environ["HEAP_PROFILE_TIME_INTERVAL"] = str(heap_profile_time_interval)
  os.environ["HEAP_PROFILE_MMAP"] = str(heap_profile_mmap).lower()
