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

"""Make sure do not include any other third-party library in this file. e.g., tensorflow, absl ..."""

import os


def find_main():
  """Find base directory of our codebase, which should be current dir
  for all binaries in monolith codebase."""
  path = os.path.abspath(__file__)

  splits = ['/__main__/', '/site-packages/', '/monolith/']
  main_dir = None
  for split in splits:
    if split in path:
      end = path.rfind(split)
      if split == '/monolith/':
        main_dir = path[0:end]
      else:
        main_dir = os.path.join(path[0:end], split.strip('/'))
      break

  if main_dir is not None and os.path.exists(os.path.join(main_dir,
                                                          'monolith')):
    return main_dir
  else:
    raise ValueError(
        "Unable to find the monolith base directory. This file directory is {}. Are you running under bazel structure?"
        .format(path))


def get_libops_path(lib_name):
  base = find_main()  # monolith base
  return os.path.join(base, lib_name)
