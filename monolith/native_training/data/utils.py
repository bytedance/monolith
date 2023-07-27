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

from typing import Dict, Union, List
TOBENV = False

USED_FREATUE_NAMES = {}
NAME_TO_SLOT = {}


def enable_to_env():
  global TOBENV
  TOBENV = True


def get_slot_feature_name(slot: int):
  if TOBENV:
    return "fc_slot_{}".format(slot)
  else:
    return "slot_{}".format(slot)


def get_slot_from_feature_name(feature_name: str):
  if feature_name in NAME_TO_SLOT:
    return NAME_TO_SLOT[feature_name]
  elif feature_name.startswith('slot_') or feature_name.startswith('fc_slot_'):
    slot = feature_name.split('_')[-1]
    return int(slot) if slot.isdigit() else None
  else:
    if feature_name in USED_FREATUE_NAMES:
      return USED_FREATUE_NAMES[feature_name]
    else:
      USED_FREATUE_NAMES[feature_name] = len(USED_FREATUE_NAMES) + 1
      return len(USED_FREATUE_NAMES)


def register_slots(sparse_features: Union[List[int], Dict[str, int]]):
  if isinstance(sparse_features, (list, tuple)):
    assert all([isinstance(x, int) for x in sparse_features])
    sparse_features = {get_slot_feature_name(slot): slot for slot in sparse_features}
  else:
    assert isinstance(sparse_features, dict)

  NAME_TO_SLOT.update(sparse_features)
