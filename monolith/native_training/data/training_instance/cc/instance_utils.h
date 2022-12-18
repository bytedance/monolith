// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_INSTANCE_UTILS_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_INSTANCE_UTILS_H_

#include <set>
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"

#include "monolith/native_training/data/training_instance/cc/reader_util.h"

namespace tensorflow {
namespace monolith_tf {

template <typename T>
std::set<T> StrToIntegerSet(const std::string& str) {
  static_assert(
      std::is_same<T, uint64_t>::value || std::is_same<T, int32_t>::value,
      "Template typename T should be uint64_t or int32_t!");
  std::set<T> integers;
  std::set<std::string> splits = absl::StrSplit(str, ",");
  for (const auto& s : splits) {
    if (!s.empty()) {
      T fid;
      if (absl::SimpleAtoi(s, &fid)) {
        integers.insert(fid);
      } else {
        throw std::invalid_argument(
            absl::StrFormat("Invalid integer string: %s", s));
      }
    }
  }

  return integers;
}

template <typename T>
typename std::enable_if<std::is_same<T, parser::proto::Instance>::value,
                        void>::type
CollectFidIntoSet(const T& instance, std::set<uint64_t>* fid_set) {
  const auto& instance_fids = instance.fid();
  fid_set->insert(instance_fids.begin(), instance_fids.end());
}

template <typename T>
typename std::enable_if<std::is_same<T, monolith::io::proto::Example>::value,
                        void>::type
CollectFidIntoSet(const T& example, std::set<uint64_t>* fid_set) {
  for (const auto& named_feature : example.named_feature()) {
    if (named_feature.feature().has_fid_v1_list()) {
      const auto& fids = named_feature.feature().fid_v1_list().value();
      fid_set->insert(fids.begin(), fids.end());
    }
    if (named_feature.feature().has_fid_v2_list()) {
      const auto& fids = named_feature.feature().fid_v2_list().value();
      fid_set->insert(fids.begin(), fids.end());
    }
  }
}

template <typename T>
typename std::enable_if<std::is_same<T, parser::proto::Instance>::value,
                        void>::type
CollectSlotIntoSet(const T& instance, std::set<uint32_t>* slot_set) {
  for (uint64_t fid : instance.fid()) {
    int slot = slot_id_v1(fid);
    slot_set->insert(slot);
  }

  for (const auto& f : instance.feature()) {
    for (uint64_t fid : f.fid()) {
      int slot = slot_id_v2(fid);
      slot_set->insert(slot);
    }
  }
}

template <typename T>
typename std::enable_if<std::is_same<T, monolith::io::proto::Example>::value,
                        void>::type
CollectSlotIntoSet(const T& example, std::set<uint32_t>* slot_set) {
  for (const auto& named_feature : example.named_feature()) {
    if (named_feature.feature().has_fid_v1_list()) {
      const auto& fids = named_feature.feature().fid_v1_list().value();
      for (uint64_t fid : fids) {
        int slot = slot_id_v1(fid);
        slot_set->insert(slot);
      }
    }

    if (named_feature.feature().has_fid_v2_list()) {
      const auto& fids = named_feature.feature().fid_v2_list().value();
      for (uint64_t fid : fids) {
        int slot = slot_id_v2(fid);
        slot_set->insert(slot);
      }
    }
  }
}

template <typename T>
bool IsInstanceOfInterest(const T& pb, const std::set<uint64_t>& filter_fids,
                          const std::set<uint64_t>& has_fids,
                          const std::set<uint64_t>& select_fids,
                          const std::set<int32_t>& has_actions,
                          int64_t req_time_min,
                          const std::set<uint32_t>& select_slots) {
  if (pb.line_id().req_time() < req_time_min) {
    return false;
  }

  std::set<uint64_t> fid_set;
  CollectFidIntoSet(pb, &fid_set);

  std::set<uint32_t> slot_set;
  CollectSlotIntoSet(pb, &slot_set);

  const auto& actions = pb.line_id().actions();
  std::set<int32_t> instance_actions_set(actions.begin(), actions.end());

  if (!filter_fids.empty()) {
    std::set<uint64_t> intersection;
    std::set_intersection(fid_set.begin(), fid_set.end(), filter_fids.begin(),
                          filter_fids.end(),
                          std::inserter(intersection, intersection.begin()));
    // If the instance contains any one of the given `filter_fids`, it will be
    // dropped.
    if (!intersection.empty()) {
      return false;
    }
  }

  if (!has_fids.empty()) {
    std::set<uint64_t> intersection;
    std::set_intersection(fid_set.begin(), fid_set.end(), has_fids.begin(),
                          has_fids.end(),
                          std::inserter(intersection, intersection.begin()));
    // If the instance does not contain any one of the given `has_fids`, it will
    // be dropped.
    if (intersection.empty()) {
      return false;
    }
  }

  if (!select_fids.empty()) {
    std::set<uint64_t> intersection;
    std::set_intersection(fid_set.begin(), fid_set.end(), select_fids.begin(),
                          select_fids.end(),
                          std::inserter(intersection, intersection.begin()));
    // If the instance does not contain all of the given `select_fids`, it will
    // be dropped.
    if (intersection.size() < select_fids.size()) {
      return false;
    }
  }

  if (!select_slots.empty()) {
    std::set<uint32_t> intersection;
    std::set_intersection(slot_set.begin(), slot_set.end(),
                          select_slots.begin(), select_slots.end(),
                          std::inserter(intersection, intersection.begin()));
    // If the instance does not contain all of the given `select_slots`, it will
    // be dropped.
    if (intersection.size() < select_slots.size()) {
      return false;
    }
  }

  if (!has_actions.empty()) {
    std::set<int32_t> intersection;
    std::set_intersection(instance_actions_set.begin(),
                          instance_actions_set.end(), has_actions.begin(),
                          has_actions.end(),
                          std::inserter(intersection, intersection.begin()));
    // If the instance does not contain any one of the given `has_actions`, it
    // will be dropped.
    if (intersection.empty()) {
      return false;
    }
  }

  return true;
}

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_INSTANCE_UTILS_H_
