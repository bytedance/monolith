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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_READER_UTIL_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_READER_UTIL_H_

#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/nlohmann/json.hpp"

constexpr uint64_t fid_v1_mask = (1LL << 54) - 1;
constexpr uint64_t fid_v2_mask = (1LL << 48) - 1;

inline int slot_id_v1(uint64_t fid) { return fid >> 54; }

inline int slot_id_v2(uint64_t fid) {
  return (fid >> 48) & (((int64_t)1 << 15) - 1);
}

inline uint64_t convert_fid_v1_to_v2(int slot, uint64_t fid) {
  uint64_t slot_long = slot;
  return ((fid & fid_v2_mask) | slot_long << 48);
}

inline uint64_t convert_fid_v1_to_v2(uint64_t fid) {
  uint64_t slot_long = fid >> 54;
  return ((fid & fid_v2_mask) | slot_long << 48);
}

inline int get_max_slot_number() { return 1 << 15; }

namespace tensorflow {
namespace monolith_tf {

inline int64_t GetFidV1(int slot, int64_t signautre) {
  return ((uint64_t)slot << 54) | (signautre & fid_v1_mask);
}

inline int64_t GetFidV2(int slot, int64_t signature) {
  return ((uint64_t)slot << 48) | (signature & fid_v2_mask);
}

class FeaturePruningByteCounter {
 public:
  ~FeaturePruningByteCounter() {
    LOG(INFO) << absl::StrFormat("Finally %s", DebugString());
  }

  void AddByteSize(uint64_t byte_size) { byte_size_ += byte_size; }

  void AddByteSizePruned(uint64_t byte_size) { byte_size_pruned_ += byte_size; }

  std::string DebugString() const {
    return absl::StrFormat(
        "read: %llu bytes (%s), after pruning: %llu bytes (%s)", byte_size_,
        PrettyBytes(byte_size_), byte_size_pruned_,
        PrettyBytes(byte_size_pruned_));
  }

 private:
  static std::string PrettyBytes(uint64_t bytes) {
    const std::vector<std::string> suffixes = {"B",  "KB", "MB", "GB",
                                               "TB", "PB", "EB"};
    int64_t s = 0;
    auto count = static_cast<double>(bytes);
    while (count >= 1024 && s < suffixes.size()) {
      s++;
      count /= 1024;
    }

    if (count - std::floor(count) == 0.0) {
      return absl::StrFormat("%llu %s", static_cast<uint64_t>(count),
                             suffixes[s]);
    } else {
      return absl::StrFormat("%.2f %s", count, suffixes[s]);
    }
  }

  uint64_t byte_size_;
  uint64_t byte_size_pruned_;
};

struct FeatureNameMapperIdInfo {
  int32_t id;
  int32_t sorted_id;
};
void to_json(nlohmann::json& j, const FeatureNameMapperIdInfo& info);

class FeatureNameMapper {
 public:
  FeatureNameMapper(FeatureNameMapper const&) = delete;
  void operator=(FeatureNameMapper const&) = delete;

  void TurnOn() { turned_on_ = true; }

  bool IsAvailable() const { return turned_on_ && initialized_; }

  bool RegisterValidIds(const std::vector<std::pair<int, int>>& valid_ids) {
    absl::WriterMutexLock l(&mu_);
    registered_feature_id_set_.insert(valid_ids.begin(), valid_ids.end());

    if (id_to_name_.empty()) {
      return true;
    }

    absl::flat_hash_set<std::pair<int, int>> invalid_ids;
    for (std::pair<int, int> p : registered_feature_id_set_) {
      if (!id_to_name_.contains(p.first) && !id_to_name_.contains(p.second)) {
        invalid_ids.insert(p);
      }
    }
    if (!invalid_ids.empty()) {
      nlohmann::json j;
      j["invalid_ids"] = invalid_ids;
      LOG(ERROR) << "ResisterValidIds: " << j.dump();
      return false;
    }

    for (std::pair<int, int> p : registered_feature_id_set_) {
      std::vector<int> ids = {p.first, p.second};
      for (int id : ids) {
        auto it = id_to_name_.find(id);
        if (it != id_to_name_.end()) {
          for (const std::string& name : it->second) {
            valid_id_to_name_[it->first].push_back(name);
            auto sorted_id = name_to_id_.at(name).sorted_id;
            valid_name_to_id_.insert({name, {it->first, sorted_id}});
          }
        }
      }
    }

    return true;
  }

  bool RegisterValidNames(const std::vector<std::string>& valid_names) {
    absl::WriterMutexLock l(&mu_);
    registered_feature_name_set_.insert(valid_names.begin(), valid_names.end());

    if (name_to_id_.empty()) {
      return true;
    }

    std::unordered_set<std::string> invalid_names;
    for (const std::string& name : registered_feature_name_set_) {
      if (!name_to_id_.contains(name)) {
        invalid_names.insert(name);
      }
    }
    if (!invalid_names.empty()) {
      nlohmann::json j;
      j["invalid_names"] = invalid_names;
      LOG(ERROR) << "ResisterValidNames: " << j.dump();
      return false;
    }

    for (const std::string& name : registered_feature_name_set_) {
      auto it = name_to_id_.find(name);
      valid_name_to_id_.insert({it->first, it->second});
      valid_id_to_name_[it->second.id].push_back(it->first);
    }

    return true;
  }

  bool SetMapping(const absl::flat_hash_map<std::string, int32_t>& name_to_id,
                  const absl::flat_hash_map<int32_t, std::vector<std::string>>&
                      id_to_name) {
    absl::WriterMutexLock l(&mu_);

    int sorted_id = 0;
    for (auto& iter : name_to_id) {
      name_to_id_[iter.first] = {iter.second, ++sorted_id};
    }
    id_to_name_ = id_to_name;
    if (name_to_id_.empty()) {
      return true;
    }

    std::unordered_set<std::string> invalid_names;
    for (const std::string& name : registered_feature_name_set_) {
      if (!name_to_id_.contains(name)) {
        invalid_names.insert(name);
      }
    }

    absl::flat_hash_set<std::pair<int, int>> invalid_ids;
    for (std::pair<int, int> p : registered_feature_id_set_) {
      if (!id_to_name_.contains(p.first) && !id_to_name_.contains(p.second)) {
        invalid_ids.insert(p);
      }
    }

    if (!invalid_names.empty() || !invalid_ids.empty()) {
      name_to_id_.clear();
      id_to_name_.clear();
      nlohmann::json j;
      j["invalid_names"] = invalid_names;
      j["invalid_ids"] = invalid_ids;
      LOG(ERROR) << "SetMapping: " << j.dump();
      return false;
    }

    for (std::pair<int, int> p : registered_feature_id_set_) {
      std::vector<int> ids = {p.first, p.second};
      for (int id : ids) {
        auto it = id_to_name_.find(id);
        if (it != id_to_name_.end()) {
          for (const std::string& name : it->second) {
            valid_id_to_name_[it->first].push_back(name);
            auto sorted_id = name_to_id_.at(name).sorted_id;
            valid_name_to_id_.insert({name, {it->first, sorted_id}});
          }
        }
      }
    }

    for (const std::string& name : registered_feature_name_set_) {
      auto it = name_to_id_.find(name);
      valid_name_to_id_.insert({it->first, it->second});
      valid_id_to_name_[it->second.id].push_back(it->first);
    }

    initialized_ = true;
    return true;
  }

  bool GetIdByName(const std::string& name, int32_t* id,
                   int32_t* sorted_id = nullptr) {
    absl::ReaderMutexLock l(&mu_);
    auto it = valid_name_to_id_.find(name);
    if (it == valid_name_to_id_.end()) {
      return false;
    }

    *id = it->second.id;
    if (sorted_id) {
      *sorted_id = it->second.sorted_id;
    }
    return true;
  }

  std::string DebugString() {
    absl::ReaderMutexLock l(&mu_);
    nlohmann::json j = valid_name_to_id_;
    return j.dump(2);
  }

  FeatureNameMapper() : turned_on_(false), initialized_(false) {}

 private:
  std::atomic_bool turned_on_;
  std::atomic_bool initialized_;
  absl::flat_hash_map<int32_t, std::vector<std::string>> id_to_name_;

  absl::flat_hash_map<std::string, FeatureNameMapperIdInfo> name_to_id_;
  absl::flat_hash_map<int32_t, std::vector<std::string>> valid_id_to_name_;
  absl::flat_hash_map<std::string, FeatureNameMapperIdInfo> valid_name_to_id_;
  absl::flat_hash_set<std::string> registered_feature_name_set_;
  absl::flat_hash_set<std::pair<int, int>> registered_feature_id_set_;
  absl::Mutex mu_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_READER_UTIL_H_
