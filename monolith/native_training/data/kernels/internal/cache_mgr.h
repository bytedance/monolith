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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_CACHE_MGR_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_CACHE_MGR_H_

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {

struct GroupStat {
  uint32_t origin_cnt = 0;
  uint32_t sample_cnt = 0;

  inline bool operator==(const GroupStat &rhs) const {
    return origin_cnt == rhs.origin_cnt && sample_cnt == rhs.sample_cnt;
  }

  inline bool Equal(const GroupStat &rhs) const {
    return origin_cnt == rhs.origin_cnt && sample_cnt == rhs.sample_cnt;
  }
};

struct ItemFeatures {
  uint64_t item_id;
  std::vector<uint64_t> fids;
  absl::flat_hash_map<std::string, ::monolith::io::proto::NamedFeature>
      example_features;

  bool Equal(const ItemFeatures &other) const;
};

class CacheWithGid {
 public:
  explicit CacheWithGid(int max_item_num, int start_num = 0);

  void Push(uint64_t item_id, std::shared_ptr<const ItemFeatures> item);

  std::shared_ptr<const ItemFeatures> RandomSelectOne(
      double *freq_factor, double *time_factor) const;

  void ToProto(::monolith::io::proto::ChannelCache *proto) const;

  void FromProto(const ::monolith::io::proto::ChannelCache &proto);

  bool Equal(const CacheWithGid &other) const;

  inline int Size() { return data_queue_.size(); }

 private:
  int start_num_;
  int max_item_num_;

  absl::flat_hash_map<uint64_t, std::shared_ptr<const ItemFeatures>> data_;
  mutable absl::flat_hash_map<uint64_t, std::shared_ptr<GroupStat>> stats_;
  std::deque<uint64_t> data_queue_;
};

class CacheManager {
 public:
  explicit CacheManager(int max_item_num_per_channel, int start_num = 0);

  std::shared_ptr<const ItemFeatures> RandomSelectOne(
      uint64_t channel_id, double *freq_factor, double *time_factor) const;

  void Push(uint64_t channel_id, uint64_t item_id,
            const std::shared_ptr<const ItemFeatures> &item);

  void Push(uint64_t channel_id, const CacheWithGid &cwg);

  absl::flat_hash_map<uint64_t, CacheWithGid> &GetCache();

  void SampleChannelID(uint64_t* channel_id);

 private:
  int start_num_;
  int max_item_num_per_channel_;

  absl::flat_hash_map<uint64_t, CacheWithGid> channel_cache_;
};

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_CACHE_MGR_H_
