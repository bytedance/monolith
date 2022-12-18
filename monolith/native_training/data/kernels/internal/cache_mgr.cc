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

#include "monolith/native_training/data/kernels/internal/cache_mgr.h"

#include <cstdlib>
#include <deque>
#include <random>

#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
using LineId = ::idl::matrix::proto::LineId;
using Action = google::protobuf::RepeatedField<int>;
using Example = ::monolith::io::proto::Example;
using EFeature = ::monolith::io::proto::NamedFeature;
using ChannelCache = ::monolith::io::proto::ChannelCache;

namespace tensorflow {
namespace monolith_tf {
namespace internal {

bool ItemFeatures::Equal(const ItemFeatures& other) const {
  if (item_id != other.item_id) {
    return false;
  }

  if (fids.size() != other.fids.size()) {
    return false;
  } else {
    std::unordered_set<uint64_t> this_fids(fids.begin(), fids.end());
    std::unordered_set<uint64_t> other_fids(other.fids.begin(),
                                            other.fids.end());

    if (this_fids.size() != other_fids.size()) {
      return false;
    } else {
      std::set<uint64_t> intersection;
      std::set_intersection(this_fids.begin(), this_fids.end(),
                            other_fids.begin(), other_fids.end(),
                            std::inserter(intersection, intersection.begin()));
      if (this_fids.size() != intersection.size()) {
        return false;
      }
    }
  }

  if (example_features.size() != other.example_features.size()) {
    for (const auto& it : example_features) {
      if (other.example_features.count(it.first) == 0) {
        return false;
      } else {
        auto this_feat = it.second.SerializeAsString();
        auto other_feat =
            other.example_features.at(it.first).SerializeAsString();
        if (this_feat != other_feat) {
          return false;
        }
      }
    }
  }

  return true;
}

CacheWithGid::CacheWithGid(int max_item_num, int start_num)
    : start_num_(start_num), max_item_num_(max_item_num) {}

void CacheWithGid::Push(uint64_t item_id,
                        std::shared_ptr<const ItemFeatures> item) {
  auto it = data_.find(item_id);
  if (it == data_.end()) {
    data_queue_.emplace_back(item_id);
    data_.emplace(item_id, item);
  }

  auto iit = stats_.find(item_id);
  if (iit == stats_.end()) {
    auto stats_ptr = std::make_shared<GroupStat>();
    stats_ptr->origin_cnt = 1;
    stats_ptr->sample_cnt = 0;
    stats_.emplace(item_id, stats_ptr);
  } else {
    iit->second->origin_cnt++;
  }

  if (data_queue_.size() > max_item_num_) {
    uint64_t item_id = data_queue_.front();
    data_.erase(item_id);
    stats_.erase(item_id);
    data_queue_.pop_front();
  }
}

std::shared_ptr<const ItemFeatures> CacheWithGid::RandomSelectOne(
    double* freq_factor, double* time_factor) const {
  if (data_queue_.size() <= start_num_) {
    return nullptr;
  }
  thread_local std::mt19937 gen((std::random_device())());
  size_t index = gen() % data_queue_.size();
  uint64_t item_id = data_queue_[index];
  auto it = data_.find(item_id);
  if (it != data_.end()) {
    *freq_factor = 1.0 / ++(stats_[item_id]->sample_cnt);
    *time_factor = (index + 1.0) / data_queue_.size();
    return it->second;
  } else {
    LOG_EVERY_N_SEC(ERROR, 1) << "item_id " << item_id
                              << "in queue but not in map";
  }
  return nullptr;
}

void CacheWithGid::ToProto(ChannelCache* proto) const {
  for (auto it : data_) {
    auto* feature_data = proto->add_feature_datas();
    feature_data->set_gid(it.first);  // gid

    for (const auto& fid : it.second->fids) {
      feature_data->add_fids(fid);
    }

    const auto& example_features =
        it.second->example_features;  // std::shared_ptr<const ItemFeatures>
    for (const auto& fc_it : example_features) {
      auto* feature_columns = feature_data->add_feature_columns();
      feature_columns->CopyFrom(fc_it.second);
    }

    const auto& stats = stats_[it.first];
    feature_data->set_origin_cnt(stats->origin_cnt);
    feature_data->set_sample_cnt(stats->sample_cnt);
  }
  LOG_EVERY_N(INFO, 1000) << "save size " << data_queue_.size() << " "
                          << data_.size();
}

void CacheWithGid::FromProto(const ChannelCache& proto) {
  data_queue_.clear();
  data_.clear();
  for (int i = 0; i < proto.feature_datas_size(); ++i) {
    const auto& feature_data = proto.feature_datas(i);
    auto gid = feature_data.gid();
    data_queue_.emplace_back(gid);

    auto group_feature_ptr = std::make_shared<ItemFeatures>();
    group_feature_ptr->item_id = gid;
    for (const auto& fid : feature_data.fids()) {
      group_feature_ptr->fids.push_back(fid);
    }
    for (const auto& fc : feature_data.feature_columns()) {
      group_feature_ptr->example_features.emplace(fc.name(), fc);
    }
    data_.emplace(gid, group_feature_ptr);

    std::shared_ptr<GroupStat> stats = std::make_shared<GroupStat>();
    stats->origin_cnt = feature_data.origin_cnt();
    stats->sample_cnt = feature_data.sample_cnt();
    stats_[gid] = stats;
  }
  LOG_EVERY_N(INFO, 1000) << "restore size " << data_queue_.size() << " "
                          << data_.size();
}

bool CacheWithGid::Equal(const CacheWithGid& other) const {
  if (start_num_ != other.start_num_) {
    return false;
  }

  if (max_item_num_ != other.max_item_num_) {
    return false;
  }

  if (stats_.size() != other.stats_.size()) {
    return false;
  } else {
    for (const auto& it : stats_) {
      auto oit = other.stats_.find(it.first);
      if (oit == other.stats_.end()) {
        return false;
      } else {
        if (it.second->Equal(*oit->second.get())) {
          return false;
        }
      }
    }
  }

  if (data_.size() != other.data_.size()) {
    return false;
  } else {
    for (const auto& it : data_) {
      if (other.data_.count(it.first) == 0) {
        return false;
      } else {
        return it.second->Equal(*other.data_.at(it.first).get());
      }
    }
  }

  return true;
}

CacheManager::CacheManager(int max_item_num_per_channel, int start_num)
    : start_num_(start_num),
      max_item_num_per_channel_(max_item_num_per_channel) {}

std::shared_ptr<const ItemFeatures> CacheManager::RandomSelectOne(
    uint64_t channel_id, double* freq_factor, double* time_factor) const {
  auto it = channel_cache_.find(channel_id);
  if (it != channel_cache_.end()) {
    return it->second.RandomSelectOne(freq_factor, time_factor);
  }
  return nullptr;
}

void CacheManager::Push(uint64_t channel_id, uint64_t item_id,
                        const std::shared_ptr<const ItemFeatures>& item) {
  auto it = channel_cache_.find(channel_id);
  if (it == channel_cache_.end()) {
    auto ret = channel_cache_.emplace(
        channel_id, CacheWithGid(max_item_num_per_channel_, start_num_));
    it = ret.first;
  }
  it->second.Push(item_id, item);
}

void CacheManager::Push(uint64_t channel_id, const CacheWithGid& cwg) {
  channel_cache_.emplace(channel_id, cwg);
}

absl::flat_hash_map<uint64_t, CacheWithGid>& CacheManager::GetCache() {
  return channel_cache_;
}

void CacheManager::SampleChannelID(uint64_t* channel_id) {
  std::vector<uint64_t> channel_ids;
  std::vector<int> cache_size;

  if (channel_cache_.size() >= 2) {
    for (auto iter = channel_cache_.begin(); iter != channel_cache_.end();
         ++iter) {
      if (iter->first != *channel_id) {
        channel_ids.emplace_back(iter->first);
        cache_size.emplace_back(iter->second.Size());
      }
    }

    std::discrete_distribution<int> discrete_dist(cache_size.begin(),
                                                  cache_size.end());

    std::mt19937 gen(std::random_device{}());
    *channel_id = channel_ids[discrete_dist(gen)];
  }
}

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
