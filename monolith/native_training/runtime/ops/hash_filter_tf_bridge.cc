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

#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"

namespace tensorflow {
namespace monolith_tf {

using ::monolith::hash_filter::Filter;
using ::monolith::hash_table::HashFilterSplitMetaDump;
using ::monolith::hash_table::HashFilterSplitDataDump;
using ::monolith::hash_table::SlotOccurrenceThresholdConfig;

HashFilterTfBridge::HashFilterTfBridge(
    std::unique_ptr<Filter> filter, const SlotOccurrenceThresholdConfig& config)
    : filter_(std::move(filter)) {
  slot_to_occurrence_threshold_.resize(get_max_slot_number(),
                                       config.default_occurrence_threshold());
  for (const auto& slot_occurrence_threshold :
       config.slot_occurrence_thresholds()) {
    slot_to_occurrence_threshold_[slot_occurrence_threshold.slot()] =
        slot_occurrence_threshold.occurrence_threshold();
  }
}

int HashFilterTfBridge::GetSlotOccurrenceThreshold(int64_t fid) const {
  return slot_to_occurrence_threshold_[slot_id_v2(fid)];
}

Status HashFilterTfBridge::Save(
    int split_idx, std::function<void(HashFilterSplitMetaDump)> write_meta_fn,
    std::function<void(HashFilterSplitDataDump)> write_data_fn) const {
  try {
    filter_->Save(split_idx, std::move(write_meta_fn),
                  std::move(write_data_fn));
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::ResourceExhausted(e.what());
  }
}

Status HashFilterTfBridge::Restore(
    int split_idx, std::function<bool(HashFilterSplitMetaDump*)> get_meta_fn,
    std::function<bool(HashFilterSplitDataDump*)> get_data_fn) const {
  try {
    filter_->Restore(split_idx, std::move(get_meta_fn), std::move(get_data_fn));
    return Status::OK();
  } catch (const std::exception& e) {
    return errors::ResourceExhausted(e.what());
  }
}

}  // namespace monolith_tf
}  // namespace tensorflow
