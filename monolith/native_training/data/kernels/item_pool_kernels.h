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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_ITEM_POOL_KERNELS_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_ITEM_POOL_KERNELS_H_

#include "monolith/native_training/data/kernels/internal/cache_mgr.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {
namespace monolith_tf {
class ItemPoolResource : public ResourceBase {
 public:
  explicit ItemPoolResource(int max_item_num_per_channel, int start_num = 0);

  std::string DebugString() const override { return "ItemPoolResource"; }

  Status Add(uint64_t channel_id, uint64_t item_id,
             const std::shared_ptr<const internal::ItemFeatures>& item);

  std::shared_ptr<const internal::ItemFeatures> Sample(uint64_t channel_id,
                                                       double* freq_factor,
                                                       double* time_factor);

  Status Save(WritableFile* ostream, int shard_index, int shard_num);

  Status Restore(RandomAccessFile* istream, int64 buffer_size);

  inline int start_num() { return start_num_; }
  inline int max_item_num_per_channel() { return max_item_num_per_channel_; }
  bool Equal(const ItemPoolResource& other) const;
  void SampleChannelID(uint64_t* channel_id);

 private:
  absl::Mutex mu_;
  int start_num_, max_item_num_per_channel_;
  std::unique_ptr<internal::CacheManager> cache_ ABSL_GUARDED_BY(mu_);
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_KERNELS_ITEM_POOL_KERNELS_H_
