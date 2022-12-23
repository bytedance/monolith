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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_GPU_MULTI_HASH_TABLE
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_GPU_MULTI_HASH_TABLE
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <vector>

#include "monolith/native_training/runtime/hash_table/GPUcucohash/cuco_multi_table_ops.cuh.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
namespace monolith_tf {

class GpuMultiHashTable : public ResourceBase {
 public:
  ::monolith::hash_table::CucoMultiHashTableOp op;
  explicit GpuMultiHashTable(
      std::vector<int> slot_occ = {},
      ::monolith::hash_table::GpucucoEmbeddingHashTableConfig config_ = {})
      : op(std::move(slot_occ), std::move(config_)) {}
  std::string DebugString() const override {
    return "This is a GPU multi hash table";
  }
};

}  // namespace monolith_tf
}  // namespace tensorflow
#endif
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_GPU_MULTI_HASH_TABLE
