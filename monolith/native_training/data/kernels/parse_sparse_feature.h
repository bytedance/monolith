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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_SPARSE_FEATURE_LIB_H_
#define MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_SPARSE_FEATURE_LIB_H_

#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "google/protobuf/descriptor.h"
#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"

#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/kernels/internal/uniq_hashtable.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/runtime/common/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

#include "absl/strings/match.h"

#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace monolith_tf {

class ShardingSparseFidsOp : public OpKernel {
 public:
  using FeatureListType = ::monolith::io::proto::FeatureListType;
  using Instance = ::parser::proto::Instance;
  using Example = ::monolith::io::proto::Example;
  using ExampleBatch = ::monolith::io::proto::ExampleBatch;
  using FeatureConfigs = ::monolith::io::proto::FeatureConfigs;

  explicit ShardingSparseFidsOp(OpKernelConstruction *ctx, int version = 1);
  ~ShardingSparseFidsOp() override { mapper_->Unref(); }
  void Compute(OpKernelContext *ctx) override;

 private:
  struct InstanceWrapper {
    struct FeaturePtr {
      FeaturePtr(const std::vector<uint64_t> *fid_v1_,
                 const idl::matrix::proto::Feature *fid_v2_)
          : fid_v1(fid_v1_), fid_v2(fid_v2_) {}
      const std::vector<uint64_t> *fid_v1 = nullptr;
      const idl::matrix::proto::Feature *fid_v2 = nullptr;
    };
    std::vector<const ::parser::proto::Instance *> instances;
    absl::flat_hash_map<std::string, std::vector<std::vector<uint64_t>>> fid_v1;
  };
  void InitInstanceWrapper(InstanceWrapper *instance_wrapper);

  template <typename TInput>
  Status FeatureParallelParse(OpKernelContext *ctx, const TInput &input,
                              OpOutputList *fid_list_out_list,
                              OpOutputList *fid_list_row_splits_out_list,
                              OpOutputList *fid_list_row_splits_size_out_list);

  int GetBatchSize(const InstanceWrapper &instance_wrapper) {
    return instance_wrapper.instances.size();
  }
  int GetBatchSize(const ::monolith::io::proto::ExampleBatch &example_batch) {
    return example_batch.batch_size();
  }
  template <typename TInput>
  int GetBatchSize(const std::vector<TInput> &inputs) {
    return inputs.size();
  }
  void ParallelRun(OpKernelContext *ctx, int task_count,
                   const std::function<void(int64, int64)> &fn);

  template <typename TData>
  struct TensorSliceAccessor {
    TData *ptr = nullptr;
    int64_t size = 0;
    TData &operator()(int64_t index) {
      // CHECK(index >= 0 && index < size);
      return *(ptr + index);
    }
  };
  void FillFidList(uint64_t value,
                   std::vector<std::vector<uint64_t>> &shard_vec,
                   MultiShardUniqHashTable &shard_uniq_hashtable,
                   tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
                   int feature_output_index, int *offset);
  void FillFidList(uint64_t value,
                   std::vector<std::vector<uint64_t>> &shard_vec,
                   tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
                   int feature_output_index, int *offset);
  void FillFidList(uint64_t value,
                   MultiShardUniqHashTable &shard_uniq_hashtable,
                   tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
                   int feature_output_index, int *offset);
  void CopyFidList(const std::vector<uint64_t> &shard_ptr, int offset,
                   TensorSliceAccessor<int64_t> *to);
  void CopyFidList(const absl::flat_hash_map<uint64_t, int> &shard_ptr,
                   int offset, TensorSliceAccessor<int64_t> *to);

  Status CreateOffsetTensor(
      OpKernelContext *ctx,
      const std::vector<std::vector<int>> &all_feature_counter,
      int all_feature_counter_size, Tensor **nfl_offset_tensor,
      Tensor **feature_offset_tensor, Tensor **fid_offset_tensor,
      OpOutputList *fid_list_row_splits_out_list,
      OpOutputList *fid_list_row_splits_size_out_list,
      std::vector<Tensor> &tmp_tensor_list,
      std::vector<TensorSliceAccessor<int64_t>> &fid_list_row_splits_flat_list,
      std::vector<int> *nfl_fid_offset,
      const std::unordered_set<int> *shared_feature);

  struct FeatureInfo {
    std::string feature_name;
    std::string table_name;
    int feature_index = -1;
    int table_index = -1;
    int feature_in_table_index = -1;

    int table_feature_count;
    int output_pre_index;
    int dims_sum;

    int version = 1;
    int GetFidOutputIndex(int ps_i) {
      if (version == 2) {
        return output_pre_index + ps_i * table_feature_count +
               feature_in_table_index;
      } else {
        return output_pre_index + ps_i;
      }
    }
    int GetPsShard(int fid_offset) {
      if (version == 2) {
        return (fid_offset - output_pre_index - feature_in_table_index) /
               table_feature_count;  // no use, only version==1 or version==3
                                     // will call this func
      } else {
        return fid_offset - output_pre_index;
      }
    }
  };

  absl::flat_hash_map<std::string, FeatureInfo> feature_conf_;
  std::vector<FeatureInfo *> feature_index_conf_;
  std::vector<FeatureInfo *> feature_cfg_list_;
  struct TableInfo {
    std::string table_name;
    int table_index = -1;
    int feature_count = 0;
    std::vector<int> feature_index_list;
  };
  absl::flat_hash_map<std::string, TableInfo> table_conf_;
  std::vector<TableInfo *> table_cfg_list_;
  absl::flat_hash_map<int, std::string>
      slot_id_to_feature_name_;  // instance fid_v1 slot 特征 映射

  int ps_num_ = 0;
  int single_thread_feature_watermark_ = 80000 * 4;
  int single_thread_assign_watermark_ = 100000 * 4;
  int int64_size_ = sizeof(int64_t);

  static constexpr uint32_t SHARED_FLAG = (1L << 31);

  bool unique_ = false;
  int parallel_flag_ = 0;
  int input_type_ = 0;
  bool enable_parallel_ = true;
  int version_ = 1;

  FeatureNameMapperTfBridge *mapper_ = nullptr;
  FeatureNameMapper *mapper_raw_ptr_ = nullptr;

  template <typename TContext>
  void SplitTask(const std::vector<TContext> &context_list, int limit,
                 std::vector<std::vector<int>> *out);

#define DFeatureParallelMakeUpTask1Context(INPUT_TYPE)                      \
  template <typename TFeatureParallelTask1Context>                          \
  void FeatureParallelMakeUpTask1Context(                                   \
      const INPUT_TYPE &input, int batch_size,                              \
      std::vector<TFeatureParallelTask1Context> *task_context_list,         \
      absl::flat_hash_map<int, TFeatureParallelTask1Context *>              \
          *feature_shard_count_map,                                         \
      absl::flat_hash_map<int, std::vector<TFeatureParallelTask1Context *>> \
          *table_feature_map,                                               \
      std::vector<std::vector<int>> *all_feature_counter,                   \
      std::unordered_set<int> *shared_feature, int *all_feature_counter_size)
  DFeatureParallelMakeUpTask1Context(::monolith::io::proto::ExampleBatch);
  DFeatureParallelMakeUpTask1Context(
      std::vector<const ::monolith::io::proto::Example *>);
  DFeatureParallelMakeUpTask1Context(InstanceWrapper);
#undef DFeatureParallelMakeUpTask1Context

#define DFeatureParallelDoTask1(INPUT_TYPE)                                \
  template <typename TFeatureParallelTask1Context>                         \
  void FeatureParallelDoTask1(                                             \
      const INPUT_TYPE &input, TFeatureParallelTask1Context *task_context, \
      std::vector<int> &nfl_fid_offset,                                    \
      tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,                  \
      tensorflow::TTypes<int32_t>::Flat feature_offset_flat)
  DFeatureParallelDoTask1(::monolith::io::proto::ExampleBatch);
  DFeatureParallelDoTask1(std::vector<const ::monolith::io::proto::Example *>);
  DFeatureParallelDoTask1(InstanceWrapper);
#undef DFeatureParallelDoTask1
};

template <typename TContext>
void ShardingSparseFidsOp::SplitTask(const std::vector<TContext> &context_list,
                                     int limit,
                                     std::vector<std::vector<int>> *out) {
  std::vector<int> full_index;
  int pre_count = 0;
  for (unsigned int i = 0; i < context_list.size(); ++i) {
    const auto &context = context_list[i];
    if (context.size == 0) {
      continue;
    } else if (context.size >= limit) {
      full_index.push_back(i);
    } else {
      if (pre_count == 0) {
        out->emplace_back(std::vector<int>());
      }
      if (pre_count + context.size <= limit) {
        out->back().push_back(i);
        pre_count += context.size;
      } else {
        out->emplace_back(std::vector<int>());
        out->back().push_back(i);
        pre_count = context.size;
      }
    }
  }
  for (auto index : full_index) {
    out->emplace_back(std::vector<int>({index}));
  }
}

template <typename TFeatureParallelTask1Context>
void ShardingSparseFidsOp::FeatureParallelMakeUpTask1Context(
    const ::monolith::io::proto::ExampleBatch &example_batch, int batch_size,
    std::vector<TFeatureParallelTask1Context> *task_context_list,
    absl::flat_hash_map<int, TFeatureParallelTask1Context *>
        *feature_shard_count_map,
    absl::flat_hash_map<int, std::vector<TFeatureParallelTask1Context *>>
        *table_feature_map,
    std::vector<std::vector<int>> *all_feature_counter,
    std::unordered_set<int> *shared_feature, int *all_feature_counter_size) {
  std::vector<int> example_batch_feature_index(feature_conf_.size(), -1);
  for (int n_i = 0; n_i < example_batch.named_feature_list_size(); ++n_i) {
    const auto &named_feature_list = example_batch.named_feature_list(n_i);
    auto &name = named_feature_list.name();
    auto find_iter = feature_conf_.find(name);
    if (find_iter == feature_conf_.end()) {
      continue;
    }
    example_batch_feature_index[find_iter->second.feature_index] = n_i;
  }
  for (uint i = 0; i < example_batch_feature_index.size(); ++i) {
    int n_i = example_batch_feature_index[i];
    auto &feature_counter = (*all_feature_counter)[i];
    if (n_i < 0) {
      feature_counter.push_back(0);
      shared_feature->insert(i);
      *all_feature_counter_size += 1;
      continue;
    }
    TFeatureParallelTask1Context *task_context = nullptr;
    auto table_index = feature_cfg_list_[i]->table_index;
    auto feature_shard_count_map_iter = feature_shard_count_map->find(i);
    if (feature_shard_count_map_iter == feature_shard_count_map->end()) {
      task_context_list->emplace_back(TFeatureParallelTask1Context());
      task_context = &(task_context_list->back());
      task_context->example_batch_feature_index = n_i;
      task_context->feature_output_index = i;
      (*feature_shard_count_map)[i] = task_context;
      (*table_feature_map)[table_index].push_back(task_context);
    } else {
      task_context = feature_shard_count_map_iter->second;
    }

    const auto &named_feature_list = example_batch.named_feature_list(n_i);
    int fid_size = 0;
    if (named_feature_list.type() == FeatureListType::SHARED) {
      const auto &feature = named_feature_list.feature(0);
      int tmp_counter = 0;
      if (feature.has_fid_v1_list()) {
        tmp_counter = feature.fid_v1_list().value_size();
      } else if (feature.has_fid_v2_list()) {
        tmp_counter = feature.fid_v2_list().value_size();
      }
      fid_size += tmp_counter;
      feature_counter.push_back(tmp_counter);
      shared_feature->insert(i);
      *all_feature_counter_size += 1;
    } else {
      feature_counter.reserve(batch_size);
      *all_feature_counter_size += batch_size;
      for (const auto &feature : named_feature_list.feature()) {
        int tmp_counter = 0;
        if (feature.has_fid_v1_list()) {
          tmp_counter = feature.fid_v1_list().value_size();
        } else if (feature.has_fid_v2_list()) {
          tmp_counter = feature.fid_v2_list().value_size();
        }
        fid_size += tmp_counter;
        feature_counter.push_back(tmp_counter);
      }
    }
    if (fid_size > 0) {
      task_context->size += fid_size;
    }
  }
}

template <typename TFeatureParallelTask1Context>
void ShardingSparseFidsOp::FeatureParallelMakeUpTask1Context(
    const std::vector<const ::monolith::io::proto::Example *> &examples,
    int batch_size,
    std::vector<TFeatureParallelTask1Context> *task_context_list,
    absl::flat_hash_map<int, TFeatureParallelTask1Context *>
        *feature_shard_count_map,
    absl::flat_hash_map<int, std::vector<TFeatureParallelTask1Context *>>
        *table_feature_map,
    std::vector<std::vector<int>> *all_feature_counter,
    std::unordered_set<int> *shared_feature, int *all_feature_counter_size) {
  task_context_list->resize(feature_conf_.size());
  for (auto &feature_counter : *all_feature_counter) {
    feature_counter.resize(batch_size, 0);
  }
  *all_feature_counter_size = batch_size * feature_conf_.size();

  for (uint i = 0; i < task_context_list->size(); ++i) {
    auto *task_context = &((*task_context_list)[i]);
    task_context->named_feature_ptr_list.reserve(examples.size());
    task_context->feature_sample_index.reserve(examples.size());
    task_context->feature_output_index = i;
    (*feature_shard_count_map)[feature_cfg_list_[i]->feature_index] =
        task_context;
    (*table_feature_map)[feature_cfg_list_[i]->table_index].push_back(
        task_context);
  }
  for (uint ex_i = 0; ex_i < examples.size(); ++ex_i) {
    const ::monolith::io::proto::Example *example = examples[ex_i];
    CHECK_NOTNULL(example);
    for (const auto &named_feature : example->named_feature()) {
      int fid_size = 0;
      const auto &feature = named_feature.feature();
      fid_size = feature.fid_v2_list().value_size();
      if (fid_size == 0) {
        fid_size = feature.fid_v1_list().value_size();
      }
      if (fid_size <= 0) continue;

      int feature_index = -1;
      auto sorted_id = named_feature.sorted_id();
      if (sorted_id > 0) {  // 优先利用id查找，比string查找更快
        if (feature_index_conf_.size()) {
          if (sorted_id >= feature_index_conf_.size()) {
            continue;
          }
          auto feature_ptr = feature_index_conf_.at(sorted_id);
          if (feature_ptr == nullptr) {
            continue;
          }
          feature_index = feature_ptr->feature_index;
          // CHECK_EQ(named_feature.name(), feature_ptr->feature_name);
        } else {
          LOG_EVERY_N_SEC(ERROR, 10) << "FeatureNameMapper error";
        }
      } else {
        const auto &name = named_feature.name();
        auto find_iter = feature_conf_.find(name);
        if (find_iter == feature_conf_.end()) {
          continue;
        }
        feature_index = find_iter->second.feature_index;
      }

      CHECK(feature_index >= 0 && feature_index < task_context_list->size());
      auto &task_context = (*task_context_list)[feature_index];
      auto &feature_counter = (*all_feature_counter)[feature_index];
      feature_counter[ex_i] = fid_size;
      task_context.size += fid_size;
      task_context.named_feature_ptr_list.push_back(&named_feature);
      task_context.feature_sample_index.push_back(feature_index * batch_size +
                                                  ex_i);
    }
  }
}

template <typename TFeatureParallelTask1Context>
void ShardingSparseFidsOp::FeatureParallelMakeUpTask1Context(
    const ShardingSparseFidsOp::InstanceWrapper &instance_wrapper,
    int batch_size,
    std::vector<TFeatureParallelTask1Context> *task_context_list,
    absl::flat_hash_map<int, TFeatureParallelTask1Context *>
        *feature_shard_count_map,
    absl::flat_hash_map<int, std::vector<TFeatureParallelTask1Context *>>
        *table_feature_map,
    std::vector<std::vector<int>> *all_feature_counter,
    std::unordered_set<int> *shared_feature, int *all_feature_counter_size) {
  task_context_list->resize(feature_conf_.size());
  for (auto &feature_counter : *all_feature_counter) {
    feature_counter.resize(batch_size, 0);
  }
  *all_feature_counter_size = batch_size * feature_conf_.size();

  std::vector<std::vector<std::pair<const InstanceWrapper::FeaturePtr, int>>>
      feature_named_feature_ptr_list(feature_conf_.size());

  for (auto &elem : feature_named_feature_ptr_list) {
    elem.reserve(batch_size);
  }

  // fid v1
  for (auto &iter : instance_wrapper.fid_v1) {
    auto find_iter = feature_conf_.find(iter.first);
    if (find_iter == feature_conf_.end()) {
      continue;
    }
    auto &named_feature_ptr_list =
        feature_named_feature_ptr_list[find_iter->second.feature_index];
    for (uint ex_i = 0; ex_i < iter.second.size(); ++ex_i) {
      named_feature_ptr_list.push_back(std::make_pair(
          InstanceWrapper::FeaturePtr(&iter.second[ex_i], nullptr), ex_i));
    }
  }
  // fid v2
  for (uint ex_i = 0; ex_i < instance_wrapper.instances.size(); ++ex_i) {
    const auto *instance = instance_wrapper.instances[ex_i];
    CHECK_NOTNULL(instance);
    for (const auto &named_feature : instance->feature()) {
      const auto &name = named_feature.name();
      auto find_iter = feature_conf_.find(name);
      if (find_iter == feature_conf_.end()) {
        continue;
      }
      auto &named_feature_ptr_list =
          feature_named_feature_ptr_list[find_iter->second.feature_index];
      named_feature_ptr_list.push_back(std::make_pair(
          InstanceWrapper::FeaturePtr({nullptr, &named_feature}), ex_i));
    }
  }
  for (uint i = 0; i < feature_named_feature_ptr_list.size(); ++i) {
    auto *task_context = &((*task_context_list)[i]);
    task_context->instance_feature_ptr_list.reserve(batch_size);
    task_context->feature_sample_index.reserve(batch_size);
    task_context->feature_output_index = i;
    (*feature_shard_count_map)[feature_cfg_list_[i]->feature_index] =
        task_context;
    (*table_feature_map)[feature_cfg_list_[i]->table_index].push_back(
        task_context);

    auto &feature_counter = (*all_feature_counter)[i];
    for (uint j = 0; j < feature_named_feature_ptr_list[i].size(); ++j) {
      auto &info = feature_named_feature_ptr_list[i][j];
      auto named_feature = info.first;
      auto ex_i = info.second;

      int fid_size = 0;
      if (named_feature.fid_v1) {
        fid_size = named_feature.fid_v1->size();
      } else {
        fid_size += named_feature.fid_v2->fid_size();
        // this is a sequence feature list.
        for (const auto &fidlist : named_feature.fid_v2->fid_list()) {
          fid_size += fidlist.value_size();
        }
      }
      if (fid_size > 0) {
        feature_counter[ex_i] = fid_size;
        task_context->size += fid_size;
        task_context->instance_feature_ptr_list.push_back(named_feature);
        task_context->feature_sample_index.push_back(i * batch_size + ex_i);
      }
    }
  }
}

template <typename TFeatureParallelTask1Context>
void ShardingSparseFidsOp::FeatureParallelDoTask1(
    const ::monolith::io::proto::ExampleBatch &example_batch,
    TFeatureParallelTask1Context *task_context,
    std::vector<int> &nfl_fid_offset,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    tensorflow::TTypes<int32_t>::Flat feature_offset_flat) {
  auto &shard_vec = task_context->fid_list;
  auto &uniq_hashtable = task_context->uniq_hashtable;
  const auto &named_feature_list = example_batch.named_feature_list(
      task_context->example_batch_feature_index);
  auto feature_output_index = task_context->feature_output_index;
  auto offset = nfl_fid_offset[feature_output_index];
  if (named_feature_list.type() == FeatureListType::SHARED) {
    const auto &feature = named_feature_list.feature(0);
    if (feature.has_fid_v1_list()) {
      for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
        auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
        FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                    feature_output_index, &offset);
      }
    } else if (feature.has_fid_v2_list()) {
      for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
        auto value = feature.fid_v2_list().value(i);
        FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                    feature_output_index, &offset);
      }
    }
  } else {
    for (const auto &feature : named_feature_list.feature()) {
      if (feature.has_fid_v1_list()) {
        for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
          auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
          FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                      feature_output_index, &offset);
        }
      } else if (feature.has_fid_v2_list()) {
        for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
          auto value = feature.fid_v2_list().value(i);
          FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                      feature_output_index, &offset);
        }
      }
    }
  }
  task_context->feature_offset = offset - nfl_fid_offset[feature_output_index];
}

template <typename TFeatureParallelTask1Context>
void ShardingSparseFidsOp::FeatureParallelDoTask1(
    const std::vector<const ::monolith::io::proto::Example *> &examples,
    TFeatureParallelTask1Context *task_context,
    std::vector<int> &nfl_fid_offset,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    tensorflow::TTypes<int32_t>::Flat feature_offset_flat) {
  auto &shard_vec = task_context->fid_list;
  auto &uniq_hashtable = task_context->uniq_hashtable;
  auto feature_output_index = task_context->feature_output_index;
  int offset = 0;
  for (uint sub_task_index = 0;
       sub_task_index < task_context->named_feature_ptr_list.size();
       ++sub_task_index) {
    const auto named_feature_ptr =
        task_context->named_feature_ptr_list[sub_task_index];
    offset =
        feature_offset_flat(task_context->feature_sample_index[sub_task_index]);
    const auto &feature = named_feature_ptr->feature();
    if (feature.has_fid_v1_list()) {
      for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
        auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
        FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                    feature_output_index, &offset);
      }
    } else if (feature.has_fid_v2_list()) {
      for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
        auto value = feature.fid_v2_list().value(i);
        FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                    feature_output_index, &offset);
      }
    }
  }
  task_context->feature_offset = offset - nfl_fid_offset[feature_output_index];
}

template <typename TFeatureParallelTask1Context>
void ShardingSparseFidsOp::FeatureParallelDoTask1(
    const ShardingSparseFidsOp::InstanceWrapper &instance_wrapper,
    TFeatureParallelTask1Context *task_context,
    std::vector<int> &nfl_fid_offset,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    tensorflow::TTypes<int32_t>::Flat feature_offset_flat) {
  auto &shard_vec = task_context->fid_list;
  auto &uniq_hashtable = task_context->uniq_hashtable;
  auto feature_output_index = task_context->feature_output_index;
  int offset = 0;
  for (uint sub_task_index = 0;
       sub_task_index < task_context->instance_feature_ptr_list.size();
       ++sub_task_index) {
    const auto &named_feature_ptr =
        task_context->instance_feature_ptr_list[sub_task_index];
    offset =
        feature_offset_flat(task_context->feature_sample_index[sub_task_index]);
    if (named_feature_ptr.fid_v1) {
      for (auto value : *named_feature_ptr.fid_v1) {
        value = convert_fid_v1_to_v2(value);
        FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                    feature_output_index, &offset);
      }
    } else {
      for (const auto &value : named_feature_ptr.fid_v2->fid()) {
        FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                    feature_output_index, &offset);
      }
      // this is a sequence feature list.
      for (const auto &fid_list : named_feature_ptr.fid_v2->fid_list()) {
        for (const auto &value : fid_list.value()) {
          FillFidList(value, shard_vec, uniq_hashtable, fid_offset_flat,
                      feature_output_index, &offset);
        }
      }
    }
  }
  task_context->feature_offset = offset - nfl_fid_offset[feature_output_index];
}

template <typename TInput>
Status ShardingSparseFidsOp::FeatureParallelParse(
    OpKernelContext *ctx, const TInput &input, OpOutputList *fid_list_out_list,
    OpOutputList *fid_list_row_splits_out_list,
    OpOutputList *fid_list_row_splits_size_out_list) {
  int batch_size = GetBatchSize(input);
  struct FeatureParallelTask1Context {
    std::vector<std::vector<uint64_t>> fid_list;
    MultiShardUniqHashTable uniq_hashtable;
    int example_batch_feature_index = -1;  // use for example_batch
    std::vector<const ::monolith::io::proto::NamedFeature *>
        named_feature_ptr_list;  // use for example
    std::vector<InstanceWrapper::FeaturePtr>
        instance_feature_ptr_list;  // for instance
    int feature_output_index = -1;
    std::vector<int> feature_sample_index;  // use for example
    int size = 0;
    std::vector<int> table_offset;  // (ps_num_, 0);
    int feature_offset = 0;
  };
  std::vector<FeatureParallelTask1Context> task_context_list;
  absl::flat_hash_map<int, FeatureParallelTask1Context *>
      feature_shard_count_map;
  absl::flat_hash_map<int, std::vector<FeatureParallelTask1Context *>>
      table_feature_map;

  Tensor *fid_offset_tensor, *feature_offset_tensor;
  std::vector<Tensor> tmp_tensor_list;
  std::vector<TensorSliceAccessor<int64_t>> fid_list_row_splits_flat_list;
  std::vector<int> nfl_fid_offset(feature_conf_.size());

  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp::Alloc"; });
    feature_shard_count_map.reserve(feature_conf_.size());
    task_context_list.reserve(feature_conf_.size());
    table_feature_map.reserve(table_cfg_list_.size());

    std::vector<std::vector<int>> all_feature_counter(feature_conf_.size());
    int all_feature_counter_size = 0;
    std::unordered_set<int> shared_feature;
    FeatureParallelMakeUpTask1Context(
        input, batch_size, &task_context_list, &feature_shard_count_map,
        &table_feature_map, &all_feature_counter, &shared_feature,
        &all_feature_counter_size);
    for (auto &task_context : task_context_list) {
      task_context.fid_list.resize(ps_num_);
      task_context.uniq_hashtable.resize(ps_num_);
      task_context.table_offset.resize(ps_num_, 0);
      if (task_context.size > 0) {
        int reserve_size = task_context.size * 6 / 5 / ps_num_;
        if (unique_) {
          task_context.uniq_hashtable.reserve(reserve_size);
        } else {
          for (auto &fid_list_part : task_context.fid_list) {
            fid_list_part.reserve(reserve_size);
          }
        }
      }
    }
    Tensor *nfl_offset_tensor;
    TF_RETURN_IF_ERROR(
        ctx,
        CreateOffsetTensor(ctx, all_feature_counter, all_feature_counter_size,
                           &nfl_offset_tensor, &feature_offset_tensor,
                           &fid_offset_tensor, fid_list_row_splits_out_list,
                           fid_list_row_splits_size_out_list, tmp_tensor_list,
                           fid_list_row_splits_flat_list, &nfl_fid_offset,
                           &shared_feature));
  }
  auto fid_offset_flat = fid_offset_tensor->flat<uint64_t>();
  auto feature_offset_flat = feature_offset_tensor->flat<int32_t>();

  std::vector<std::vector<int>> task_split;
  SplitTask<FeatureParallelTask1Context>(
      task_context_list, single_thread_feature_watermark_, &task_split);
  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp::AddVec"; });
    activity.AppendMetadata([&task_split, &task_context_list] {
      return profiler::TraceMeEncode({{"task_num", task_context_list.size()},
                                      {"split_num", task_split.size()}});
    });
    std::vector<size_t> capacities(task_split.size(), 0);
    auto task_func = [this, &task_context_list, &input, &task_split,
                      &nfl_fid_offset, &fid_offset_flat, &feature_offset_flat,
                      &capacities](const int64 begin, const int64 end) {
      UniqHashTable uniq_hashtable;
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task_context_list[index];
          if (unique_) {
            uniq_hashtable.Reset();
            task_context.uniq_hashtable.init(&uniq_hashtable);
          }
          FeatureParallelDoTask1(input, &task_context, nfl_fid_offset,
                                 fid_offset_flat, feature_offset_flat);
        }
        capacities[task_index] = uniq_hashtable.Capacity();
      }
    };

    ParallelRun(ctx, task_split.size(), task_func);

    double avg_capacity = 0;
    if (capacities.size() > 0) {
      avg_capacity = static_cast<double>(std::accumulate(capacities.begin(),
                                                         capacities.end(), 0)) /
                     capacities.size();
    }
    activity.AppendMetadata([&avg_capacity] {
      return profiler::TraceMeEncode({{"hashtable_size", avg_capacity}});
    });
    monolith::GetMetrics()->emit_timer("sharding_sparse_fids_op_hashtable_capacity",
                                        avg_capacity);
  }

  struct TaskContext2 {
    // fill fid_list
    TaskContext2(const TensorSliceAccessor<int64_t> &accessor_,
                 std::vector<uint64_t> *shard_fid_list_, std::vector<uint64_t> *shard_ptr_,
                 int size_, int offset_)
        : accessor(accessor_),
          shard_fid_list(shard_fid_list_),
          shard_ptr(shard_ptr_),
          size(size_),
          offset(offset_) {}

    // rewrite fid_offset
    explicit TaskContext2(FeatureParallelTask1Context *task_context_)
        : task1_context(task_context_), size(task_context_->feature_offset) {}
    TensorSliceAccessor<int64_t> accessor;
    std::vector<uint64_t> *shard_fid_list = nullptr;
    std::vector<uint64_t> *shard_ptr = nullptr;
    int size = -1;
    int offset = -1;
    FeatureParallelTask1Context *task1_context = nullptr;
  };
  std::vector<TaskContext2> task2_context_list;

  Tensor *fid_list_table_row_length_tensor = nullptr;
  Tensor *fid_list_shard_row_lenth_tensor = nullptr;
  Tensor *fid_list_emb_row_lenth_tensor = nullptr;
  std::vector<TensorSliceAccessor<int64_t>> fid_list_tensor_vec(
      table_cfg_list_.size() * ps_num_);
  if (version_ == 4) {
    int size_record_total = 0;
    std::vector<int> size_record(table_cfg_list_.size() * ps_num_, 0);
    for (uint table_index = 0; table_index < table_cfg_list_.size();
         ++table_index) {
      // auto &table_name = table_names_[table_index];
      auto table_feature_map_find_iter = table_feature_map.find(table_index);
      std::vector<FeatureParallelTask1Context *> *feature_vec_ptr = nullptr;
      if (table_feature_map_find_iter != table_feature_map.end()) {
        feature_vec_ptr = &(table_feature_map_find_iter->second);
      }
      for (int ps_num_i = 0; ps_num_i < ps_num_; ++ps_num_i) {
        int &size = size_record[table_index * ps_num_ + ps_num_i];
        if (feature_vec_ptr != nullptr) {
          for (auto task_context_ptr : *feature_vec_ptr) {
            if (unique_) {
              size += task_context_ptr->uniq_hashtable.fid_num(ps_num_i);
            } else {
              size += task_context_ptr->fid_list[ps_num_i].size();
            }
          }
        }
        size_record_total += size;
      }
    }
    Tensor *fid_list_tensor;
    TF_RETURN_IF_ERROR(ctx,
                       ctx->allocate_output("fid_list", TensorShape({
                                                            size_record_total,
                                                        }),
                                            &fid_list_tensor));
    fid_list_tensor->flat<int64_t>().setZero();

    int pre_count = 0;
    for (uint ps_num_i = 0; ps_num_i < ps_num_; ++ps_num_i) {
      for (uint table_index = 0; table_index < table_cfg_list_.size();
           ++table_index) {
        int size = size_record[table_index * ps_num_ + ps_num_i];
        fid_list_tensor_vec[table_index * ps_num_ + ps_num_i] =
            TensorSliceAccessor<int64_t>(
                {static_cast<int64_t *>(fid_list_tensor->data()) + pre_count,
                 size});
        // fid_list_tensor->Slice(pre_count, pre_count + size);
        pre_count += size;
      }
    }

    TF_RETURN_IF_ERROR(
        ctx,
        ctx->allocate_output("fid_list_table_row_length",
                             tensorflow::TensorShape({
                                 ps_num_ * table_cfg_list_.size(),
                             }),
                             &fid_list_table_row_length_tensor));
    fid_list_table_row_length_tensor->flat<int32_t>().setZero();

    TF_RETURN_IF_ERROR(ctx,
                       ctx->allocate_output("fid_list_shard_row_lenth",
                                            tensorflow::TensorShape({
                                                ps_num_,
                                            }),
                                            &fid_list_shard_row_lenth_tensor));
    fid_list_shard_row_lenth_tensor->flat<int32_t>().setZero();

    TF_RETURN_IF_ERROR(
        ctx,
        ctx->allocate_output("fid_list_emb_row_lenth",
                             tensorflow::TensorShape({
                                 ps_num_ * table_cfg_list_.size(),
                             }),
                             &fid_list_emb_row_lenth_tensor));
    fid_list_emb_row_lenth_tensor->flat<int32_t>().setZero();
  }
  int index = -1;
  Tensor *emb_size_tensor;
  if (version_ == 5) {
    TF_RETURN_IF_ERROR(ctx, ctx->allocate_output("emb_size",
                                                 TensorShape({
                                                   table_cfg_list_.size() * ps_num_,
                                                 }),
                                                 &emb_size_tensor));
  }
  for (uint table_index = 0; table_index < table_cfg_list_.size();
       ++table_index) {
    // auto &table_name = table_names_[table_index];
    auto table_feature_map_find_iter = table_feature_map.find(table_index);
    std::vector<FeatureParallelTask1Context *> *feature_vec_ptr = nullptr;
    if (table_feature_map_find_iter != table_feature_map.end()) {
      feature_vec_ptr = &(table_feature_map_find_iter->second);
    }
    for (int ps_num_i = 0; ps_num_i < ps_num_; ++ps_num_i) {
      auto &cur_tensor_flat =
          fid_list_row_splits_flat_list[table_index * ps_num_ + ps_num_i];
      int size = 0;
      int pre_offset = 0;
      if (feature_vec_ptr != nullptr) {
        for (auto task_context_ptr : *feature_vec_ptr) {
          task_context_ptr->table_offset[ps_num_i] = pre_offset;
          int cur_fid_size = 0;
          if (unique_) {
            cur_fid_size = task_context_ptr->uniq_hashtable.fid_num(ps_num_i);
          } else {
            cur_fid_size = task_context_ptr->fid_list[ps_num_i].size();
          }
          size += cur_fid_size;
          // std::cerr << "cur_fid_size: " << cur_fid_size << " size: " << size
          // << std::endl << std::flush;
          if (version_ == 3 || version_ == 4 || version_ == 5) {
            int emb_size =
                cur_fid_size *
                feature_cfg_list_[task_context_ptr->feature_output_index]
                    ->dims_sum;
            if (version_ == 4) {
              auto fid_list_emb_row_lenth_flat =
                  fid_list_emb_row_lenth_tensor->flat<int32_t>();
              fid_list_emb_row_lenth_flat(ps_num_i * table_cfg_list_.size() +
                                          table_index) += emb_size;
            }
            pre_offset += emb_size;
          } else {
            pre_offset = size;
          }
          auto cur_tensor_flat_index =
              feature_cfg_list_[task_context_ptr->feature_output_index]
                  ->feature_in_table_index +
              1;
          cur_tensor_flat(cur_tensor_flat_index) = cur_fid_size;
        }
      }
      if (version_ == 5) {
        emb_size_tensor->flat<int32>()(table_index * ps_num_ + ps_num_i) = pre_offset;
      }
      TensorSliceAccessor<int64_t> cur_accessor;
      for (uint z = 2; z <= table_cfg_list_[table_index]->feature_count; ++z) {
        cur_tensor_flat(z) += cur_tensor_flat(z - 1);
      }
      if (version_ != 4) {
        Tensor *cur_tensor;
        TF_RETURN_IF_ERROR(
            ctx,
            fid_list_out_list->allocate(++index, tensorflow::TensorShape{size},
                                        &cur_tensor));
        if (size == 0) {
          std::memset(cur_tensor->data(), 0, cur_tensor->TotalBytes());
          continue;
        }
        cur_accessor = TensorSliceAccessor<int64_t>(
            {static_cast<int64_t *>(cur_tensor->data()),
             cur_tensor->NumElements()});
      } else {
        cur_accessor = fid_list_tensor_vec[++index];

        fid_list_table_row_length_tensor->flat<int32_t>()(
            ps_num_i * table_cfg_list_.size() + table_index) += size;
        fid_list_shard_row_lenth_tensor->flat<int32_t>()(ps_num_i) += size;
      }
      int offset = 0;
      for (auto task_context_ptr : *feature_vec_ptr) {
        std::vector<uint64_t> *shard_fid_list = nullptr;
        std::vector<uint64_t> *shard_ptr = nullptr;
        int tmp_size = 0;
        if (unique_) {
          shard_fid_list =
              &(task_context_ptr->uniq_hashtable.fid_list(ps_num_i));
          tmp_size = shard_fid_list->size();
        } else {
          shard_ptr = &(task_context_ptr->fid_list[ps_num_i]);
          tmp_size = shard_ptr->size();
        }
        DCHECK_LE(offset + tmp_size, size);
        TaskContext2 tmp_task_context(cur_accessor, shard_fid_list, shard_ptr,
                                      tmp_size, offset);
        task2_context_list.emplace_back(tmp_task_context);
        offset += tmp_size;
      }
    }
    if (version_ != 2) {
      if (feature_vec_ptr) {
        for (auto task_context_ptr : *feature_vec_ptr) {
          TaskContext2 tmp_task_context(task_context_ptr);
          task2_context_list.emplace_back(tmp_task_context);
        }
      }
    }
  }

  std::vector<std::vector<int>> task2_split;
  SplitTask<TaskContext2>(task2_context_list, single_thread_assign_watermark_,
                          &task2_split);

  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp::Copy"; });
    auto tensor_assign_func = [this, ctx, &task2_context_list, &task2_split,
                               &nfl_fid_offset, &fid_offset_flat](
        const int64 begin, const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task2_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task2_context_list[index];
          if (task_context.task1_context) {
            auto &task1_context = *task_context.task1_context;
            auto offset = nfl_fid_offset[task1_context.feature_output_index];
            /*
            auto table_index =
                feature_cfg_list_[task1_context.feature_output_index]
                    ->table_index *
                ps_num_;
            for (int i = 0; i < task1_context.feature_offset; ++i, ++offset)
            { auto cur = fid_offset_flat(offset); cur +=
            task1_context.table_offset.at(static_cast<int>(cur >> 32) -
                                                   table_index);
              fid_offset_flat(offset) = cur;
            }*/
            auto feature_cfg =
                feature_cfg_list_[task1_context.feature_output_index];
            for (int i = 0; i < task1_context.feature_offset; ++i, ++offset) {
              auto cur = fid_offset_flat(offset);
              cur += task1_context.table_offset.at(
                  feature_cfg->GetPsShard(static_cast<int>(cur >> 32)));
              fid_offset_flat(offset) = cur;
            }
          } else if (unique_) {
            CopyFidList(*task_context.shard_fid_list, task_context.offset,
                        &task_context.accessor);
          } else {
            CopyFidList(*task_context.shard_ptr, task_context.offset,
                        &task_context.accessor);
          }
        }
      }
    };
    ParallelRun(ctx, task2_split.size(), tensor_assign_func);
  }
  return Status::OK();
}

}  // namespace monolith_tf
}  // namespace tensorflow
#endif MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_SPARSE_FEATURE_LIB_H_
