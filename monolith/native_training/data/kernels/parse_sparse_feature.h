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

#include <unordered_map>
#include <unordered_set>

#include "google/protobuf/descriptor.h"
#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"

#include "monolith/native_training/data/kernels/feature_name_mapper_tf_bridge.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
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
  Status TableParallelParse(OpKernelContext *ctx, const TInput &input,
                            OpOutputList *fid_list_out_list,
                            OpOutputList *fid_list_row_splits_out_list);
  //!!! TableParallelUniqueParse fid_list_row_splits_out_list may have bug when
  // two fid in different feature are same, because unique fid in table with all
  // features' fid
  template <typename TInput>
  Status TableParallelUniqueParse(OpKernelContext *ctx, const TInput &input,
                                  OpOutputList *fid_list_out_list,
                                  OpOutputList *fid_list_row_splits_out_list);

  template <typename TInput>
  Status FeatureParallelParse(OpKernelContext *ctx, const TInput &input,
                              OpOutputList *fid_list_out_list,
                              OpOutputList *fid_list_row_splits_out_list);
  template <typename TInput>
  Status FeatureParallelUniqueParse(OpKernelContext *ctx, const TInput &input,
                                    OpOutputList *fid_list_out_list,
                                    OpOutputList *fid_list_row_splits_out_list);

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

  void FillFidList(uint64_t value,
                   std::vector<std::vector<uint64_t>> &shard_vec,
                   tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
                   int feature_output_index, int *offset,
                   std::vector<int> *pre_feature_count = nullptr,
                   void *task_context = nullptr);
  void FillFidList(uint64_t value,
                   std::vector<absl::flat_hash_map<uint64_t, int>> &shard_vec,
                   tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
                   int feature_output_index, int *offset,
                   std::vector<int> *pre_feature_count = nullptr,
                   void *task_context = nullptr);

  Status CreateOffsetTensor(
      OpKernelContext *ctx,
      const std::vector<std::vector<int>> &all_feature_counter,
      int all_feature_counter_size, Tensor **nfl_offset_tensor,
      Tensor **feature_offset_tensor, Tensor **fid_offset_tensor,
      OpOutputList *fid_list_row_splits_out_list,
      std::vector<Tensor> &fid_list_row_tensor_list,
      std::vector<tensorflow::TTypes<int64_t>::Flat>
          &fid_list_row_splits_flat_list,
      std::vector<int> *nfl_fid_offset = nullptr,
      const std::unordered_set<int> *shared_feature = nullptr);

  struct FeatureInfo {
    std::string feature_name;
    std::string table_name;
    int feature_index = -1;
    int table_index = -1;
    int feature_in_table_index = -1;

    int table_feature_count;
    int output_pre_index;

    int version = 1;
    int GetFidOutputIndex(int ps_i) {
      if (version == 2) {
        return output_pre_index + ps_i * table_feature_count +
               feature_in_table_index;
      } else {
        return output_pre_index + ps_i;
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
  int single_thread_feature_watermark_ = 80000;
  int single_thread_assign_watermark_ = 100000;
  int int64_size_ = sizeof(int64_t);

  uint32_t shard_flag_ = (1L << 31);

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
#define DTableParallelMakeUpTask1Context(INPUT_TYPE)       \
  template <typename TTableParallelTask1Context>           \
  void TableParallelMakeUpTask1Context(                    \
      const INPUT_TYPE &Input, int batch_size,             \
      absl::flat_hash_map<int, TTableParallelTask1Context> \
          *table_shard_count_map,                          \
      std::vector<std::vector<int>> *all_feature_counter,  \
      std::unordered_set<int> *shared_feature, int *all_feature_counter_size)
  DTableParallelMakeUpTask1Context(::monolith::io::proto::ExampleBatch);
  DTableParallelMakeUpTask1Context(
      std::vector<const ::monolith::io::proto::Example *>);
  DTableParallelMakeUpTask1Context(InstanceWrapper);
#undef DTableParallelMakeUpTask1Context

#define DTableParallelDoTask1(INPUT_TYPE)                                \
  template <typename TTableParallelTask1Context>                         \
  void TableParallelDoTask1(                                             \
      const INPUT_TYPE &input, TTableParallelTask1Context *task_context, \
      std::vector<int> &nfl_fid_offset,                                  \
      tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,                \
      tensorflow::TTypes<int32_t>::Flat feature_offset_flat,             \
      std::vector<tensorflow::TTypes<int64_t>::Flat>                     \
          &fid_list_row_splits_flat_list)
  DTableParallelDoTask1(::monolith::io::proto::ExampleBatch);
  DTableParallelDoTask1(std::vector<const ::monolith::io::proto::Example *>);
  DTableParallelDoTask1(InstanceWrapper);
#undef DTableParallelDoTask1

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

template <typename TTableParallelTask1Context>
void ShardingSparseFidsOp::TableParallelMakeUpTask1Context(
    const ::monolith::io::proto::ExampleBatch &example_batch, int batch_size,
    absl::flat_hash_map<int, TTableParallelTask1Context> *table_shard_count_map,
    std::vector<std::vector<int>> *all_feature_counter,
    std::unordered_set<int> *shared_feature, int *all_feature_counter_size) {
  std::vector<int> example_batch_feature_index(feature_conf_.size(), -1);
  for (uint n_i = 0; n_i < example_batch.named_feature_list_size(); ++n_i) {
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
    const auto &named_feature_list = example_batch.named_feature_list(n_i);
    auto &task_context =
        (*table_shard_count_map)[feature_cfg_list_[i]->table_index];
    int feature_size = 0;
    if (named_feature_list.type() == FeatureListType::SHARED) {
      const auto &feature = named_feature_list.feature(0);
      int tmp_counter = 0;
      if (feature.has_fid_v1_list()) {
        tmp_counter = feature.fid_v1_list().value_size();
      } else if (feature.has_fid_v2_list()) {
        tmp_counter = feature.fid_v2_list().value_size();
      }
      feature_size += tmp_counter;
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
        feature_size += tmp_counter;
        feature_counter.push_back(tmp_counter);
      }
    }
    if (feature_size > 0) {
      task_context.size += feature_size;
      task_context.example_batch_feature_index.push_back(n_i);
      task_context.feature_output_index.push_back(i);
    }
  }
}
template <typename TTableParallelTask1Context>
void ShardingSparseFidsOp::TableParallelMakeUpTask1Context(
    const std::vector<const ::monolith::io::proto::Example *> &examples,
    int batch_size,
    absl::flat_hash_map<int, TTableParallelTask1Context> *table_shard_count_map,
    std::vector<std::vector<int>> *all_feature_counter,
    std::unordered_set<int> *shared_feature, int *all_feature_counter_size) {
  for (auto &feature_counter : *all_feature_counter) {
    feature_counter.resize(batch_size, 0);
  }
  *all_feature_counter_size = batch_size * feature_conf_.size();
  std::vector<
      std::vector<std::pair<const ::monolith::io::proto::NamedFeature *, int>>>
      feature_named_feature_ptr_list(feature_conf_.size());

  for (uint ex_i = 0; ex_i < examples.size(); ++ex_i) {
    const ::monolith::io::proto::Example *example = examples[ex_i];
    CHECK_NOTNULL(example);
    for (const auto &named_feature : example->named_feature()) {
      const auto &name = named_feature.name();
      auto find_iter = feature_conf_.find(name);
      if (find_iter == feature_conf_.end()) {
        continue;
      }
      auto &named_feature_ptr_list =
          feature_named_feature_ptr_list[find_iter->second.feature_index];
      named_feature_ptr_list.push_back(std::make_pair(&named_feature, ex_i));
    }
  }
  for (uint i = 0; i < feature_named_feature_ptr_list.size(); ++i) {
    auto &task_context =
        (*table_shard_count_map)[feature_cfg_list_[i]->table_index];
    auto &feature_counter = (*all_feature_counter)[i];
    for (uint j = 0; j < feature_named_feature_ptr_list[i].size(); ++j) {
      auto &info = feature_named_feature_ptr_list[i][j];
      auto named_feature = info.first;
      auto ex_i = info.second;
      int feature_size = 0;
      const auto &feature = named_feature->feature();
      if (feature.has_fid_v1_list()) {
        feature_size = feature.fid_v1_list().value_size();
      } else if (feature.has_fid_v2_list()) {
        feature_size = feature.fid_v2_list().value_size();
      }
      feature_counter[ex_i] = feature_size;

      if (feature_size > 0) {
        task_context.size += feature_size;
        task_context.named_feature_ptr_list.push_back(named_feature);
        task_context.feature_output_index.push_back(i);
        task_context.feature_sample_index.push_back(i * batch_size + ex_i);
      }
    }
  }
}
template <typename TTableParallelTask1Context>
void ShardingSparseFidsOp::TableParallelMakeUpTask1Context(
    const ShardingSparseFidsOp::InstanceWrapper &instance_wrapper,
    int batch_size,
    absl::flat_hash_map<int, TTableParallelTask1Context> *table_shard_count_map,
    std::vector<std::vector<int>> *all_feature_counter,
    std::unordered_set<int> *shared_feature, int *all_feature_counter_size) {
  for (auto &feature_counter : *all_feature_counter) {
    feature_counter.resize(batch_size, 0);
  }
  *all_feature_counter_size = batch_size * feature_conf_.size();
  std::vector<std::vector<std::pair<const InstanceWrapper::FeaturePtr, int>>>
      feature_named_feature_ptr_list(feature_conf_.size());
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
          InstanceWrapper::FeaturePtr(nullptr, &named_feature), ex_i));
    }
  }
  for (uint i = 0; i < feature_named_feature_ptr_list.size(); ++i) {
    auto &task_context =
        (*table_shard_count_map)[feature_cfg_list_[i]->table_index];
    auto &feature_counter = (*all_feature_counter)[i];
    for (uint j = 0; j < feature_named_feature_ptr_list[i].size(); ++j) {
      auto &info = feature_named_feature_ptr_list[i][j];
      auto named_feature = info.first;
      auto ex_i = info.second;
      int feature_size = 0;
      if (named_feature.fid_v1) {
        feature_size = named_feature.fid_v1->size();
      } else {
        feature_size += named_feature.fid_v2->fid_size();
        // this is a sequence feature list.
        for (const auto &fidlist : named_feature.fid_v2->fid_list()) {
          feature_size += fidlist.value_size();
        }
      }
      feature_counter[ex_i] = feature_size;

      if (feature_size > 0) {
        task_context.size += feature_size;
        task_context.instance_feature_ptr_list.push_back(named_feature);
        task_context.feature_output_index.push_back(i);
        task_context.feature_sample_index.push_back(i * batch_size + ex_i);
      }
    }
  }
}

template <typename TTableParallelTask1Context>
void ShardingSparseFidsOp::TableParallelDoTask1(
    const ::monolith::io::proto::ExampleBatch &example_batch,
    TTableParallelTask1Context *task_context, std::vector<int> &nfl_fid_offset,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    tensorflow::TTypes<int32_t>::Flat feature_offset_flat,
    std::vector<tensorflow::TTypes<int64_t>::Flat>
        &fid_list_row_splits_flat_list) {
  auto &shard_vec = task_context->fid_list;
  for (uint sub_task_index = 0;
       sub_task_index < task_context->example_batch_feature_index.size();
       ++sub_task_index) {
    const auto &named_feature_list = example_batch.named_feature_list(
        task_context->example_batch_feature_index[sub_task_index]);
    auto feature_output_index =
        task_context->feature_output_index[sub_task_index];
    auto offset = nfl_fid_offset[feature_output_index];
    auto table_index = feature_cfg_list_[feature_output_index]->table_index;
    table_index *= ps_num_;
    auto feature_in_table_index =
        feature_cfg_list_[feature_output_index]->feature_in_table_index;
    std::vector<int> shard_record(ps_num_, 0);
    for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
      shard_record[ps_i] = shard_vec[ps_i].size();
    }
    if (named_feature_list.type() == FeatureListType::SHARED) {
      const auto &feature = named_feature_list.feature(0);
      if (feature.has_fid_v1_list()) {
        for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
          auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
          FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                      &offset, &shard_record);
        }
      } else if (feature.has_fid_v2_list()) {
        for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
          auto value = feature.fid_v2_list().value(i);
          FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                      &offset, &shard_record);
        }
      }
    } else {
      for (int feature_index = 0;
           feature_index < named_feature_list.feature_size(); ++feature_index) {
        const auto &feature = named_feature_list.feature(feature_index);
        if (feature.has_fid_v1_list()) {
          for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
            auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
            FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                        &offset, &shard_record);
          }
        } else if (feature.has_fid_v2_list()) {
          for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
            auto value = feature.fid_v2_list().value(i);
            FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                        &offset, &shard_record);
          }
        }
      }
    }

    for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
      fid_list_row_splits_flat_list[table_index + ps_i](feature_in_table_index +
                                                        1) =
          shard_vec[ps_i].size() - shard_record[ps_i];
    }
  }
}

template <typename TTableParallelTask1Context>
void ShardingSparseFidsOp::TableParallelDoTask1(
    const std::vector<const ::monolith::io::proto::Example *> &examples,
    TTableParallelTask1Context *task_context, std::vector<int> &nfl_fid_offset,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    tensorflow::TTypes<int32_t>::Flat feature_offset_flat,
    std::vector<tensorflow::TTypes<int64_t>::Flat>
        &fid_list_row_splits_flat_list) {
  auto &shard_vec = task_context->fid_list;
  std::vector<int> shard_record(ps_num_, 0);
  int pre_feature_in_table_index = -1;
  int table_index = -1;
  for (uint sub_task_index = 0;
       sub_task_index < task_context->named_feature_ptr_list.size();
       ++sub_task_index) {
    const auto named_feature_ptr =
        task_context->named_feature_ptr_list[sub_task_index];
    auto feature_output_index =
        task_context->feature_output_index[sub_task_index];
    table_index = feature_cfg_list_[feature_output_index]->table_index;
    table_index *= ps_num_;
    auto feature_in_table_index =
        feature_cfg_list_[feature_output_index]->feature_in_table_index;
    if (pre_feature_in_table_index != feature_in_table_index) {
      if (pre_feature_in_table_index != -1) {
        for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
          fid_list_row_splits_flat_list[table_index + ps_i](
              feature_in_table_index) =
              shard_vec[ps_i].size() - shard_record[ps_i];
          shard_record[ps_i] = shard_vec[ps_i].size();
        }
      }
      pre_feature_in_table_index = feature_in_table_index;
    }
    auto offset =
        feature_offset_flat(task_context->feature_sample_index[sub_task_index]);
    const auto &feature = named_feature_ptr->feature();
    if (feature.has_fid_v1_list()) {
      for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
        auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset, &shard_record, task_context);
      }
    } else if (feature.has_fid_v2_list()) {
      for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
        auto value = feature.fid_v2_list().value(i);
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset, &shard_record, task_context);
      }
    }
  }
  if (pre_feature_in_table_index != -1) {
    for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
      fid_list_row_splits_flat_list[table_index + ps_i](
          pre_feature_in_table_index + 1) =
          shard_vec[ps_i].size() - shard_record[ps_i];
    }
  }
}

template <typename TTableParallelTask1Context>
void ShardingSparseFidsOp::TableParallelDoTask1(
    const ShardingSparseFidsOp::InstanceWrapper &instance_wrapper,
    TTableParallelTask1Context *task_context, std::vector<int> &nfl_fid_offset,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    tensorflow::TTypes<int32_t>::Flat feature_offset_flat,
    std::vector<tensorflow::TTypes<int64_t>::Flat>
        &fid_list_row_splits_flat_list) {
  auto &shard_vec = task_context->fid_list;
  std::vector<int> shard_record(ps_num_, 0);
  int pre_feature_in_table_index = -1;
  int table_index = -1;
  for (uint sub_task_index = 0;
       sub_task_index < task_context->instance_feature_ptr_list.size();
       ++sub_task_index) {
    const auto &named_feature_ptr =
        task_context->instance_feature_ptr_list[sub_task_index];
    auto feature_output_index =
        task_context->feature_output_index[sub_task_index];
    auto table_index = feature_cfg_list_[feature_output_index]->table_index;
    table_index *= ps_num_;
    auto feature_in_table_index =
        feature_cfg_list_[feature_output_index]->feature_in_table_index;
    if (pre_feature_in_table_index != feature_in_table_index) {
      if (pre_feature_in_table_index != -1) {
        for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
          fid_list_row_splits_flat_list[table_index + ps_i](
              feature_in_table_index) =
              shard_vec[ps_i].size() - shard_record[ps_i];
          shard_record[ps_i] = shard_vec[ps_i].size();
        }
      }
      pre_feature_in_table_index = feature_in_table_index;
    }
    auto offset =
        feature_offset_flat(task_context->feature_sample_index[sub_task_index]);
    if (named_feature_ptr.fid_v1) {
      for (auto value : *named_feature_ptr.fid_v1) {
        value = convert_fid_v1_to_v2(value);
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset, &shard_record);
      }
    } else {
      for (const auto &value : named_feature_ptr.fid_v2->fid()) {
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset, &shard_record);
      }
      // this is a sequence feature list.
      for (const auto &fidlist : named_feature_ptr.fid_v2->fid_list()) {
        for (const auto &value : fidlist.value()) {
          FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                      &offset, &shard_record);
        }
      }
    }

    if (pre_feature_in_table_index != -1) {
      for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
        fid_list_row_splits_flat_list[table_index + ps_i](
            pre_feature_in_table_index + 1) =
            shard_vec[ps_i].size() - shard_record[ps_i];
      }
    }
  }
}

template <typename TInput>
Status ShardingSparseFidsOp::TableParallelParse(
    OpKernelContext *ctx, const TInput &input, OpOutputList *fid_list_out_list,
    OpOutputList *fid_list_row_splits_out_list) {
  int batch_size = GetBatchSize(input);
  struct TableParallelTask1Context {
    std::vector<std::vector<uint64_t>> fid_list;
    std::vector<int> example_batch_feature_index;  // for example_batch
    std::vector<const ::monolith::io::proto::NamedFeature *>
        named_feature_ptr_list;  // for examples
    std::vector<InstanceWrapper::FeaturePtr>
        instance_feature_ptr_list;  // for instance
    std::vector<int> feature_output_index;
    std::vector<int> feature_sample_index;  // for examples + instances
    int size = 0;
  };
  absl::flat_hash_map<int, TableParallelTask1Context> table_shard_count_map;
  std::vector<TableParallelTask1Context *> task_context_list;

  Tensor *fid_offset_tensor, *feature_offset_tensor;
  std::vector<int> nfl_fid_offset(feature_conf_.size());
  std::vector<Tensor> fid_list_row_splits_tenor_list;
  std::vector<tensorflow::TTypes<int64_t>::Flat> fid_list_row_splits_flat_list;

  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp1::Alloc"; });
    table_shard_count_map.reserve(table_cfg_list_.size());
    for (uint table_i = 0; table_i < table_cfg_list_.size(); ++table_i) {
      auto &task_context = table_shard_count_map[table_i];
      task_context.fid_list.resize(ps_num_);
    }

    std::vector<std::vector<int>> all_feature_counter(feature_conf_.size());
    int all_feature_counter_size = 0;
    std::unordered_set<int> shared_feature;

    TableParallelMakeUpTask1Context(input, batch_size, &table_shard_count_map,
                                    &all_feature_counter, &shared_feature,
                                    &all_feature_counter_size);

    for (auto &table_shard_count_iter : table_shard_count_map) {
      auto &task_context = table_shard_count_iter.second;
      if (task_context.size > 0) {
        int reserve_size = task_context.size * 3 / 2 / ps_num_;
        for (auto &fid_list_part : task_context.fid_list) {
          fid_list_part.reserve(reserve_size);
        }
        task_context_list.push_back(&task_context);
      }
    }
    Tensor *nfl_offset_tensor;
    TF_RETURN_IF_ERROR(
        ctx,
        CreateOffsetTensor(
            ctx, all_feature_counter, all_feature_counter_size,
            &nfl_offset_tensor, &feature_offset_tensor, &fid_offset_tensor,
            fid_list_row_splits_out_list, fid_list_row_splits_tenor_list,
            fid_list_row_splits_flat_list, &nfl_fid_offset, &shared_feature));
  }
  auto fid_offset_flat = fid_offset_tensor->flat<uint64_t>();
  auto feature_offset_flat = feature_offset_tensor->flat<int32_t>();

  {
    profiler::TraceMe activity(
        []() { return "ShardingSparseFidsOp1::AddVec"; });
    auto task_func = [this, &task_context_list, &input, &nfl_fid_offset,
                      &fid_offset_flat, &feature_offset_flat,
                      &fid_list_row_splits_flat_list](const int64 begin,
                                                      const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto *task_context = task_context_list[task_index];
        TableParallelDoTask1(input, task_context, nfl_fid_offset,
                             fid_offset_flat, feature_offset_flat,
                             fid_list_row_splits_flat_list);
      }
    };
    ParallelRun(ctx, task_context_list.size(), task_func);
  }

  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp1::Copy"; });
    int index = -1;
    for (uint table_i = 0; table_i < table_cfg_list_.size(); ++table_i) {
      auto &task_context = table_shard_count_map[table_i];
      for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
        auto &shard = task_context.fid_list[ps_i];
        int size = shard.size();
        {
          auto &cur_tensor_flat =
              fid_list_row_splits_flat_list[table_i * ps_num_ + ps_i];
          for (uint z = 2; z <= table_cfg_list_[table_i]->feature_count; ++z) {
            cur_tensor_flat(z) += cur_tensor_flat(z - 1);
          }
        }
        Tensor *cur_tensor;
        ++index;
        TF_RETURN_IF_ERROR(
            ctx, fid_list_out_list->allocate(
                     index, tensorflow::TensorShape{size}, &cur_tensor));
        if (size == 0) {
          std::memset(cur_tensor->data(), 0, cur_tensor->TotalBytes());
        } else {
          std::memcpy(cur_tensor->data(), shard.data(),
                      cur_tensor->TotalBytes());
        }
      }
    }
  }
  return Status::OK();
}

template <typename TInput>
Status ShardingSparseFidsOp::TableParallelUniqueParse(
    OpKernelContext *ctx, const TInput &input, OpOutputList *fid_list_out_list,
    OpOutputList *fid_list_row_splits_out_list) {
  int batch_size = GetBatchSize(input);

  struct TableParallelTask1ContextUnique {
    std::vector<absl::flat_hash_map<uint64_t, int>> fid_list;
    std::vector<int> example_batch_feature_index;  // use for example_batch
    std::vector<const ::monolith::io::proto::NamedFeature *>
        named_feature_ptr_list;  // use for example
    std::vector<InstanceWrapper::FeaturePtr>
        instance_feature_ptr_list;  // for instance
    std::vector<int> feature_output_index;
    std::vector<int> feature_sample_index;  // use for example or instance
    int size = 0;
  };
  absl::flat_hash_map<int, TableParallelTask1ContextUnique>
      table_shard_count_map;
  std::vector<TableParallelTask1ContextUnique *> task_context_list;

  Tensor *fid_offset_tensor, *feature_offset_tensor;
  std::vector<Tensor> fid_list_row_splits_tenor_list;
  std::vector<tensorflow::TTypes<int64_t>::Flat> fid_list_row_splits_flat_list;
  std::vector<int> nfl_fid_offset(feature_conf_.size());

  {
    profiler::TraceMe activity(
        []() { return "ShardingSparseFidsOpU1::Alloc"; });
    for (uint table_i = 0; table_i < table_cfg_list_.size(); ++table_i) {
      auto &task_context = table_shard_count_map[table_i];
      task_context.fid_list.resize(ps_num_);
    }
    std::unordered_set<int> shared_feature;
    std::vector<std::vector<int>> all_feature_counter(feature_conf_.size());
    int all_feature_counter_size = 0;
    TableParallelMakeUpTask1Context(input, batch_size, &table_shard_count_map,
                                    &all_feature_counter, &shared_feature,
                                    &all_feature_counter_size);
    for (auto &table_shard_count_iter : table_shard_count_map) {
      auto &task_context = table_shard_count_iter.second;
      if (task_context.size > 0) {
        int reserve_size = task_context.size * 3 / 2 / ps_num_;
        for (auto &fid_list_part : task_context.fid_list) {
          fid_list_part.reserve(reserve_size);
        }
        task_context_list.push_back(&task_context);
      }
    }
    Tensor *nfl_offset_tensor;
    TF_RETURN_IF_ERROR(
        ctx,
        CreateOffsetTensor(
            ctx, all_feature_counter, all_feature_counter_size,
            &nfl_offset_tensor, &feature_offset_tensor, &fid_offset_tensor,
            fid_list_row_splits_out_list, fid_list_row_splits_tenor_list,
            fid_list_row_splits_flat_list, &nfl_fid_offset, &shared_feature));
  }
  auto fid_offset_flat = fid_offset_tensor->flat<uint64_t>();
  auto feature_offset_flat = feature_offset_tensor->flat<int32_t>();

  {
    profiler::TraceMe activity(
        []() { return "ShardingSparseFidsOpU1::AddSet"; });
    auto task_func = [this, &task_context_list, &input, &nfl_fid_offset,
                      &fid_offset_flat, &feature_offset_flat,
                      &fid_list_row_splits_flat_list](const int64 begin,
                                                      const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto *task_context = task_context_list[task_index];
        TableParallelDoTask1(input, task_context, nfl_fid_offset,
                             fid_offset_flat, feature_offset_flat,
                             fid_list_row_splits_flat_list);
      }
    };
    ParallelRun(ctx, task_context_list.size(), task_func);
  }

  struct TaskContext2 {
    TaskContext2(Tensor *cur_tensor_,
                 absl::flat_hash_map<uint64_t, int> *shard_ptr_, int size_)
        : cur_tensor_flat(cur_tensor_->flat<int64_t>()),
          cur_tensor(cur_tensor_),
          shard_ptr(shard_ptr_),
          size(size_) {}
    Tensor *cur_tensor;
    tensorflow::TTypes<int64_t>::Flat cur_tensor_flat;
    absl::flat_hash_map<uint64_t, int> *shard_ptr;
    int size;
  };
  std::vector<TaskContext2> task2_context_list;

  int index = -1;
  for (uint table_i = 0; table_i < table_cfg_list_.size(); ++table_i) {
    auto &task_context = table_shard_count_map[table_i];
    for (uint ps_i = 0; ps_i < ps_num_; ++ps_i) {
      auto &shard = task_context.fid_list[ps_i];
      int size = shard.size();
      {
        auto &cur_tensor_flat =
            fid_list_row_splits_flat_list[table_i * ps_num_ + ps_i];
        for (uint z = 2; z <= table_cfg_list_[table_i]->feature_count; ++z) {
          cur_tensor_flat(z) += cur_tensor_flat(z - 1);
        }
      }
      Tensor *cur_tensor;
      TF_RETURN_IF_ERROR(
          ctx, fid_list_out_list->allocate(
                   ++index, tensorflow::TensorShape{size}, &cur_tensor));
      if (size == 0) {
        std::memset(cur_tensor->data(), 0, cur_tensor->TotalBytes());
      } else {
        TaskContext2 tmp_task_context(cur_tensor, &shard, size);
        task2_context_list.emplace_back(tmp_task_context);
      }
    }
  }

  std::vector<std::vector<int>> task2_split;
  SplitTask<TaskContext2>(task2_context_list, single_thread_assign_watermark_,
                          &task2_split);
  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOpU1::Copy"; });
    auto tensor_assgin_func = [this, ctx, &task2_context_list, &task2_split](
                                  const int64 begin, const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task2_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task2_context_list[index];
          // int set_index = -1;
          for (auto &fid : *task_context.shard_ptr) {
            task_context.cur_tensor_flat(fid.second) = fid.first;
          }
        }
      }
    };
    ParallelRun(ctx, task2_split.size(), tensor_assgin_func);
  }
  return Status::OK();
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
    int feature_size = 0;
    if (named_feature_list.type() == FeatureListType::SHARED) {
      const auto &feature = named_feature_list.feature(0);
      int tmp_counter = 0;
      if (feature.has_fid_v1_list()) {
        tmp_counter = feature.fid_v1_list().value_size();
      } else if (feature.has_fid_v2_list()) {
        tmp_counter = feature.fid_v2_list().value_size();
      }
      feature_size += tmp_counter;
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
        feature_size += tmp_counter;
        feature_counter.push_back(tmp_counter);
      }
    }
    if (feature_size > 0) {
      task_context->size += feature_size;
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
      }

      if (feature_index < 0) {
        const auto &name = named_feature.name();
        auto find_iter = feature_conf_.find(name);
        if (find_iter == feature_conf_.end()) {
          continue;
        }
        feature_index = find_iter->second.feature_index;
      }
      auto *task_context = &((*task_context_list)[feature_index]);
      auto &feature_counter = (*all_feature_counter)[feature_index];
      int feature_size = 0;
      const auto &feature = named_feature.feature();
      if (feature.has_fid_v1_list()) {
        feature_size = feature.fid_v1_list().value_size();
      } else if (feature.has_fid_v2_list()) {
        feature_size = feature.fid_v2_list().value_size();
      }
      feature_counter[ex_i] = feature_size;

      if (feature_size > 0) {
        task_context->size += feature_size;
        task_context->named_feature_ptr_list.push_back(&named_feature);
        task_context->feature_sample_index.push_back(
            feature_index * batch_size + ex_i);
      }
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

      int feature_size = 0;
      if (named_feature.fid_v1) {
        feature_size = named_feature.fid_v1->size();
      } else {
        feature_size += named_feature.fid_v2->fid_size();
        // this is a sequence feature list.
        for (const auto &fidlist : named_feature.fid_v2->fid_list()) {
          feature_size += fidlist.value_size();
        }
      }
      feature_counter[ex_i] = feature_size;
      if (feature_size > 0) {
        task_context->size += feature_size;
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
  const auto &named_feature_list = example_batch.named_feature_list(
      task_context->example_batch_feature_index);
  auto feature_output_index = task_context->feature_output_index;
  auto offset = nfl_fid_offset[feature_output_index];
  if (named_feature_list.type() == FeatureListType::SHARED) {
    const auto &feature = named_feature_list.feature(0);
    if (feature.has_fid_v1_list()) {
      for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
        auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset);
      }
    } else if (feature.has_fid_v2_list()) {
      for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
        auto value = feature.fid_v2_list().value(i);
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset);
      }
    }
  } else {
    for (const auto &feature : named_feature_list.feature()) {
      if (feature.has_fid_v1_list()) {
        for (int i = 0; i < feature.fid_v1_list().value_size(); ++i) {
          auto value = convert_fid_v1_to_v2(feature.fid_v1_list().value(i));
          FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                      &offset);
        }
      } else if (feature.has_fid_v2_list()) {
        for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
          auto value = feature.fid_v2_list().value(i);
          FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                      &offset);
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
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset);
      }
    } else if (feature.has_fid_v2_list()) {
      for (int i = 0; i < feature.fid_v2_list().value_size(); ++i) {
        auto value = feature.fid_v2_list().value(i);
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset);
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
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset);
      }
    } else {
      for (const auto &value : named_feature_ptr.fid_v2->fid()) {
        FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                    &offset);
      }
      // this is a sequence feature list.
      for (const auto &fidlist : named_feature_ptr.fid_v2->fid_list()) {
        for (const auto &value : fidlist.value()) {
          FillFidList(value, shard_vec, fid_offset_flat, feature_output_index,
                      &offset);
        }
      }
    }
  }
  task_context->feature_offset = offset - nfl_fid_offset[feature_output_index];
}

template <typename TInput>
Status ShardingSparseFidsOp::FeatureParallelParse(
    OpKernelContext *ctx, const TInput &input, OpOutputList *fid_list_out_list,
    OpOutputList *fid_list_row_splits_out_list) {
  int batch_size = GetBatchSize(input);
  struct FeatureParallelTask1Context {
    std::vector<std::vector<uint64_t>> fid_list;
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
  std::vector<Tensor> fid_list_row_splits_tenor_list;
  std::vector<tensorflow::TTypes<int64_t>::Flat> fid_list_row_splits_flat_list;
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
      task_context.table_offset.resize(ps_num_, 0);
      if (task_context.size > 0) {
        int reserve_size = task_context.size * 6 / 5 / ps_num_;
        for (auto &fid_list_part : task_context.fid_list) {
          fid_list_part.reserve(reserve_size);
        }
      }
    }
    Tensor *nfl_offset_tensor;
    TF_RETURN_IF_ERROR(
        ctx,
        CreateOffsetTensor(
            ctx, all_feature_counter, all_feature_counter_size,
            &nfl_offset_tensor, &feature_offset_tensor, &fid_offset_tensor,
            fid_list_row_splits_out_list, fid_list_row_splits_tenor_list,
            fid_list_row_splits_flat_list, &nfl_fid_offset, &shared_feature));
  }
  auto fid_offset_flat = fid_offset_tensor->flat<uint64_t>();
  auto feature_offset_flat = feature_offset_tensor->flat<int32_t>();

  std::vector<std::vector<int>> task_split;
  SplitTask<FeatureParallelTask1Context>(
      task_context_list, single_thread_feature_watermark_, &task_split);
  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp::AddVec"; });
    auto task_func = [this, &task_context_list, &input, &task_split,
                      &nfl_fid_offset, &fid_offset_flat, &feature_offset_flat](
                         const int64 begin, const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task_context_list[index];
          FeatureParallelDoTask1(input, &task_context, nfl_fid_offset,
                                 fid_offset_flat, feature_offset_flat);
        }
      }
    };

    ParallelRun(ctx, task_split.size(), task_func);
  }

  struct TaskContext2 {
    TaskContext2(Tensor *cur_tensor_, std::vector<uint64_t> *shard_ptr_,
                 int size_, int offset_)
        : data_ptr(cur_tensor_->data()),
          cur_tensor(cur_tensor_),
          shard_ptr(shard_ptr_),
          size(size_),
          offset(offset_) {}
    explicit TaskContext2(FeatureParallelTask1Context *task_context_)
        : task1_context(task_context_), size(task_context_->feature_offset) {}
    Tensor *cur_tensor;
    void *data_ptr;
    std::vector<uint64_t> *shard_ptr;
    int size = -1;
    int offset = -1;
    FeatureParallelTask1Context *task1_context = nullptr;
  };
  std::vector<TaskContext2> task2_context_list;

  int index = -1;
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
      if (feature_vec_ptr != nullptr) {
        for (auto task_context_ptr : *feature_vec_ptr) {
          task_context_ptr->table_offset[ps_num_i] = size;
          size += task_context_ptr->fid_list[ps_num_i].size();
          cur_tensor_flat(
              feature_cfg_list_[task_context_ptr->feature_output_index]
                  ->feature_in_table_index +
              1) = task_context_ptr->fid_list[ps_num_i].size();
        }
      }
      {
        for (uint z = 2; z <= table_cfg_list_[table_index]->feature_count;
             ++z) {
          cur_tensor_flat(z) += cur_tensor_flat(z - 1);
        }
        // cur_tensor_flat(0) = 0;
        // cur_tensor_flat(z) = size;
      }
      Tensor *cur_tensor;
      TF_RETURN_IF_ERROR(
          ctx, fid_list_out_list->allocate(
                   ++index, tensorflow::TensorShape{size}, &cur_tensor));
      if (size == 0) {
        std::memset(cur_tensor->data(), 0, cur_tensor->TotalBytes());
        continue;
      }
      int offset = 0;
      for (auto task_context_ptr : *feature_vec_ptr) {
        auto &fid_list = task_context_ptr->fid_list[ps_num_i];
        TaskContext2 tmp_task_context(cur_tensor, &fid_list, fid_list.size(),
                                      offset);
        task2_context_list.emplace_back(tmp_task_context);
        offset += fid_list.size();
      }
    }
    if (feature_vec_ptr != nullptr) {
      for (auto task_context_ptr : *feature_vec_ptr) {
        TaskContext2 tmp_task_context(task_context_ptr);
        task2_context_list.emplace_back(tmp_task_context);
      }
    }
  }

  std::vector<std::vector<int>> task2_split;
  SplitTask<TaskContext2>(task2_context_list, single_thread_assign_watermark_,
                          &task2_split);

  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOp::Copy"; });
    auto tensor_assgin_func = [this, ctx, &task2_context_list, &task2_split,
                               &nfl_fid_offset, &fid_offset_flat](
                                  const int64 begin, const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task2_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task2_context_list[index];
          if (task_context.task1_context) {
            if (version_ == 1) {
              auto &task1_context = *task_context.task1_context;
              auto offset = nfl_fid_offset[task1_context.feature_output_index];
              auto table_index =
                  feature_cfg_list_[task1_context.feature_output_index]
                      ->table_index *
                  ps_num_;
              for (int i = 0; i < task1_context.feature_offset; ++i, ++offset) {
                auto cur = fid_offset_flat(offset);
                cur += task1_context.table_offset.at(
                    static_cast<int>(cur >> 32) - table_index);
                fid_offset_flat(offset) = cur;
              }
            }
          } else {
            std::memcpy(reinterpret_cast<char *>(task_context.data_ptr) +
                            int64_size_ * task_context.offset,
                        task_context.shard_ptr->data(),
                        task_context.shard_ptr->size() * int64_size_);
          }
        }
      }
    };
    ParallelRun(ctx, task2_split.size(), tensor_assgin_func);
  }
  return Status::OK();
}

template <typename TInput>
Status ShardingSparseFidsOp::FeatureParallelUniqueParse(
    OpKernelContext *ctx, const TInput &input, OpOutputList *fid_list_out_list,
    OpOutputList *fid_list_row_splits_out_list) {
  int batch_size = GetBatchSize(input);

  struct FeatureParallelTask1ContextUnique {
    std::vector<absl::flat_hash_map<uint64_t, int>> fid_list;
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

  std::vector<FeatureParallelTask1ContextUnique> task_context_list;
  absl::flat_hash_map<int, FeatureParallelTask1ContextUnique *>
      feature_shard_count_map;
  absl::flat_hash_map<int, std::vector<FeatureParallelTask1ContextUnique *>>
      table_feature_map;

  Tensor *fid_offset_tensor, *feature_offset_tensor;
  std::vector<Tensor> fid_list_row_splits_tenor_list;
  std::vector<tensorflow::TTypes<int64_t>::Flat> fid_list_row_splits_flat_list;
  std::vector<int> nfl_fid_offset(feature_conf_.size());

  {
    profiler::TraceMe activity(
        []() { return "ShardingSparseFidsOpU2::Alloc"; });
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
      task_context.table_offset.resize(ps_num_, 0);
      if (task_context.size > 0) {
        int reserve_size = task_context.size * 6 / 5 / ps_num_;
        for (auto &fid_list_part : task_context.fid_list) {
          fid_list_part.reserve(reserve_size);
        }
      }
    }

    Tensor *nfl_offset_tensor;
    TF_RETURN_IF_ERROR(
        ctx,
        CreateOffsetTensor(
            ctx, all_feature_counter, all_feature_counter_size,
            &nfl_offset_tensor, &feature_offset_tensor, &fid_offset_tensor,
            fid_list_row_splits_out_list, fid_list_row_splits_tenor_list,
            fid_list_row_splits_flat_list, &nfl_fid_offset, &shared_feature));
  }

  auto fid_offset_flat = fid_offset_tensor->flat<uint64_t>();
  auto feature_offset_flat = feature_offset_tensor->flat<int32_t>();

  std::vector<std::vector<int>> task_split;
  SplitTask<FeatureParallelTask1ContextUnique>(
      task_context_list, single_thread_feature_watermark_, &task_split);

  {
    profiler::TraceMe activity(
        []() { return "ShardingSparseFidsOpU2::AddSet"; });
    auto task_func = [this, &task_context_list, &input, &task_split,
                      &nfl_fid_offset, &fid_offset_flat, &feature_offset_flat](
                         const int64 begin, const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task_context_list[index];
          FeatureParallelDoTask1(input, &task_context, nfl_fid_offset,
                                 fid_offset_flat, feature_offset_flat);
        }
      }
    };
    ParallelRun(ctx, task_split.size(), task_func);
  }

  struct TaskContext2 {
    TaskContext2(Tensor *cur_tensor_,
                 absl::flat_hash_map<uint64_t, int> *shard_ptr_, int size_,
                 int offset_)
        : cur_tensor(cur_tensor_),
          shard_ptr(shard_ptr_),
          size(size_),
          offset(offset_) {}
    explicit TaskContext2(FeatureParallelTask1ContextUnique *task_context_)
        : task1_context(task_context_), size(task_context_->feature_offset) {}
    Tensor *cur_tensor;
    // tensorflow::TTypes<int64_t>::Flat cur_tensor_flat;
    absl::flat_hash_map<uint64_t, int> *shard_ptr;
    int size = -1;
    int offset = -1;
    FeatureParallelTask1ContextUnique *task1_context = nullptr;
  };
  std::vector<TaskContext2> task2_context_list;

  int index = -1;
  for (uint table_index = 0; table_index < table_cfg_list_.size();
       ++table_index) {
    // auto &table_name = table_names_[table_index];
    auto table_feature_map_find_iter = table_feature_map.find(table_index);
    std::vector<FeatureParallelTask1ContextUnique *> *feature_vec_ptr = nullptr;
    if (table_feature_map_find_iter != table_feature_map.end()) {
      feature_vec_ptr = &(table_feature_map_find_iter->second);
    }
    for (int ps_num_i = 0; ps_num_i < ps_num_; ++ps_num_i) {
      auto &cur_tensor_flat =
          fid_list_row_splits_flat_list[table_index * ps_num_ + ps_num_i];
      int size = 0;
      if (feature_vec_ptr != nullptr) {
        for (auto task_context_ptr : *feature_vec_ptr) {
          task_context_ptr->table_offset[ps_num_i] = size;
          size += task_context_ptr->fid_list[ps_num_i].size();
          cur_tensor_flat(
              feature_cfg_list_[task_context_ptr->feature_output_index]
                  ->feature_in_table_index +
              1) = task_context_ptr->fid_list[ps_num_i].size();
        }
      }
      {
        for (uint z = 2; z <= table_cfg_list_[table_index]->feature_count;
             ++z) {
          cur_tensor_flat(z) += cur_tensor_flat(z - 1);
        }
        // cur_tensor_flat(0) = size;
        // cur_tensor_flat(z) = size;
      }
      Tensor *cur_tensor;
      TF_RETURN_IF_ERROR(
          ctx, fid_list_out_list->allocate(
                   ++index, tensorflow::TensorShape{size}, &cur_tensor));
      if (size == 0) {
        std::memset(cur_tensor->data(), 0, cur_tensor->TotalBytes());
        continue;
      }
      int offset = 0;
      for (auto task_context_ptr : *feature_vec_ptr) {
        auto &fid_list = task_context_ptr->fid_list[ps_num_i];
        TaskContext2 tmp_task_context(cur_tensor, &fid_list, fid_list.size(),
                                      offset);
        task2_context_list.emplace_back(tmp_task_context);
        offset += fid_list.size();
      }
    }
    if (feature_vec_ptr != nullptr) {
      for (auto task_context_ptr : *feature_vec_ptr) {
        TaskContext2 tmp_task_context(task_context_ptr);
        task2_context_list.emplace_back(tmp_task_context);
      }
    }
  }

  std::vector<std::vector<int>> task2_split;
  SplitTask<TaskContext2>(task2_context_list, single_thread_assign_watermark_,
                          &task2_split);

  {
    profiler::TraceMe activity([]() { return "ShardingSparseFidsOpU2::Copy"; });
    auto tensor_assgin_func = [this, ctx, &task2_context_list, &task2_split,
                               &nfl_fid_offset, &fid_offset_flat](
                                  const int64 begin, const int64 end) {
      for (int64 task_index = begin; task_index < end; ++task_index) {
        auto &task_index_list = task2_split[task_index];
        for (auto index : task_index_list) {
          auto &task_context = task2_context_list[index];
          if (task_context.task1_context) {
            if (version_ == 1) {
              auto &task1_context = *task_context.task1_context;
              auto offset = nfl_fid_offset[task1_context.feature_output_index];
              auto table_index =
                  feature_cfg_list_[task1_context.feature_output_index]
                      ->table_index *
                  ps_num_;
              for (int i = 0; i < task1_context.feature_offset; ++i, ++offset) {
                auto cur = fid_offset_flat(offset);
                cur += task1_context.table_offset.at(
                    static_cast<int>(cur >> 32) - table_index);
                fid_offset_flat(offset) = cur;
              }
            }
          } else {
            // int set_index = -1;
            auto cur_tensor_flat =
                task_context.cur_tensor->template flat<int64_t>();
            for (auto &fid : *task_context.shard_ptr) {
              cur_tensor_flat(fid.second + task_context.offset) = fid.first;
            }
          }
        }
      }
    };
    ParallelRun(ctx, task2_split.size(), tensor_assgin_func);
  }
  return Status::OK();
}

}  // namespace monolith_tf
}  // namespace tensorflow
#endif MONOLITH_NATIVE_TRAINING_DATA_KERNELS_PARSE_SPARSE_FEATURE_LIB_H_
