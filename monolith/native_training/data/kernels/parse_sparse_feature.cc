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

#include "monolith/native_training/data/kernels/parse_sparse_feature.h"
#include <algorithm>
#include <tuple>
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"

namespace tensorflow {
namespace monolith_tf {
ShardingSparseFidsOp::ShardingSparseFidsOp(OpKernelConstruction *ctx,
                                           int version /* = 1*/)
    : OpKernel(ctx), version_(version) {
  std::string feature_cfgs_str;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("ps_num", &ps_num_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_cfgs", &feature_cfgs_str));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("unique", &unique_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("parallel_flag", &parallel_flag_));
  std::string input_type;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_type", &input_type));
  if (input_type == "example") {
    input_type_ = 0;
  } else if (input_type == "examplebatch") {
    input_type_ = 1;
  } else if (input_type == "instance") {
    input_type_ = 2;
  } else {
    OP_REQUIRES(ctx, false,
                errors::FailedPrecondition(
                    "input_type only support example/examplebatch."));
  }
  ::monolith::io::proto::FeatureConfigs feature_cfgs;
  OP_REQUIRES(
      ctx, feature_cfgs.ParseFromString(feature_cfgs_str),
      errors::FailedPrecondition("Failed to parse the FeatureConfigs."));

  enable_parallel_ =
      (parallel_flag_ > 0);  // parallel_flag_ == 0 default not parallel

  auto creator = [this](FeatureNameMapperTfBridge **out_mapper) {
    TF_RETURN_IF_ERROR(FeatureNameMapperTfBridge::New(out_mapper));
    return Status::OK();
  };
  ResourceMgr *resource_mgr = ctx->resource_manager();
  OP_REQUIRES_OK(ctx, resource_mgr->LookupOrCreate<FeatureNameMapperTfBridge>(
                          resource_mgr->default_container(),
                          FeatureNameMapperTfBridge::kName, &mapper_, creator));
  std::vector<std::string> feature_names;
  for (const auto &pair : feature_cfgs.feature_configs()) {
    feature_names.push_back(pair.first);
  }
  mapper_raw_ptr_ = mapper_->GetFeatureNameMapper();
  CHECK(mapper_raw_ptr_->RegisterValidNames(feature_names));

  feature_index_conf_.reserve(feature_cfgs.feature_configs_size() * 2);
  static std::vector<std::string> slot_id_feature_prfix({"fc_slot_", "slot_"});
  for (auto &iter : feature_cfgs.feature_configs()) {
    auto &feature_cfg = feature_conf_[iter.first];
    feature_cfg.table_name = iter.second.table();
    feature_cfg.feature_name = iter.first;
    feature_cfg.version = version_;

    int dims_sum = 0;
    for (size_t slice_idx = 0; slice_idx < iter.second.slice_dims_size();
         slice_idx++) {
      dims_sum += iter.second.slice_dims(slice_idx);
    }
    feature_cfg.dims_sum = dims_sum;

    auto &table_cfg = table_conf_[iter.second.table()];
    table_cfg.table_name = iter.second.table();

    for (auto &feature_prfix : slot_id_feature_prfix) {
      if (absl::StartsWith(iter.first, feature_prfix)) {
        std::string sub_str(iter.first.substr(feature_prfix.size()));
        try {
          int slot_id = std::stoi(sub_str);
          slot_id_to_feature_name_[slot_id] = iter.first;
        } catch (std::exception const &ex) {
          LOG(ERROR) << "slot_id_to_feature_name_ err:" << ex.what() << ":"
                     << iter.first << "," << sub_str;
          continue;
        }
      }
    }
  }
  for (auto &iter : table_conf_) {
    table_cfg_list_.push_back(&iter.second);
  }
  std::sort(
      table_cfg_list_.begin(), table_cfg_list_.end(),
      [](TableInfo *a, TableInfo *b) { return a->table_name < b->table_name; });
  for (uint i = 0; i < table_cfg_list_.size(); ++i) {
    auto &conf = *(table_cfg_list_[i]);
    conf.table_index = i;
  }

  for (auto &iter : feature_conf_) {
    feature_cfg_list_.push_back(&iter.second);
  }
  std::sort(feature_cfg_list_.begin(), feature_cfg_list_.end(),
            [](FeatureInfo *a, FeatureInfo *b) {
              return a->feature_name < b->feature_name;
            });
  for (uint i = 0; i < feature_cfg_list_.size(); ++i) {
    auto &conf = *(feature_cfg_list_[i]);
    conf.feature_index = i;
    auto &table_cfg = table_conf_[conf.table_name];
    conf.table_index = table_cfg.table_index;
    conf.feature_in_table_index = table_cfg.feature_count++;
    table_cfg.feature_index_list.push_back(i);
  }

  for (uint i = 0; i < feature_cfg_list_.size(); ++i) {
    auto &conf = *(feature_cfg_list_[i]);
    auto &table_cfg = table_conf_[conf.table_name];
    conf.table_feature_count = table_cfg.feature_count;
  }

  if (version_ == 2) {
    int output_index = 0;
    for (uint i = 0; i < table_cfg_list_.size(); ++i) {
      auto &table_cfg = *(table_cfg_list_[i]);
      for (auto feature_index : table_cfg.feature_index_list) {
        auto &feature_cfg = *(feature_cfg_list_[feature_index]);
        feature_cfg.output_pre_index = output_index;
      }
      output_index +=
          std::max(table_cfg.feature_index_list.size(), 1UL) * ps_num_;
    }
  } else {
    for (auto feature_cfg_ptr : feature_cfg_list_) {
      feature_cfg_ptr->output_pre_index =
          feature_cfg_ptr->table_index * ps_num_;
    }
  }

  if (mapper_raw_ptr_->IsAvailable()) {
    int32_t max_sorted_id = -1;
    LOG_FIRST_N(INFO, 1) << mapper_raw_ptr_->DebugString();

    absl::flat_hash_map<int32_t, FeatureInfo *> feature_index_conf_tmp;
    feature_index_conf_tmp.reserve(feature_conf_.size() * 2);
    for (auto &iter : feature_conf_) {
      int32_t id = -1;
      int32_t sorted_id = -1;
      bool found = mapper_raw_ptr_->GetIdByName(iter.first, &id, &sorted_id);
      if (found && !feature_index_conf_tmp.contains(sorted_id)) {
        feature_index_conf_tmp[sorted_id] = &iter.second;
        max_sorted_id = std::max(max_sorted_id, sorted_id);
      } else {
        feature_index_conf_tmp.clear();
        LOG(ERROR) << "mapper_raw_ptr_ not find:" << iter.first;
        break;
      }
    }
    if (feature_index_conf_tmp.size() > 0) {
      feature_index_conf_.resize(max_sorted_id + 1, nullptr);
      for (auto &iter : feature_index_conf_tmp) {
        feature_index_conf_[iter.first] = iter.second;
      }
    }
  } else {
    LOG(WARNING) << "mapper_raw_ptr_ not Available()";
  }
}

void ShardingSparseFidsOp::FillFidList(
    uint64_t value, std::vector<std::vector<uint64_t>> &shard_vec,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    int feature_output_index, int *offset) {
  auto mod = value % ps_num_;
  int output_offset =
      feature_cfg_list_[feature_output_index]->GetFidOutputIndex(mod);
  int feature_offset = shard_vec[mod].size();
  if (version_ == 3 || version_ == 4) {
    feature_offset *= feature_cfg_list_[feature_output_index]->dims_sum;
  }
  fid_offset_flat(*offset) = ((uint64_t(output_offset) << 32) | feature_offset);
  shard_vec[mod].push_back(value);
  ++(*offset);
}

void ShardingSparseFidsOp::FillFidList(
    uint64_t value, std::vector<absl::flat_hash_map<uint64_t, int>> &shard_vec,
    tensorflow::TTypes<uint64_t>::Flat fid_offset_flat,
    int feature_output_index, int *offset) {
  auto mod = value % ps_num_;
  int output_offset =
      feature_cfg_list_[feature_output_index]->GetFidOutputIndex(mod);
  auto fid_find_iter = shard_vec[mod].find(value);
  int feature_offset = -1;
  if (fid_find_iter == shard_vec[mod].end()) {
    feature_offset = shard_vec[mod].size();
    shard_vec[mod].emplace(value, feature_offset);
  } else {
    feature_offset = fid_find_iter->second;
  }
  if (version_ == 3 || version_ == 4) {
    feature_offset *= feature_cfg_list_[feature_output_index]->dims_sum;
  }
  fid_offset_flat(*offset) = (uint64_t(output_offset) << 32) | feature_offset;
  ++(*offset);
}

void ShardingSparseFidsOp::CopyFidList(
    const std::vector<uint64_t> &shard_ptr, int offset,
    TensorSliceAccessor<int64_t> *cur_tensor) {
  void *data_ptr = cur_tensor->ptr;
  std::memcpy(reinterpret_cast<char *>(data_ptr) + int64_size_ * offset,
              shard_ptr.data(), shard_ptr.size() * int64_size_);
}

void ShardingSparseFidsOp::CopyFidList(
    const absl::flat_hash_map<uint64_t, int> &shard_ptr, int offset,
    TensorSliceAccessor<int64_t> *cur_tensor) {
  // auto cur_tensor_flat = cur_tensor->template flat<int64_t>();
  for (auto &fid : shard_ptr) {
    (*cur_tensor)(fid.second + offset) = fid.first;
  }
}

Status ShardingSparseFidsOp::CreateOffsetTensor(
    OpKernelContext *ctx,
    const std::vector<std::vector<int>> &all_feature_counter,
    int all_feature_counter_size, Tensor **nfl_offset_tensor,
    Tensor **feature_offset_tensor, Tensor **fid_offset_tensor,
    OpOutputList *fid_list_row_splits_out_list,
    std::vector<Tensor> &tmp_tensor_list,
    std::vector<ShardingSparseFidsOp::TensorSliceAccessor<int64_t>>
        &fid_list_row_splits_flat_list,
    std::vector<int> *nfl_fid_offset,
    const std::unordered_set<int> *shared_feature) {
  TF_RETURN_IF_ERROR(ctx, ctx->allocate_output("nfl_offset",
                                               TensorShape({
                                                   feature_conf_.size(),
                                               }),
                                               nfl_offset_tensor));
  auto nfl_offset_flat = (*nfl_offset_tensor)->flat<uint32_t>();

  TF_RETURN_IF_ERROR(ctx, ctx->allocate_output("feature_offset",
                                               TensorShape({
                                                   all_feature_counter_size,
                                               }),
                                               feature_offset_tensor));
  auto feature_offset_flat = (*feature_offset_tensor)->flat<int32_t>();
  int all_feature_size = 0;
  int feature_offset_index = 0;
  for (uint i = 0; i < all_feature_counter.size(); ++i) {
    auto &feature_counter = all_feature_counter[i];
    nfl_offset_flat(i) = feature_offset_index;
    if (shared_feature->size() > 0 && shared_feature->count(i) > 0) {
      nfl_offset_flat(i) |= shard_flag_;
    }
    (*nfl_fid_offset)[i] = all_feature_size;

    for (uint j = 0; j < feature_counter.size(); ++j) {
      feature_offset_flat(feature_offset_index) = all_feature_size;
      all_feature_size += abs(feature_counter[j]);
      ++feature_offset_index;
    }
  }
  TF_RETURN_IF_ERROR(ctx, ctx->allocate_output("fid_offset",
                                               TensorShape({
                                                   all_feature_size,
                                               }),
                                               fid_offset_tensor));

  if (version_ == 4) {
    Tensor *fid_list_row_lengths_tensor;
    int all_feature_count = 0;
    for (uint i = 0; i < table_cfg_list_.size(); ++i) {
      all_feature_count += table_cfg_list_[i]->feature_count + 1;
    }
    all_feature_count *= ps_num_;
    TF_RETURN_IF_ERROR(ctx, ctx->allocate_output("fid_list_row_splits",
                                                 TensorShape({
                                                     all_feature_count,
                                                 }),
                                                 &fid_list_row_lengths_tensor));
    auto cur_tensor_flat = fid_list_row_lengths_tensor->flat<int64_t>();
    cur_tensor_flat.setZero();
    fid_list_row_splits_flat_list.resize(table_cfg_list_.size() * ps_num_);
    int pre_feature_count = 0;
    for (uint ps_num_i = 0; ps_num_i < ps_num_; ++ps_num_i) {
      for (uint table_index = 0; table_index < table_cfg_list_.size();
           ++table_index) {
        /*
          LOG(ERROR) << "xxxx all_feature_count:" << all_feature_count
                    << ",pre_feature_count:" << pre_feature_count
                    << ",feature_count:"
                    << table_cfg_list_[table_index]->feature_count;
          auto cur_tensor = fid_list_row_lengths_tensor->Slice(
          pre_feature_count,
          pre_feature_count + table_cfg_list_[table_index]->feature_count);
        */
        int cur_count = table_cfg_list_[table_index]->feature_count + 1;
        fid_list_row_splits_flat_list[table_index * ps_num_ + ps_num_i] =
            TensorSliceAccessor<int64_t>(
                {static_cast<int64_t *>(fid_list_row_lengths_tensor->data()) +
                     pre_feature_count,
                 cur_count});
        pre_feature_count += cur_count;
      }
    }
  } else {
    tmp_tensor_list.resize(table_cfg_list_.size() * ps_num_);
    int fid_list_row_splits_flat_list_index = -1;
    for (uint i = 0; i < table_cfg_list_.size(); ++i) {
      for (uint j = 0; j < ps_num_; ++j) {
        Tensor *cur_tensor;
        ++fid_list_row_splits_flat_list_index;
        if (version_ == 2 || version_ == 3) {
          TF_RETURN_IF_ERROR(ctx,
                             fid_list_row_splits_out_list->allocate(
                                 fid_list_row_splits_flat_list_index,
                                 tensorflow::TensorShape{
                                     table_cfg_list_[i]->feature_count + 1},
                                 &cur_tensor));
        } else {
          cur_tensor = &(tmp_tensor_list[i * ps_num_ + j]);
          TF_RETURN_IF_ERROR(
              ctx,
              ctx->allocate_temp(DT_INT64,
                                 tensorflow::TensorShape{
                                     table_cfg_list_[i]->feature_count + 1},
                                 cur_tensor));
        }
        auto cur_tensor_flat = cur_tensor->flat<int64_t>();
        cur_tensor_flat.setZero();
        fid_list_row_splits_flat_list.emplace_back(TensorSliceAccessor<int64_t>(
            {static_cast<int64_t *>(cur_tensor->data()),
             cur_tensor->NumElements()}));
      }
    }
  }

  return Status::OK();
}

void ShardingSparseFidsOp::ParallelRun(
    OpKernelContext *ctx, int task_count,
    const std::function<void(int64, int64)> &fn) {
  if (enable_parallel_ && task_count > 1) {
    auto workers = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    workers->ParallelFor(
        task_count,
        thread::ThreadPool::SchedulingParams(
            thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
            absl::nullopt, 1),
        fn);
  } else {
    for (int i = 0; i < task_count; ++i) {
      fn(i, i + 1);
    }
  }
}

void ShardingSparseFidsOp::InitInstanceWrapper(
    ShardingSparseFidsOp::InstanceWrapper *instance_wrapper) {
  auto &instances = instance_wrapper->instances;
  if (slot_id_to_feature_name_.size() == 0) {
    return;
  }
  instance_wrapper->fid_v1.reserve(slot_id_to_feature_name_.size());
  for (auto &slot_name_iter : slot_id_to_feature_name_) {
    auto &part = instance_wrapper->fid_v1[slot_name_iter.second];
    part.resize(instances.size());
    for (int i = 0; i < part.size(); ++i) {
      part[i].reserve(instances[i]->fid_size());
    }
  }
  for (int i = 0; i < instances.size(); ++i) {
    for (auto fid : instances[i]->fid()) {
      int slot_id = slot_id_v1(fid);
      auto find_iter = slot_id_to_feature_name_.find(slot_id);
      if (find_iter == slot_id_to_feature_name_.end()) {
        continue;
      }
      instance_wrapper->fid_v1[find_iter->second][i].push_back(fid);
    }
  }
}

void ShardingSparseFidsOp::Compute(OpKernelContext *ctx) {
  const Tensor *pb_input;
  OP_REQUIRES_OK(ctx, ctx->input("pb_input", &pb_input));
  OpOutputList out_list;
  if (version_ != 4) {
    OP_REQUIRES_OK(ctx, ctx->output_list("fid_list", &out_list));
  }
  OpOutputList fid_list_row_splits_out_list;
  if (version_ == 2 || version_ == 3) {
    OP_REQUIRES_OK(ctx, ctx->output_list("fid_list_row_splits",
                                         &fid_list_row_splits_out_list));
  }
  int batch_size = 0;
  Status st;
  if (input_type_ == 0) {
    const auto &pb_variant_tensor = pb_input->vec<Variant>();
    batch_size = pb_variant_tensor.dimension(0);
    std::vector<const Example *> examples;
    examples.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      const auto *example = pb_variant_tensor(i).get<Example>();
      CHECK_NOTNULL(example);
      examples.push_back(example);
    }
    if (unique_) {
      st = FeatureParallelUniqueParse(ctx, examples, &out_list,
                                      &fid_list_row_splits_out_list);
    } else {
      st = FeatureParallelParse(ctx, examples, &out_list,
                                &fid_list_row_splits_out_list);
    }

  } else if (input_type_ == 1) {
    const auto &example_batch =
        *(pb_input->scalar<Variant>()().get<ExampleBatch>());
    batch_size = example_batch.batch_size();
    if (unique_) {
      st = FeatureParallelUniqueParse(ctx, example_batch, &out_list,
                                      &fid_list_row_splits_out_list);
    } else {
      st = FeatureParallelParse(ctx, example_batch, &out_list,
                                &fid_list_row_splits_out_list);
    }
  } else if (input_type_ == 2) {
    const auto &pb_variant_tensor = pb_input->vec<Variant>();
    batch_size = pb_variant_tensor.dimension(0);
    InstanceWrapper instance_wapper;
    instance_wapper.instances.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      const auto *instance = pb_variant_tensor(i).get<Instance>();
      CHECK_NOTNULL(instance);
      instance_wapper.instances.push_back(instance);
    }
    InitInstanceWrapper(&instance_wapper);

    if (unique_) {
      st = FeatureParallelUniqueParse(ctx, instance_wapper, &out_list,
                                      &fid_list_row_splits_out_list);
    } else {
      st = FeatureParallelParse(ctx, instance_wapper, &out_list,
                                &fid_list_row_splits_out_list);
    }
  }
  OP_REQUIRES_OK(ctx, st);

  Tensor *batch_size_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("batch_size", TensorShape({}),
                                           &batch_size_tensor));
  batch_size_tensor->scalar<int32>()() = batch_size;
}

REGISTER_KERNEL_BUILDER(Name("ShardingSparseFids").Device(DEVICE_CPU),
                        ShardingSparseFidsOp);

class ShardingSparseFidsOpV2 : public ShardingSparseFidsOp {
 public:
  explicit ShardingSparseFidsOpV2(OpKernelConstruction *ctx)
      : ShardingSparseFidsOp(ctx, 2) {}
};

REGISTER_KERNEL_BUILDER(Name("ShardingSparseFidsV2").Device(DEVICE_CPU),
                        ShardingSparseFidsOpV2);

class ShardingSparseFidsOpV3 : public ShardingSparseFidsOp {
 public:
  explicit ShardingSparseFidsOpV3(OpKernelConstruction *ctx)
      : ShardingSparseFidsOp(ctx, 3) {}
};

REGISTER_KERNEL_BUILDER(Name("ShardingSparseFidsV3").Device(DEVICE_CPU),
                        ShardingSparseFidsOpV3);

class ShardingSparseFidsOpV4 : public ShardingSparseFidsOp {
 public:
  explicit ShardingSparseFidsOpV4(OpKernelConstruction *ctx)
      : ShardingSparseFidsOp(ctx, 4) {}
};

REGISTER_KERNEL_BUILDER(Name("ShardingSparseFidsV4").Device(DEVICE_CPU),
                        ShardingSparseFidsOpV4);

// 该函数包含6个输出
// fid_list:             shape(table_count*ps_num, 若干fid),
//                       将不同feature样本的全部fid聚合，并按照ps_num分shard，按照feature->table的映射聚合填充
// fid_list_row_splits   shape(table_count*ps_num, table内feature个数+1),
//                       与fid_list组成ragged_tensor，主要作用是将相同table内不同feature区分开
// fid_offset            shape(特征数*(1 if shard feature else
// batch_size)*fid数),
//                       样本的一维平铺，最后不是存储的fid，而是在fid_list中的偏移，用于在fid_list寻址
//                       高32位为fid_list 第一维度与feature在table内index 的组合
//                       低32位为fid在当前feature fid_list的第几位
// feature_offset        shape(特征数*(1 if shard feature else batch_size)),
//                       对fid_offset一维平铺的拆解，标识每个样本的分界点
// nfl_offset            shape(特征数),
//                       对fid_offset/feature_offset一维平铺的拆解，标识每个特征的分界点
// batch_size

REGISTER_OP("ShardingSparseFids")
    .Input("pb_input: variant")
    .Output("fid_list: N * int64")
    .Output("fid_offset: uint64")
    .Output("feature_offset: int32")
    .Output("nfl_offset: uint32")
    .Output("batch_size: int32")
    .Attr("ps_num: int")
    .Attr("feature_cfgs: string")
    .Attr("N: int")
    .Attr("unique: bool")
    .Attr("input_type: string")
    .Attr("parallel_flag: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      // fid_list
      int N = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));  // fid_list
      }
      ctx->set_output(N,
                      ctx->Vector(ctx->UnknownDim()));  // fid_offset
      ctx->set_output(N + 1,
                      ctx->Vector(ctx->UnknownDim()));  // feature_offset
      ctx->set_output(N + 2,
                      ctx->Vector(ctx->UnknownDim()));  // nfl_offset
      ctx->set_output(N + 3, ctx->Scalar());            // batch_size

      return Status::OK();
    });

REGISTER_OP("ShardingSparseFidsV2")
    .Input("pb_input: variant")
    .Output("fid_list: N * int64")
    .Output("fid_list_row_splits: N * int64")
    .Output("fid_offset: uint64")
    .Output("feature_offset: int32")
    .Output("nfl_offset: uint32")
    .Output("batch_size: int32")
    .Attr("ps_num: int")
    .Attr("feature_cfgs: string")
    .Attr("N: int")
    .Attr("unique: bool")
    .Attr("input_type: string")
    .Attr("parallel_flag: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      // fid_list
      int N = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));  // fid_list
        ctx->set_output(N + i,
                        ctx->Vector(ctx->UnknownDim()));  // fid_list_row_splits
      }
      N *= 2;
      ctx->set_output(N,
                      ctx->Vector(ctx->UnknownDim()));  // fid_offset
      ctx->set_output(N + 1,
                      ctx->Vector(ctx->UnknownDim()));  // feature_offset
      ctx->set_output(N + 2,
                      ctx->Vector(ctx->UnknownDim()));  // nfl_offset
      ctx->set_output(N + 3, ctx->Scalar());            // batch_size

      return Status::OK();
    });

// 与v2的区别在于fid_offset包含偏移都乘以了feature的dim

REGISTER_OP("ShardingSparseFidsV3")
    .Input("pb_input: variant")
    .Output("fid_list: N * int64")
    .Output("fid_list_row_splits: N * int64")
    .Output("fid_offset: uint64")
    .Output("feature_offset: int32")
    .Output("nfl_offset: uint32")
    .Output("batch_size: int32")
    .Attr("ps_num: int")
    .Attr("feature_cfgs: string")
    .Attr("N: int")
    .Attr("unique: bool")
    .Attr("input_type: string")
    .Attr("parallel_flag: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      // fid_list
      int N = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        ctx->set_output(i, ctx->Vector(ctx->UnknownDim()));  // fid_list
        ctx->set_output(N + i,
                        ctx->Vector(ctx->UnknownDim()));  // fid_list_row_splits
      }
      N *= 2;
      ctx->set_output(N,
                      ctx->Vector(ctx->UnknownDim()));  // fid_offset
      ctx->set_output(N + 1,
                      ctx->Vector(ctx->UnknownDim()));  // feature_offset
      ctx->set_output(N + 2,
                      ctx->Vector(ctx->UnknownDim()));  // nfl_offset
      ctx->set_output(N + 3, ctx->Scalar());            // batch_size
      int ps_num = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("ps_num", &ps_num));
      std::string feature_cfgs_str;
      TF_RETURN_IF_ERROR(ctx->GetAttr("feature_cfgs", &feature_cfgs_str));
      ::monolith::io::proto::FeatureConfigs feature_cfgs;
      CHECK(feature_cfgs.ParseFromString(feature_cfgs_str));
      std::unordered_set<std::string> table_name;
      for (auto &iter : feature_cfgs.feature_configs()) {
        table_name.insert(iter.second.table());
      }

      return Status::OK();
    });

// 为适配gpu emb，将多个tenor合并优化性能，并设配输入输出结构
// fid_list_row_splits [ps_shard, table, feature]
// fid_list_table_row_length [ps_shard, table]
//  [ps_shard]
// fid_list_emb_row_lenth [ps_shard, table] 对应emb在ps_shard的切分

REGISTER_OP("ShardingSparseFidsV4")
    .Input("pb_input: variant")
    .Output("fid_list: int64")
    .Output("fid_list_row_splits: int64")
    .Output("fid_list_table_row_length: int32")
    .Output("fid_list_shard_row_lenth: int32")
    .Output("fid_list_emb_row_lenth: int32")
    .Output("fid_offset: uint64")
    .Output("feature_offset: int32")
    .Output("nfl_offset: uint32")
    .Output("batch_size: int32")
    .Attr("ps_num: int")
    .Attr("feature_cfgs: string")
    .Attr("unique: bool")
    .Attr("input_type: string")
    .Attr("parallel_flag: int")
    .SetDoNotOptimize()  // Source dataset ops must disable constant folding.
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      int ps_num = 0;
      TF_RETURN_IF_ERROR(ctx->GetAttr("ps_num", &ps_num));
      std::string feature_cfgs_str;
      TF_RETURN_IF_ERROR(ctx->GetAttr("feature_cfgs", &feature_cfgs_str));
      ::monolith::io::proto::FeatureConfigs feature_cfgs;
      CHECK(feature_cfgs.ParseFromString(feature_cfgs_str));
      std::unordered_set<std::string> table_name;
      for (auto &iter : feature_cfgs.feature_configs()) {
        table_name.insert(iter.second.table());
      }
      // fid_list
      int N = 0;
      ctx->set_output(0, ctx->Vector(ctx->UnknownDim()));  // fid_list
      ctx->set_output(1,
                      ctx->Vector(ctx->UnknownDim()));  // fid_list_row_splits
      ctx->set_output(
          2,
          ctx->Vector(ps_num *
                      table_name.size()));  // fid_list_table_row_length
      ctx->set_output(3,
                      ctx->Vector(ps_num));  // fid_list_shard_row_lenth
      ctx->set_output(
          4,
          ctx->Vector(ps_num * table_name.size()));  // fid_list_emb_row_lenth

      N = 5;
      ctx->set_output(N,
                      ctx->Vector(ctx->UnknownDim()));  // fid_offset
      ctx->set_output(N + 1,
                      ctx->Vector(ctx->UnknownDim()));  // feature_offset
      ctx->set_output(N + 2,
                      ctx->Vector(ctx->UnknownDim()));  // nfl_offset
      ctx->set_output(N + 3, ctx->Scalar());            // batch_size

      return Status::OK();
    });

}  // namespace monolith_tf
}  // namespace tensorflow
