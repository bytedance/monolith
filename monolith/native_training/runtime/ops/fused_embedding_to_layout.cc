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

#include <cstring>
#include <tuple>  // for tuple
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

#include "idl/matrix/proto/example.pb.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/runtime/hash_table/optimizer/avx_utils.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace monolith_tf {

namespace {
using FidList = ::monolith::io::proto::FidList;
using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using FeatureConfigs = ::monolith::io::proto::FeatureConfigs;
using PoolingType = ::monolith::io::proto::PoolingType;
using OutType = ::monolith::io::proto::OutType;
using SliceConfig = ::monolith::io::proto::SliceConfig;
using LayoutShape = ::monolith::io::proto::TensorShape;
using OutConfig = ::monolith::io::proto::OutConfig;
using Feature = ::monolith::io::proto::Feature;
using FeatureListType = ::monolith::io::proto::FeatureListType;
using NamedFeatureList = ::monolith::io::proto::NamedFeatureList;

using MiniBatch = std::unordered_map<std::string, std::vector<const Feature *>>;
using Fid2EmbIdxMap = std::unordered_map<int64, std::pair<int, int>>;
using Fid2EmbMap = std::unordered_map<int64, const float *>;

static constexpr int NUM_LOCKS = 512;

using mutex = std::mutex;

struct PtrWrapper {
  const float *ptr;
  uint offset;
  uint count;
};
struct GroupA {
  GroupA(int dim1, int dim2) : b(dim1 * dim2, true), dim_1(dim1), dim_2(dim2) {}
  char *Get(int dim1, int dim2) { return &(b.at(dim1 * dim_2 + dim2)); }
  std::vector<char> b;
  int dim_1;
  int dim_2;
};

class Layout {
 public:
  Layout(const std::string &name, const OutConfig &out_conf)
      : name_(name), out_config_(out_conf) {}

  virtual ~Layout() {}

  virtual PtrWrapper GetSlice(int row_id, const SliceConfig &slice_conf) = 0;

  std::string GetKey(const SliceConfig &slice_conf) {
    return absl::StrCat(name_, "_", slice_conf.feature_name(), "_",
                        slice_conf.start(), "_", slice_conf.end());
  }

  const ::google::protobuf::RepeatedPtrField<SliceConfig> &GetSliceConfig() {
    return out_config_.slice_configs();
  }

  const OutType out_type() { return out_config_.out_type(); }

 protected:
  std::string name_;
  const OutConfig &out_config_;
};

class NoneLayout : public Layout {
 public:
  // op input TODO
  NoneLayout(const std::string &name, const OutConfig &out_conf,
             OpInputList &tensor_list, int &start_idx)
      : Layout(name, out_conf) {
    int offset = 0;
    CHECK(out_conf.slice_configs_size() == out_conf.shape_size());
    for (const SliceConfig &slice_conf : out_conf.slice_configs()) {
      slice_to_tensor_.insert(
          {GetKey(slice_conf), {&tensor_list[start_idx++], offset++}});
    }
  }

  // op output
  NoneLayout(const std::string &name, const OutConfig &out_conf,
             OpOutputList &tensor_list, int &start_idx)
      : Layout(name, out_conf) {
    int offset = 0;
    CHECK(out_conf.slice_configs_size() == out_conf.shape_size());
    for (const SliceConfig &slice_conf : out_conf.slice_configs()) {
      slice_to_tensor_.insert(
          {GetKey(slice_conf), {tensor_list[start_idx++], offset++}});
    }
  }

  virtual ~NoneLayout() {}

  PtrWrapper GetSlice(int row_id, const SliceConfig &slice_conf) override {
    std::string key = GetKey(slice_conf);
    auto it = slice_to_tensor_.find(key);
    if (it != slice_to_tensor_.end()) {
      auto layout_info = it->second;
      const LayoutShape shape = out_config_.shape(layout_info.second);
      if (slice_conf.pooling_type() == PoolingType::FIRSTN) {
        CHECK_EQ(shape.dims_size(), 3);
        // none seq [batch_size, max_seq_len, num_dim]
        const auto tensor = layout_info.first->tensor<float, 3>();
        return PtrWrapper{&tensor(row_id, 0, 0), shape.dims(1) * shape.dims(2),
                          layout_info.first->NumElements()};

      } else {
        CHECK_EQ(shape.dims_size(), 2);  // none [batch_size, num_dim]
        const auto mat = layout_info.first->matrix<float>();
        return PtrWrapper{&mat(row_id, 0), shape.dims(1),
                          layout_info.first->NumElements()};
      }
    }
  }

 private:
  std::unordered_map<std::string, std::pair<const Tensor *, const int>>
      slice_to_tensor_;
};

class DefaultLayout : public Layout {
 public:
  DefaultLayout(const std::string &name, const OutConfig &out_conf,
                OpInputList &tensor_list, int &start_idx)
      : Layout(name, out_conf) {
    int offset = 0;
    CHECK_EQ(out_conf.shape_size(), 1);
    CHECK_NE(out_conf.out_type(), OutType::NONE);

    for (const SliceConfig slice_conf : out_conf.slice_configs()) {
      slice_to_tensor_.insert(
          {GetKey(slice_conf), {&tensor_list[start_idx], offset}});
      if (out_conf.out_type() == OutType::STACK) {
        offset += 1;
      } else if (out_conf.out_type() == OutType::CONCAT) {
        offset += slice_conf.end() - slice_conf.start();
      } else {
        CHECK(out_conf.out_type() == OutType::ADDN);
      }
    }

    start_idx++;
  }

  DefaultLayout(const std::string &name, const OutConfig &out_conf,
                OpOutputList &tensor_list, int &start_idx)
      : Layout(name, out_conf) {
    int offset = 0;
    CHECK_EQ(out_conf.shape_size(), 1);
    CHECK_NE(out_conf.out_type(), OutType::NONE);

    for (const SliceConfig slice_conf : out_conf.slice_configs()) {
      slice_to_tensor_.insert(
          {GetKey(slice_conf), {tensor_list[start_idx], offset}});
      if (out_conf.out_type() == OutType::STACK) {
        offset += 1;
      } else if (out_conf.out_type() == OutType::CONCAT) {
        offset += slice_conf.end() - slice_conf.start();
      } else {
        CHECK(out_conf.out_type() == OutType::ADDN);
      }
    }

    start_idx++;
  }

  virtual ~DefaultLayout() {}

  PtrWrapper GetSlice(int row_id, const SliceConfig &slice_conf) override {
    std::string key = GetKey(slice_conf);
    auto it = slice_to_tensor_.find(key);
    if (it != slice_to_tensor_.end()) {
      auto layout_info = it->second;
      CHECK_EQ(out_config_.shape_size(), 1);
      const LayoutShape shape = out_config_.shape(0);

      // TODO(zhangru): support concat/stack seq
      if (slice_conf.pooling_type() == PoolingType::FIRSTN) {
        CHECK(shape.dims_size() > 2 && shape.dims_size() < 5);
        if (shape.dims_size() == 3) {
          // concat [batch_size, max_seq_len, num_dims];
          // add_n [batch_size, , num_dim];
          const auto tensor = layout_info.first->tensor<float, 3>();
          return PtrWrapper{&tensor(row_id, 0, layout_info.second),
                            shape.dims(1) * shape.dims(2),
                            layout_info.first->NumElements()};
        } else {  // if (shape.dims_size() == 4) {
          // stack [batch_size, features_size, max_seq_len , num_dim];
          const auto tensor = layout_info.first->tensor<float, 4>();
          return PtrWrapper{&tensor(row_id, 0, 0, layout_info.second),
                            shape.dims(1) * shape.dims(2) * shape.dims(3),
                            layout_info.first->NumElements()};
        }
      } else {
        CHECK(shape.dims_size() > 1 && shape.dims_size() < 4);
        if (shape.dims_size() == 2) {
          // concat [batch_size, num_dims];
          // add_n [batch_size, num_dim];
          const auto mat = layout_info.first->matrix<float>();
          return PtrWrapper{&mat(row_id, layout_info.second), shape.dims(1),
                            layout_info.first->NumElements()};
        } else {  // if (shape.dims_size() == 3) {
          // stack [batch_size, features_size , num_dim];
          const auto tensor = layout_info.first->tensor<float, 3>();
          return PtrWrapper{&tensor(row_id, layout_info.second, 0),
                            shape.dims(1) * shape.dims(2),
                            layout_info.first->NumElements()};
        }
      }
    }
  }

 private:
  std::unordered_map<std::string, std::pair<const Tensor *, const int>>
      slice_to_tensor_;
};

class MonolithEmbeddingToLayoutBase : public OpKernel {
 public:
  explicit MonolithEmbeddingToLayoutBase(OpKernelConstruction *ctx, int version)
      : OpKernel(ctx), version_(version) {
    std::string serialized;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_cfgs", &serialized));
    OP_REQUIRES(
        ctx, feature_cfgs_.ParseFromArray(serialized.data(), serialized.size()),
        errors::FailedPrecondition("Failed to parse the feature_cfgs_."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variant_type", &variant_type_));
    if (version_ == 2) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ps_num", &ps_num_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("parallel_flag", &parallel_flag_));
    }

    // set max_sequence_length/pooling_type/slice_idx/feature_idx here:
    // use the index in the sorted feature_names_used as feature_idx.
    const auto &feature_names_used = feature_cfgs_.feature_configs();
    std::vector<std::string> feature_names;
    std::map<std::string, std::map<std::string, int>> table_feature_dim_map;
    for (const auto &feature_conf_pair : feature_names_used) {
      feature_names.push_back(feature_conf_pair.first);
      int dims_sum = 0;
      for (size_t slice_idx = 0;
           slice_idx < feature_conf_pair.second.slice_dims_size();
           slice_idx++) {
        dims_sum += feature_conf_pair.second.slice_dims(slice_idx);
      }
      table_feature_dim_map[feature_conf_pair.second.table()]
                           [feature_conf_pair.first] = dims_sum;
    }
    std::sort(feature_names.begin(), feature_names.end());

    {
      table_feature_dim_.resize(table_feature_dim_map.size());
      int i = 0;
      // table_feature_dim_map is map, already sort
      for (auto &iter : table_feature_dim_map) {
        auto &table_name = iter.first;
        auto &record_dims = table_feature_dim_[i];
        std::vector<std::string> feature_name_tmp;
        auto &feature_dim_map = iter.second;
        for (auto &sub_iter : feature_dim_map) {
          feature_name_tmp.push_back(sub_iter.first);
        }
        std::sort(feature_name_tmp.begin(), feature_name_tmp.end());
        record_dims.resize(feature_name_tmp.size());
        for (size_t j = 0; j < feature_name_tmp.size(); ++j) {
          record_dims[j] = feature_dim_map[feature_name_tmp[j]];
        }
        ++i;
      }
    }
    std::vector<std::unordered_map<int, int>>
        slice_idx_per_feature;  // feature_index: {start: slice_idx}
    slice_idx_per_feature.resize(feature_names_used.size());
    for (size_t feature_idx = 0; feature_idx < feature_names.size();
         feature_idx++) {
      std::unordered_map<int, int> start2slice_idx;
      const auto &feature_name = feature_names[feature_idx];
      const auto &feat_conf = feature_names_used.at(feature_name);
      int slice_prefix_sum_ = 0;
      for (size_t slice_idx = 0; slice_idx < feat_conf.slice_dims_size();
           slice_idx++) {
        start2slice_idx[slice_prefix_sum_] = slice_idx;
        slice_prefix_sum_ += feat_conf.slice_dims(slice_idx);
      }
      max_slice_num_ = std::max(max_slice_num_, feat_conf.slice_dims_size());
      slice_idx_per_feature[feature_idx] = start2slice_idx;
    }

    auto *out_configs = feature_cfgs_.mutable_out_configs();
    for (auto &pair : *out_configs) {
      layout_names_.push_back(pair.first);
      for (auto &slice_config : *pair.second.mutable_slice_configs()) {
        const auto &feature_name = slice_config.feature_name();
        const auto &feat_conf = feature_names_used.at(feature_name);
        slice_config.set_max_sequence_length(feat_conf.max_sequence_length());
        slice_config.set_pooling_type(feat_conf.pooling_type());
        auto it =
            std::find(feature_names.begin(), feature_names.end(), feature_name);
        if (it != feature_names.end()) {
          int feature_idx = it - feature_names.begin();
          slice_config.set_feature_idx(feature_idx);
          slice_config.set_slice_idx(
              slice_idx_per_feature[feature_idx][slice_config.start()]);
        }
      }
    }
    std::sort(layout_names_.begin(), layout_names_.end());
  }

 private:
  std::string variant_type_;
  FeatureConfigs feature_cfgs_;
  std::vector<std::string> layout_names_;
  int max_slice_num_ = 0;
  std::vector<std::vector<int>> table_feature_dim_;
  int ps_num_ = 0;
  int parallel_flag_ = 0;
  int version_ = 1;

 protected:
  int GetMaxSliceNum() { return max_slice_num_; }
  const std::string &GetVariantType() { return variant_type_; }
  const std::vector<std::string> &GetLayoutNames() { return layout_names_; }
  const FeatureConfigs &GetFeatureCfgs() { return feature_cfgs_; }
  int GetPsNum() { return ps_num_; }
  int GetParallelFlag() { return parallel_flag_; }
  int GetVersion() { return version_; }
  const std::vector<std::vector<int>> &GetFeatureInTableDim() {
    return table_feature_dim_;
  }

  void ParseFidOffset(const uint64 &fids_offset, int32 *index1, int32 *index2) {
    *index1 = fids_offset >> 32;
    *index2 = fids_offset << 32 >> 32;
  }

  void ParseNflOffset(const uint32 &nfl_offset_encode, bool *is_shared,
                      int *nfl_offset) {
    *is_shared = nfl_offset_encode >> 31;
    *nfl_offset = nfl_offset_encode & 0x7fffffff;
  }

  void GetFeatureInfo(const int64 &nfl_idx,
                      const typename TTypes<uint32>::ConstFlat &nfl_offset_vec,
                      const int &total_nfl_num, const int &total_feature_num,
                      bool &is_shared, int &nfl_offset, int &feature_num) {
    ParseNflOffset(nfl_offset_vec(nfl_idx), &is_shared, &nfl_offset);
    if (nfl_idx < total_nfl_num - 1) {
      bool is_shared_later;
      int nfl_offset_later;
      ParseNflOffset(nfl_offset_vec(nfl_idx + 1), &is_shared_later,
                     &nfl_offset_later);
      feature_num = nfl_offset_later - nfl_offset;
    } else {
      feature_num = total_feature_num - nfl_offset;
    }
  }

  template <class TInit>
  void OptimizedSumpooling(const float *src, const int &dim_num, TInit *init,
                           float *dst, mutex *one_mutex = nullptr) {
    if (one_mutex) {
      one_mutex->lock();
    }
    if (*init) {
      std::memcpy(dst, src, dim_num * sizeof(float));
      *init = false;
    } else {
      ::monolith::hash_table::ReduceSum(src, dst, dst, dim_num);
    }
    if (one_mutex) {
      one_mutex->unlock();
    }
  }
};

class MonolithEmbeddingToLayoutOp : public MonolithEmbeddingToLayoutBase {
 public:
  explicit MonolithEmbeddingToLayoutOp(OpKernelConstruction *ctx,
                                       int version = 1)
      : MonolithEmbeddingToLayoutBase(ctx, version) {}

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    OpInputList embeddings_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("embeddings_list", &embeddings_list));

    const Tensor *fids_offset_input;
    OP_REQUIRES_OK(ctx, ctx->input("fid_offset", &fids_offset_input));

    const Tensor *feature_offset_input;
    OP_REQUIRES_OK(ctx, ctx->input("feature_offset", &feature_offset_input));

    const Tensor *nfl_offset_input;
    OP_REQUIRES_OK(ctx, ctx->input("nfl_offset", &nfl_offset_input));

    const Tensor *batch_size_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch_size", &batch_size_tensor));

    const auto &fids_offset_vec = fids_offset_input->flat<uint64>();
    const auto &feature_offset_vec = feature_offset_input->flat<int32>();
    const auto nfl_offset_vec = nfl_offset_input->flat<uint32>();
    const int32 batch_size = batch_size_tensor->scalar<int32>()();
    // value = vector_tensor->flat<int64>().data();
    int total_nfl_num = nfl_offset_input->dim_size(0);
    int total_feature_num = feature_offset_input->dim_size(0);
    int total_fid_num = fids_offset_input->dim_size(0);
    OpOutputList layout_tensor_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &layout_tensor_list));

    std::vector<PtrWrapper> embeddings_data;
    if (GetVersion() == 2) {
      OpInputList fid_list_row_split;
      OP_REQUIRES_OK(
          ctx, ctx->input_list("fid_list_row_split", &fid_list_row_split));

      int ps_num = GetPsNum();
      const std::vector<std::vector<int>> &table_feature_dim =
          GetFeatureInTableDim();
      embeddings_data.reserve(GetFeatureCfgs().feature_configs_size() * ps_num);
      CHECK_EQ(embeddings_list.size(), ps_num * table_feature_dim.size());
      CHECK_EQ(embeddings_list.size(), fid_list_row_split.size());
      for (size_t table_i = 0; table_i < table_feature_dim.size(); ++table_i) {
        auto &feature_dims = table_feature_dim[table_i];
        for (size_t ps_i = 0; ps_i < ps_num; ++ps_i) {
          int emb_index = table_i * ps_num + ps_i;
          auto embeddings_flat = embeddings_list[emb_index].flat<float>();
          auto embeddings_size = embeddings_flat.size();
          auto embeddings_ptr = embeddings_flat.data();

          auto fid_list_row_split_flat =
              fid_list_row_split[emb_index].flat<int64_t>();

          CHECK_EQ(static_cast<int>(feature_dims.size() + 1),
                   fid_list_row_split_flat.size());
          int pre_offset = 0;
          int pre_emb_offset = 0;
          for (size_t feature_i = 0; feature_i < feature_dims.size();
               ++feature_i) {
            int dim = feature_dims[feature_i];
            int offset = fid_list_row_split_flat(feature_i + 1);
            int fid_count = (offset - pre_offset);
            embeddings_data.push_back(PtrWrapper{
                embeddings_ptr + pre_emb_offset, dim, fid_count * dim});
            pre_offset = offset;
            pre_emb_offset += fid_count * dim;
            CHECK(pre_emb_offset <= embeddings_size);
          }
        }
      }
    } else {
      embeddings_data.reserve(embeddings_list.size());
      for (size_t i = 0; i < embeddings_list.size(); ++i) {
        const auto &embeddings_mat_ptr_ =
            embeddings_list[i].flat<float>().data();
        embeddings_data.push_back(PtrWrapper(
            {embeddings_mat_ptr_, embeddings_list[i].dim_size(1),
             embeddings_list[i].dim_size(0) * embeddings_list[i].dim_size(1)}));
      }
    }

    {
      auto activity = std::make_unique<profiler::TraceMe>(
          []() { return "AllocateTensors"; });
      int offset = 0;
      const auto &out_configs = GetFeatureCfgs().out_configs();
      for (const auto &layout_name : GetLayoutNames()) {
        const OutConfig &out_conf = out_configs.at(layout_name);
        for (const auto shape : out_conf.shape()) {
          Tensor *tensor;
          TensorShape tensor_shape;
          for (size_t i = 0; i < shape.dims_size(); ++i) {
            if (i == 0) {
              tensor_shape.AddDim(shape.dims(i) == -1 ? batch_size
                                                      : shape.dims(i));
            } else {
              CHECK_GT(shape.dims(i), 0);
              tensor_shape.AddDim(shape.dims(i));
            }
          }
          OP_REQUIRES_OK(ctx, layout_tensor_list.allocate(
                                  offset++, tensor_shape, &tensor));
          tensor->flat<float>().setZero();
        }
      }
    }

    int offset = 0;
    std::vector<std::shared_ptr<Layout>> layouts;
    {
      auto activity =
          std::make_unique<profiler::TraceMe>([]() { return "CreateLayout"; });
      for (const auto &layout_name : GetLayoutNames()) {
        const OutConfig &out_conf =
            GetFeatureCfgs().out_configs().at(layout_name);
        switch (out_conf.out_type()) {
          case OutType::NONE:
            layouts.push_back(std::make_shared<NoneLayout>(
                layout_name, out_conf, layout_tensor_list, offset));
            break;
          default:
            layouts.push_back(std::make_shared<DefaultLayout>(
                layout_name, out_conf, layout_tensor_list, offset));
            break;
        }
      }
    }

    auto gather_emb_fn = [&, this](int start, int end) {
      for (int64 para_i = start; para_i < end; ++para_i) {
        auto &layout = layouts.at(para_i);
        // CHECK(end - start == 1);
        const ::google::protobuf::RepeatedPtrField<SliceConfig>
            &layout_slice_configs = layout->GetSliceConfig();
        for (uint slice_conf_i = 0; slice_conf_i < layout_slice_configs.size();
             ++slice_conf_i) {
          const SliceConfig &slice_conf = layout_slice_configs[slice_conf_i];
          int dim_num = slice_conf.end() - slice_conf.start();
          PtrWrapper ptr_info = layout->GetSlice(0, slice_conf);
          const int64 &nfl_idx = slice_conf.feature_idx();
          bool is_shared;
          int nfl_offset, feature_num;
          GetFeatureInfo(nfl_idx, nfl_offset_vec, total_nfl_num,
                         total_feature_num, is_shared, nfl_offset, feature_num);
          if (!feature_num) continue;  // nfl exits

          std::unique_ptr<float> tmp;
          if (layout->out_type() == OutType::ADDN &&
              (slice_conf.pooling_type() == PoolingType::MEAN || is_shared)) {
            tmp.reset(new float[dim_num]());
            // std::memset(tmp, 0, sizeof(float) * dim_num);
            // init = true;first time copy, no need memset
          }
          int feature_idx = nfl_offset + 0;
          for (size_t index = 0; index < batch_size; ++index) {
            int temp_offset = index * ptr_info.offset;
            if (slice_conf.pooling_type() == PoolingType::FIRSTN) {
              CHECK(temp_offset + slice_conf.max_sequence_length() * dim_num <=
                    ptr_info.count);
            } else {
              CHECK(temp_offset + dim_num <= ptr_info.count);
            }
            if (!is_shared || index == 0) {
              bool init = (layout->out_type() != OutType::ADDN) || tmp;
              GatherEmb(feature_idx, slice_conf.max_sequence_length(),
                        slice_conf.pooling_type(), slice_conf, embeddings_data,
                        feature_offset_vec, fids_offset_vec, total_feature_num,
                        total_fid_num,
                        const_cast<float *>(tmp ? tmp.get()
                                                : ptr_info.ptr + temp_offset),
                        &init);
              if (tmp) {
                bool init_tmp = (slice_conf_i == 0);
                OptimizedSumpooling(
                    tmp.get(), dim_num, &init_tmp,
                    const_cast<float *>(ptr_info.ptr + temp_offset));
              }
              feature_idx++;
            } else {
              if (tmp) {
                bool init_tmp = (slice_conf_i == 0);  // && index == 0
                OptimizedSumpooling(
                    tmp.get(), dim_num, &init_tmp,
                    const_cast<float *>(ptr_info.ptr + temp_offset));
              } else {
                CopyEmb(slice_conf, slice_conf.max_sequence_length(),
                        slice_conf.pooling_type(),
                        const_cast<float *>(ptr_info.ptr),
                        const_cast<float *>(ptr_info.ptr + temp_offset));
              }
            }
          }
        }
      }
    };

    {
      auto activity =
          std::make_unique<profiler::TraceMe>([]() { return "GatherEmbFn"; });
      int parallel_flag = GetParallelFlag();
      if (parallel_flag == 0) {
        for (int i = 0; i < layouts.size(); ++i) {
          gather_emb_fn(i, i + 1);
        }
      } else {
        auto workers = ctx->device()->tensorflow_cpu_worker_threads()->workers;
        workers->ParallelFor(
            layouts.size(),
            thread::ThreadPool::SchedulingParams(
                thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
                absl::nullopt,
                1),  // block_size
            gather_emb_fn);
      }
    }
  }

  void GatherEmb(const int &feature_idx, const int &max_sequence_length,
                 const PoolingType &pooling_type, const SliceConfig &slice_conf,
                 const std::vector<PtrWrapper> &embeddings_data,
                 const typename TTypes<int32>::ConstFlat &feature_offset_vec,
                 const typename TTypes<uint64>::ConstFlat &fids_offset_vec,
                 const int &total_feature_num, const int &total_fid_num,
                 float *out_ptr, bool *init) {
    int fid_num = (feature_idx < total_feature_num - 1)
                      ? feature_offset_vec(feature_idx + 1) -
                            feature_offset_vec(feature_idx)
                      : total_fid_num - feature_offset_vec(feature_idx);
    if (fid_num == 0) return;
    const auto &start_fid_offset_idx = feature_offset_vec(feature_idx);

    int seq_idx = 0;
    int dims = slice_conf.end() - slice_conf.start();

    auto gather_emb_fid = [&embeddings_data, &out_ptr, &pooling_type,
                           &max_sequence_length, &seq_idx, &dims, &init,
                           &slice_conf, fids_offset_vec, &feature_idx,
                           this](const auto &fid_offset_idx) {
      int32 index1, index2;
      const uint64 &fids_offset = fids_offset_vec(fid_offset_idx);
      ParseFidOffset(fids_offset, &index1, &index2);
      const auto &ptr_info = embeddings_data.at(index1);
      int tmp_offset = index2 * ptr_info.offset + slice_conf.start();
      CHECK(tmp_offset + dims <= ptr_info.count);
      float *src = const_cast<float *>(ptr_info.ptr) + tmp_offset;
      switch (pooling_type) {
        case PoolingType::SUM:
        case PoolingType::MEAN:
          OptimizedSumpooling(src, dims, init, out_ptr);
          break;
        case PoolingType::FIRSTN:
          if (seq_idx < max_sequence_length) {
            std::memcpy(out_ptr + seq_idx * dims, src, dims * sizeof(float));
          }
          seq_idx++;
          break;
        default:
          break;
      }
    };

    for (int fid_idx = 0; fid_idx < fid_num; fid_idx++) {
      gather_emb_fid(start_fid_offset_idx + fid_idx);
    }
    if (pooling_type == PoolingType::MEAN) {
      for (int mi = 0; mi < dims; mi++) {
        *(out_ptr + mi) = *(out_ptr + mi) / fid_num;
      }
    }
  }

  void CopyEmb(const SliceConfig &slice_conf, const int &max_sequence_length,
               const PoolingType &pooling_type, const float *in_ptr,
               float *out_ptr) {
    int dims = slice_conf.end() - slice_conf.start();
    switch (pooling_type) {
      case PoolingType::SUM:
      case PoolingType::MEAN:
        std::memcpy(out_ptr, in_ptr, dims * sizeof(float));
        break;
      case PoolingType::FIRSTN:
        std::memcpy(out_ptr, in_ptr,
                    dims * max_sequence_length * sizeof(float));
        break;
      default:
        break;
    }
  }
};

class MonolithEmbeddingToLayoutOpV2 : public MonolithEmbeddingToLayoutOp {
 public:
  explicit MonolithEmbeddingToLayoutOpV2(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOp(ctx, 2) {}
};

class MonolithEmbeddingToLayoutGradOp : public MonolithEmbeddingToLayoutBase {
 public:
  explicit MonolithEmbeddingToLayoutGradOp(OpKernelConstruction *ctx,
                                           int version = 1)
      : MonolithEmbeddingToLayoutBase(ctx, version) {}

  void Compute(OpKernelContext *ctx) override {
    // Grab the input tensor
    OpInputList embeddings_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("embeddings_list", &embeddings_list));

    const Tensor *fids_offset_input;
    OP_REQUIRES_OK(ctx, ctx->input("fid_offset", &fids_offset_input));

    const Tensor *feature_offset_input;
    OP_REQUIRES_OK(ctx, ctx->input("feature_offset", &feature_offset_input));

    const Tensor *nfl_offset_input;
    OP_REQUIRES_OK(ctx, ctx->input("nfl_offset", &nfl_offset_input));

    const Tensor *batch_size_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch_size", &batch_size_tensor));

    OpInputList tensors_grad;
    OP_REQUIRES_OK(ctx, ctx->input_list("tensors_grad", &tensors_grad));

    const auto &fids_offset_vec = fids_offset_input->flat<uint64>();
    const auto &feature_offset_vec = feature_offset_input->flat<int32>();
    const auto nfl_offset_vec = nfl_offset_input->flat<uint32>();
    const int32 batch_size = batch_size_tensor->scalar<int32>()();
    int total_nfl_num = nfl_offset_input->dim_size(0);
    int total_feature_num = feature_offset_input->dim_size(0);
    int total_fid_num = fids_offset_input->dim_size(0);

    std::vector<std::pair<int, int>> ufid_grads_info;

    OpOutputList embeddings_grad_list;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("embeddings_grad_list", &embeddings_grad_list));
    std::vector<PtrWrapper> embeddings_grads_data;
    int init_counter = 0;
    if (GetVersion() == 2) {
      OpInputList fid_list_row_split;
      OP_REQUIRES_OK(
          ctx, ctx->input_list("fid_list_row_split", &fid_list_row_split));
      int ps_num = GetPsNum();
      const std::vector<std::vector<int>> &table_feature_dim =
          GetFeatureInTableDim();
      ufid_grads_info.reserve(GetFeatureCfgs().feature_configs_size() * ps_num);
      embeddings_grads_data.reserve(GetFeatureCfgs().feature_configs_size() *
                                    ps_num);
      CHECK_EQ(embeddings_list.size(), ps_num * table_feature_dim.size());
      CHECK_EQ(embeddings_list.size(), fid_list_row_split.size());
      for (size_t table_i = 0; table_i < table_feature_dim.size(); ++table_i) {
        auto &feature_dims = table_feature_dim[table_i];
        for (size_t ps_i = 0; ps_i < ps_num; ++ps_i) {
          int emb_index = table_i * ps_num + ps_i;
          Tensor *tensor;
          OP_REQUIRES_OK(
              ctx, embeddings_grad_list.allocate(
                       emb_index, embeddings_list[emb_index].shape(), &tensor));
          tensor->flat<float>().setConstant(0);

          auto embeddings_grad_flat =
              embeddings_grad_list[emb_index]->flat<float>();
          auto embeddings_grad_size = embeddings_grad_flat.size();
          auto embeddings_grad_ptr = embeddings_grad_flat.data();

          auto fid_list_row_split_flat =
              fid_list_row_split[emb_index].flat<int64_t>();

          CHECK_EQ(static_cast<int>(feature_dims.size() + 1),
                   fid_list_row_split_flat.size());

          int pre_offset = 0;
          int pre_emb_offset = 0;
          for (size_t feature_i = 0; feature_i < feature_dims.size();
               ++feature_i) {
            int dim = feature_dims[feature_i];
            int offset = fid_list_row_split_flat(feature_i + 1);
            int fid_count = (offset - pre_offset);
            embeddings_grads_data.push_back(PtrWrapper{
                embeddings_grad_ptr + pre_emb_offset, dim, fid_count * dim});
            ufid_grads_info.emplace_back(
                std::make_pair(init_counter, fid_count));
            pre_offset = offset;
            pre_emb_offset += fid_count * dim;
            CHECK(pre_emb_offset <= embeddings_grad_size);
            init_counter += fid_count;
          }
        }
      }
    } else {
      embeddings_grads_data.reserve(embeddings_list.size());
      for (size_t i = 0; i < embeddings_list.size(); ++i) {
        Tensor *tensor;
        OP_REQUIRES_OK(ctx, embeddings_grad_list.allocate(
                                i, embeddings_list[i].shape(), &tensor));
        tensor->flat<float>().setConstant(0);
        int dim = embeddings_list[i].dim_size(1);
        int fid_count = embeddings_list[i].dim_size(0);
        embeddings_grads_data.push_back(
            PtrWrapper{embeddings_grad_list[i]->flat<float>().data(), dim,
                       fid_count * dim});
        ufid_grads_info.emplace_back(std::make_pair(init_counter, fid_count));
        init_counter += fid_count;
      }
    }

    // wrapper of bool for avoid :
    // invalid initialization of non-const reference of type 'bool&' from an
    // rvalue of type 'bool'
    GroupA init(init_counter, GetMaxSliceNum());

    int offset = 0;
    std::vector<std::shared_ptr<Layout>> layouts;
    for (const auto &layout_name : GetLayoutNames()) {
      const OutConfig &out_conf =
          GetFeatureCfgs().out_configs().at(layout_name);

      switch (out_conf.out_type()) {
        case OutType::NONE:
          layouts.push_back(std::make_shared<NoneLayout>(layout_name, out_conf,
                                                         tensors_grad, offset));
          break;
        default:
          layouts.push_back(std::make_shared<DefaultLayout>(
              layout_name, out_conf, tensors_grad, offset));
          break;
      }
    }

    int parallel_flag = GetParallelFlag();

    // mutex/init per op compute, because there are several(>1) grad op
    // calculated togather.
    std::unique_ptr<mutex[]> mutex_list;
    if (parallel_flag != 0) {
      mutex_list = std::make_unique<mutex[]>(NUM_LOCKS);
    }

    auto scatter_grad_fn = [&, this](int start, int end) {
      for (int64 para_i = start; para_i < end; ++para_i) {
        auto &layout = layouts.at(para_i);
        // CHECK(end - start == 1);
        const ::google::protobuf::RepeatedPtrField<SliceConfig>
            &layout_slice_configs = layout->GetSliceConfig();
        for (const SliceConfig &slice_conf : layout_slice_configs) {
          int dim_num = slice_conf.end() - slice_conf.start();
          PtrWrapper ptr_info = layout->GetSlice(0, slice_conf);
          const int64 &nfl_idx = slice_conf.feature_idx();
          bool is_shared;
          int nfl_offset, feature_num;
          GetFeatureInfo(nfl_idx, nfl_offset_vec, total_nfl_num,
                         total_feature_num, is_shared, nfl_offset, feature_num);
          if (!feature_num) continue;  // nfl exits
          int feature_idx = nfl_offset + 0;
          for (size_t index = 0; index < batch_size; ++index) {
            int temp_offset = index * ptr_info.offset;
            if (slice_conf.pooling_type() == PoolingType::FIRSTN) {
              CHECK(temp_offset + slice_conf.max_sequence_length() * dim_num <=
                    ptr_info.count);
            } else {
              CHECK(temp_offset + dim_num <= ptr_info.count);
            }
            ScatterGrad(feature_idx, slice_conf.max_sequence_length(),
                        slice_conf.pooling_type(), ptr_info.ptr + temp_offset,
                        slice_conf, embeddings_grads_data, feature_offset_vec,
                        fids_offset_vec, total_feature_num, total_fid_num,
                        ufid_grads_info,
                        (mutex_list ? mutex_list.get() : nullptr), &init);
            if (!is_shared) {  // train don't have shared feature
              feature_idx++;
            }
          }
        }
      }
    };

    if (parallel_flag == 0) {
      for (int i = 0; i < layouts.size(); ++i) {
        scatter_grad_fn(i, i + 1);
      }
    } else {
      auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
      worker_threads->workers->ParallelFor(
          layouts.size(),
          thread::ThreadPool::SchedulingParams(
              thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
              absl::nullopt,
              1),  // block_size
          scatter_grad_fn);
    }
  }

  void ScatterGrad(const int &feature_idx, const int &max_sequence_length,
                   const PoolingType &pooling_type, const float *grad_ptr,
                   const SliceConfig &slice_conf,
                   const std::vector<PtrWrapper> &embeddings_grads_data,
                   const typename TTypes<int32>::ConstFlat &feature_offset_vec,
                   const typename TTypes<uint64>::ConstFlat &fids_offset_vec,
                   const int &total_feature_num, const int &total_fid_num,
                   const std::vector<std::pair<int, int>> &ufid_grads_info,
                   mutex *mutex_list, GroupA *init) {
    int fid_num = (feature_idx < total_feature_num - 1)
                      ? feature_offset_vec(feature_idx + 1) -
                            feature_offset_vec(feature_idx)
                      : total_fid_num - feature_offset_vec(feature_idx);
    if (fid_num == 0) return;
    const auto &start_fid_offset_idx = feature_offset_vec(feature_idx);
    int seq_idx = 0;
    int dims = slice_conf.end() - slice_conf.start();
    auto scatter_grad_fid = [&embeddings_grads_data, &grad_ptr, &pooling_type,
                             &max_sequence_length, &seq_idx, &dims, &init,
                             &slice_conf, this](int real_ufid_idx, float *dst,
                                                const int &fid_num,
                                                mutex *one_mutex) {
      // const std::lock_guard<std::mutex> lock(*one_mutex);
      //   embeddings_grads_data: fid grad data
      auto init_p = init->Get(real_ufid_idx, slice_conf.slice_idx());
      // bool init_p = false;
      switch (pooling_type) {
        case PoolingType::SUM: {
          OptimizedSumpooling(grad_ptr, dims, init_p, dst, one_mutex);
          break;
        }
        case PoolingType::MEAN: {
          // Maybe there exists one slice used twice
          std::vector<float> tmp(dims);
          for (int mi = 0; mi < dims; mi++) {
            tmp[mi] = *(grad_ptr + mi) / fid_num;
          }
          OptimizedSumpooling(grad_ptr, dims, init_p, dst, one_mutex);
          break;
        }
        case PoolingType::FIRSTN: {
          if (seq_idx < max_sequence_length) {
            OptimizedSumpooling(grad_ptr + seq_idx * dims, dims, init_p, dst,
                                one_mutex);
          }
          seq_idx++;
          break;
        }

        default:
          break;
      }
    };

    for (int fid_idx = 0; fid_idx < fid_num; fid_idx++) {
      int32 fid_offset_idx = start_fid_offset_idx + fid_idx;
      int32 index1, index2;
      const uint64 &fids_offset = fids_offset_vec(fid_offset_idx);
      ParseFidOffset(fids_offset, &index1, &index2);
      // embeddings_grads_data: fid grad data
      const auto &ptr_info = embeddings_grads_data.at(index1);
      int tmp_offset = index2 * ptr_info.offset + slice_conf.start();
      CHECK(tmp_offset + dims <= ptr_info.count);
      float *dst = const_cast<float *>(ptr_info.ptr) + tmp_offset;
      const auto &fid_info = ufid_grads_info.at(index1);
      CHECK(index2 < fid_info.second);
      int real_ufid_idx = fid_info.first + index2;
      mutex *one_mutex = nullptr;
      if (mutex_list) {
        // [NOTE]: lock per unique fid not per fid_offset.
        float *fid_dst =
            const_cast<float *>(ptr_info.ptr) + index2 * ptr_info.offset;
        int mutex_idx = real_ufid_idx % NUM_LOCKS;
        one_mutex = mutex_list + mutex_idx;
      }
      scatter_grad_fid(real_ufid_idx, dst, fid_num, one_mutex);
    }
  }
};

class MonolithEmbeddingToLayoutGradOpV2
    : public MonolithEmbeddingToLayoutGradOp {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV2(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOp(ctx, 2) {}
};

REGISTER_OP("MonolithEmbeddingToLayout")
    .Input("embeddings_list: M * float")
    .Input("fid_offset: uint64")
    .Input("feature_offset: int32")
    .Input("nfl_offset: uint32")
    .Input("batch_size: int32")
    .Output("tensors: num_out * float")
    .Attr("M: int")  // num of fids_list (shard x subtable)
    .Attr("num_out: int")
    .Attr("variant_type: string")
    .Attr("feature_cfgs: string")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::string serialized;
      TF_RETURN_IF_ERROR(ctx->GetAttr("feature_cfgs", &serialized));
      FeatureConfigs feature_cfgs;
      CHECK(feature_cfgs.ParseFromArray(serialized.data(), serialized.size()));

      std::vector<std::string> layout_names;
      const auto &out_configs = feature_cfgs.out_configs();
      for (const auto &pair : out_configs) {
        layout_names.push_back(pair.first);
      }
      std::sort(layout_names.begin(), layout_names.end());

      std::vector<shape_inference::ShapeHandle> tensors_shape;
      for (const auto &layout_name : layout_names) {
        const OutConfig &out_conf = out_configs.at(layout_name);
        for (const auto shape : out_conf.shape()) {
          std::vector<shape_inference::DimensionHandle> dims;
          for (size_t i = 0; i < shape.dims_size(); ++i) {
            if (i == 0) {
              dims.push_back(ctx->UnknownDim());
            } else {
              CHECK_GT(shape.dims(i), 0);
              dims.push_back(ctx->MakeDim(shape.dims(i)));
            }
          }
          tensors_shape.push_back(ctx->MakeShape(dims));
        }
      }
      TF_RETURN_IF_ERROR(ctx->set_output("tensors", tensors_shape));
      return Status::OK();
    });

REGISTER_OP("MonolithEmbeddingToLayoutGrad")
    .Input("embeddings_list: M * float")
    .Input("fid_offset: uint64")
    .Input("feature_offset: int32")
    .Input("nfl_offset: uint32")
    .Input("batch_size: int32")
    .Input("tensors_grad: num_input * float")
    .Output("embeddings_grad_list: M * float")
    .Attr("M: int")          // num of fids_list (shard x subtable)
    .Attr("num_input: int")  // num of tensors_grad input
    .Attr("variant_type: string")
    .Attr("feature_cfgs: string")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::vector<shape_inference::ShapeHandle> embeddings_list_shape;
      TF_RETURN_IF_ERROR(ctx->input("embeddings_list", &embeddings_list_shape));
      TF_RETURN_IF_ERROR(
          ctx->set_output("embeddings_grad_list", embeddings_list_shape));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayout").Device(DEVICE_CPU),
                        MonolithEmbeddingToLayoutOp);

REGISTER_KERNEL_BUILDER(
    Name("MonolithEmbeddingToLayoutGrad").Device(DEVICE_CPU),
    MonolithEmbeddingToLayoutGradOp);

REGISTER_OP("MonolithEmbeddingToLayoutV2")
    .Input("embeddings_list: M * float")
    .Input("fid_list_row_split: M * int64")
    .Input("fid_offset: uint64")
    .Input("feature_offset: int32")
    .Input("nfl_offset: uint32")
    .Input("batch_size: int32")
    .Output("tensors: num_out * float")
    .Attr("M: int")  // num of fids_list (shard x subtable)
    .Attr("num_out: int")
    .Attr("variant_type: string")
    .Attr("feature_cfgs: string")
    .Attr("ps_num: int")
    .Attr("parallel_flag: int = 0")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::string serialized;
      TF_RETURN_IF_ERROR(ctx->GetAttr("feature_cfgs", &serialized));
      FeatureConfigs feature_cfgs;
      CHECK(feature_cfgs.ParseFromArray(serialized.data(), serialized.size()));

      std::vector<std::string> layout_names;
      const auto &out_configs = feature_cfgs.out_configs();
      for (const auto &pair : out_configs) {
        layout_names.push_back(pair.first);
      }
      std::sort(layout_names.begin(), layout_names.end());

      std::vector<shape_inference::ShapeHandle> tensors_shape;
      for (const auto &layout_name : layout_names) {
        const OutConfig &out_conf = out_configs.at(layout_name);
        for (const auto shape : out_conf.shape()) {
          std::vector<shape_inference::DimensionHandle> dims;
          for (size_t i = 0; i < shape.dims_size(); ++i) {
            if (i == 0) {
              dims.push_back(ctx->UnknownDim());
            } else {
              CHECK_GT(shape.dims(i), 0);
              dims.push_back(ctx->MakeDim(shape.dims(i)));
            }
          }
          tensors_shape.push_back(ctx->MakeShape(dims));
        }
      }
      TF_RETURN_IF_ERROR(ctx->set_output("tensors", tensors_shape));
      return Status::OK();
    });

REGISTER_OP("MonolithEmbeddingToLayoutGradV2")
    .Input("embeddings_list: M * float")
    .Input("fid_list_row_split: M * int64")
    .Input("fid_offset: uint64")
    .Input("feature_offset: int32")
    .Input("nfl_offset: uint32")
    .Input("batch_size: int32")
    .Input("tensors_grad: num_input * float")
    .Output("embeddings_grad_list: M * float")
    .Attr("M: int")          // num of fids_list (shard x subtable)
    .Attr("num_input: int")  // num of tensors_grad input
    .Attr("variant_type: string")
    .Attr("feature_cfgs: string")
    .Attr("ps_num: int")
    .Attr("parallel_flag: int = 0")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext *ctx) {
      std::vector<shape_inference::ShapeHandle> embeddings_list_shape;
      TF_RETURN_IF_ERROR(ctx->input("embeddings_list", &embeddings_list_shape));
      TF_RETURN_IF_ERROR(
          ctx->set_output("embeddings_grad_list", embeddings_list_shape));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV2").Device(DEVICE_CPU),
                        MonolithEmbeddingToLayoutOpV2);

REGISTER_KERNEL_BUILDER(
    Name("MonolithEmbeddingToLayoutGradV2").Device(DEVICE_CPU),
    MonolithEmbeddingToLayoutGradOpV2);

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
