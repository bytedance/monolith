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

#include "monolith/native_training/runtime/ops/fused_embedding_to_layout.h"

namespace tensorflow {
namespace monolith_tf {

namespace fused_layout {

void *MemCopy(float *dest, const float *src, std::size_t count) {
  return std::memcpy(dest, src, count * sizeof(float));
}

template <class TInit>
void OptimizedSumpooling(const float *src, const int dim_num, void *init_ptr,
                         float *dst, void *one_mutex_ptr = nullptr,
                         int mean_pool_fid_num = 0) {
  std::mutex *one_mutex = static_cast<std::mutex *>(one_mutex_ptr);
  TInit *init = static_cast<TInit *>(init_ptr);
  if (one_mutex) {
    one_mutex->lock();
  }
  if (init && *init) {
    if (mean_pool_fid_num) {
      for (size_t i = 0; i < dim_num; ++i) {
        dst[i] = (src[i] / mean_pool_fid_num);
      }
    } else {
      MemCopy(dst, src, dim_num);
    }
    *init = false;
  } else {
    // ::monolith::hash_table::ReduceSum(src, dst, dst, dim_num);
    if (mean_pool_fid_num) {
      for (size_t i = 0; i < dim_num; ++i) {
        dst[i] += (src[i] / mean_pool_fid_num);
      }
    } else {
      for (size_t i = 0; i < dim_num; ++i) {
        dst[i] += src[i];
      }
    }
  }
  if (one_mutex) {
    one_mutex->unlock();
  }
}

NoneLayout::NoneLayout(const std::string &name, const OutConfig &out_conf,
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
NoneLayout::NoneLayout(const std::string &name, const OutConfig &out_conf,
                       OpOutputList &tensor_list, int &start_idx)
    : Layout(name, out_conf) {
  int offset = 0;
  CHECK(out_conf.slice_configs_size() == out_conf.shape_size());
  for (const SliceConfig &slice_conf : out_conf.slice_configs()) {
    slice_to_tensor_.insert(
        {GetKey(slice_conf), {tensor_list[start_idx++], offset++}});
  }
}

PtrWrapper NoneLayout::GetSlice(int row_id, const SliceConfig &slice_conf) {
  auto key = GetKey(slice_conf);
  auto it = slice_to_tensor_.find(key);
  if (it != slice_to_tensor_.end()) {
    auto &layout_info = it->second;
    const LayoutShape &shape = out_config_.shape(layout_info.second);
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

DefaultLayout::DefaultLayout(const std::string &name, const OutConfig &out_conf,
                             OpInputList &tensor_list, int &start_idx)
    : Layout(name, out_conf) {
  int offset = 0;
  CHECK_EQ(out_conf.shape_size(), 1);
  CHECK_NE(out_conf.out_type(), OutType::NONE);

  for (const SliceConfig &slice_conf : out_conf.slice_configs()) {
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

DefaultLayout::DefaultLayout(const std::string &name, const OutConfig &out_conf,
                             OpOutputList &tensor_list, int &start_idx)
    : Layout(name, out_conf) {
  int offset = 0;
  CHECK_EQ(out_conf.shape_size(), 1);
  CHECK_NE(out_conf.out_type(), OutType::NONE);

  for (const SliceConfig &slice_conf : out_conf.slice_configs()) {
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

PtrWrapper DefaultLayout::GetSlice(int row_id, const SliceConfig &slice_conf) {
  auto key = GetKey(slice_conf);
  auto it = slice_to_tensor_.find(key);
  if (it != slice_to_tensor_.end()) {
    auto &layout_info = it->second;
    CHECK_EQ(out_config_.shape_size(), 1);
    const LayoutShape &shape = out_config_.shape(0);

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

MonolithEmbeddingToLayoutBase::MonolithEmbeddingToLayoutBase(
    OpKernelConstruction *ctx, int version)
    : OpKernel(ctx), version_(version) {
  std::string serialized;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_cfgs", &serialized));
  OP_REQUIRES(
      ctx, feature_cfgs_.ParseFromArray(serialized.data(), serialized.size()),
      errors::FailedPrecondition("Failed to parse the feature_cfgs_."));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("variant_type", &variant_type_));
  if (version_ >= 2) {
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
         slice_idx < feature_conf_pair.second.slice_dims_size(); slice_idx++) {
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
      CHECK(!(pair.second.out_type() == OutType::ADDN &&
              slice_config.pooling_type() == PoolingType::FIRSTN));
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

MonolithEmbeddingToLayoutOp::MonolithEmbeddingToLayoutOp(
    OpKernelConstruction *ctx, int version /* = 1*/)
    : MonolithEmbeddingToLayoutBase(ctx, version) {}

void MonolithEmbeddingToLayoutOp::Compute(OpKernelContext *ctx) {
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

  const Tensor *nfl_size_tensor;
  const Tensor *feature_size_tensor;
  const Tensor *fid_size_tensor;
  const Tensor *emb_size_tensor;
  if (GetVersion() == 5) {
    OP_REQUIRES_OK(ctx, ctx->input("nfl_size", &nfl_size_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("feature_size", &feature_size_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("fid_size", &fid_size_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("emb_size", &emb_size_tensor));
  }

  const auto fids_offset_vec = fids_offset_input->flat<uint64>();
  int total_fid_num = fids_offset_input->dim_size(0);
  const auto feature_offset_vec = feature_offset_input->flat<int32>();
  int total_feature_num = feature_offset_input->dim_size(0);
  const auto nfl_offset_vec = nfl_offset_input->flat<uint32>();
  int total_nfl_num = nfl_offset_input->dim_size(0);
  int req_num = 1;
  int32 max_batch_size = 0;
  std::vector<int> each_req_batch_size_offset(1, 0);
  std::vector<int> each_req_nfl_offset(1, 0);
  std::vector<int> each_req_feature_offset(1, 0);
  std::vector<int> each_req_fid_offset(1, 0);
  if (GetVersion() == 5) {
    const auto batch_size_vec = batch_size_tensor->flat<int32>();
    req_num = batch_size_tensor->dim_size(0);

    for (size_t i = 0; i < req_num; ++i) {
      each_req_batch_size_offset.push_back(
          each_req_batch_size_offset[i] + batch_size_vec(i));
      max_batch_size = std::max(batch_size_vec(i), max_batch_size);
    }

    const auto nfl_size_vec = nfl_size_tensor->flat<int32>();
    for (size_t i = 0; i < req_num; ++i) {
      each_req_nfl_offset.push_back(each_req_nfl_offset[i] + nfl_size_vec(i));
    }
    CHECK_EQ(each_req_nfl_offset.back(), total_nfl_num);

    const auto feature_size_vec = feature_size_tensor->flat<int32>();
    for (size_t i = 0; i < req_num; ++i) {
      each_req_feature_offset.push_back(
          each_req_feature_offset[i] + feature_size_vec(i));
    }
    CHECK_EQ(each_req_feature_offset.back(), total_feature_num);

    const auto fid_size_vec = fid_size_tensor->flat<int32>();
    for (size_t i = 0; i < req_num; ++i) {
      each_req_fid_offset.push_back(each_req_fid_offset[i] + fid_size_vec(i));
    }
    CHECK_EQ(each_req_fid_offset.back(), total_fid_num);
  } else {
    max_batch_size = batch_size_tensor->scalar<int32>()();
    each_req_batch_size_offset.push_back(max_batch_size);
    each_req_nfl_offset.push_back(total_nfl_num);
    each_req_feature_offset.push_back(total_feature_num);
    each_req_fid_offset.push_back(total_fid_num);
  }
  req_sum_ += req_num;
  process_num_++;
  LOG_EVERY_N_SEC(INFO, 60) << "input avg req num: "
                            << req_sum_ * 1.0 / process_num_;

  OpOutputList layout_tensor_list;
  OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &layout_tensor_list));

  std::vector<PtrWrapper> embeddings_data;
  if (GetVersion() == 2) {
    CHECK_EQ(req_num, 1);
    OpInputList fid_list_row_split;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("fid_list_row_split", &fid_list_row_split));

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
          embeddings_data.push_back(PtrWrapper{embeddings_ptr + pre_emb_offset,
                                               dim, fid_count * dim});
          pre_offset = offset;
          pre_emb_offset += fid_count * dim;
          CHECK(pre_emb_offset <= embeddings_size);
        }
      }
    }
  } else if (GetVersion() == 3) {
    embeddings_data.reserve(embeddings_list.size());
    for (size_t i = 0; i < embeddings_list.size(); ++i) {
      const auto &embeddings_mat_ptr_ = embeddings_list[i].flat<float>().data();
      embeddings_data.push_back(PtrWrapper(
          {embeddings_mat_ptr_, 1, embeddings_list[i].flat<float>().size()}));
    }
  } else if (GetVersion() == 4) {
    int ps_num = GetPsNum();
    const std::vector<std::vector<int>> &table_feature_dim =
        GetFeatureInTableDim();

    CHECK_EQ(embeddings_list.size(), 1);
    const auto embeddings_list_flat = embeddings_list[0].flat<float>();

    const Tensor *fid_list_emb_row_lenth_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("fid_list_emb_row_lenth",
                                   &fid_list_emb_row_lenth_tensor));
    const auto fid_list_emb_row_lenth_flat =
        fid_list_emb_row_lenth_tensor->flat<int32_t>();
    CHECK_EQ(fid_list_emb_row_lenth_flat.size(),
             table_feature_dim.size() * ps_num);

    embeddings_data.resize(req_num * fid_list_emb_row_lenth_flat.size());
    int pre_count = 0;
    for (size_t i = 0; i < req_num * fid_list_emb_row_lenth_flat.size(); ++i) {
      int req_i = i / fid_list_emb_row_lenth_flat.size();
      int table_idx = (i % fid_list_emb_row_lenth_flat.size()) % table_feature_dim.size();
      int ps_index = (i % fid_list_emb_row_lenth_flat.size()) / table_feature_dim.size();
      int index = req_i * fid_list_emb_row_lenth_flat.size() + table_idx * ps_num + ps_index;

      embeddings_data[index].ptr = embeddings_list_flat.data() + pre_count;
      embeddings_data[index].offset = 1;
      embeddings_data[index].count = fid_list_emb_row_lenth_flat(i);

      pre_count += fid_list_emb_row_lenth_flat(i);
    }
    CHECK_EQ(pre_count, req_num * embeddings_list_flat.size());
  } else if (GetVersion() == 5) {
    const auto emb_size_vec = emb_size_tensor->flat<int32>();
    std::vector<std::vector<int>> each_req_emb_offset(
        embeddings_list.size(), std::vector<int>(req_num + 1, 0));
    for (size_t i = 0; i < embeddings_list.size(); ++i) {
      for (size_t req_i = 0; req_i < req_num; ++req_i) {
        each_req_emb_offset[i][req_i + 1] =
            each_req_emb_offset[i][req_i] +
            emb_size_vec(i + req_i * embeddings_list.size());
      }
      CHECK_EQ(each_req_emb_offset[i].back(),
               embeddings_list[i].flat<float>().size());
    }

    embeddings_data.reserve(req_num * embeddings_list.size());
    for (size_t req_i = 0; req_i < req_num; req_i++) {
      for (size_t i = 0; i < embeddings_list.size(); ++i) {
        const auto &embeddings_mat_ptr_ =
            embeddings_list[i].flat<float>().data();
        embeddings_data.push_back(
            PtrWrapper({embeddings_mat_ptr_ + each_req_emb_offset[i][req_i], 1,
                        each_req_emb_offset[i][req_i + 1] -
                            each_req_emb_offset[i][req_i]}));
      }
    }
  } else {
    CHECK_EQ(req_num, 1);
    embeddings_data.reserve(embeddings_list.size());
    for (size_t i = 0; i < embeddings_list.size(); ++i) {
      const auto &embeddings_mat_ptr_ = embeddings_list[i].flat<float>().data();
      embeddings_data.push_back(PtrWrapper(
          {embeddings_mat_ptr_, embeddings_list[i].dim_size(1),
           embeddings_list[i].dim_size(0) * embeddings_list[i].dim_size(1)}));
    }
  }

  {
    auto activity =
        std::make_unique<profiler::TraceMe>([]() { return "AllocateTensors"; });
    int offset = 0;
    const auto &out_configs = GetFeatureCfgs().out_configs();
    for (const auto &layout_name : GetLayoutNames()) {
      const OutConfig &out_conf = out_configs.at(layout_name);
      for (const auto shape : out_conf.shape()) {
        Tensor *tensor;
        TensorShape tensor_shape;
        for (size_t i = 0; i < shape.dims_size(); ++i) {
          if (i == 0) {
            tensor_shape.AddDim(shape.dims(i) == -1
                                    ? each_req_batch_size_offset.back()
                                    : shape.dims(i));
          } else {
            CHECK_GT(shape.dims(i), 0);
            tensor_shape.AddDim(shape.dims(i));
          }
        }
        OP_REQUIRES_OK(
            ctx, layout_tensor_list.allocate(offset++, tensor_shape, &tensor));
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

  TaskRun(layouts, embeddings_data, fids_offset_vec.data(), total_fid_num,
          feature_offset_vec.data(), total_feature_num, nfl_offset_vec.data(),
          total_nfl_num, max_batch_size, each_req_batch_size_offset,
          each_req_nfl_offset, each_req_feature_offset, each_req_fid_offset,
          req_num, ctx, &layout_tensor_list);
}

void ForwardTaskRunImpl(int slice_conf_i, int dim_num, int64 nfl_idx,
                        ::monolith::io::proto::OutType out_type,
                        ::monolith::io::proto::PoolingType pooling_type,
                        int max_sequence_length, int start,
                        const uint64 *fids_offset_vec, int total_fid_num,
                        const int32 *feature_offset_vec, int total_feature_num,
                        const uint32 *nfl_offset_vec, int total_nfl_num,
                        int batch_size, const PtrWrapper *embeddings_data,
                        int embeddings_data_size, PtrWrapper *ptr_info_ptr) {
  PtrWrapper &ptr_info = *ptr_info_ptr;
  bool is_shared;
  int nfl_offset, feature_num;
  GetFeatureInfo(nfl_idx, nfl_offset_vec, total_nfl_num, total_feature_num,
                 &is_shared, &nfl_offset, &feature_num);
  if (!feature_num) return;  // nfl exits

  std::unique_ptr<float[]> tmp;
  if (is_shared && (out_type == OutType::ADDN)) {
    tmp.reset(new float[dim_num]());
  }
  int feature_idx = nfl_offset + 0;
  for (size_t index = 0; index < batch_size; ++index) {
    int temp_offset = index * ptr_info.offset;
    if (pooling_type == PoolingType::FIRSTN) {
      CHECK(temp_offset + max_sequence_length * dim_num <= ptr_info.count);
    } else {
      CHECK(temp_offset + dim_num <= ptr_info.count);
    }
    if (!is_shared || index == 0) {
      bool init = (out_type != OutType::ADDN) || tmp;
      GatherEmb(
          feature_idx, max_sequence_length, pooling_type, dim_num, start,
          embeddings_data, embeddings_data_size, fids_offset_vec, total_fid_num,
          feature_offset_vec, total_feature_num,
          const_cast<float *>(tmp ? tmp.get() : ptr_info.ptr + temp_offset),
          OptimizedSumpooling<bool>, MemCopy, nullptr, nullptr,
          DefaultGetInitFunc, &init);
      if (tmp) {
        bool init_tmp = (out_type != OutType::ADDN) || (slice_conf_i == 0);
        OptimizedSumpooling<bool>(
            tmp.get(), dim_num, &init_tmp,
            const_cast<float *>(ptr_info.ptr + temp_offset));
      }
      feature_idx++;
    } else {
      if (tmp) {
        bool init_tmp = (slice_conf_i == 0);  // && index == 0
        OptimizedSumpooling<bool>(
            tmp.get(), dim_num, &init_tmp,
            const_cast<float *>(ptr_info.ptr + temp_offset));
      } else {
        switch (pooling_type) {
          case PoolingType::SUM:
          case PoolingType::MEAN:
            MemCopy(const_cast<float *>(ptr_info.ptr + temp_offset),
                    ptr_info.ptr, dim_num);
            break;
          case PoolingType::FIRSTN:
            MemCopy(const_cast<float *>(ptr_info.ptr + temp_offset),
                    ptr_info.ptr, dim_num * max_sequence_length);
            break;
          default:
            break;
        }
      }
    }
  }
}

void MonolithEmbeddingToLayoutOp::TaskRun(
    const std::vector<std::shared_ptr<Layout>> &layouts,
    const std::vector<PtrWrapper> &embeddings_data,
    const uint64 *fids_offset_vec, int total_fid_num,
    const int32 *feature_offset_vec, int total_feature_num,
    const uint32 *nfl_offset_vec, int total_nfl_num, int batch_size,
    const std::vector<int> &each_req_batch_size_offset,
    const std::vector<int> &each_req_nfl_offset,
    const std::vector<int> &each_req_feature_offset,
    const std::vector<int> &each_req_fid_offset, int req_num,
    OpKernelContext *ctx, OpOutputList *layout_tensor_list) {
  CHECK_EQ(req_num, 1);
  for (int32 idx = 0; idx < layout_tensor_list->size(); ++idx) {
    (*layout_tensor_list)[idx]->flat<float>().setZero();
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
        const int64 nfl_idx = slice_conf.feature_idx();

        ForwardTaskRunImpl(slice_conf_i, dim_num, nfl_idx, layout->out_type(),
                           slice_conf.pooling_type(),
                           slice_conf.max_sequence_length(), slice_conf.start(),
                           fids_offset_vec, total_fid_num, feature_offset_vec,
                           total_feature_num, nfl_offset_vec, total_nfl_num,
                           batch_size, embeddings_data.data(),
                           embeddings_data.size(), &ptr_info);
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

class MonolithEmbeddingToLayoutOpV2 : public MonolithEmbeddingToLayoutOp {
 public:
  explicit MonolithEmbeddingToLayoutOpV2(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOp(ctx, 2) {}
};

class MonolithEmbeddingToLayoutOpV3 : public MonolithEmbeddingToLayoutOp {
 public:
  explicit MonolithEmbeddingToLayoutOpV3(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOp(ctx, 3) {}
};

class MonolithEmbeddingToLayoutOpV4 : public MonolithEmbeddingToLayoutOp {
 public:
  explicit MonolithEmbeddingToLayoutOpV4(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOp(ctx, 4) {}
};

class MonolithEmbeddingToLayoutOpV5 : public MonolithEmbeddingToLayoutOp {
 public:
  explicit MonolithEmbeddingToLayoutOpV5(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOp(ctx, 5) {}
};

MonolithEmbeddingToLayoutGradOp::MonolithEmbeddingToLayoutGradOp(
    OpKernelConstruction *ctx, int version /* = 1*/)
    : MonolithEmbeddingToLayoutBase(ctx, version) {}

void MonolithEmbeddingToLayoutGradOp::Compute(OpKernelContext *ctx) {
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

  const auto fids_offset_vec = fids_offset_input->flat<uint64>();
  int total_fid_num = fids_offset_input->dim_size(0);
  const auto feature_offset_vec = feature_offset_input->flat<int32>();
  int total_feature_num = feature_offset_input->dim_size(0);
  const auto nfl_offset_vec = nfl_offset_input->flat<uint32>();
  int total_nfl_num = nfl_offset_input->dim_size(0);
  int32 batch_size = batch_size_tensor->scalar<int32>()();

  std::vector<std::pair<int, int>> ufid_grads_info;

  OpOutputList embeddings_grad_list;
  OP_REQUIRES_OK(
      ctx, ctx->output_list("embeddings_grad_list", &embeddings_grad_list));
  std::vector<PtrWrapper> embeddings_grads_data;
  int init_counter = 0;
  if (GetVersion() == 2) {
    OpInputList fid_list_row_split;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("fid_list_row_split", &fid_list_row_split));
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
          ufid_grads_info.emplace_back(std::make_pair(init_counter, fid_count));
          pre_offset = offset;
          pre_emb_offset += fid_count * dim;
          CHECK(pre_emb_offset <= embeddings_grad_size);
          init_counter += fid_count;
        }
      }
    }
  } else if (GetVersion() == 3 || GetVersion() == 5) {
    embeddings_grads_data.reserve(embeddings_list.size());
    for (size_t i = 0; i < embeddings_list.size(); ++i) {
      Tensor *tensor;
      OP_REQUIRES_OK(ctx, embeddings_grad_list.allocate(
                              i, embeddings_list[i].shape(), &tensor));
      embeddings_grads_data.push_back(
          PtrWrapper({embeddings_grad_list[i]->flat<float>().data(), 1,
                      embeddings_grad_list[i]->flat<float>().size()}));
    }
  } else if (GetVersion() == 4) {
    int ps_num = GetPsNum();
    const std::vector<std::vector<int>> &table_feature_dim =
        GetFeatureInTableDim();

    CHECK_EQ(embeddings_list.size(), 1);
    Tensor *embeddings_grad_list_tensor;
    OP_REQUIRES_OK(
        ctx, embeddings_grad_list.allocate(0, {embeddings_list[0].shape()},
                                           &embeddings_grad_list_tensor));
    const auto embeddings_grad_list_flat =
        embeddings_grad_list_tensor->flat<float>();

    const Tensor *fid_list_emb_row_lenth_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("fid_list_emb_row_lenth",
                                   &fid_list_emb_row_lenth_tensor));
    const auto fid_list_emb_row_lenth_flat =
        fid_list_emb_row_lenth_tensor->flat<int32_t>();
    CHECK_EQ(fid_list_emb_row_lenth_flat.size(),
             table_feature_dim.size() * ps_num);

    embeddings_grads_data.resize(fid_list_emb_row_lenth_flat.size());
    int pre_count = 0;
    for (size_t i = 0; i < fid_list_emb_row_lenth_flat.size(); ++i) {
      int table_idx = i % table_feature_dim.size();
      int ps_index = i / table_feature_dim.size();
      int index = table_idx * ps_num + ps_index;

      embeddings_grads_data[index].ptr =
          embeddings_grad_list_flat.data() + pre_count;
      embeddings_grads_data[index].offset = 1;
      embeddings_grads_data[index].count = fid_list_emb_row_lenth_flat(i);

      pre_count += fid_list_emb_row_lenth_flat(i);
    }
    CHECK_EQ(pre_count, embeddings_grad_list_flat.size());
  } else {
    embeddings_grads_data.reserve(embeddings_list.size());
    for (size_t i = 0; i < embeddings_list.size(); ++i) {
      Tensor *tensor;
      OP_REQUIRES_OK(ctx, embeddings_grad_list.allocate(
                              i, embeddings_list[i].shape(), &tensor));
      int dim = embeddings_list[i].dim_size(1);
      int fid_count = embeddings_list[i].dim_size(0);
      embeddings_grads_data.push_back(PtrWrapper{
          embeddings_grad_list[i]->flat<float>().data(), dim, fid_count * dim});
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
    const OutConfig &out_conf = GetFeatureCfgs().out_configs().at(layout_name);

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

  TaskRun(layouts, &ufid_grads_info, fids_offset_vec.data(), total_fid_num,
          feature_offset_vec.data(), total_feature_num, nfl_offset_vec.data(),
          total_nfl_num, batch_size, ctx, &embeddings_grad_list,
          &embeddings_grads_data,
          (GetVersion() == 3 || GetVersion() == 4 || GetVersion() == 5) ? nullptr : &init);
}

static constexpr int NUM_LOCKS = 512;
void *ScatterGradGetMutexFuncFunc(void *main_params, int32 index1,
                                  int32 index2) {
  std::mutex *mutex_list = static_cast<std::mutex *>(main_params);
  int lock_idx = index1 * index2;
  int mutex_idx = lock_idx % NUM_LOCKS;
  auto one_mutex = mutex_list + mutex_idx;
  return one_mutex;
}
struct ScatterGradGetInitFuncParams {
  int slice_conf_slice_idx;
  const std::vector<std::pair<int, int>> *ufid_grads_info;
  GroupA *init;
};
void *ScatterGradGetInitFunc(void *main_params, int32 index1, int32 index2) {
  ScatterGradGetInitFuncParams *params =
      static_cast<ScatterGradGetInitFuncParams *>(main_params);
  const auto &fid_info = params->ufid_grads_info->at(index1);
  CUSTOM_CHECK(index2 < fid_info.second);
  int real_ufid_idx = fid_info.first + index2;
  auto init_p = params->init->Get(real_ufid_idx, params->slice_conf_slice_idx);
  return init_p;
}

void MonolithEmbeddingToLayoutGradOp::TaskRun(
    const std::vector<std::shared_ptr<Layout>> &layouts,
    const std::vector<std::pair<int, int>> *ufid_grads_info,
    const uint64 *fids_offset_vec, int total_fid_num,
    const int32 *feature_offset_vec, int total_feature_num,
    const uint32 *nfl_offset_vec, int total_nfl_num, int batch_size,
    OpKernelContext *ctx, OpOutputList *embeddings_grad_list,
    std::vector<PtrWrapper> *embeddings_grads_data, GroupA *init) {
  for (int32 idx = 0; idx < embeddings_grad_list->size(); ++idx) {
    (*embeddings_grad_list)[idx]->flat<float>().setConstant(0);
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
                       total_feature_num, &is_shared, &nfl_offset,
                       &feature_num);
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
          ScatterGradGetInitFuncParams init_params(
              {slice_conf.slice_idx(), ufid_grads_info, init});
          ScatterGrad(feature_idx, slice_conf.max_sequence_length(),
                      slice_conf.pooling_type(), ptr_info.ptr + temp_offset,
                      dim_num, slice_conf.start(), fids_offset_vec,
                      total_fid_num, feature_offset_vec, total_feature_num,
                      embeddings_grads_data->size(),
                      embeddings_grads_data->data(), OptimizedSumpooling<char>,
                      (mutex_list ? ScatterGradGetMutexFuncFunc : nullptr),
                      (mutex_list ? mutex_list.get() : nullptr),
                      (init ? ScatterGradGetInitFunc : nullptr), &init_params);
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

class MonolithEmbeddingToLayoutGradOpV2
    : public MonolithEmbeddingToLayoutGradOp {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV2(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOp(ctx, 2) {}
};

class MonolithEmbeddingToLayoutGradOpV3
    : public MonolithEmbeddingToLayoutGradOp {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV3(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOp(ctx, 3) {}
};

class MonolithEmbeddingToLayoutGradOpV4
    : public MonolithEmbeddingToLayoutGradOp {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV4(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOp(ctx, 4) {}
};

class MonolithEmbeddingToLayoutGradOpV5
    : public MonolithEmbeddingToLayoutGradOp {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV5(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOp(ctx, 5) {}
};

auto forward_shape_inference_fn = [](shape_inference::InferenceContext *ctx) {
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
    .SetShapeFn(forward_shape_inference_fn);

auto backward_shape_inference_fn = [](shape_inference::InferenceContext *ctx) {
  std::vector<shape_inference::ShapeHandle> embeddings_list_shape;
  TF_RETURN_IF_ERROR(ctx->input("embeddings_list", &embeddings_list_shape));
  TF_RETURN_IF_ERROR(
      ctx->set_output("embeddings_grad_list", embeddings_list_shape));
  return Status::OK();
};

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
    .SetShapeFn(backward_shape_inference_fn);

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
    .SetShapeFn(forward_shape_inference_fn);

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
    .SetShapeFn(backward_shape_inference_fn);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV2").Device(DEVICE_CPU),
                        MonolithEmbeddingToLayoutOpV2);

REGISTER_KERNEL_BUILDER(
    Name("MonolithEmbeddingToLayoutGradV2").Device(DEVICE_CPU),
    MonolithEmbeddingToLayoutGradOpV2);

REGISTER_OP("MonolithEmbeddingToLayoutV3")
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
    .Attr("ps_num: int")
    .Attr("parallel_flag: int = 0")
    .SetDoNotOptimize()
    .SetShapeFn(forward_shape_inference_fn);

REGISTER_OP("MonolithEmbeddingToLayoutGradV3")
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
    .Attr("ps_num: int")
    .Attr("parallel_flag: int = 0")
    .SetDoNotOptimize()
    .SetShapeFn(backward_shape_inference_fn);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV3").Device(DEVICE_CPU),
                        MonolithEmbeddingToLayoutOpV3);

REGISTER_KERNEL_BUILDER(
    Name("MonolithEmbeddingToLayoutGradV3").Device(DEVICE_CPU),
    MonolithEmbeddingToLayoutGradOpV3);

REGISTER_OP("MonolithEmbeddingToLayoutV4")
    .Input("embeddings_list: M * float")
    .Input("fid_list_emb_row_lenth: int32")
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
    .SetShapeFn(forward_shape_inference_fn);

REGISTER_OP("MonolithEmbeddingToLayoutGradV4")
    .Input("embeddings_list: M * float")
    .Input("fid_list_emb_row_lenth: int32")
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
    .SetShapeFn(backward_shape_inference_fn);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV4").Device(DEVICE_CPU),
                        MonolithEmbeddingToLayoutOpV4);

REGISTER_KERNEL_BUILDER(
    Name("MonolithEmbeddingToLayoutGradV4").Device(DEVICE_CPU),
    MonolithEmbeddingToLayoutGradOpV4);

REGISTER_OP("MonolithEmbeddingToLayoutV5")
    .Input("embeddings_list: M * float")
    .Input("fid_offset: uint64")
    .Input("feature_offset: int32")
    .Input("nfl_offset: uint32")
    .Input("batch_size: int32")
    .Input("nfl_size: int32")
    .Input("feature_size: int32")
    .Input("fid_size: int32")
    .Input("emb_size: int32")
    .Output("tensors: num_out * float")
    .Attr("M: int")  // num of fids_list (shard x subtable)
    .Attr("num_out: int")
    .Attr("variant_type: string")
    .Attr("feature_cfgs: string")
    .Attr("ps_num: int")
    .Attr("parallel_flag: int = 0")
    .SetDoNotOptimize()
    .SetShapeFn(forward_shape_inference_fn);

REGISTER_OP("MonolithEmbeddingToLayoutGradV5")
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
    .Attr("ps_num: int")
    .Attr("parallel_flag: int = 0")
    .SetDoNotOptimize()
    .SetShapeFn(backward_shape_inference_fn);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV5").Device(DEVICE_CPU),
                        MonolithEmbeddingToLayoutOpV5);

REGISTER_KERNEL_BUILDER(
    Name("MonolithEmbeddingToLayoutGradV5").Device(DEVICE_CPU),
    MonolithEmbeddingToLayoutGradOpV5);


}  // namespace fused_layout
}  // namespace monolith_tf
}  // namespace tensorflow
