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

namespace fused_layout {
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

EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void ParseFidOffset(
    const uint64 &fids_offset, int32 *index1, int32 *index2) {
  *index1 = fids_offset >> 32;
  *index2 = fids_offset << 32 >> 32;
}

EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void ParseNflOffset(
    const uint32 nfl_offset_encode, bool *is_shared, int *nfl_offset) {
  *is_shared = nfl_offset_encode >> 31;
  *nfl_offset = nfl_offset_encode & 0x7fffffff;
}

EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void GetFeatureInfo(
    const int64 nfl_idx, const uint32 *nfl_offset_vec, const int total_nfl_num,
    const int total_feature_num, bool *is_shared, int *nfl_offset,
    int *feature_num) {
  ParseNflOffset(*(nfl_offset_vec + nfl_idx), is_shared, nfl_offset);
  if (nfl_idx < total_nfl_num - 1) {
    bool is_shared_later;
    int nfl_offset_later;
    ParseNflOffset(*(nfl_offset_vec + nfl_idx + 1), &is_shared_later,
                   &nfl_offset_later);
    *feature_num = nfl_offset_later - *nfl_offset;
  } else {
    *feature_num = total_feature_num - *nfl_offset;
  }
}

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

  const SliceConfig *GetKey(const SliceConfig &slice_conf) {
    return &slice_conf;
    // return absl::StrCat(name_, "_", slice_conf.feature_name(), "_",
    //                     slice_conf.start(), "_", slice_conf.end());
  }

  const ::google::protobuf::RepeatedPtrField<SliceConfig> &GetSliceConfig() {
    return out_config_.slice_configs();
  }

  const OutType out_type() { return out_config_.out_type(); }

 protected:
  const std::string &name_;
  const OutConfig &out_config_;
};

class NoneLayout : public Layout {
 public:
  // op input TODO
  NoneLayout(const std::string &name, const OutConfig &out_conf,
             OpInputList &tensor_list, int &start_idx);

  // op output
  NoneLayout(const std::string &name, const OutConfig &out_conf,
             OpOutputList &tensor_list, int &start_idx);

  virtual ~NoneLayout() {}

  PtrWrapper GetSlice(int row_id, const SliceConfig &slice_conf) override;

 private:
  absl::flat_hash_map<const SliceConfig *, std::pair<const Tensor *, const int>>
      slice_to_tensor_;
};

class DefaultLayout : public Layout {
 public:
  DefaultLayout(const std::string &name, const OutConfig &out_conf,
                OpInputList &tensor_list, int &start_idx);

  DefaultLayout(const std::string &name, const OutConfig &out_conf,
                OpOutputList &tensor_list, int &start_idx);

  virtual ~DefaultLayout() {}

  PtrWrapper GetSlice(int row_id, const SliceConfig &slice_conf) override;

 private:
  absl::flat_hash_map<const SliceConfig *, std::pair<const Tensor *, const int>>
      slice_to_tensor_;
};

class MonolithEmbeddingToLayoutBase : public OpKernel {
 public:
  explicit MonolithEmbeddingToLayoutBase(OpKernelConstruction *ctx,
                                         int version);

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
};

#ifdef EIGEN_USE_GPU
#define CUSTOM_CHECK(cond)                                               \
  if (!(cond)) {                                                         \
    printf("ERROR %s %s:%d CHECK Fail\n", __FILE__, __func__, __LINE__); \
    return;                                                              \
  }
#else
#define CUSTOM_CHECK(cond) CHECK(cond)
#endif

typedef void (*OptimizedSumpoolingFunc)(const float *src, const int dim_num,
                                        void *init, float *dst, void *one_mutex,
                                        int mean_pool_fid_num);

typedef void *(MemCopyFunc)(float *dest, const float *src, std::size_t count);
typedef void *(GetMutexFunc)(void *main_params, int32 index1, int32 index2);
typedef void *(GetInitFunc)(void *main_params, int32 index1, int32 index2);
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void *DefaultGetInitFunc(
    void *main_params, int32 index1, int32 index2) {
  return main_params;
}

EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void GatherEmb(
    const int feature_idx, const int max_sequence_length,
    const PoolingType pooling_type, int dims, int slice_conf_start,
    const PtrWrapper *embeddings_data, int embeddings_data_size,
    const uint64 *fids_offset_vec, const int total_fid_num,
    const int32 *feature_offset_vec, const int total_feature_num,
    float *out_ptr, OptimizedSumpoolingFunc opt_sumpool_fn,
    MemCopyFunc mem_copy_fn, GetMutexFunc get_mutex_func,
    void *get_mutex_func_main_params, GetInitFunc get_init_func,
    void *get_init_func_main_params) {
  CUSTOM_CHECK(feature_idx < total_feature_num);
  int fid_num = (feature_idx < total_feature_num - 1)
                    ? *(feature_offset_vec + feature_idx + 1) -
                          *(feature_offset_vec + feature_idx)
                    : total_fid_num - *(feature_offset_vec + feature_idx);
  CUSTOM_CHECK(fid_num >= 0);
  if (fid_num == 0) return;
  const auto start_fid_offset_idx = *(feature_offset_vec + feature_idx);

  int seq_idx = 0;

  for (int fid_idx = 0; fid_idx < fid_num; fid_idx++) {
    auto fid_offset_idx = start_fid_offset_idx + fid_idx;
    int32 index1, index2;
    CUSTOM_CHECK(fid_offset_idx < total_fid_num);
    const uint64 fids_offset = *(fids_offset_vec + fid_offset_idx);
    ParseFidOffset(fids_offset, &index1, &index2);
    CUSTOM_CHECK(index1 < embeddings_data_size);
    const auto &ptr_info = *(embeddings_data + index1);
    int tmp_offset = index2 * ptr_info.offset + slice_conf_start;
    CUSTOM_CHECK(tmp_offset + dims <= ptr_info.count);
    const float *src = ptr_info.ptr + tmp_offset;
    void *one_mutex = nullptr;
    if (get_mutex_func) {
      one_mutex = get_mutex_func(get_mutex_func_main_params, index1, index2);
    }
    void *init = nullptr;
    if (get_init_func) {
      init = get_init_func(get_init_func_main_params, index1, index2);
    }
    switch (pooling_type) {
      case PoolingType::SUM:
        opt_sumpool_fn(src, dims, init, out_ptr, one_mutex, 0);
        break;
      case PoolingType::MEAN:
        opt_sumpool_fn(src, dims, init, out_ptr, one_mutex, fid_num);
        break;
      case PoolingType::FIRSTN:
        if (seq_idx < max_sequence_length) {
          mem_copy_fn(out_ptr + seq_idx * dims, src, dims);
        }
        seq_idx++;
        break;
      default:
        break;
    }
  }
}

class MonolithEmbeddingToLayoutOp : public MonolithEmbeddingToLayoutBase {
 public:
  explicit MonolithEmbeddingToLayoutOp(OpKernelConstruction *ctx,
                                       int version = 1);

  void Compute(OpKernelContext *ctx) override;
  virtual void TaskRun(const std::vector<std::shared_ptr<Layout>> &layouts,
                       const std::vector<PtrWrapper> &embeddings_data,
                       const uint64 *fids_offset_vec, int total_fid_num,
                       const int32 *feature_offset_vec, int total_feature_num,
                       const uint32 *nfl_offset_vec, int total_nfl_num,
                       int batch_size,
                       const std::vector<int> &each_req_batch_size_offset,
                       const std::vector<int> &each_req_nfl_offset,
                       const std::vector<int> &each_req_feature_offset,
                       const std::vector<int> &each_req_fid_offset, int req_num,
                       OpKernelContext *ctx, OpOutputList *layout_tensor_list);

 private:
  int req_sum_ = 0;
  int process_num_ = 0;
};

EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC void ScatterGrad(
    const int feature_idx, const int max_sequence_length,
    const PoolingType pooling_type, const float *grad_ptr, int dims,
    int slice_conf_start, const uint64 *fids_offset_vec,
    const int total_fid_num, const int32 *feature_offset_vec,
    const int total_feature_num, const int embeddings_grads_data_num,
    PtrWrapper *embeddings_grads_data, OptimizedSumpoolingFunc opt_sumpool_fn,
    GetMutexFunc get_mutex_func, void *get_mutex_func_main_params,
    GetInitFunc get_init_func, void *get_init_func_main_params) {
  CUSTOM_CHECK(feature_idx < total_feature_num);
  int fid_num = (feature_idx < total_feature_num - 1)
                    ? *(feature_offset_vec + feature_idx + 1) -
                          *(feature_offset_vec + feature_idx)
                    : total_fid_num - *(feature_offset_vec + feature_idx);
  CUSTOM_CHECK(fid_num >= 0);
  if (fid_num == 0) return;
  const auto start_fid_offset_idx = *(feature_offset_vec + feature_idx);
  int seq_idx = 0;

  for (int fid_idx = 0; fid_idx < fid_num; fid_idx++) {
    int32 fid_offset_idx = start_fid_offset_idx + fid_idx;
    int32 index1, index2;
    CUSTOM_CHECK(fid_offset_idx < total_fid_num);
    const uint64 fids_offset = *(fids_offset_vec + fid_offset_idx);
    ParseFidOffset(fids_offset, &index1, &index2);
    CUSTOM_CHECK(index1 < embeddings_grads_data_num);
    // embeddings_grads_data: fid grad data
    const auto &ptr_info = *(embeddings_grads_data + index1);
    int tmp_offset = index2 * ptr_info.offset + slice_conf_start;
    CUSTOM_CHECK(tmp_offset + dims <= ptr_info.count);
    void *one_mutex = nullptr;
    if (get_mutex_func) {
      one_mutex = get_mutex_func(get_mutex_func_main_params, index1, index2);
    }
    void *init_p = nullptr;
    if (get_init_func) {
      init_p = get_init_func(get_init_func_main_params, index1, index2);
    }
    float *dst = const_cast<float *>(ptr_info.ptr) + tmp_offset;
    switch (pooling_type) {
      case PoolingType::SUM: {
        opt_sumpool_fn(grad_ptr, dims, init_p, dst, one_mutex, 0);
        break;
      }
      case PoolingType::MEAN: {
        opt_sumpool_fn(grad_ptr, dims, init_p, dst, one_mutex, fid_num);
        break;
      }
      case PoolingType::FIRSTN: {
        if (seq_idx < max_sequence_length) {
          opt_sumpool_fn(grad_ptr + seq_idx * dims, dims, init_p, dst,
                         one_mutex, false);
        }
        seq_idx++;
        break;
      }

      default:
        break;
    }
  }
}

class MonolithEmbeddingToLayoutGradOp : public MonolithEmbeddingToLayoutBase {
 public:
  explicit MonolithEmbeddingToLayoutGradOp(OpKernelConstruction *ctx,
                                           int version = 1);

  void Compute(OpKernelContext *ctx) override;
  virtual void TaskRun(const std::vector<std::shared_ptr<Layout>> &layouts,
                       const std::vector<std::pair<int, int>> *ufid_grads_info,
                       const uint64 *fids_offset_vec, int total_fid_num,
                       const int32 *feature_offset_vec, int total_feature_num,
                       const uint32 *nfl_offset_vec, int total_nfl_num,
                       int batch_size, OpKernelContext *ctx,
                       OpOutputList *embeddings_grad_list,
                       std::vector<PtrWrapper> *embeddings_grads_data,
                       GroupA *init);
};

}  // namespace fused_layout
}  // namespace monolith_tf
}  // namespace tensorflow
