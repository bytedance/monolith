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


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "monolith/native_training/runtime/ops/fused_embedding_to_layout.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith_tf {
namespace fused_layout {

typedef Eigen::GpuDevice GPUDevice;

struct ForwardTaskInfo {
  int dim_num;
  PtrWrapper ptr_info;
  int64 nfl_idx;

  ::monolith::io::proto::OutType out_type;
  ::monolith::io::proto::PoolingType pooling_type;
  int max_sequence_length;
  int start;
  int req_i;
};

__device__ void *MemCopyGPU(float *dest, const float *src, std::size_t count) {
  for (int32 idx = 0; idx < count; ++idx) {
    *(dest + idx) = *(src + idx);
  }
  return dest;
}

__device__ void OptimizedSumpoolingGPU(const float *src, const int dim_num,
                                       void *init_ptr, float *dst,
                                       void *one_mutex = nullptr,
                                       int mean_pool_fid_num = 0) {
  bool *init = static_cast<bool *>(init_ptr);
  if (init && *init) {
    if (mean_pool_fid_num) {
      for (size_t i = 0; i < dim_num; ++i) {
        dst[i] = (src[i] / mean_pool_fid_num);
      }
    } else {
      MemCopyGPU(dst, src, dim_num);
    }
    *init = false;
  } else {
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
}

__device__ void OptimizedSumpoolingGPUWithLock(const float *src,
                                               const int dim_num,
                                               void *init_ptr, float *dst,
                                               void *one_mutex = nullptr,
                                               int mean_pool_fid_num = 0) {
  if (mean_pool_fid_num) {
    for (int32 idx = 0; idx < dim_num; ++idx) {
      GpuAtomicAdd(dst + idx, (*(src + idx)) / mean_pool_fid_num);
    }
  } else {
    for (int32 idx = 0; idx < dim_num; ++idx) {
      GpuAtomicAdd(dst + idx, *(src + idx));
    }
  }
}

__global__ void ForwardBatchKernel(
    const Gpu2DLaunchConfig config,
    GpuDeviceArrayStruct<PtrWrapper> embeddings_data_list,
    const uint64 *fids_offset_vec, int total_fid_num,
    const int32 *feature_offset_vec, int total_feature_num,
    const uint32 *nfl_offset_vec, int total_nfl_num,
    GpuDeviceArrayStruct<ForwardTaskInfo> task_info_list,
    GpuDeviceArrayStruct<int> each_req_batch_size_list,
    GpuDeviceArrayStruct<int> each_req_nfl_list,
    GpuDeviceArrayStruct<int> each_req_feature_list,
    GpuDeviceArrayStruct<int> each_req_fid_list,
    GpuDeviceArrayStruct<int> each_req_emb_list) {
  ForwardTaskInfo *task_info_list_ptr =
      GetGpuDeviceArrayOnDevice(&task_info_list);
  int *each_req_batch_size_offset =
      GetGpuDeviceArrayOnDevice(&each_req_batch_size_list);
  int *each_req_nfl_offset = GetGpuDeviceArrayOnDevice(&each_req_nfl_list);
  int *each_req_feature_offset =
      GetGpuDeviceArrayOnDevice(&each_req_feature_list);
  int *each_req_fid_offset = GetGpuDeviceArrayOnDevice(&each_req_fid_list);
  int *each_req_emb_offset = GetGpuDeviceArrayOnDevice(&each_req_emb_list);

  const PtrWrapper *embeddings_data_list_ptr =
      GetGpuDeviceArrayOnDevice(&embeddings_data_list);

  bool is_shared;
  int nfl_offset;
  int feature_num;
  ForwardTaskInfo *task_info = nullptr;
  int feature_idx;
  int temp_offset;
  bool init;

  GPU_AXIS_KERNEL_LOOP(task_idx, config.virtual_thread_count.y, Y) {
    task_info = task_info_list_ptr + task_idx;
    GetFeatureInfo(task_info->nfl_idx,
                   nfl_offset_vec + *(each_req_nfl_offset + task_info->req_i),
                   *(each_req_nfl_offset + task_info->req_i + 1) -
                       *(each_req_nfl_offset + task_info->req_i),
                   *(each_req_feature_offset + task_info->req_i + 1) -
                       *(each_req_feature_offset + task_info->req_i),
                   &is_shared, &nfl_offset, &feature_num);
    if (!feature_num) return;  // nfl exits

    GPU_AXIS_KERNEL_LOOP(batch_idx, config.virtual_thread_count.x, X) {
      if (batch_idx >= *(each_req_batch_size_offset + task_info->req_i + 1) -
                           *(each_req_batch_size_offset + task_info->req_i))
        return;                  // out of range
      feature_idx = nfl_offset;  // in single req scope
      if (!is_shared) {
        feature_idx += batch_idx;
      }
      temp_offset =
          (batch_idx + *(each_req_batch_size_offset + task_info->req_i)) *
          task_info->ptr_info.offset;

      if (task_info->out_type == OutType::ADDN) {
        if (task_info->pooling_type == PoolingType::FIRSTN) {  // not support
        } else {
          GatherEmb(feature_idx, task_info->max_sequence_length,
                    task_info->pooling_type, task_info->dim_num,
                    task_info->start,
                    embeddings_data_list_ptr +
                        *(each_req_emb_offset + task_info->req_i),
                    *(each_req_emb_offset + task_info->req_i + 1) -
                        *(each_req_emb_offset + task_info->req_i),
                    fids_offset_vec + *(each_req_fid_offset + task_info->req_i),
                    *(each_req_fid_offset + task_info->req_i + 1) -
                        *(each_req_fid_offset + task_info->req_i),
                    feature_offset_vec +
                        *(each_req_feature_offset + task_info->req_i),
                    *(each_req_feature_offset + task_info->req_i + 1) -
                        *(each_req_feature_offset + task_info->req_i),
                    const_cast<float *>(task_info->ptr_info.ptr + temp_offset),
                    OptimizedSumpoolingGPUWithLock, MemCopyGPU, nullptr,
                    nullptr, nullptr, nullptr);
        }
      } else {
        init = true;
        GatherEmb(
            feature_idx, task_info->max_sequence_length,
            task_info->pooling_type, task_info->dim_num, task_info->start,
            embeddings_data_list_ptr +
                *(each_req_emb_offset + task_info->req_i),
            *(each_req_emb_offset + task_info->req_i + 1) -
                *(each_req_emb_offset + task_info->req_i),
            fids_offset_vec + *(each_req_fid_offset + task_info->req_i),
            *(each_req_fid_offset + task_info->req_i + 1) -
                *(each_req_fid_offset + task_info->req_i),
            feature_offset_vec + *(each_req_feature_offset + task_info->req_i),
            *(each_req_feature_offset + task_info->req_i + 1) -
                *(each_req_feature_offset + task_info->req_i),
            const_cast<float *>(task_info->ptr_info.ptr + temp_offset),
            OptimizedSumpoolingGPU, MemCopyGPU, nullptr, nullptr,
            DefaultGetInitFunc, &init);
      }
    }
  }
}

template <typename T>
struct SetZeroFunctor {
  void operator()(const GPUDevice &d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
  }
};

class MonolithEmbeddingToLayoutOpV3GPU : public MonolithEmbeddingToLayoutOp {
 public:
  explicit MonolithEmbeddingToLayoutOpV3GPU(OpKernelConstruction *ctx,
                                            int verison = 3)
      : MonolithEmbeddingToLayoutOp(ctx, verison) {}
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
                       OpKernelContext *ctx, OpOutputList *layout_tensor_list) {
    GPUDevice gpu_device = ctx->eigen_device<GPUDevice>();
    SetZeroFunctor<float> zero_functor;
    for (int32 idx = 0; idx < layout_tensor_list->size(); ++idx) {
      zero_functor(gpu_device, (*layout_tensor_list)[idx]->flat<float>());
    }

    int each_req_emb_num = embeddings_data.size() / req_num;
    std::vector<ForwardTaskInfo> task_info_vec;
    {
      auto activity =
          std::make_unique<profiler::TraceMe>([]() { return "BuildGPUTask"; });
      for (int req_i = 0; req_i < req_num; req_i++) {
        for (int para_i = 0; para_i < layouts.size(); ++para_i) {
          auto &layout = layouts.at(para_i);
          // CHECK(end - start == 1);
          const ::google::protobuf::RepeatedPtrField<SliceConfig>
              &layout_slice_configs = layout->GetSliceConfig();
          for (uint slice_conf_i = 0;
               slice_conf_i < layout_slice_configs.size(); ++slice_conf_i) {
            const SliceConfig &slice_conf = layout_slice_configs[slice_conf_i];
            int dim_num = slice_conf.end() - slice_conf.start();
            PtrWrapper ptr_info = layout->GetSlice(0, slice_conf);
            const int64 nfl_idx = slice_conf.feature_idx();
            task_info_vec.push_back(ForwardTaskInfo(
                {dim_num, ptr_info, nfl_idx, layout->out_type(),
                 slice_conf.pooling_type(), slice_conf.max_sequence_length(),
                 slice_conf.start(), req_i}));
          }
        }
      }
    }

    GpuDeviceArrayOnHost<ForwardTaskInfo> task_info_list(ctx,
                                                         task_info_vec.size());
    GpuDeviceArrayOnHost<PtrWrapper> embeddings_data_list(
        ctx, embeddings_data.size());
    GpuDeviceArrayOnHost<int> each_req_batch_size_list(
        ctx, each_req_batch_size_offset.size());
    GpuDeviceArrayOnHost<int> each_req_nfl_list(ctx,
                                                each_req_nfl_offset.size());
    GpuDeviceArrayOnHost<int> each_req_feature_list(
        ctx, each_req_feature_offset.size());
    GpuDeviceArrayOnHost<int> each_req_fid_list(ctx,
                                                each_req_fid_offset.size());
    GpuDeviceArrayOnHost<int> each_req_emb_list(ctx, req_num + 1);
    {
      auto activity = std::make_unique<profiler::TraceMe>(
          []() { return "CopyHostValueToDevice"; });
      OP_REQUIRES_OK(ctx, task_info_list.Init());
      for (int i = 0; i < task_info_vec.size(); ++i) {
        task_info_list.Set(i, task_info_vec[i]);
      }
      OP_REQUIRES_OK(ctx, task_info_list.Finalize());

      OP_REQUIRES_OK(ctx, embeddings_data_list.Init());
      for (int i = 0; i < embeddings_data.size(); ++i) {
        embeddings_data_list.Set(i, embeddings_data[i]);
      }
      OP_REQUIRES_OK(ctx, embeddings_data_list.Finalize());

      OP_REQUIRES_OK(ctx, each_req_batch_size_list.Init());
      for (int i = 0; i < each_req_batch_size_offset.size(); ++i) {
        each_req_batch_size_list.Set(i, each_req_batch_size_offset[i]);
      }
      OP_REQUIRES_OK(ctx, each_req_batch_size_list.Finalize());

      OP_REQUIRES_OK(ctx, each_req_nfl_list.Init());
      for (int i = 0; i < each_req_nfl_offset.size(); ++i) {
        each_req_nfl_list.Set(i, each_req_nfl_offset[i]);
      }
      OP_REQUIRES_OK(ctx, each_req_nfl_list.Finalize());

      OP_REQUIRES_OK(ctx, each_req_feature_list.Init());
      for (int i = 0; i < each_req_feature_offset.size(); ++i) {
        each_req_feature_list.Set(i, each_req_feature_offset[i]);
      }
      OP_REQUIRES_OK(ctx, each_req_feature_list.Finalize());

      OP_REQUIRES_OK(ctx, each_req_fid_list.Init());
      for (int i = 0; i < each_req_fid_offset.size(); ++i) {
        each_req_fid_list.Set(i, each_req_fid_offset[i]);
      }
      OP_REQUIRES_OK(ctx, each_req_fid_list.Finalize());

      OP_REQUIRES_OK(ctx, each_req_emb_list.Init());
      for (int i = 0; i < req_num + 1; ++i) {
        each_req_emb_list.Set(i, i * each_req_emb_num);
      }
      OP_REQUIRES_OK(ctx, each_req_emb_list.Finalize());
    }

    auto config =
        GetGpu2DLaunchConfig(batch_size, task_info_vec.size(), gpu_device);
    GpuLaunchKernel(ForwardBatchKernel, config.block_count,
                    config.thread_per_block, 0, gpu_device.stream(), config,
                    embeddings_data_list.data(), fids_offset_vec, total_fid_num,
                    feature_offset_vec, total_feature_num, nfl_offset_vec,
                    total_nfl_num, task_info_list.data(),
                    each_req_batch_size_list.data(), each_req_nfl_list.data(),
                    each_req_feature_list.data(), each_req_fid_list.data(),
                    each_req_emb_list.data());
  }
};

class MonolithEmbeddingToLayoutOpV4GPU
    : public MonolithEmbeddingToLayoutOpV3GPU {
 public:
  explicit MonolithEmbeddingToLayoutOpV4GPU(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOpV3GPU(ctx, 4) {}
};

class MonolithEmbeddingToLayoutOpV5GPU
    : public MonolithEmbeddingToLayoutOpV3GPU {
 public:
  explicit MonolithEmbeddingToLayoutOpV5GPU(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutOpV3GPU(ctx, 5) {}
};

__global__ void BackwardBatchKernel(
    const Gpu2DLaunchConfig config,
    GpuDeviceArrayStruct<ForwardTaskInfo> task_info_list,
    const uint64 *fids_offset_vec, int total_fid_num,
    const int32 *feature_offset_vec, int total_feature_num,
    const uint32 *nfl_offset_vec, int total_nfl_num, int batch_size,
    GpuDeviceArrayStruct<PtrWrapper> embeddings_grads_data) {
  const ForwardTaskInfo *task_info_list_ptr =
      GetGpuDeviceArrayOnDevice(&task_info_list);

  PtrWrapper *embeddings_grads_data_list_ptr =
      GetGpuDeviceArrayOnDevice(&embeddings_grads_data);

  bool is_shared;
  int nfl_offset;
  int feature_num;
  const ForwardTaskInfo *task_info = nullptr;
  int feature_idx;
  int temp_offset;

  GPU_AXIS_KERNEL_LOOP(task_idx, config.virtual_thread_count.y, Y) {
    task_info = task_info_list_ptr + task_idx;

    GetFeatureInfo(task_info->nfl_idx, nfl_offset_vec, total_nfl_num,
                   total_feature_num, &is_shared, &nfl_offset, &feature_num);

    if (!feature_num) return;  // nfl exits

    GPU_AXIS_KERNEL_LOOP(batch_idx, config.virtual_thread_count.x, X) {
      feature_idx = nfl_offset;
      if (!is_shared) {
        feature_idx += batch_idx;
      }
      temp_offset = batch_idx * task_info->ptr_info.offset;

      ScatterGrad(
          feature_idx, task_info->max_sequence_length, task_info->pooling_type,
          task_info->ptr_info.ptr + temp_offset, task_info->dim_num,
          task_info->start, fids_offset_vec, total_fid_num, feature_offset_vec,
          total_feature_num, embeddings_grads_data.size,
          embeddings_grads_data_list_ptr, OptimizedSumpoolingGPUWithLock,
          nullptr, nullptr, nullptr, nullptr);
    }
  }
}

class MonolithEmbeddingToLayoutGradOpV3GPU
    : public MonolithEmbeddingToLayoutGradOp {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV3GPU(OpKernelConstruction *ctx,
                                                int verison = 3)
      : MonolithEmbeddingToLayoutGradOp(ctx, verison) {}
  void TaskRun(const std::vector<std::shared_ptr<Layout>> &layouts,
               const std::vector<std::pair<int, int>> *ufid_grads_info,
               const uint64 *fids_offset_vec, int total_fid_num,
               const int32 *feature_offset_vec, int total_feature_num,
               const uint32 *nfl_offset_vec, int total_nfl_num, int batch_size,
               OpKernelContext *ctx, OpOutputList *embeddings_grad_list,
               std::vector<PtrWrapper> *embeddings_grads_data, GroupA *init) {
    GPUDevice gpu_device = ctx->eigen_device<GPUDevice>();
    SetZeroFunctor<float> zero_functor;
    for (int32 idx = 0; idx < embeddings_grad_list->size(); ++idx) {
      zero_functor(gpu_device, (*embeddings_grad_list)[idx]->flat<float>());
    }
    std::vector<ForwardTaskInfo> task_info_vec;
    for (int64 para_i = 0; para_i < layouts.size(); ++para_i) {
      auto &layout = layouts.at(para_i);
      // CHECK(end - start == 1);
      const ::google::protobuf::RepeatedPtrField<SliceConfig>
          &layout_slice_configs = layout->GetSliceConfig();
      for (const SliceConfig &slice_conf : layout_slice_configs) {
        int dim_num = slice_conf.end() - slice_conf.start();
        PtrWrapper ptr_info = layout->GetSlice(0, slice_conf);
        const int64 nfl_idx = slice_conf.feature_idx();
        task_info_vec.push_back(ForwardTaskInfo(
            {dim_num, ptr_info, nfl_idx, layout->out_type(),
             slice_conf.pooling_type(), slice_conf.max_sequence_length(),
             slice_conf.start()}));
      }
    }
    GpuDeviceArrayOnHost<ForwardTaskInfo> task_info_list(ctx,
                                                         task_info_vec.size());
    OP_REQUIRES_OK(ctx, task_info_list.Init());
    for (int i = 0; i < task_info_vec.size(); ++i) {
      task_info_list.Set(i, task_info_vec[i]);
    }
    OP_REQUIRES_OK(ctx, task_info_list.Finalize());

    GpuDeviceArrayOnHost<PtrWrapper> embeddings_grads_data_list(
        ctx, embeddings_grads_data->size());
    OP_REQUIRES_OK(ctx, embeddings_grads_data_list.Init());
    for (int i = 0; i < embeddings_grads_data->size(); ++i) {
      embeddings_grads_data_list.Set(i, (*embeddings_grads_data)[i]);
    }
    OP_REQUIRES_OK(ctx, embeddings_grads_data_list.Finalize());

    auto config =
        GetGpu2DLaunchConfig(batch_size, task_info_vec.size(), gpu_device);
    GpuLaunchKernel(
        BackwardBatchKernel, config.block_count, config.thread_per_block, 0,
        gpu_device.stream(), config, task_info_list.data(), fids_offset_vec,
        total_fid_num, feature_offset_vec, total_feature_num, nfl_offset_vec,
        total_nfl_num, batch_size, embeddings_grads_data_list.data());
  }
};

class MonolithEmbeddingToLayoutGradOpV4GPU
    : public MonolithEmbeddingToLayoutGradOpV3GPU {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV4GPU(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOpV3GPU(ctx, 4) {}
};

class MonolithEmbeddingToLayoutGradOpV5GPU
    : public MonolithEmbeddingToLayoutGradOpV3GPU {
 public:
  explicit MonolithEmbeddingToLayoutGradOpV5GPU(OpKernelConstruction *ctx)
      : MonolithEmbeddingToLayoutGradOpV3GPU(ctx, 5) {}
};

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV3")
                            .Device(DEVICE_GPU)
                            .HostMemory("batch_size"),
                        MonolithEmbeddingToLayoutOpV3GPU);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutGradV3")
                            .Device(DEVICE_GPU)
                            .HostMemory("batch_size"),
                        MonolithEmbeddingToLayoutGradOpV3GPU);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV4")
                            .Device(DEVICE_GPU)
                            .HostMemory("batch_size")
                            .HostMemory("fid_list_emb_row_lenth"),
                        MonolithEmbeddingToLayoutOpV4GPU);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutGradV4")
                            .Device(DEVICE_GPU)
                            .HostMemory("batch_size")
                            .HostMemory("fid_list_emb_row_lenth"),
                        MonolithEmbeddingToLayoutGradOpV4GPU);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutV5")
                            .Device(DEVICE_GPU)
                            .HostMemory("batch_size")
                            .HostMemory("nfl_size")
                            .HostMemory("feature_size")
                            .HostMemory("fid_size")
                            .HostMemory("emb_size"),
                        MonolithEmbeddingToLayoutOpV5GPU);

REGISTER_KERNEL_BUILDER(Name("MonolithEmbeddingToLayoutGradV5")
                            .Device(DEVICE_GPU)
                            .HostMemory("batch_size"),
                        MonolithEmbeddingToLayoutGradOpV5GPU);

}  // namespace fused_layout
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
