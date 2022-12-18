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

#include <memory>

#include "monolith/native_training/runtime/hash_filter/dummy_hash_filter.h"
#include "monolith/native_training/runtime/hash_filter/sliding_hash_filter.h"
#include "monolith/native_training/runtime/ops/file_utils.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace monolith_tf {

using ::monolith::hash_table::SlidingHashFilterMetaDump;
using ::monolith::hash_table::HashFilterSplitMetaDump;
using ::monolith::hash_table::HashFilterSplitDataDump;

class HashFilterSaveOp : public AsyncOpKernel {
 public:
  explicit HashFilterSaveOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashFilterTfBridge* hash_filter = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &hash_filter), done);
    core::ScopedUnref unref(hash_filter);
    const Tensor& basename_tensor = ctx->input(1);
    const std::string basename = basename_tensor.scalar<tstring>()();
    const std::string dirname = std::string(io::Dirname(basename));
    OP_REQUIRES_OK_ASYNC(ctx, ctx->env()->RecursivelyCreateDir(dirname), done);
    ctx->set_output(0, ctx->input(0));
    int nsplits = hash_filter->GetSplitNum();
    if (nsplits == 0) {
      done();
      return;
    }
    auto pack = new HashFilterAsyncPack(ctx, hash_filter, basename,
                                        std::move(done), nsplits);
    for (int i = 0; i < nsplits; ++i) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, i, nsplits, pack] { WorkerThread(i, nsplits, pack); });
    }
  }

 private:
  void WorkerThread(int split_idx, int nsplits, HashFilterAsyncPack* p) {
    p->status[split_idx] = SaveOneSplit(split_idx, nsplits, p);
    if (p->finish_num.fetch_add(1) == p->thread_num - 1) {
      Cleanup(p);
    }
  }

  Status SaveOneSplit(int split_idx, int nsplits, HashFilterAsyncPack* p) {
    std::string filename = GetShardedFileName(p->basename, split_idx, nsplits);
    std::string tmp_filename = absl::StrCat(filename, "-tmp-", random::New64());
    std::unique_ptr<WritableFile> f;
    TF_RETURN_IF_ERROR(p->ctx->env()->NewWritableFile(tmp_filename, &f));
    io::RecordWriter writer(f.get());
    Status write_status;
    // In theory, we should stop writing once write failed.
    // But this requires a lot of refactoring and currently we only do 2 writes.
    // So we keep it as it is here.
    auto write_data_fn = [this, &writer,
                          &write_status](HashFilterSplitDataDump dump) {
      Status s = writer.WriteRecord(dump.SerializeAsString());
      if (TF_PREDICT_FALSE(!s.ok())) {
        write_status.Update(s);
      }
    };
    auto write_meta_fn = [this, &writer,
                          &write_status](HashFilterSplitMetaDump dump) {
      Status s = writer.WriteRecord(dump.SerializeAsString());
      if (TF_PREDICT_FALSE(!s.ok())) {
        write_status.Update(s);
      }
    };
    TF_RETURN_IF_ERROR(
        p->hash_filter->Save(split_idx, write_meta_fn, write_data_fn));
    TF_RETURN_IF_ERROR(write_status);
    TF_RETURN_IF_ERROR(writer.Close());
    TF_RETURN_IF_ERROR(f->Close());
    TF_RETURN_IF_ERROR(p->ctx->env()->RenameFile(tmp_filename, filename));
    return Status::OK();
  }

  // Clean up when all shards are done.
  void Cleanup(HashFilterAsyncPack* p) {
    auto done = [p]() {
      // We want to delete p first and then call done.
      auto done = std::move(p->done);
      delete p;
      done();
    };
    for (int i = 0; i < p->thread_num; ++i) {
      OP_REQUIRES_OK_ASYNC(p->ctx, p->status[i], done);
    }
    done();
  }
};

REGISTER_OP("MonolithHashFilterSave")
    .Input("handle: resource")
    .Input("basename: string")
    .Output("output_handle: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithHashFilterSave").Device(DEVICE_CPU),
                        HashFilterSaveOp);

}  // namespace monolith_tf
}  // namespace tensorflow
