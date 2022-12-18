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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/threadpool.h"

#include "monolith/native_training/runtime/hash_filter/dummy_hash_filter.h"
#include "monolith/native_training/runtime/hash_filter/sliding_hash_filter.h"
#include "monolith/native_training/runtime/ops/file_utils.h"
#include "monolith/native_training/runtime/ops/hash_filter_tf_bridge.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace tensorflow {
namespace monolith_tf {

using ::monolith::hash_table::SlidingHashFilterMetaDump;
using ::monolith::hash_table::HashFilterSplitMetaDump;
using ::monolith::hash_table::HashFilterSplitDataDump;

class HashFilterRestoreOp : public AsyncOpKernel {
 public:
  explicit HashFilterRestoreOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashFilterTfBridge* hash_filter = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &hash_filter), done);
    core::ScopedUnref unref(hash_filter);
    const Tensor& basename_tensor = ctx->input(1);
    const std::string basename = basename_tensor.scalar<tstring>()();
    std::vector<std::string> files;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->env()->GetMatchingPaths(absl::StrCat(basename, "-*"), &files),
        done);

    OP_REQUIRES_OK_ASYNC(ctx, ValidateShardedFiles(basename, files), done);
    OP_REQUIRES_ASYNC(ctx, !files.empty(),
                      errors::NotFound("Unable to find the dump files for: ",
                                       name(), " in ", basename),
                      done);
    ctx->set_output(0, ctx->input(0));
    int nsplits = files.size();
    auto pack = new HashFilterAsyncPack(ctx, hash_filter, basename,
                                        std::move(done), nsplits);
    for (int i = 0; i < nsplits; ++i) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i, nsplits] { WorkerThread(i, nsplits, pack); });
    }
  }

 private:
  void WorkerThread(int split_idx, int nsplits, HashFilterAsyncPack* p) {
    p->status[split_idx] = RestoreOneSplit(split_idx, nsplits, p);
    if (p->finish_num.fetch_add(1) == p->thread_num - 1) {
      Cleanup(p);
    }
  }

  Status RestoreOneSplit(int split_idx, int nsplits, HashFilterAsyncPack* p) {
    std::string filename = GetShardedFileName(p->basename, split_idx, nsplits);
    std::unique_ptr<RandomAccessFile> f;

    TF_RETURN_IF_ERROR(p->ctx->env()->NewRandomAccessFile(filename, &f));
    io::RecordReaderOptions opts;
    opts.buffer_size = 10 * 1024 * 1024;
    io::SequentialRecordReader reader(f.get(), opts);
    Status restore_status;
    auto get_meta_fn = [&reader,
                        &restore_status](HashFilterSplitMetaDump* dump) {
      Status s = GetMetaRecord(&reader, dump);
      if (TF_PREDICT_FALSE(!s.ok())) {
        if (!errors::IsOutOfRange(s)) {
          restore_status = s;
        }
        return false;
      }
      return true;
    };
    auto get_data_fn = [&reader,
                        &restore_status](HashFilterSplitDataDump* dump) {
      Status s = GetDataRecord(&reader, dump);
      if (TF_PREDICT_FALSE(!s.ok())) {
        if (!errors::IsOutOfRange(s)) {
          restore_status = s;
        }
        return false;
      }
      return true;
    };

    TF_RETURN_IF_ERROR(
        p->hash_filter->Restore(split_idx, get_meta_fn, get_data_fn));
    TF_RETURN_IF_ERROR(restore_status);
    return Status::OK();
  }

  static Status GetMetaRecord(io::SequentialRecordReader* reader,
                              HashFilterSplitMetaDump* dump) {
    tstring s;
    TF_RETURN_IF_ERROR(reader->ReadRecord(&s));
    if (!dump->ParseFromArray(s.data(), s.size())) {
      return errors::FailedPrecondition(
          "Unable to parse data. Data might be corrupted");
    }
    return Status::OK();
  }

  static Status GetDataRecord(io::SequentialRecordReader* reader,
                              HashFilterSplitDataDump* dump) {
    tstring s;
    TF_RETURN_IF_ERROR(reader->ReadRecord(&s));
    if (!dump->ParseFromArray(s.data(), s.size())) {
      return errors::FailedPrecondition(
          "Unable to parse data. Data might be corrupted");
    }
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

REGISTER_OP("MonolithHashFilterRestore")
    .Input("handle: resource")
    .Input("basename: string")
    .Output("output_handle: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithHashFilterRestore").Device(DEVICE_CPU),
                        HashFilterRestoreOp);

}  // namespace monolith_tf
}  // namespace tensorflow
