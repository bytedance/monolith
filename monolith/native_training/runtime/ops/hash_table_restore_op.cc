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

#include <atomic>
#include <cstring>
#include <string>

#include "absl/strings/str_cat.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/file_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/threadpool.h"
namespace tensorflow {
namespace monolith_tf {
namespace {

// Carries the data through async process.
// It will ref and unref |p_hash_table|.
struct AsyncPack {
  AsyncPack(OpKernelContext* p_ctx, EmbeddingHashTableTfBridge* p_hash_table,
            std::string p_basename, std::function<void()> p_done,
            int p_thread_num)
      : ctx(p_ctx),
        basename(p_basename),
        hash_table(p_hash_table),
        done(std::move(p_done)),
        thread_num(p_thread_num),
        finish_num(0),
        status(thread_num) {
    hash_table->Ref();
  }

  ~AsyncPack() { hash_table->Unref(); }

  OpKernelContext* ctx;
  std::string basename;
  EmbeddingHashTableTfBridge* hash_table;
  std::function<void()> done;
  const int thread_num;
  std::atomic_int finish_num;
  std::vector<Status> status;
};

}  // namespace

class HashTableRestoreOp : public AsyncOpKernel {
 public:
  explicit HashTableRestoreOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table), done);
    core::ScopedUnref unref(hash_table);
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
    hash_table->Clear();
    int nshards = files.size();
    auto pack =
        new AsyncPack(ctx, hash_table, basename, std::move(done), nshards);
    for (int i = 0; i < nshards; ++i) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i, nshards] {
            WorkerThread({i, nshards}, pack);
          });
    }
  }

 private:
  void WorkerThread(EmbeddingHashTableTfBridge::DumpShard shard, AsyncPack* p) {
    p->status[shard.idx] = RestoreOneShard(shard, p);
    if (p->finish_num.fetch_add(1) == p->thread_num - 1) {
      Cleanup(p);
    }
  }

  Status RestoreOneShard(EmbeddingHashTableTfBridge::DumpShard shard,
                         AsyncPack* p) {
    std::string filename =
        GetShardedFileName(p->basename, shard.idx, shard.total);
    std::unique_ptr<RandomAccessFile> f;

    TF_RETURN_IF_ERROR(p->ctx->env()->NewRandomAccessFile(filename, &f));
    io::RecordReaderOptions opts;
    opts.buffer_size = 10 * 1024 * 1024;
    io::SequentialRecordReader reader(f.get(), opts);
    Status restore_status;
    auto get_fn = [&reader, &restore_status, &filename](
        EmbeddingHashTableTfBridge::EntryDump* dump, int64_t* max_update_ts) {
      Status s = GetRecord(&reader, dump);
      if (TF_PREDICT_FALSE(!s.ok())) {
        if (!errors::IsOutOfRange(s)) {
          restore_status = s;
        }
        return false;
      }
      if (!dump->has_last_update_ts_sec()) {
        dump->set_last_update_ts_sec(0);
      }

      *max_update_ts = std::max(dump->last_update_ts_sec(), *max_update_ts);
      return true;
    };

    TF_RETURN_IF_ERROR(p->hash_table->Restore(p->ctx, shard, get_fn));
    TF_RETURN_IF_ERROR(restore_status);
    return Status::OK();
  }

  static Status GetRecord(io::SequentialRecordReader* reader,
                          EmbeddingHashTableTfBridge::EntryDump* dump) {
    tstring s;
    TF_RETURN_IF_ERROR(reader->ReadRecord(&s));
    if (!dump->ParseFromArray(s.data(), s.size())) {
      return errors::FailedPrecondition(
          "Unable to parse data. Data might be corrupted");
    }
    return Status::OK();
  }

  // Clean up when all shards are done.
  void Cleanup(AsyncPack* p) {
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

REGISTER_OP("MonolithHashTableRestore")
    .Input("handle: resource")
    .Input("basename: string")
    .Output("output_handle: resource")
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("MonolithHashTableRestore").Device(DEVICE_CPU),
                        HashTableRestoreOp);
}  // namespace tensorflow
}  // namespace monolith_tf