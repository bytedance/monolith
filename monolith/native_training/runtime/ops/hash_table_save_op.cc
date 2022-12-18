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

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/file_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
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

const int kAutoTune = -1;

}  // namespace

class HashTableSaveOp : public AsyncOpKernel {
 public:
  explicit HashTableSaveOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nshards", &nshards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("random_sleep_ms", &random_sleep_ms_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_expire_time_config",
                                     &slot_expire_time_config_serialized_));
    if (!slot_expire_time_config_serialized_.empty()) {
      OP_REQUIRES(
          ctx,
          slot_expire_time_config_.ParseFromString(
              slot_expire_time_config_serialized_),
          errors::InvalidArgument("Unable to parse config. Make sure it "
                                  "is serialized version of "
                                  "SlotExpireTimeConfig."));
    }

    slot_to_expire_time_.resize(get_max_slot_number(),
                                slot_expire_time_config_.default_expire_time());
    for (const auto& slot_expire_time :
         slot_expire_time_config_.slot_expire_times()) {
      slot_to_expire_time_[slot_expire_time.slot()] =
          slot_expire_time.expire_time();
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    EmbeddingHashTableTfBridge* hash_table = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &hash_table), done);
    core::ScopedUnref unref(hash_table);
    const Tensor& basename_tensor = ctx->input(1);
    const std::string basename = basename_tensor.scalar<tstring>()();
    const std::string dirname = std::string(io::Dirname(basename));
    OP_REQUIRES_OK_ASYNC(ctx, ctx->env()->RecursivelyCreateDir(dirname), done);
    ctx->set_output(0, ctx->input(0));
    int real_nshards = PickNshards(hash_table);
    auto pack =
        new AsyncPack(ctx, hash_table, basename, std::move(done), real_nshards);
    for (int i = 0; i < real_nshards; ++i) {
      // !important: When using GPU, tensorflow_cpu_worker_threads' are bound to
      // device 0 regardless of the correct device id of the current process.
      // That means one has to use the CUDA_VISIBLE_DEVICES
      // environment vairable to make sure that only one GPU is visible each
      // process. Otherwise something like this will happen: some device memory
      // is allocated on device 3, but accessing that memory in this thread as
      // device 0 will cause illegal memory errors
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i, real_nshards] {
            WorkerThread({i, real_nshards}, pack);
          });
    }
  }

  int PickNshards(EmbeddingHashTableTfBridge* table) {
    if (nshards_ >= 0) return nshards_;
    const int64 size = table->Size();
    const int64 kBaseline = 1000000ll;
    return std::min(4LL, std::max(1LL, size / kBaseline));
  }

 private:
  void WorkerThread(EmbeddingHashTableTfBridge::DumpShard shard, AsyncPack* p) {
    absl::BitGen bitgen;
    absl::SleepFor(
        absl::Milliseconds(absl::Uniform(bitgen, 0, random_sleep_ms_)));
    p->status[shard.idx] = SaveOneShard(shard, p);
    if (p->finish_num.fetch_add(1) == p->thread_num - 1) {
      Cleanup(p);
    }
  }

  Status SaveOneShard(EmbeddingHashTableTfBridge::DumpShard shard,
                      AsyncPack* p) {
    std::string filename =
        GetShardedFileName(p->basename, shard.idx, shard.total);
    std::string tmp_filename = absl::StrCat(filename, "-tmp-", random::New64());
    std::unique_ptr<WritableFile> f;
    TF_RETURN_IF_ERROR(p->ctx->env()->NewWritableFile(tmp_filename, &f));
    io::RecordWriter writer(f.get());
    Status write_status;
    int64_t max_update_ts_sec = p->hash_table->max_update_ts_sec();
    auto write_fn = [this, &max_update_ts_sec, &writer, &write_status](
                        EmbeddingHashTableTfBridge::EntryDump dump) {
      int64_t slot_id = slot_id_v2(dump.id());
      // Elements of slot_to_expire_time_ are in days.
      // last_update_ts_sec is seconds since the Epoch.
      if (max_update_ts_sec - dump.last_update_ts_sec() >=
          slot_to_expire_time_[slot_id] * 24 * 3600) {
        return true;
      }
      Status s = writer.WriteRecord(dump.SerializeAsString());
      if (TF_PREDICT_FALSE(!s.ok())) {
        // OK to throw here since it will be catched.
        write_status = s;
        return false;
      }
      return true;
    };
    EmbeddingHashTableTfBridge::DumpIterator iter;
    TF_RETURN_IF_ERROR(p->hash_table->Save(p->ctx, shard, write_fn, &iter));
    TF_RETURN_IF_ERROR(writer.Close());
    TF_RETURN_IF_ERROR(f->Close());
    TF_RETURN_IF_ERROR(p->ctx->env()->RenameFile(tmp_filename, filename));
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

  int nshards_;
  int64 random_sleep_ms_;
  std::string slot_expire_time_config_serialized_;
  monolith::hash_table::SlotExpireTimeConfig slot_expire_time_config_;
  std::vector<int64_t> slot_to_expire_time_;
};

REGISTER_OP("MonolithHashTableSave")
    .Input("handle: resource")
    .Input("basename: string")
    .Output("output_handle: resource")
    .Attr("nshards: int=-1")
    .Attr("random_sleep_ms: int=0")
    .Attr("slot_expire_time_config: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("MonolithHashTableSave").Device(DEVICE_CPU),
                        HashTableSaveOp);
}  // namespace monolith_tf
}  // namespace tensorflow
