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
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "monolith/native_training/runtime/ops/file_utils.h"
#include "monolith/native_training/runtime/ops/multi_hash_table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

// Carries the data through async process.
struct AsyncPack {
  AsyncPack(OpKernelContext* p_ctx, core::RefCountPtr<MultiHashTable> p_mtable,
            std::string p_basename, std::function<void()> p_done,
            int p_thread_num)
      : ctx(p_ctx),
        basename(p_basename),
        mtable(std::move(p_mtable)),
        done(std::move(p_done)),
        status(p_thread_num) {}

  ~AsyncPack() {
    for (const auto& s : status) {
      OP_REQUIRES_OK_ASYNC(ctx, s, done);
    }
    done();
  }

  OpKernelContext* ctx;
  std::string basename;
  core::RefCountPtr<MultiHashTable> mtable;
  std::function<void()> done;
  mutable std::vector<Status> status;
};

struct EntryDumpIter {
  explicit EntryDumpIter(io::SequentialRecordReader* reader_, int64_t limit_)
      : reader(reader_), limit(limit_), offset(0) {}
  bool GetNext(EmbeddingHashTableTfBridge::EntryDump* dump, Status* status) {
    *status = Status::OK();
    if (offset >= limit) return false;
    tstring s;
    *status = reader->ReadRecord(&s);
    if (!status->ok() || !dump->ParseFromArray(s.data(), s.size())) {
      *status = errors::DataLoss("Parse entry failed!");
      return false;
    }
    offset++;
    return true;
  }

  io::SequentialRecordReader* reader;
  int64_t limit;
  int64_t offset;
};

const char* const kShardedMetadataFileFormat = "%s.meta-%05d-of-%05d";

std::string GetShardedMetadataFileName(absl::string_view basename, int shard,
                                       int nshards) {
  return absl::StrFormat(kShardedMetadataFileFormat, basename, shard, nshards);
}

}  // namespace

class MultiHashTableSaveOp : public AsyncOpKernel {
 public:
  explicit MultiHashTableSaveOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nshards", &nshards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_expire_time_config",
                                     &slot_expire_time_config_serialized_));
    if (!slot_expire_time_config_serialized_.empty()) {
      OP_REQUIRES(
          ctx, slot_expire_time_config_.ParseFromString(
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
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &mtable), done);
    const Tensor& basename_tensor = ctx->input(1);
    const std::string basename = basename_tensor.scalar<tstring>()();
    const std::string dirname = std::string(io::Dirname(basename));
    OP_REQUIRES_OK_ASYNC(ctx, ctx->env()->RecursivelyCreateDir(dirname), done);

    int real_nshards = PickNshards(*mtable);
    auto pack = std::make_shared<const AsyncPack>(
        ctx, std::move(mtable), basename, std::move(done), real_nshards);
    for (int i = 0; i < real_nshards; ++i) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i, real_nshards] {
            WorkerThread({i, real_nshards}, pack);
          });
    }
    ctx->set_output(0, ctx->input(0));
  }

 private:
  void WorkerThread(EmbeddingHashTableTfBridge::DumpShard shard,
                    std::shared_ptr<const AsyncPack> p) {
    p->status[shard.idx] = SaveOneShard(shard, p.get());
  }

  Status SaveOneShard(EmbeddingHashTableTfBridge::DumpShard shard,
                      const AsyncPack* p) {
    const std::string filename =
        GetShardedFileName(p->basename, shard.idx, shard.total);
    const std::string meta_filename =
        GetShardedMetadataFileName(p->basename, shard.idx, shard.total);
    const std::string tmp_filename =
        GetShardedFileName(absl::StrCat(p->basename, "-tmp-", random::New64()),
                           shard.idx, shard.total);
    const std::string tmp_meta_filename = GetShardedMetadataFileName(
        absl::StrCat(p->basename, "-tmp-", random::New64()), shard.idx,
        shard.total);
    std::unique_ptr<WritableFile> fp;
    TF_RETURN_IF_ERROR(p->ctx->env()->NewWritableFile(tmp_filename, &fp));
    std::unique_ptr<WritableFile> fp_meta;
    TF_RETURN_IF_ERROR(
        p->ctx->env()->NewWritableFile(tmp_meta_filename, &fp_meta));

    io::RecordWriterOptions options;
    options.compression_type = io::RecordWriterOptions::SNAPPY_COMPRESSION;
    io::RecordWriterOptions options_meta;
    io::RecordWriter writer(fp.get(), options);
    io::RecordWriter meta_writer(fp_meta.get(), options_meta);
    Status write_status;

    for (int table_idx = 0; table_idx < p->mtable->size(); table_idx++) {
      int64_t num_entries = 0;
      const EmbeddingHashTableTfBridge* table = p->mtable->table(table_idx);
      const std::string& table_name = p->mtable->name(table_idx);
      int64_t max_update_ts_sec = table->max_update_ts_sec();
      auto write_fn = [&](EmbeddingHashTableTfBridge::EntryDump dump) {
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
        num_entries++;
        return true;
      };
      EmbeddingHashTableTfBridge::DumpIterator iter;
      TF_RETURN_IF_ERROR(table->Save(p->ctx, shard, write_fn, &iter));

      monolith::hash_table::MultiHashTableMetadata meta;
      meta.set_table_name(table_name);
      meta.set_num_entries(num_entries);
      TF_RETURN_IF_ERROR(meta_writer.WriteRecord(meta.SerializeAsString()));
    }

    TF_RETURN_IF_ERROR(writer.Close());
    TF_RETURN_IF_ERROR(meta_writer.Close());
    TF_RETURN_IF_ERROR(fp->Close());
    TF_RETURN_IF_ERROR(fp_meta->Close());
    TF_RETURN_IF_ERROR(p->ctx->env()->RenameFile(tmp_filename, filename));
    TF_RETURN_IF_ERROR(
        p->ctx->env()->RenameFile(tmp_meta_filename, meta_filename));
    return Status::OK();
  }

  int PickNshards(const MultiHashTable& mtable) {
    if (nshards_ >= 0) return nshards_;
    int64 total_size = 0;
    const int64 kBaseline = 1000000ll;
    for (size_t i = 0; i < mtable.size(); i++) {
      total_size += mtable.table(i)->Size();
    }
    return std::min(4LL, std::max(1LL, total_size / kBaseline));
  }

  int nshards_;
  std::string slot_expire_time_config_serialized_;
  monolith::hash_table::SlotExpireTimeConfig slot_expire_time_config_;
  std::vector<int64_t> slot_to_expire_time_;
};

class MultiHashTableRestoreOp : public AsyncOpKernel {
 public:
  explicit MultiHashTableRestoreOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    core::RefCountPtr<MultiHashTable> mtable;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &mtable), done);

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

    int nshards = files.size();
    auto pack = std::make_shared<const AsyncPack>(
        ctx, std::move(mtable), basename, std::move(done), nshards);
    for (int i = 0; i < nshards; ++i) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i, nshards] {
            WorkerThread({i, nshards}, pack);
          });
    }
    ctx->set_output(0, ctx->input(0));
  }

 private:
  void WorkerThread(EmbeddingHashTableTfBridge::DumpShard shard,
                    std::shared_ptr<const AsyncPack> p) {
    p->status[shard.idx] = RestoreOneShard(shard, p.get());
  }

  Status RestoreOneShard(EmbeddingHashTableTfBridge::DumpShard shard,
                         const AsyncPack* p) {
    std::string filename =
        GetShardedFileName(p->basename, shard.idx, shard.total);
    std::string meta_filename =
        GetShardedMetadataFileName(p->basename, shard.idx, shard.total);
    std::unique_ptr<RandomAccessFile> fp;
    std::unique_ptr<RandomAccessFile> fp_meta;
    TF_RETURN_IF_ERROR(p->ctx->env()->NewRandomAccessFile(filename, &fp));
    TF_RETURN_IF_ERROR(
        p->ctx->env()->NewRandomAccessFile(meta_filename, &fp_meta));

    io::RecordReaderOptions options;
    options.compression_type = io::RecordReaderOptions::SNAPPY_COMPRESSION;
    options.buffer_size = 10 * 1024 * 1024;
    io::SequentialRecordReader reader(fp.get(), options);
    io::RecordReaderOptions options_meta;
    io::SequentialRecordReader meta_reader(fp_meta.get(), options_meta);

    absl::flat_hash_set<std::string> tables_in_shard;
    absl::flat_hash_map<std::string, int> name_to_idx;
    for (int i = 0; i < p->mtable->size(); ++i) {
      name_to_idx.insert({p->mtable->name(i), i});
    }
    bool eof = false;
    Status restore_status;
    while (!eof) {
      tstring meta_pb;
      Status meta_status = meta_reader.ReadRecord(&meta_pb);
      if (!meta_status.ok()) {
        if (errors::IsOutOfRange(meta_status)) {
          eof = true;
          break;
        } else {
          return errors::DataLoss("Read table metadata failed!");
        }
      }

      monolith::hash_table::MultiHashTableMetadata meta;
      if (!meta.ParseFromArray(meta_pb.data(), meta_pb.size())) {
        return errors::DataLoss("Parse table metadata failed!");
      }
      auto name_iter = name_to_idx.find(meta.table_name());
      if (name_iter == name_to_idx.end()) {
        if (shard.idx == 0) {
          LOG(INFO) << "Table " << meta.table_name()
                    << " in checkpoint. skipped.";
        }
        tstring dummy_str;
        for (int64_t i = 0; i < meta.num_entries(); i++) {
          TF_RETURN_IF_ERROR(reader.ReadRecord(&dummy_str));
        }
        continue;
      }
      tables_in_shard.insert(meta.table_name());
      EmbeddingHashTableTfBridge* table = p->mtable->table(name_iter->second);

      EntryDumpIter entry_iter(&reader, meta.num_entries());
      auto get_fn = [&](EmbeddingHashTableTfBridge::EntryDump* dump,
                        int64_t* max_update_ts) {
        if (!entry_iter.GetNext(dump, &restore_status)) return false;
        if (!dump->has_last_update_ts_sec()) {
          dump->set_last_update_ts_sec(0);
        }
        *max_update_ts = std::max(dump->last_update_ts_sec(), *max_update_ts);
        return true;
      };
      TF_RETURN_IF_ERROR(table->Restore(p->ctx, shard, get_fn));
      TF_RETURN_IF_ERROR(restore_status);
    }
    if (shard.idx == 0) {
      for (const std::string& table_name : p->mtable->names()) {
        if (!tables_in_shard.contains(table_name)) {
          LOG(WARNING) << "Table " << table_name << " not found checkpoint.";
        }
      }
    }
    if (!eof)
      return errors::DataLoss("Couldn't read all of checkpoint shard ",
                              shard.idx);
    return Status::OK();
  }
};

class MultiHashTableFeatureStatOp : public OpKernel {
 public:
  explicit MultiHashTableFeatureStatOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& basename_tensor = ctx->input(0);
    const std::string basename = basename_tensor.scalar<tstring>()();
    std::vector<std::string> files;
    OP_REQUIRES_OK(ctx, ctx->env()->GetMatchingPaths(
                            absl::StrCat(basename, "-*"), &files));

    OP_REQUIRES_OK(ctx, ValidateShardedFiles(basename, files));
    OP_REQUIRES(ctx, !files.empty(),
                errors::NotFound("Unable to find the dump files for: ", name(),
                                 " in ", basename));

    absl::flat_hash_map<std::string, uint64_t> feature_count;
    int nshards = files.size();
    for (int idx = 0; idx < nshards; ++idx) {
      std::string filename = GetShardedMetadataFileName(basename, idx, nshards);
      std::unique_ptr<RandomAccessFile> fp;
      OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(filename, &fp));

      io::RecordReaderOptions options;
      io::SequentialRecordReader reader(fp.get(), options);

      bool eof = false;
      while (!eof) {
        tstring meta_pb;
        Status s = reader.ReadRecord(&meta_pb);
        if (!s.ok()) {
          if (errors::IsOutOfRange(s)) {
            eof = true;
            break;
          } else {
            OP_REQUIRES(ctx, s.ok(),
                        errors::DataLoss("Read table metadata failed!"));
          }
        }

        monolith::hash_table::MultiHashTableMetadata meta;
        OP_REQUIRES(ctx, meta.ParseFromArray(meta_pb.data(), meta_pb.size()),
                    errors::DataLoss("Parse table metadata failed!"));
        if (!feature_count.contains(meta.table_name())) {
          feature_count[meta.table_name()] = 0;
        }
        feature_count[meta.table_name()] += meta.num_entries();
      }

      OP_REQUIRES(ctx, eof, errors::DataLoss(
                                "Couldn't read all of checkpoint shard ", idx));
    }

    int num_tables = feature_count.size();
    Tensor* features;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({
                                                    num_tables,
                                                }),
                                             &features));
    auto features_vec = features->vec<tstring>();
    Tensor* counts;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({
                                                    num_tables,
                                                }),
                                             &counts));
    auto counts_vec = counts->vec<uint64_t>();
    int feature_iter = 0;
    for (const auto& it : feature_count) {
      features_vec(feature_iter) = it.first;
      counts_vec(feature_iter) = it.second;
      feature_iter++;
    }
  }
};

REGISTER_OP("MonolithMultiHashTableSave")
    .Input("mtable: resource")
    .Input("basename: string")
    .Output("output_mtable: resource")
    .Attr("nshards: int=-1")
    .Attr("slot_expire_time_config: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithMultiHashTableSave").Device(DEVICE_CPU),
                        MultiHashTableSaveOp);

REGISTER_OP("MonolithMultiHashTableRestore")
    .Input("mtable: resource")
    .Input("basename: string")
    .Output("output_mtable: resource")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableRestore").Device(DEVICE_CPU),
    MultiHashTableRestoreOp);

REGISTER_OP("MonolithMultiHashTableFeatureStat")
    .Input("basename: string")
    .Output("features: string")
    .Output("counts: uint64")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->Vector(ctx->UnknownDim()));
      ctx->set_output(1, ctx->Vector(ctx->UnknownDim()));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("MonolithMultiHashTableFeatureStat").Device(DEVICE_CPU),
    MultiHashTableFeatureStatOp);

}  // namespace monolith_tf
}  // namespace tensorflow
