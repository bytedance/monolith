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
#include <cstdlib>

#include "monolith/native_training/data/kernels/item_pool_kernels.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"

#include "absl/random/random.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/threadpool.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
using NamedFeature = ::monolith::io::proto::NamedFeature;
using ChannelCache = ::monolith::io::proto::ChannelCache;
static const std::string FILE_NAME_PREFIX = "item_pool_";
static constexpr uint64_t MASK = (1L << 48) - 1;

namespace tensorflow {
namespace monolith_tf {

// Carries the data through async process.
// It will ref and unref |p_hash_table|.
struct AsyncPack {
  AsyncPack(OpKernelContext* p_ctx, ItemPoolResource* p_pool,
            std::function<void()> p_done, int p_thread_num)
      : ctx(p_ctx),
        pool(p_pool),
        done(std::move(p_done)),
        thread_num(p_thread_num),
        finish_num(0),
        status(thread_num) {
    pool->Ref();
  }

  ~AsyncPack() { pool->Unref(); }

  OpKernelContext* ctx;
  ItemPoolResource* pool;
  std::function<void()> done;
  const int thread_num;
  std::atomic_int finish_num;
  std::vector<Status> status;
};

ItemPoolResource::ItemPoolResource(int max_item_num_per_channel, int start_num)
    : start_num_(start_num),
      max_item_num_per_channel_(max_item_num_per_channel),
      cache_(std::make_unique<internal::CacheManager>(max_item_num_per_channel,
                                                      start_num)) {}

Status ItemPoolResource::Add(
    uint64_t channel_id, uint64_t item_id,
    const std::shared_ptr<const internal::ItemFeatures>& item) {
  absl::MutexLock l(&mu_);
  cache_->Push(channel_id, item_id, item);
  return Status::OK();
}

std::shared_ptr<const internal::ItemFeatures> ItemPoolResource::Sample(
    uint64_t channel_id, double* freq_factor, double* time_factor) {
  absl::MutexLock l(&mu_);
  return cache_->RandomSelectOne(channel_id, freq_factor, time_factor);
}

Status ItemPoolResource::Save(WritableFile* ostream, int shard_index,
                              int shard_num) {
  absl::MutexLock l(&mu_);
  const absl::flat_hash_map<uint64_t, internal::CacheWithGid>& channel_cache_ =
      cache_->GetCache();

  io::RecordWriter writer(ostream);
  Status write_status = Status::OK();
  for (const auto& pair : channel_cache_) {
    if (pair.first % shard_num != shard_index) {
      continue;
    }
    ChannelCache channel_cache;
    channel_cache.set_channel_id(pair.first);
    pair.second.ToProto(&channel_cache);
    Status s = writer.WriteRecord(channel_cache.SerializeAsString());
    if (TF_PREDICT_FALSE(!s.ok())) {
      write_status.Update(s);
      break;
    }
  }

  TF_RETURN_IF_ERROR(write_status);
  TF_RETURN_IF_ERROR(writer.Close());

  return Status::OK();
}

Status ItemPoolResource::Restore(RandomAccessFile* istream, int64 buffer_size) {
  absl::MutexLock l(&mu_);
  io::RecordReaderOptions opts;
  opts.buffer_size = buffer_size;
  io::SequentialRecordReader reader(istream, opts);

  Status restore_status = Status::OK();
  while (true) {
    tstring s;
    ChannelCache channel_cache;
    // read record
    Status rs = reader.ReadRecord(&s);
    if (errors::IsOutOfRange(rs)) {
      LOG(INFO) << "EOF, read file done...";
      break;
    } else {
      restore_status.Update(rs);
    }

    if (!channel_cache.ParseFromArray(s.data(), s.size())) {
      restore_status.Update(errors::FailedPrecondition(
          "Unable to parse data. Data might be corrupted"));
      break;
    } else {
      restore_status.Update(Status::OK());
    }

    internal::CacheWithGid cache_with_gid(max_item_num_per_channel_,
                                          start_num_);
    cache_with_gid.FromProto(channel_cache);
    cache_->Push(channel_cache.channel_id(), cache_with_gid);
  }

  TF_RETURN_IF_ERROR(restore_status);

  return Status::OK();
}

bool ItemPoolResource::Equal(const ItemPoolResource& other) const {
  if (other.max_item_num_per_channel_ != max_item_num_per_channel_) {
    return false;
  }

  if (other.start_num_ != start_num_) {
    return false;
  }

  auto this_cache = cache_->GetCache();
  auto other_cache = other.cache_->GetCache();
  if (this_cache.size() != other_cache.size()) {
    return false;
  } else {
    for (const auto& it : this_cache) {
      if (other_cache.count(it.first) == 0) {
        return false;
      } else {
        auto this_channel = it.second;
        auto other_channel = other_cache.at(it.first);
        return this_channel.Equal(other_channel);
      }
    }
  }

  return true;
}

void ItemPoolResource::SampleChannelID(uint64_t* channel_id) {
  absl::MutexLock l(&mu_);
  cache_->SampleChannelID(channel_id);
}

void get_index_and_worker_num(int* index, int* worker_num) {
  const char* env_p = std::getenv("TF_CONFIG");
  if (env_p == nullptr) {
    *index = 0;
    *worker_num = 1;
  } else {
    auto tf_config = json::parse(env_p);
    // assert TF_CONFIG only has ps + chief + worker
    for (const auto& conf_item : tf_config["cluster"].items()) {
      if (conf_item.key() != "ps" && conf_item.key() != "chief" &&
          conf_item.key() != "worker") {
        LOG(ERROR) << "Unknown Cluster Type: " << conf_item.key();
      }
    }
    auto chief = tf_config["cluster"]["chief"];
    auto workers = tf_config["cluster"]["worker"];
    *worker_num = chief.size() + workers.size();
    if (tf_config["task"]["type"] == "worker") {
      *index = static_cast<int>(tf_config["task"]["index"]) + 1;
    } else {
      *index = 0;
    }
  }
}

class ItemPoolCreateOp : public ResourceOpKernel<ItemPoolResource> {
 public:
  explicit ItemPoolCreateOp(OpKernelConstruction* ctx) : ResourceOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("start_num", &start_num_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_item_num_per_channel",
                                     &max_item_num_per_channel_));
  }

 private:
  Status CreateResource(ItemPoolResource** wrapper)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *wrapper = new ItemPoolResource(max_item_num_per_channel_, start_num_);
    return Status::OK();
  }

  int start_num_, max_item_num_per_channel_;
};

// for test only
class ItemPoolRandomFillOp : public OpKernel {
 public:
  explicit ItemPoolRandomFillOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    ItemPoolResource* pool;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &pool));
    core::ScopedUnref unref(pool);
    ctx->set_output(0, ctx->input(0));

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 50; ++j) {
        std::shared_ptr<internal::ItemFeatures> item =
            std::make_shared<internal::ItemFeatures>();
        GenItemFeatures(item.get());
        pool->Add(i, j, item);
      }
    }
  }

 private:
  void GenNamedFeature(NamedFeature* nf) {
    int slot = std::rand() % 1024;
    nf->set_name(absl::StrCat("fc_", slot));
    auto* fid_v2_list = nf->mutable_feature()->mutable_fid_v2_list();
    int num_fids = std::abs(std::rand() % 20) + 1;
    for (int i = 0; i < num_fids; ++i) {
      uint64_t fid = ((uint64_t)slot << 48) | ((std::rand() % 100000) & MASK);
      fid_v2_list->add_value(fid);
    }
  }

  void GenItemFeatures(internal::ItemFeatures* item) {
    int num_feats = std::abs(std::rand() % 20) + 1;
    for (int i = 0; i < num_feats; ++i) {
      NamedFeature nf;
      GenNamedFeature(&nf);
      if (!item->example_features.contains(nf.name())) {
        item->example_features.insert({nf.name(), nf});
      }
    }
  }
};

// for test only
class ItemPoolCheckOp : public OpKernel {
 public:
  explicit ItemPoolCheckOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_path", &model_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nshards", &nshards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &buffer_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    ItemPoolResource* pool;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &pool));
    core::ScopedUnref unref(pool);
    ctx->set_output(0, ctx->input(0));
    const Tensor& global_step_tensor = ctx->input(1);
    global_step_ = global_step_tensor.scalar<int64>()();

    ItemPoolResource pool2(pool->max_item_num_per_channel(), pool->start_num());
    for (int idx = 0; idx < nshards_; ++idx) {
      std::string filename = GetRestoreFileName(ctx, idx);
      if (!filename.empty()) {
        std::unique_ptr<RandomAccessFile> istream;
        OP_REQUIRES_OK(ctx,
                       ctx->env()->NewRandomAccessFile(filename, &istream));
        OP_REQUIRES_OK(ctx, pool2.Restore(istream.get(), buffer_size_));
      }
    }

    if (!pool->Equal(pool2)) {
      LOG(INFO) << "resotre not equal~ ...";
    } else {
      LOG(INFO) << "resotre equal~ ...";
    }
  }

 private:
  std::string model_path_;
  int64 buffer_size_;
  int nshards_;
  int64 global_step_;

  std::string GetRestoreFileName(OpKernelContext* ctx, int shard_index) {
    int index, worker_num;
    get_index_and_worker_num(&index, &worker_num);
    std::vector<std::string> files;

    Status s = ctx->env()->GetMatchingPaths(
        absl::StrCat(model_path_, "/model.ckpt-", global_step_, "_",
                     FILE_NAME_PREFIX, "*"),
        &files);
    if (!s.ok()) {
      LOG(INFO) << "GetMatchingPaths Error: " << s;
      return "";
    }

    int last_worker_num = 1;
    int64 mtime_nsec = 0;
    for (const auto& file : files) {
      FileStatistics stat;
      ctx->env()->Stat(file, &stat);
      if (mtime_nsec < stat.mtime_nsec) {
        std::vector<absl::string_view> items = absl::StrSplit(file, "_");
        absl::SimpleAtoi(items.back(), &last_worker_num);
        mtime_nsec = stat.mtime_nsec;
      }
    }

    if (files.size() > 0) {
      // {model_path}/item_pool_{index}_{worker_num}
      return absl::StrCat(model_path_, "/model.ckpt-", global_step_, "_",
                          FILE_NAME_PREFIX, index % last_worker_num, "_",
                          shard_index, "_", last_worker_num);
    } else {
      return "";
    }
  }
};

class ItemPoolSaveOp : public AsyncOpKernel {
 public:
  explicit ItemPoolSaveOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_path", &model_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nshards", &nshards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("random_sleep_ms", &random_sleep_ms_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    ItemPoolResource* pool;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &pool));
    core::ScopedUnref unref(pool);
    const Tensor& global_step_tensor = ctx->input(1);
    global_step_ = global_step_tensor.scalar<int64>()();

    ctx->set_output(0, ctx->input(0));
    if (!ctx->env()->FileExists(model_path_).ok()) {
      OP_REQUIRES_OK_ASYNC(ctx, ctx->env()->RecursivelyCreateDir(model_path_),
                           done);
    }
    // add multi-thread
    auto pack = new AsyncPack(ctx, pool, std::move(done), nshards_);
    for (int i = 0; i < nshards_; ++i) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i] { WorkerThread(i, pack); });
    }
  }

 private:
  std::string model_path_;
  int64 global_step_;
  int nshards_;
  int64 random_sleep_ms_;

  std::string GetSaveFileName(int shard_index) {
    int index, worker_num;
    get_index_and_worker_num(&index, &worker_num);
    return absl::StrCat(model_path_, "/model.ckpt-", global_step_, "_",
                        FILE_NAME_PREFIX, index, "_", shard_index, "_",
                        worker_num);
  }

  void WorkerThread(int shard_index, AsyncPack* p) {
    absl::BitGen bitgen;
    p->status[shard_index] = SaveOneShard(shard_index, p);
    if (p->finish_num.fetch_add(1) == p->thread_num - 1) {
      Cleanup(p);
    }
  }

  Status SaveOneShard(int shard_index, AsyncPack* p) {
    std::string filename = GetSaveFileName(shard_index);
    std::string tmp_filename = absl::StrCat(filename, "_tmp");
    std::unique_ptr<WritableFile> ostream;
    TF_RETURN_IF_ERROR(p->ctx->env()->NewWritableFile(tmp_filename, &ostream));
    TF_RETURN_IF_ERROR(
        p->pool->Save(ostream.get(), shard_index, p->thread_num));
    TF_RETURN_IF_ERROR(ostream->Close());
    if (p->ctx->env()->FileExists(filename).ok()) {
      TF_RETURN_IF_ERROR(
          p->ctx->env()->RenameFile(filename, absl::StrCat(filename, "_old")));
      TF_RETURN_IF_ERROR(p->ctx->env()->RenameFile(tmp_filename, filename));
      TF_RETURN_IF_ERROR(
          p->ctx->env()->DeleteFile(absl::StrCat(filename, "_old")));
    } else {
      TF_RETURN_IF_ERROR(p->ctx->env()->RenameFile(tmp_filename, filename));
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

class ItemPoolRestoreOp : public AsyncOpKernel {
 public:
  explicit ItemPoolRestoreOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_path", &model_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &buffer_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("nshards", &nshards_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("random_sleep_ms", &random_sleep_ms_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    ItemPoolResource* pool;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &pool));
    core::ScopedUnref unref(pool);
    const Tensor& global_step_tensor = ctx->input(1);
    global_step_ = global_step_tensor.scalar<int64>()();

    ctx->set_output(0, ctx->input(0));
    auto pack = new AsyncPack(ctx, pool, std::move(done), nshards_);
    for (int i = 0; i < nshards_; i++) {
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(
          [this, pack, i] { WorkerThread(i, pack); });
    }
  }

 private:
  std::string model_path_;
  int64 global_step_;
  int64 buffer_size_;
  int nshards_;
  int64 random_sleep_ms_;

  void WorkerThread(int shard_index, AsyncPack* p) {
    absl::BitGen bitgen;
    p->status[shard_index] = RestoreOneShard(shard_index, p);
    if (p->finish_num.fetch_add(1) == p->thread_num - 1) {
      Cleanup(p);
    }
  }

  Status RestoreOneShard(int shard_index, AsyncPack* p) {
    std::string filename = GetRestoreFileName(p->ctx, shard_index);
    if (filename == "") {
      LOG(INFO) << "Cannot find file to restore, skip!";
    } else if (p->ctx->env()->FileExists(filename).ok()) {
      LOG(INFO) << "Restoring file: " << filename;
      std::unique_ptr<RandomAccessFile> istream;
      TF_RETURN_IF_ERROR(
          p->ctx->env()->NewRandomAccessFile(filename, &istream));
      TF_RETURN_IF_ERROR(p->pool->Restore(istream.get(), buffer_size_));
    } else {
      LOG(INFO) << "File dose not exist: " << filename;
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

  int FindLastNumber(std::vector<std::string> const &files, OpKernelContext* ctx) {
    // 支持 restore 时的 worker_num 可以和 save 时不同
    int last_worker_num = 1;
    int64 mtime_nsec = 0;
    for (const auto& file : files) {
      if (absl::EndsWith(file, "tmp")) {
        LOG(INFO) << "Files vector contains file with tmp suffix.";
        continue;
      }
      FileStatistics stat;
      ctx->env()->Stat(file, &stat);
      if (mtime_nsec < stat.mtime_nsec) {
        std::vector<absl::string_view> items = absl::StrSplit(file, "_");
        absl::SimpleAtoi(items.back(), &last_worker_num);
        mtime_nsec = stat.mtime_nsec;
      }
    }
    return last_worker_num;
  }

  std::string GetRestoreFileName(OpKernelContext* ctx, int shard_index) {
    int index, worker_num;
    get_index_and_worker_num(&index, &worker_num);
    std::vector<std::string> files_new;
    std::vector<std::string> files_old;

    Status s_new = ctx->env()->GetMatchingPaths(
        absl::StrCat(model_path_, "/model.ckpt-", global_step_, "_",
                     FILE_NAME_PREFIX, "*"),
        &files_new);
    Status s_old = ctx->env()->GetMatchingPaths(
        absl::StrCat(model_path_, "/", FILE_NAME_PREFIX, "*"), &files_old);
    if (!s_new.ok() && !s_old.ok()) {
      LOG(INFO) << "GetMatchingPaths Error: [new] " << s_new << " and [old] " << s_old;
      return "";
    }

    if (files_new.size() > 0) {
      // {model_path}/item_pool_{index}_{worker_num}
      LOG(INFO) << "new version files > 0";
      int last_worker_num = FindLastNumber(files_new, ctx);
      LOG(INFO) << "last worker num is: " << last_worker_num;
      return absl::StrCat(model_path_, "/model.ckpt-", global_step_, "_",
                          FILE_NAME_PREFIX, index % last_worker_num, "_",
                          shard_index, "_", last_worker_num);
    } else if (files_old.size() > 0) {
      LOG(INFO) << "old version files > 0";
      int last_worker_num = FindLastNumber(files_old, ctx);
      LOG(INFO) << "last worker num is: " << last_worker_num;
      return absl::StrCat(model_path_, "/", FILE_NAME_PREFIX,
                          index % last_worker_num, "_", shard_index, "_",
                          last_worker_num);
    } else {
      return "";
    }
  }
};

namespace {
REGISTER_KERNEL_BUILDER(Name("ItemPoolCreate").Device(DEVICE_CPU),
                        ItemPoolCreateOp);

// for test only
REGISTER_KERNEL_BUILDER(Name("ItemPoolCheck").Device(DEVICE_CPU),
                        ItemPoolCheckOp);

// for test only
REGISTER_KERNEL_BUILDER(Name("ItemPoolRandomFill").Device(DEVICE_CPU),
                        ItemPoolRandomFillOp);

REGISTER_KERNEL_BUILDER(Name("ItemPoolSave").Device(DEVICE_CPU),
                        ItemPoolSaveOp);

REGISTER_KERNEL_BUILDER(Name("ItemPoolRestore").Device(DEVICE_CPU),
                        ItemPoolRestoreOp);
}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
