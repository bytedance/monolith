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

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"

#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

// It is a thin wrapper of GFile. Make it compatible with ResourceKernelOp
// and thread safe.
class FileResource : public ResourceBase {
 public:
  explicit FileResource(std::unique_ptr<WritableFile> f,
                        absl::string_view debugging_info)
      : f_(std::move(f)), debugging_info_(debugging_info), closed_(false) {}

  std::string DebugString() const override { return debugging_info_; }

  Status Close() {
    absl::MutexLock l(&mu_);
    closed_ = true;
    return f_->Close();
  }

  Status Append(StringPiece data) {
    absl::MutexLock l(&mu_);
    return f_->Append(data);
  }

  Status AppendRecord(const string& serialized) {
    absl::MutexLock l(&mu_);
    if (!record_writer_) {
      record_writer_.reset(new tensorflow::io::RecordWriter(
          f_.get(),
          tensorflow::io::RecordWriterOptions::CreateRecordWriterOptions("")));
    }
    return record_writer_->WriteRecord(serialized);
  }

  ~FileResource() {
    absl::MutexLock l(&mu_);
    if (!closed_) {
      auto s = f_->Close();
      if (!s.ok()) {
        LOG(ERROR) << "Unable to close file " << debugging_info_ << " :"
                   << s.ToString();
      }
    }
  }

 private:
  absl::Mutex mu_;
  std::unique_ptr<WritableFile> f_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<tensorflow::io::RecordWriter> record_writer_
      ABSL_GUARDED_BY(mu_);
  const std::string debugging_info_;
  bool closed_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

class MonolithWritableFileOp : public ResourceOpKernel<FileResource> {
 public:
  explicit MonolithWritableFileOp(OpKernelConstruction* c)
      : ResourceOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("filename", &filename_));
    env_ = c->env();
  }

  ~MonolithWritableFileOp() override {}

 private:
  Status CreateResource(FileResource** file_wrapper)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    const std::string dir = std::string(io::Dirname(filename_));
    if (!env_->FileExists(dir).ok()) {
      TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(dir));
    }
    std::unique_ptr<WritableFile> f;
    TF_RETURN_IF_ERROR(env_->NewWritableFile(filename_, &f));
    *file_wrapper = new FileResource(std::move(f), filename_);
    return Status::OK();
  }

  std::string filename_;
  Env* env_;
};

REGISTER_OP("MonolithWritableFile")
    .Output("handle: resource")
    .Attr("filename: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MonolithWritableFile").Device(DEVICE_CPU),
                        MonolithWritableFileOp);

class MonolithWritableFileCloseOp : public OpKernel {
 public:
  explicit MonolithWritableFileCloseOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    FileResource* f;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &f));
    core::ScopedUnref unref(f);
    OP_REQUIRES_OK(c, f->Close());
  }
};

REGISTER_OP("MonolithWritableFileClose")
    .Input("handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("MonolithWritableFileClose").Device(DEVICE_CPU),
                        MonolithWritableFileCloseOp);

class MonolithWritableFileAppendOp : public OpKernel {
 public:
  explicit MonolithWritableFileAppendOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    FileResource* f;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &f));
    core::ScopedUnref unref(f);
    const auto& content = c->input(1).scalar<tstring>()();
    OP_REQUIRES_OK(c, f->Append(content));
  }
};

REGISTER_OP("MonolithWritableFileAppend")
    .Input("handle: resource")
    .Input("content: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("MonolithWritableFileAppend").Device(DEVICE_CPU),
                        MonolithWritableFileAppendOp);

class MonolithEntryDumpFileAppendOp : public OpKernel {
 public:
  explicit MonolithEntryDumpFileAppendOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    FileResource* f;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &f));
    core::ScopedUnref unref(f);
    const auto& item_id = c->input(1).flat<int64>();
    const auto& bias = c->input(2).flat<float>();
    const auto& embedding = c->input(3).flat<float>();

    size_t batch_size = item_id.size();
    CHECK_GT(batch_size, 0);
    CHECK_EQ(embedding.size() % batch_size, 0);
    size_t embedding_len = embedding.size() / batch_size;
    for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
      monolith::hash_table::EntryDump d;
      d.set_id(item_id(batch_id));
      d.add_num(bias(batch_id));
      for (size_t i = 0; i < embedding_len; i++) {
        d.add_num(embedding(batch_id * embedding_len + i));
      }
      OP_REQUIRES_OK(c, f->AppendRecord(d.SerializeAsString()));
    }
  }
};

REGISTER_OP("MonolithEntryDumpFileAppend")
    .Input("handle: resource")
    .Input("item_id: int64")
    .Input("bias: float")
    .Input("embedding: float")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("MonolithEntryDumpFileAppend").Device(DEVICE_CPU),
                        MonolithEntryDumpFileAppendOp);

}  // namespace monolith_tf
}  // namespace tensorflow
