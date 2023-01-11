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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_IO_ARROW_KERNELS_H_
#define TENSORFLOW_IO_ARROW_KERNELS_H_

#include "arrow/buffer.h"
#include "arrow/io/api.h"
#include "arrow/type.h"
#include "parquet/windows_compatibility.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow_io/core/kernels/io_stream.h"

namespace tensorflow {

class RandomAccessFile;

namespace data {

// NOTE: Both SizedRandomAccessFile and ArrowRandomAccessFile overlap
// with another PR. Will remove duplicate once PR merged

class ArrowRandomAccessFile : public ::arrow::io::RandomAccessFile {
 public:
  explicit ArrowRandomAccessFile(tensorflow::RandomAccessFile* file, int64 size)
      : file_(file), size_(size), position_(0) {}

  ~ArrowRandomAccessFile() {}
  arrow::Status Close() override { return arrow::Status::OK(); }
  bool closed() const override { return false; }
  arrow::Result<int64_t> Tell() const override { return position_; }
  arrow::Status Seek(int64_t position) override {
    return arrow::Status::NotImplemented("Seek");
  }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    StringPiece result;
    Status status =
        file_->Read(position_, nbytes, &result, reinterpret_cast<char*>(out));
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return arrow::Status::IOError(status.error_message());
    }
    position_ += result.size();
    return result.size();
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> result =
        arrow::AllocateResizableBuffer(nbytes);
    ARROW_RETURN_NOT_OK(result);
    std::shared_ptr<arrow::ResizableBuffer> buffer =
        std::move(result).ValueUnsafe();

    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read,
                          Read(nbytes, buffer->mutable_data()));
    RETURN_NOT_OK(buffer->Resize(bytes_read));
    return buffer;
  }
  arrow::Result<int64_t> GetSize() override { return size_; }
  bool supports_zero_copy() const override { return false; }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes,
                                void* out) override {
    StringPiece result;
    Status status =
        file_->Read(position, nbytes, &result, reinterpret_cast<char*>(out));
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return arrow::Status::IOError(status.error_message());
    }
    return result.size();
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(
      int64_t position, int64_t nbytes) override {
    string buffer;
    buffer.resize(nbytes);
    StringPiece result;
    Status status = file_->Read(position, nbytes, &result,
                                reinterpret_cast<char*>(&buffer[0]));
    if (!(status.ok() || errors::IsOutOfRange(status))) {
      return arrow::Status::IOError(status.error_message());
    }
    buffer.resize(result.size());
    return arrow::Buffer::FromString(std::move(buffer));
  }

 private:
  tensorflow::RandomAccessFile* file_;
  int64 size_;
  int64 position_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_IO_ARROW_KERNELS_H_
