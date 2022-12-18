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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_READER_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_READER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include "monolith/native_training/data/training_instance/cc/data_format_options.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"

namespace tensorflow {
namespace monolith_tf {

enum FeaturePruningType {
  AS_IS = 0,
  PRUNING_FEATURE = 1,
  PRUNING_RAW_FEATURE = 2
};

void ExtendExample(::monolith::io::proto::Example *pb,
                   FeatureNameMapper *mapper = nullptr);
Status ExampleToInstance(::monolith::io::proto::Example *example,
                         ::parser::proto::Instance *instance);
Status InstanceToExample(::parser::proto::Instance *instance,
                         ::monolith::io::proto::Example *example);

template <class T>
class BaseStreamReaderTmpl {
 public:
  explicit BaseStreamReaderTmpl(const DataFormatOptions &options)
      : options_(options) {}
  virtual ~BaseStreamReaderTmpl() = default;

  Status ReadPBBytes(uint8_t *pb_type, uint32_t *data_source_key, T *record) {
    TF_RETURN_IF_ERROR(ReadDataHeader(pb_type, data_source_key));
    size_t size;
    TF_RETURN_IF_ERROR(ReadBinarySize(&size));
    // Don't know whether FALLBACK_RESERVE_VALUE is in use.
    if (size == 0xfefefefe) {
      return errors::InvalidArgument("DEADBEEF value found");
    }
    TF_RETURN_IF_ERROR(ReadNBytes(size, record));
    return Status::OK();
  }
  virtual uint64 GetOffset() = 0;
  virtual Status SetOffset(uint64 *offset) = 0;

 protected:
  virtual Status ReadNBytes(size_t n, T *result) = 0;

 private:
  Status ReadDataHeader(uint8_t *pb_type, uint32_t *data_source_key) {
    size_t size = 0, aggregate_page_sortid_size = 0;
    if (options_.lagrangex_header) {
      // *dtype = ins_type == 0 ? PROTO_INSTANCE : EXAMPLE_PB;
      TF_RETURN_IF_ERROR(ReadBinarySize(&size));
      uint64_t lgx_header = static_cast<uint64_t>(size);
      *pb_type = static_cast<uint8_t>(lgx_header & 0xff);
      uint32_t source = static_cast<uint32_t>(lgx_header);
      *data_source_key = (source >> 8) << 8;
    } else {
      *pb_type = 0;
      if (options_.kafka_dump_prefix) {
        TF_RETURN_IF_ERROR(ReadBinarySize(&size));
        if (size == 0) {
          TF_RETURN_IF_ERROR(ReadBinarySize(&size));
        } else {
          aggregate_page_sortid_size = size;
        }
      }
      if (options_.has_sort_id) {
        if (aggregate_page_sortid_size == 0) {
          TF_RETURN_IF_ERROR(ReadBinarySize(&size));
        } else {
          size = aggregate_page_sortid_size;
        }
        T sort_id;
        TF_RETURN_IF_ERROR(ReadNBytes(size, &sort_id));
      }
      if (options_.kafka_dump) {
        TF_RETURN_IF_ERROR(ReadBinarySize(&size));
      }
    }

    return Status::OK();
  }

  Status ReadBinarySize(size_t *size) {
    T result;
    TF_RETURN_IF_ERROR(ReadNBytes(sizeof(size_t), &result));
    *size = static_cast<size_t>(core::DecodeFixed64(result.data()));
    return Status::OK();
  }

  DataFormatOptions options_;
};

using BaseStreamReader = BaseStreamReaderTmpl<tstring>;

class StdinStreamReader : public BaseStreamReader {
 public:
  explicit StdinStreamReader(const DataFormatOptions &options,
                             int64 buffer_size = 64 * 1024 * 1024);
  ~StdinStreamReader() override = default;
  uint64 GetOffset() override;
  Status SetOffset(uint64 *offset) override;

 protected:
  Status ReadNBytes(size_t n, tstring *result) override;

 private:
  std::shared_ptr<std::istream> input_stream_;
  std::unique_ptr<char[]> buffer_;
  uint64 offset_;
  int64 buffer_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(StdinStreamReader);
};

class InputStreamReader : public BaseStreamReader {
 public:
  explicit InputStreamReader(
      const DataFormatOptions &options,
      std::unique_ptr<io::InputStreamInterface> input_stream);
  ~InputStreamReader() override = default;
  uint64 GetOffset() override;
  Status SetOffset(uint64 *offset) override;

 private:
  Status ReadNBytes(size_t n, tstring *result) override;

  std::unique_ptr<io::InputStreamInterface> input_stream_;
  bool last_read_failed_;

  TF_DISALLOW_COPY_AND_ASSIGN(InputStreamReader);
};

class FileStreamReader : public InputStreamReader {
 public:
  explicit FileStreamReader(const DataFormatOptions &options,
                            std::unique_ptr<RandomAccessFile> f,
                            bool use_snappy,
                            int64 buffer_size = 64 * 1024 * 1024);

 private:
  std::unique_ptr<RandomAccessFile> f_;
};

template <class T>
class StringStreamReader : public BaseStreamReaderTmpl<T> {
 public:
  explicit StringStreamReader(const DataFormatOptions &options, T content)
      : BaseStreamReaderTmpl<T>(options),
        content_(std::move(content)),
        cur_(0) {}
  Status ReadNBytes(size_t n, T *result) override {
    if (cur_ + n > content_.size()) {
      return errors::OutOfRange("Size exceeds he content size.");
    }
    *result = content_.substr(cur_, n);
    cur_ += n;
    return Status::OK();
  }

  uint64 GetOffset() override { return cur_; }
  Status SetOffset(uint64 *offset) override {
    cur_ = *offset;
    return Status::OK();
  }

 private:
  T content_;
  int64 cur_;
};

using ZeroCopyStringViewStreamReader = StringStreamReader<absl::string_view>;

class PBIterator {
 public:
  PBIterator() = default;
  explicit PBIterator(std::unique_ptr<BaseStreamReader> reader,
                      FeaturePruningType feature_pruning_type);
  virtual ~PBIterator() = default;

  virtual Status next(uint64 *offset, uint32_t *data_source_key,
                      tstring *serialized);

  virtual Status next(uint64 *offset, ::parser::proto::Instance *pb);

  virtual Status next(uint64 *offset, ::monolith::io::proto::Example *pb);

  virtual Status next(uint64 *offset, ::monolith::io::proto::ExampleBatch *pb);

  uint64 GetOffset();
  Status SetOffset(uint64 *offset);

 protected:
  FeaturePruningType feature_pruning_type_ = PRUNING_RAW_FEATURE;
  std::unique_ptr<BaseStreamReader> reader_;
  std::unique_ptr<FeaturePruningByteCounter> counter_;

  TF_DISALLOW_COPY_AND_ASSIGN(PBIterator);
};

class ExampleBatchIterator : public PBIterator {
 public:
  ExampleBatchIterator() = default;
  explicit ExampleBatchIterator(std::unique_ptr<BaseStreamReader> reader,
                                FeaturePruningType feature_pruning_type,
                                FeatureNameMapper *mapper);

  Status next(uint64 *offset, uint32_t *data_source_key, tstring *serialized);
  Status next(uint64 *offset, ::monolith::io::proto::ExampleBatch *pb);
  Status next(uint64 *offset, ::parser::proto::Instance *pb) override;
  Status next(uint64 *offset, ::monolith::io::proto::Example *pb) override;

 private:
  Status next_internal(uint64 *offset);
  int index_ = 0, batch_size_ = 0;
  monolith::io::proto::ExampleBatch *cur_;
  std::unique_ptr<google::protobuf::Arena> arena_;
  FeatureNameMapper *mapper_;
  TF_DISALLOW_COPY_AND_ASSIGN(ExampleBatchIterator);
};
}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_READER_H_
