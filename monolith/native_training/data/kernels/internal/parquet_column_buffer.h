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

#ifndef PARQUET_COLUMN_BUFFER_H_
#define PARQUET_COLUMN_BUFFER_H_

#include "parquet/api/reader.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace data {

class ColumnBuffer {
 public:
  explicit ColumnBuffer(std::shared_ptr<parquet::ColumnReader>& col_reader)
      : col_reader_(col_reader),
        buffer_limit_(0),
        values_limit_(0),
        values_p_(0),
        levels_p_(0) {
    max_definition_level_ = col_reader->descr()->max_definition_level();
    max_repetition_level_ = col_reader->descr()->max_repetition_level();
  }

  virtual ~ColumnBuffer() = default;

 protected:
  std::shared_ptr<parquet::ColumnReader> col_reader_;
  std::unique_ptr<int16_t[]> def_levels_buffer_;
  std::unique_ptr<int16_t[]> rep_levels_buffer_;
  int64_t buffer_limit_;
  int64_t values_limit_;
  int64_t values_p_;
  int64_t levels_p_;
  const int64_t BUFFER_SIZE = 256;
  int16_t max_definition_level_;
  int16_t max_repetition_level_;
};

template <typename DType>
class TypedColumnBuffer : public ColumnBuffer {
 public:
  typedef typename DType::c_type T;

  explicit TypedColumnBuffer(std::shared_ptr<parquet::ColumnReader>& col_reader)
      : ColumnBuffer(col_reader) {
    typed_col_reader_ =
        static_cast<parquet::TypedColumnReader<DType>*>(col_reader.get());
    value_buffer_.reset(new T[BUFFER_SIZE]);
    def_levels_buffer_.reset(new int16_t[BUFFER_SIZE]);
    rep_levels_buffer_.reset(new int16_t[BUFFER_SIZE]);
  }

  Status GetNextValues(std::vector<T>& values) {
    T value;
    int16_t def_value, rep_level;
    bool is_null;

    if (max_repetition_level_ == 0 && max_definition_level_ == 0) {
      TF_RETURN_IF_ERROR(
          ReadNextValue(&value, &def_value, &rep_level, &is_null));
      values.push_back(value);
    } else if (max_repetition_level_ == 0 && max_definition_level_ > 0) {
      TF_RETURN_IF_ERROR(
          ReadNextValue(&value, &def_value, &rep_level, &is_null));
      if (!is_null) {
        values.push_back(value);
      }
    } else {
      do {
        ReadNextValue(&value, &def_value, &rep_level, &is_null);
        // debug use
        // LOG(INFO) << "In GetNextValues " << def_value << " " << rep_level <<
        // " " << is_null;
        if (is_null) {
          break;
        }
        values.push_back(value);
      } while (HasNextRepeatedValue());
    }
    return Status::OK();
  }

  bool HasNextRepeatedValue() {
    if (levels_p_ >= buffer_limit_) {
      Status status = ReadBuffer();
      if (!status.ok()) {
        return false;
      }
    }
    return rep_levels_buffer_[levels_p_] ? true : false;
  }

  Status ReadNextValue(T* out, int16_t* def_level, int16_t* rep_level,
                       bool* is_null) {
    if (max_repetition_level_ == 0 && max_definition_level_ == 0) {
      // required column
      while (values_p_ >= values_limit_) {
        TF_RETURN_IF_ERROR(ReadBuffer());
      }
      *is_null = false;
      *out = value_buffer_[values_p_++];
    } else if (max_repetition_level_ == 0 && max_definition_level_ > 0) {
      // optional column
      while (levels_p_ >= buffer_limit_) {
        TF_RETURN_IF_ERROR(ReadBuffer());
      }
      *def_level = def_levels_buffer_[levels_p_];
      if (*def_level == 0) {
        *is_null = true;
      } else {
        *is_null = false;
        if (values_p_ >= values_limit_) {
          return errors::InvalidArgument("No extra values in buffer.");
        }
        *out = value_buffer_[values_p_++];
      }
      levels_p_++;
    } else {
      // repeated column
      while (levels_p_ >= buffer_limit_) {
        TF_RETURN_IF_ERROR(ReadBuffer());
      }
      *def_level = def_levels_buffer_[levels_p_];
      *rep_level = rep_levels_buffer_[levels_p_];
      if (*def_level == 0 && *rep_level == 0) {
        *is_null = true;
      } else {
        *is_null = false;
        if (values_p_ >= values_limit_) {
          return errors::InvalidArgument("No extra values in buffer.");
        }
        *out = value_buffer_[values_p_++];
      }
      levels_p_++;
    }
    return Status::OK();
  }

  Status ReadBuffer() {
    if (!typed_col_reader_->HasNext()) {
      return errors::OutOfRange("Column values all consumed, out of range");
    }
    int64_t values_read;
    int64_t levels_read = typed_col_reader_->ReadBatch(
        BUFFER_SIZE, def_levels_buffer_.get(), rep_levels_buffer_.get(),
        value_buffer_.get(), &values_read);
    buffer_limit_ = levels_read;
    values_limit_ = values_read;
    values_p_ = 0;
    levels_p_ = 0;
    return Status::OK();
  }

 private:
  parquet::TypedColumnReader<DType>* typed_col_reader_;
  std::unique_ptr<T[]> value_buffer_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // PARQUET_COLUMN_BUFFER_H_
