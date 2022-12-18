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

#include "monolith/native_training/data/training_instance/cc/data_writer.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/coding.h"

namespace tensorflow {
namespace monolith_tf {

Status BaseStreamWriter::PrepareHeader() {
  if (options_.lagrangex_header) {
    TF_RETURN_IF_ERROR(Write(std::string(8, 0)));
  } else {
    if (options_.kafka_dump_prefix) {
      TF_RETURN_IF_ERROR(Write(std::string(16, 0)));
    }
    if (options_.has_sort_id) {
      TF_RETURN_IF_ERROR(Write(std::string(8, 0)));
    }
    if (options_.kafka_dump) {
      TF_RETURN_IF_ERROR(Write(std::string(8, 0)));
    }
  }
  return Status::OK();
}

BaseStreamWriter::BaseStreamWriter(DataFormatOptions options)
    : options_(std::move(options)) {}

Status BaseStreamWriter::WriteRecord(absl::string_view record) {
  TF_RETURN_IF_ERROR(PrepareHeader());
  char size_encoded[8];
  core::EncodeFixed64(size_encoded, record.size());
  TF_RETURN_IF_ERROR(Write(absl::string_view(size_encoded, 8)));
  TF_RETURN_IF_ERROR(Write(record));
  return Status::OK();
}

Status StringStreamWriter::Write(absl::string_view s) {
  absl::StrAppend(out_, s);
  return Status::OK();
}

}  // namespace monolith_tf
}  // namespace tensorflow
