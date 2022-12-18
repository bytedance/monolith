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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_WRITER_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_WRITER_H_

#include "absl/strings/string_view.h"
#include "monolith/native_training/data/training_instance/cc/data_format_options.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace monolith_tf {

class BaseStreamWriter {
 public:
  explicit BaseStreamWriter(DataFormatOptions options);

  Status WriteRecord(absl::string_view record);

 protected:
  virtual Status Write(absl::string_view s) = 0;

 private:
  Status PrepareHeader();

 private:
  DataFormatOptions options_;
};

class StringStreamWriter : public BaseStreamWriter {
 public:
  explicit StringStreamWriter(DataFormatOptions options, std::string* out)
      : BaseStreamWriter(std::move(options)), out_(out) {}

 private:
  Status Write(absl::string_view s) override;
  std::string* out_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_WRITER_H_
