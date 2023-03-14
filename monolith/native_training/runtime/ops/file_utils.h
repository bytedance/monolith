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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_FILE_UTILS
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_FILE_UTILS

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace monolith_tf {

// Returns sharded file name.
std::string GetShardedFileName(absl::string_view basename, int shard,
                               int nshards);

// A spec reprsents a set of files.
class FileSpec final {
 public:
  FileSpec() {}

  static FileSpec ShardedFileSpec(absl::string_view prefix, int nshards);

  std::vector<std::string> GetFilenames() const;

  int nshards() const { return nshards_; }

 private:
  enum Type { UNKNOWN, SHARDED_FILES };

  Type type_ = UNKNOWN;
  std::string prefix_;
  int nshards_ = 0;
};

// Validates if filenames construct a valid file spec for base name.
Status ValidateShardedFiles(absl::string_view basename,
                            absl::Span<const std::string> filenames,
                            FileSpec* spec = nullptr);

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_FILE_UTILS
