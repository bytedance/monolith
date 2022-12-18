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

#include "monolith/native_training/runtime/ops/file_utils.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace monolith_tf {

namespace {

const char* const kShardedFileFormat = "%s-%05d-of-%05d";
}

std::string GetShardedFileName(absl::string_view basename, int shard,
                               int nshards) {
  return absl::StrFormat(kShardedFileFormat, basename, shard, nshards);
}

Status ValidateShardedFiles(absl::string_view basename,
                            absl::Span<const std::string> filenames) {
  std::vector<bool> show;
  for (absl::string_view filename : filenames) {
    if (filename.substr(0, basename.size()) != basename) {
      return errors::InvalidArgument("Filename ", filename,
                                     " doesn't belong to ", basename);
    }
    absl::string_view suffix = filename.substr(basename.size());
    int shard, nshards;
    // TODO(leqi.zou): should use RE2 here.
    if (!absl::SimpleAtoi(suffix.substr(1, 5), &shard) ||
        suffix.substr(6, 4) != "-of-" ||
        !absl::SimpleAtoi(suffix.substr(10), &nshards)) {
      return errors::InvalidArgument("Filename ", filename, " is invalid");
    }
    if (show.empty()) {
      show.resize(nshards);
    }
    if (nshards != (int)show.size()) {
      return errors::InvalidArgument("Filename ", filename,
                                     " doesn't match nshards. ", show.size());
    }
    show[shard] = true;
  }
  for (int i = 0; i < (int)show.size(); ++i) {
    if (!show[i]) {
      return errors::InvalidArgument("Shard ", i, " doesn't show up for ",
                                     basename);
    }
  }
  return Status::OK();
}

}  // namespace monolith_tf
}  // namespace tensorflow
