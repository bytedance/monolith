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
#include "re2/re2.h"
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
                            absl::Span<const std::string> filenames,
                            FileSpec* spec) {
  std::vector<bool> show;
  for (absl::string_view filename : filenames) {
    if (filename.substr(0, basename.size()) != basename) {
      return errors::InvalidArgument("Filename ", filename,
                                     " doesn't belong to ", basename);
    }
    absl::string_view suffix = filename.substr(basename.size());
    int shard, nshards;
    // Ignore invalid files.
    if (!RE2::FullMatch(suffix, R"raw(-(\d{5})?-of-(\d{5})?)raw", &shard,
                        &nshards)) {
      continue;
    }
    if (show.empty()) {
      show.resize(nshards);
    }
    if (nshards != (int)show.size()) {
      return errors::InvalidArgument("Filename ", filename,
                                     " doesn't match nshards. ", show.size());
    }
    if (shard >= nshards) {
      return errors::InvalidArgument("Shard ", shard, "exceeds ", nshards,
                                     " for ", filename);
    }
    show[shard] = true;
  }
  if (show.empty()) {
    return errors::InvalidArgument("There is no valid sharded files for ",
                                   basename);
  }
  for (int i = 0; i < (int)show.size(); ++i) {
    if (!show[i]) {
      return errors::InvalidArgument("Shard ", i, " doesn't show up for ",
                                     basename);
    }
  }

  if (spec != nullptr) {
    *spec = FileSpec::ShardedFileSpec(basename, show.size());
  }
  return Status::OK();
}

FileSpec FileSpec::ShardedFileSpec(absl::string_view prefix, int nshards) {
  FileSpec spec;
  spec.type_ = FileSpec::SHARDED_FILES;
  spec.prefix_ = std::string(prefix);
  spec.nshards_ = nshards;
  return spec;
}

std::vector<std::string> FileSpec::GetFilenames() const {
  std::vector<std::string> filenames;
  switch (type_) {
    case FileSpec::SHARDED_FILES:
      for (int i = 0; i < nshards_; ++i) {
        filenames.push_back(GetShardedFileName(prefix_, i, nshards_));
      }
      break;
    default:
      break;
  }
  return filenames;
}

}  // namespace monolith_tf
}  // namespace tensorflow
