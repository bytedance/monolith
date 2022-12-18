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

#include "monolith/native_training/data/kernels/internal/file_match_split_provider.h"

#include <unistd.h>
#include <memory>
#include <string>
#include <vector>
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

using std::chrono::milliseconds;

namespace tensorflow {
namespace data {
namespace monolith_tf {
namespace {

TEST(FileMatchSplitProvider, Create) {
  char tmp[256];
  getcwd(tmp, 256);
  std::vector<std::string> patterns = {
      absl::StrCat(tmp, "/monolith/native_training/data/kernels/*.h"),
      absl::StrCat(tmp, "/monolith/native_training/data/kernels/*.cc")};
  FileMatchSplitProvider split_provider(patterns);

  Tensor split;
  bool end_of_splits = false;
  int cnt = 0;
  while (!end_of_splits) {
    Status s = split_provider.GetNext(&split, &end_of_splits);
    if (!s.ok() || end_of_splits) {
      return;
    } else {
      cnt++;
      LOG(INFO) << split.scalar<tstring>()();
    }
  }

  EXPECT_GE(cnt, 0);
}

TEST(FileMatchSplitProvider, Reset) {
  char tmp[256];
  getcwd(tmp, 256);
  std::vector<std::string> patterns = {
      absl::StrCat(tmp, "/monolith/native_training/data/kernels/internal/*")};
  FileMatchSplitProvider split_provider(patterns);
  Status s;

  Tensor split;
  bool end_of_splits = false;
  s.Update(split_provider.GetNext(&split, &end_of_splits));

  split_provider.Reset();
  int cnt = 0;
  end_of_splits = false;
  while (!end_of_splits) {
    s.Update(split_provider.GetNext(&split, &end_of_splits));
    if (!s.ok() || end_of_splits) {
      return;
    } else {
      cnt++;
      LOG(INFO) << split.scalar<tstring>()();
    }
  }
  EXPECT_GE(cnt, 1);
}

}  // namespace
}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
