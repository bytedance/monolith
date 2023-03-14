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

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using ::testing::ElementsAre;

TEST(ValidateShardedFilesTest, Basic) {
  FileSpec spec;
  TF_EXPECT_OK(ValidateShardedFiles("a/b", {"a/b-00000-of-00001"}));
  TF_EXPECT_OK(ValidateShardedFiles(
      "a/b", {"a/b-00000-of-00002", "a/b-00001-of-00002"}, &spec));
  EXPECT_THAT(spec.nshards(), 2);
  TF_EXPECT_OK(ValidateShardedFiles(
      "a", {"a-00000-of-00001", "a-00000-of-00001-tmp-1234"}));
  TF_EXPECT_OK(ValidateShardedFiles(
      "a", {"a-00000-of-00001", "a-00000-of-00002-tmp-1234"}));
  EXPECT_FALSE(ValidateShardedFiles("a/b", {"a/b-00000-of-00002"}).ok());
  EXPECT_FALSE(
      ValidateShardedFiles("a/b", {"a/b-00000-of-00001", "a/b-00001-of-00001"})
          .ok());
  EXPECT_FALSE(
      ValidateShardedFiles("a/b", {"a/b-00000-of-00001", "a/b-00000-of-00002",
                                   "a/b-00001-of-00002"})
          .ok());
  EXPECT_FALSE(ValidateShardedFiles("a/b", {"random-string"}).ok());
  EXPECT_FALSE(ValidateShardedFiles("a/b", {"a/b-random-string"}).ok());
}

TEST(ValidateShardedFilesTest, FileSpecTest) {
  auto spec = FileSpec::ShardedFileSpec("a/b", 2);
  EXPECT_THAT(spec.GetFilenames(),
              ElementsAre("a/b-00000-of-00002", "a/b-00001-of-00002"));
}

TEST(ValidateShardedFilesTest, LargeFileSet) {
  std::vector<std::string> filenames;
  for (int i = 0; i < 100; ++i) {
    filenames.push_back(GetShardedFileName("/a", i, 100));
  }
  TF_EXPECT_OK(ValidateShardedFiles("/a", filenames));
}

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
