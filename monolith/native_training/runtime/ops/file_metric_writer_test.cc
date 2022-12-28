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

#include "monolith/native_training/runtime/ops/file_metric_writer.h"

#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "tensorflow/core/lib/io/record_writer.h"

namespace monolith {
namespace deep_insight {

using tensorflow::Env;
using tensorflow::Status;
using tensorflow::io::RecordWriter;
using tensorflow::io::RecordWriterOptions;

TEST(FileMetricWriterTest, Basic) {
  Env* env = Env::Default();
  std::string filename;
  CHECK(env->LocalTempFilename(&filename));
  FileMetricWriter writer(filename);
  writer.Write("hello");
  writer.Write("world");
}

}  // namespace deep_insight
}  // namespace monolith
