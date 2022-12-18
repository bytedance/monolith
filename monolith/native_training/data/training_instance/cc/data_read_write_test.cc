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

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/data_writer.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

class ReadWriteTest : public ::testing::TestWithParam<DataFormatOptions> {};

Status ReadBytes(BaseStreamReaderTmpl<absl::string_view>* reader,
                 absl::string_view* out) {
  uint8_t pb_type;
  uint32_t data_source_key;
  return reader->ReadPBBytes(&pb_type, &data_source_key, out);
}

TEST_P(ReadWriteTest, Basic) {
  DataFormatOptions options = GetParam();
  std::string s;
  StringStreamWriter writer(options, &s);
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(writer.WriteRecord(std::string(i, 'a')).ok());
  }
  ZeroCopyStringViewStreamReader reader(options, s);
  absl::string_view out;
  for (int i = 0; i < 16; ++i) {
    auto status = ReadBytes(&reader, &out);
    EXPECT_TRUE(status.ok()) << status;
    EXPECT_THAT(out, std::string(i, 'a')) << i;
  }
}

std::vector<DataFormatOptions> GenerateOptions() {
  std::vector<DataFormatOptions> res;
  for (int i = 0; i < 16; ++i) {
    DataFormatOptions options;
    options.lagrangex_header = i & 1;
    options.kafka_dump_prefix = i / 2 & 1;
    options.has_sort_id = i / 4 & 1;
    options.kafka_dump = i / 8 & 1;
    res.push_back(options);
  }
  return res;
}

INSTANTIATE_TEST_SUITE_P(ReadWriteTestAll, ReadWriteTest,
                         testing::ValuesIn(GenerateOptions()));

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
