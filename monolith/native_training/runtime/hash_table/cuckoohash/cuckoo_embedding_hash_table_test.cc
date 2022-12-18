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

#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_test.h"

namespace monolith {
namespace hash_table {
namespace {

namespace proto2 = google::protobuf;
using ::testing::ElementsAre;

std::tuple<EmbeddingHashTableConfig, std::vector<float>>
GetTestOneDimSgdHashTable(EmbeddingHashTableConfig::EntryType type =
                              EmbeddingHashTableConfig::PACKED) {
  EmbeddingHashTableConfig config;
  EXPECT_TRUE(proto2::TextFormat::ParseFromString(R"(
    entry_config {
      segments {
        dim_size: 1
        init_config { zeros {} }
        opt_config { sgd {} }
      }
    }
    initial_capacity: 1
    cuckoo {}
  )",
                                                  &config));
  config.set_entry_type(type);
  std::vector<float> learning_rates(1, 0.01f);
  return std::make_tuple(config, learning_rates);
}

INSTANTIATE_TEST_CASE_P(
    CuckooHashmapReadWrite, ReadWriteEmbeddingHashTableTest,
    ::testing::Values(
        GetTestOneDimSgdHashTable(EmbeddingHashTableConfig::PACKED),
        GetTestOneDimSgdHashTable(EmbeddingHashTableConfig::RAW)));

INSTANTIATE_TEST_CASE_P(CuckooHashmapRestore, SaveRestoreEmbeddingHashTestTest,
                        ::testing::Values(GetTestOneDimSgdHashTable()));

INSTANTIATE_TEST_CASE_P(OneTimeEvict, EmbeddingHashTableEvictTest,
                        ::testing::Values(GetTestOneDimSgdHashTable()));

INSTANTIATE_TEST_CASE_P(EvictWhileRehash, EmbeddingHashTableEvictTest,
                        ::testing::Values(GetTestOneDimSgdHashTable()));

}  // namespace
}  // namespace hash_table
}  // namespace monolith
