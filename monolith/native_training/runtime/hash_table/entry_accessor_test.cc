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

#include "monolith/native_training/runtime/hash_table/entry_accessor.h"

#include <memory>

#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table.pb.h"

namespace monolith {
namespace hash_table {
namespace {

namespace proto2 = google::protobuf;

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::FloatNear;

TEST(EntryAccessorTest, FromConfig) {
  EntryConfig config;
  ASSERT_TRUE(proto2::TextFormat::ParseFromString(R"(
    segments {
      dim_size: 1
      init_config { zeros {} }
      opt_config { sgd {} }
    }
    segments {
      dim_size: 2
      init_config { zeros {} }
      opt_config { sgd {} }
    }
  )",
                                                  &config));
  auto accessor = NewEntryAccessor(config);
  auto entry = std::make_unique<char[]>(accessor->SizeBytes());
  accessor->Init(entry.get());
  accessor->Optimize(entry.get(), {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f}, 0);
  std::vector<float> num(3);
  accessor->Fill(entry.get(), absl::MakeSpan(num));
  EXPECT_THAT(absl::MakeSpan(num), ElementsAre(-1.0f, -4.0f, -6.0f));
}

TEST(EntryAccessorTest, SaveRestore) {
  EntryConfig config;
  ASSERT_TRUE(proto2::TextFormat::ParseFromString(R"(
    segments {
      dim_size: 1
      init_config { zeros {} }
      opt_config { adagrad { initial_accumulator_value: 0.1
      } }
    }
  )",
                                                  &config));
  auto accessor = NewEntryAccessor(config);
  auto entry1 = std::make_unique<char[]>(accessor->SizeBytes());
  accessor->Init(entry1.get());
  accessor->Optimize(entry1.get(), {1.0f}, {1.0f});
  std::vector<float> num(1);
  accessor->Fill(entry1.get(), absl::MakeSpan(num));
  ASSERT_THAT(absl::MakeSpan(num), ElementsAre(FloatEq(-0.95346254f)));
  EntryDump dump = accessor->Save(entry1.get(), 100);
  auto entry2 = std::make_unique<char[]>(accessor->SizeBytes());
  uint32_t timestamp_sec;
  accessor->Restore(entry2.get(), &timestamp_sec, dump);
  EXPECT_EQ(timestamp_sec, 100);
  accessor->Optimize(entry2.get(), {1.0f}, {1.0f}, 0);
  accessor->Fill(entry2.get(), absl::MakeSpan(num));
  ASSERT_THAT(absl::MakeSpan(num), ElementsAre(FloatEq(-1.643528f)));
}

TEST(EntryAccessorTest, Update) {
  std::unordered_map<std::string, std::string> configs = {{"fp32", R"(
        segments {
          dim_size: 3
          init_config { zeros {} }
          opt_config { sgd {} }
        }
      )"},
                                                          {"fp16", R"(
        segments {
          dim_size: 3
          init_config { zeros {} }
          opt_config { sgd {} }
        }
      )"}};

  for (const auto& kv : configs) {
    EntryConfig config;
    ASSERT_TRUE(proto2::TextFormat::ParseFromString(kv.second, &config));
    auto accessor = NewEntryAccessor(config);
    auto entry = std::make_unique<char[]>(accessor->SizeBytes());

    std::vector<float> num = {0.1, 0.2, 0.3};
    accessor->Assign(absl::MakeSpan(num), entry.get());
    std::vector<float> embedding(3);
    accessor->Fill(entry.get(), absl::MakeSpan(embedding));

    if (kv.first == "fp32") {
      EXPECT_THAT(embedding, ElementsAre(0.1, 0.2, 0.3));
    }
    if (kv.first == "fp16") {
      float eps = 0.0001;
      EXPECT_THAT(embedding,
                  ElementsAre(FloatNear(0.1, eps), FloatNear(0.2, eps),
                              FloatNear(0.3, eps)));
    }
  }
}

TEST(ServingEntryAccessorTest, Basic) {
  EntryConfig config;
  ASSERT_TRUE(proto2::TextFormat::ParseFromString(R"(
    segments {
      dim_size: 1
      comp_config { fp32 {} }
    }
    entry_type: SERVING
  )",
                                                  &config));
  auto accessor = NewEntryAccessor(config);
  auto entry = std::make_unique<char[]>(accessor->SizeBytes());
  EntryDump dump;
  dump.add_num(1.0);
  dump.set_last_update_ts_sec(100);
  uint32_t timestamp_sec;
  accessor->Restore(entry.get(), &timestamp_sec, dump);
  EXPECT_THAT(timestamp_sec, timestamp_sec);
  std::vector<float> out(1);
  accessor->Fill(entry.get(), absl::MakeSpan(out));
  EXPECT_THAT(out, ElementsAre(1.0));
}

TEST(ServingEntryAccessorTest, Update) {
  std::unordered_map<std::string, std::string> configs = {{"fp32", R"(
        segments {
          dim_size: 3
          comp_config { fp32 {} }
        }
        entry_type: SERVING
      )"},
                                                          {"fp16", R"(
        segments {
          dim_size: 3
          comp_config { fp16 {} }
        }
        entry_type: SERVING
      )"}};

  for (const auto& kv : configs) {
    EntryConfig config;
    ASSERT_TRUE(proto2::TextFormat::ParseFromString(kv.second, &config));
    auto accessor = NewEntryAccessor(config);
    auto entry = std::make_unique<char[]>(accessor->SizeBytes());

    std::vector<float> num = {0.1, 0.2, 0.3};
    accessor->Assign(absl::MakeSpan(num), entry.get());
    std::vector<float> embedding(3);
    accessor->Fill(entry.get(), absl::MakeSpan(embedding));

    if (kv.first == "fp32") {
      EXPECT_THAT(embedding, ElementsAre(0.1, 0.2, 0.3));
    }
    if (kv.first == "fp16") {
      float eps = 0.0001;
      EXPECT_THAT(embedding,
                  ElementsAre(FloatNear(0.1, eps), FloatNear(0.2, eps),
                              FloatNear(0.3, eps)));
    }
  }
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
