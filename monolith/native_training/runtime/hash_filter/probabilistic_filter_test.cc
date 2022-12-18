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

#include "monolith/native_training/runtime/hash_filter/probabilistic_filter.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_factory.h"

namespace monolith {
namespace hash_filter {
namespace {

using ::monolith::hash_table::EmbeddingHashTableConfig;
using ::monolith::hash_table::EmbeddingHashTableInterface;
namespace proto2 = google::protobuf;

TEST(ProbabilisticFilterTest, UnequalProbability) {
  ProbabilisticFilter filter;

  int fid_num = 10000, slot_occurrence_threshold = 7;
  int count1 = 0, count2 = 0;
  float tolerance_ratio = 0.01f;
  for (int i = 0; i < fid_num; ++i) {
    if (filter.InsertedIntoHashTableUnequalProbability(
            1, slot_occurrence_threshold)) {
      ++count1;
    }
    if (filter.InsertedIntoHashTableUnequalProbability(
            2, slot_occurrence_threshold)) {
      ++count2;
    }
  }

  EXPECT_NEAR(fid_num / slot_occurrence_threshold, count1,
              fid_num * tolerance_ratio);
  EXPECT_NEAR(fid_num / slot_occurrence_threshold * 2, count2,
              fid_num * tolerance_ratio * 2);
}

TEST(ProbabilisticFilterTest, EqualProbability) {
  ProbabilisticFilter filter;

  int fid_num = 10000, slot_occurrence_threshold = 7;
  int count1 = 0, count2 = 0;
  float tolerance_ratio = 0.05;
  float p = 1 - std::pow(tolerance_ratio,
                         1.f / static_cast<float>(slot_occurrence_threshold));

  for (int i = 0; i < fid_num; ++i) {
    if (filter.InsertedIntoHashTableEqualProbability(
            1, slot_occurrence_threshold)) {
      ++count1;
    }
    if (filter.InsertedIntoHashTableEqualProbability(
            2, slot_occurrence_threshold)) {
      ++count2;
    }
  }

  EXPECT_NEAR(fid_num * p, count1, fid_num * tolerance_ratio);
  EXPECT_NEAR(fid_num * (1.f - std::pow(1 - p, 2)), count2,
              fid_num * tolerance_ratio * 2);
}

TEST(ProbabilisticFilterTest, ShouldBeFiltered) {
  EmbeddingHashTableConfig config;
  EXPECT_TRUE(proto2::TextFormat::ParseFromString(R"(
    entry_config {
      segments {
        dim_size: 1
        init_config { zeros {} }
        opt_config { sgd {} }
      }
    }
    cuckoo {}
  )",
                                                  &config));
  std::unique_ptr<EmbeddingHashTableInterface> table =
      NewEmbeddingHashTableFromConfig(config);

  ProbabilisticFilter filter(false);

  int fid_num = 10000, slot_occurrence_threshold = 7;
  int count1 = 0, count2 = 0, count3 = 0;
  float tolerance_ratio = 0.01f;
  for (int i = 0; i < fid_num; ++i) {
    if (!filter.ShouldBeFiltered(i, 1, slot_occurrence_threshold,
                                 table.get())) {
      ++count1;
    }

    if (!filter.ShouldBeFiltered(i, 2, slot_occurrence_threshold,
                                 table.get())) {
      ++count2;
    }

    if (!filter.ShouldBeFiltered(i, 3, slot_occurrence_threshold, nullptr)) {
      ++count3;
    }
  }

  EXPECT_NEAR(fid_num / slot_occurrence_threshold, count1,
              fid_num * tolerance_ratio);
  EXPECT_NEAR(fid_num / slot_occurrence_threshold * 2, count2,
              fid_num * tolerance_ratio);
  EXPECT_EQ(fid_num, count3);
}

}  // namespace
}  // namespace hash_filter
}  // namespace monolith
