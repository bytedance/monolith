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
#include "monolith/native_training/data/training_instance/cc/fid.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"

namespace {
using tensorflow::monolith_tf::GetFidV1;
using tensorflow::monolith_tf::GetFidV2;

TEST(FIDTest, FIDV1) {
  // 8 bytes
  EXPECT_EQ(sizeof(FIDV1), 8);

  // normal case
  FIDV1 fid1(1, 100);
  EXPECT_EQ(fid1.slot(), 1);
  EXPECT_EQ(fid1.signature(), 100);
  EXPECT_EQ(fid1, GetFidV1(1, 100));

  // corner case1
  FIDV1 fid2(1023, 1LL << 54);
  EXPECT_EQ(fid2.slot(), 1023);
  EXPECT_EQ(fid2.signature(), 0);
  EXPECT_EQ(fid2, GetFidV1(1023, 1LL << 54));

  // corner case2
  EXPECT_THROW(
      {
        FIDV1 fid3(1024, 1LL << 54);
        EXPECT_EQ(fid3.slot(), 0);
        EXPECT_EQ(fid3.signature(), 0);
        EXPECT_EQ(fid3, GetFidV1(1024, 1LL << 54));
      },
      std::invalid_argument);

  // corner case3
  EXPECT_THROW(
      {
        FIDV1 fid4(1025, 1LL << 54 | 1);
        EXPECT_EQ(fid4.slot(), 1);
        EXPECT_EQ(fid4.signature(), 1);
        EXPECT_EQ(fid4, GetFidV1(1025, 1LL << 54 | 1));
      },
      std::invalid_argument);
}

TEST(FIDTest, FIDV2) {
  // 8 bytes
  EXPECT_EQ(sizeof(FIDV2), 8);

  // normal case
  FIDV2 fid1(1, 100);
  EXPECT_EQ(fid1.slot(), 1);
  EXPECT_EQ(fid1.signature(), 100);
  EXPECT_EQ(fid1, GetFidV2(1, 100));

  // corner case1
  FIDV2 fid2(1024, 1LL << 54);
  EXPECT_EQ(fid2.slot(), 1024);
  EXPECT_EQ(fid2.signature(), 0);
  EXPECT_EQ(fid2, GetFidV2(1024, 1LL << 54));

  // corner case2
  FIDV2 fid3(32767, 1LL << 48);
  EXPECT_EQ(fid3.slot(), 32767);
  EXPECT_EQ(fid3.signature(), 0);
  EXPECT_EQ(fid3, GetFidV2(32767, 1LL << 48));

  // corner case3
  EXPECT_THROW(
      {
        FIDV2 fid4(32768, 1LL << 48);
        EXPECT_EQ(fid4.slot(), 0);
        EXPECT_EQ(fid4.signature(), 0);
        // GetFidV2 has a tiny bug
        EXPECT_EQ(fid4, (GetFidV2(32768, 1LL << 48) << 1) >> 1);
      },
      std::invalid_argument);

  // corner case4
  EXPECT_THROW(
      {
        FIDV2 fid5(32769, 1LL << 48 | 1);
        EXPECT_EQ(fid5.slot(), 1);
        EXPECT_EQ(fid5.signature(), 1);
        // GetFidV2 has a tiny bug
        EXPECT_EQ(fid5, (GetFidV2(32769, 1LL << 48 | 1) << 1) >> 1);
      },
      std::invalid_argument);
}

TEST(FIDTest, FIDV1ConvertV2) {
  // normal case
  FIDV1 fid_v1(1, 100);
  FIDV2 fid_v2 = fid_v1.ConvertAsV2();
  EXPECT_EQ(fid_v2.slot(), 1);
  EXPECT_EQ(fid_v2.signature(), 100);
  EXPECT_EQ(fid_v2, convert_fid_v1_to_v2(fid_v1));

  // corner case1
  FIDV1 fid_v1_1(1023, 1LL << 54);
  FIDV2 fid_v2_1 = fid_v1_1.ConvertAsV2();
  EXPECT_EQ(fid_v2_1.slot(), 1023);
  EXPECT_EQ(fid_v2_1.signature(), 0);
  EXPECT_EQ(fid_v2_1, convert_fid_v1_to_v2(fid_v1_1));

  // corner case2
  EXPECT_THROW(
      {
        FIDV1 fid_v1_2(1024, 1LL << 54);
        FIDV2 fid_v2_2 = fid_v1_2.ConvertAsV2();
        EXPECT_EQ(fid_v2_2.slot(), 0);
        EXPECT_EQ(fid_v2_2.signature(), 0);
        EXPECT_EQ(fid_v2_2, convert_fid_v1_to_v2(fid_v1_2));
      },
      std::invalid_argument);
}

}  // namespace
