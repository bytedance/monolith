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

#include "monolith/native_training/data/training_instance/cc/reader_util.h"
#include <string>
#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {

const uint32_t kSlotBits = 15;
const uint32_t kFeatureV2Bits = 48;

TEST(ReaderUtilTest, GetSlotID) {
  int64_t slot_id = 123;
  int64_t fid = slot_id << kFeatureV2Bits;
  int64_t slot_id_ret = slot_id_v2(fid);
  ASSERT_TRUE(slot_id_ret == slot_id);

  slot_id = (1 << kSlotBits) - 1;
  fid = slot_id << kFeatureV2Bits;
  slot_id_ret = slot_id_v2(fid);
  ASSERT_TRUE(slot_id_ret == slot_id);
}


TEST(ReaderUtilTest, FeatureNameMapperNormalCase1) {
  auto mapper = std::make_unique<FeatureNameMapper>();
  ASSERT_FALSE(mapper->IsAvailable());
  mapper->TurnOn();
  ASSERT_FALSE(mapper->IsAvailable());

  std::string name1 = "slot_1", name2 = "slot_2", name3 = "slot_3",
              name4 = "slot_4";
  int id1 = -1, id2 = -1, id3 = -1, id4 = -1;

  ASSERT_TRUE(mapper->RegisterValidNames({name1, name2}));
  ASSERT_TRUE(mapper->RegisterValidIds({{3, 10003}}));

  absl::flat_hash_map<std::string, int32_t> m;
  absl::flat_hash_map<int32_t, std::vector<std::string>> m2;
  m.insert({name1, 1});
  m.insert({name2, 2});
  m.insert({name3, 3});
  m.insert({name4, 4});
  m2.insert({1, {name1}});
  m2.insert({2, {name2}});
  m2.insert({3, {name3}});
  m2.insert({4, {name4}});
  ASSERT_TRUE(mapper->SetMapping(m, m2));
  ASSERT_TRUE(mapper->GetIdByName(name1, &id1));
  ASSERT_TRUE(mapper->GetIdByName(name2, &id2));
  ASSERT_TRUE(mapper->GetIdByName(name3, &id3));
  ASSERT_FALSE(mapper->GetIdByName(name4, &id4));
  ASSERT_EQ(id1, 1);
  ASSERT_EQ(id2, 2);
  ASSERT_EQ(id3, 3);

  LOG(INFO) << mapper->DebugString();
}

TEST(ReaderUtilTest, FeatureNameMapperNormalCase2) {
  auto mapper = std::make_unique<FeatureNameMapper>();
  ASSERT_FALSE(mapper->IsAvailable());
  mapper->TurnOn();
  ASSERT_FALSE(mapper->IsAvailable());

  std::string name1 = "slot_1", name2 = "slot_2", name3 = "slot_3",
              name4 = "slot_4";
  int id1 = -1, id2 = -1, id3 = -1, id4 = -1;

  absl::flat_hash_map<std::string, int32_t> m;
  absl::flat_hash_map<int32_t, std::vector<std::string>> m2;
  m.insert({name1, 1});
  m.insert({name2, 2});
  m.insert({name3, 3});
  m.insert({name4, 4});
  m2.insert({1, {name1}});
  m2.insert({2, {name2}});
  m2.insert({3, {name3}});
  m2.insert({4, {name4}});
  ASSERT_TRUE(mapper->SetMapping(m, m2));
  ASSERT_FALSE(mapper->GetIdByName(name1, &id1));
  ASSERT_FALSE(mapper->GetIdByName(name2, &id2));
  ASSERT_FALSE(mapper->GetIdByName(name3, &id3));
  ASSERT_FALSE(mapper->GetIdByName(name4, &id4));

  ASSERT_TRUE(mapper->RegisterValidNames({name1, name2}));
  ASSERT_TRUE(mapper->RegisterValidIds({{3, 10003}}));

  ASSERT_TRUE(mapper->GetIdByName(name1, &id1));
  ASSERT_TRUE(mapper->GetIdByName(name2, &id2));
  ASSERT_TRUE(mapper->GetIdByName(name3, &id3));
  ASSERT_FALSE(mapper->GetIdByName(name4, &id4));
  ASSERT_EQ(id1, 1);
  ASSERT_EQ(id2, 2);
  ASSERT_EQ(id3, 3);

  LOG(INFO) << mapper->DebugString();
}

TEST(ReaderUtilTest, FeatureNameMapperCornerCase1) {
  auto mapper = std::make_unique<FeatureNameMapper>();
  ASSERT_FALSE(mapper->IsAvailable());
  mapper->TurnOn();
  ASSERT_FALSE(mapper->IsAvailable());

  std::string name1 = "slot_1", name2 = "slot_2", name3 = "slot_3",
              name4 = "slot_4";
  int id1 = -1, id2 = -1, id3 = -1, id4 = -1;

  ASSERT_TRUE(mapper->RegisterValidNames({name1, name2}));
  absl::flat_hash_map<std::string, int32_t> m;
  absl::flat_hash_map<int32_t, std::vector<std::string>> m2;
  m.insert({name1, 1});
  m.insert({name2, 2});
  m.insert({name3, 3});
  m.insert({name4, 4});
  m2.insert({1, {name1}});
  m2.insert({2, {name2}});
  m2.insert({3, {name3}});
  m2.insert({4, {name4}});
  ASSERT_TRUE(mapper->SetMapping(m, m2));
  ASSERT_TRUE(mapper->RegisterValidIds({{3, 10003}}));

  ASSERT_TRUE(mapper->GetIdByName(name1, &id1));
  ASSERT_TRUE(mapper->GetIdByName(name2, &id2));
  ASSERT_TRUE(mapper->GetIdByName(name3, &id3));
  ASSERT_FALSE(mapper->GetIdByName(name4, &id4));
  ASSERT_EQ(id1, 1);
  ASSERT_EQ(id2, 2);
  ASSERT_EQ(id3, 3);

  LOG(INFO) << mapper->DebugString();
}

TEST(ReaderUtilTest, FeatureNameMapperCornerCase2) {
  auto mapper = std::make_unique<FeatureNameMapper>();
  ASSERT_FALSE(mapper->IsAvailable());
  mapper->TurnOn();
  ASSERT_FALSE(mapper->IsAvailable());

  std::string name1 = "slot_1", name2 = "slot_2", name3 = "slot_3",
              name4 = "slot_4";
  int id1 = -1, id2 = -1, id3 = -1, id4 = -1;

  ASSERT_TRUE(mapper->RegisterValidIds({{3, 10003}}));
  absl::flat_hash_map<std::string, int32_t> m;
  absl::flat_hash_map<int32_t, std::vector<std::string>> m2;
  m.insert({name1, 1});
  m.insert({name2, 2});
  m.insert({name3, 3});
  m.insert({name4, 4});
  m2.insert({1, {name1}});
  m2.insert({2, {name2}});
  m2.insert({3, {name3}});
  m2.insert({4, {name4}});
  ASSERT_TRUE(mapper->SetMapping(m, m2));
  ASSERT_TRUE(mapper->RegisterValidNames({name1, name2}));

  ASSERT_TRUE(mapper->GetIdByName(name1, &id1));
  ASSERT_TRUE(mapper->GetIdByName(name2, &id2));
  ASSERT_TRUE(mapper->GetIdByName(name3, &id3));
  ASSERT_FALSE(mapper->GetIdByName(name4, &id4));
  ASSERT_EQ(id1, 1);
  ASSERT_EQ(id2, 2);
  ASSERT_EQ(id3, 3);

  LOG(INFO) << mapper->DebugString();
}

TEST(ReaderUtilTest, FeatureNameMapperCornerCase3) {
  auto mapper = std::make_unique<FeatureNameMapper>();
  ASSERT_FALSE(mapper->IsAvailable());
  mapper->TurnOn();
  ASSERT_FALSE(mapper->IsAvailable());

  std::string name1 = "slot_1";

  ASSERT_TRUE(mapper->RegisterValidIds({{2, 10002}}));
  absl::flat_hash_map<std::string, int32_t> m;
  absl::flat_hash_map<int32_t, std::vector<std::string>> m2;
  m.insert({name1, 1});
  m2.insert({1, {name1}});
  ASSERT_FALSE(mapper->SetMapping(m, m2));

  LOG(INFO) << mapper->DebugString();
}

TEST(ReaderUtilTest, FeatureNameMapperCornerCase4) {
  auto mapper = std::make_unique<FeatureNameMapper>();
  ASSERT_FALSE(mapper->IsAvailable());
  mapper->TurnOn();
  ASSERT_FALSE(mapper->IsAvailable());

  std::string name1 = "slot_1";

  absl::flat_hash_map<std::string, int32_t> m;
  absl::flat_hash_map<int32_t, std::vector<std::string>> m2;
  m.insert({name1, 1});
  m2.insert({1, {name1}});
  ASSERT_TRUE(mapper->SetMapping(m, m2));
  ASSERT_FALSE(mapper->RegisterValidIds({{2, 10002}}));

  LOG(INFO) << mapper->DebugString();
}

}  // namespace monolith_tf
}  // namespace tensorflow
