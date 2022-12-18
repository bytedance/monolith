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

#include "monolith/native_training/data/training_instance/cc/instance_utils.h"

#include "absl/time/clock.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {
namespace {

using ::testing::Eq;

TEST(StrToIntegerSet, StrToFIDs) {
  std::set<uint64_t> fids1 = StrToIntegerSet<uint64_t>("");
  EXPECT_TRUE(fids1.empty());

  std::set<uint64_t> fids2 = StrToIntegerSet<uint64_t>("6461985998153810495");
  EXPECT_EQ(fids2.size(), 1);
  EXPECT_EQ(*fids2.begin(), 6461985998153810495ull);

  std::set<uint64_t> fids3 =
      StrToIntegerSet<uint64_t>("6457882839108881377,6436927642569553426,");
  EXPECT_EQ(fids3.size(), 2);
  EXPECT_EQ(*fids3.begin(), 6436927642569553426ull);
  EXPECT_EQ(*std::next(fids3.begin(), 1), 6457882839108881377ul);

  std::set<uint64_t> fids4 = StrToIntegerSet<uint64_t>("6461985998153810495,");
  EXPECT_EQ(fids4.size(), 1);
  EXPECT_EQ(*fids4.begin(), 6461985998153810495ull);

  try {
    StrToIntegerSet<uint64_t>("6461985998153810495,abc");
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string(e.what()), Eq("Invalid integer string: abc"));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Unexpected exception thrown: " << e.what() << std::endl;
  }
}

TEST(StrToIntegerSet, StrToActions) {
  std::set<int32_t> actions1 = StrToIntegerSet<int32_t>("");
  EXPECT_TRUE(actions1.empty());

  std::set<int32_t> actions2 = StrToIntegerSet<int32_t>("-1");
  EXPECT_EQ(actions2.size(), 1);
  EXPECT_EQ(*actions2.begin(), -1);

  std::set<int32_t> actions3 = StrToIntegerSet<int32_t>("-1,3,");
  EXPECT_EQ(actions3.size(), 2);
  EXPECT_EQ(*actions3.begin(), -1);
  EXPECT_EQ(*std::next(actions3.begin(), 1), 3);

  std::set<int32_t> actions4 = StrToIntegerSet<int32_t>("1,");
  EXPECT_EQ(actions4.size(), 1);
  EXPECT_EQ(*actions4.begin(), 1);

  try {
    StrToIntegerSet<int32_t>("1,abc");
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string(e.what()), Eq("Invalid integer string: abc"));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Unexpected exception thrown: " << e.what() << std::endl;
  }
}

TEST(IsInstanceOfInterest, Basic) {
  parser::proto::Instance instance;
  instance.mutable_fid()->Add(6436927642569553426ull);
  instance.mutable_fid()->Add(6457882839108881377ull);
  instance.mutable_fid()->Add(6461985998153810495ull);

  int64_t now = absl::ToUnixSeconds(absl::Now());
  instance.mutable_line_id()->set_req_time(now);
  instance.mutable_line_id()->mutable_actions()->Add(-1);
  instance.mutable_line_id()->mutable_actions()->Add(1);

  std::set<uint64_t> filter_fids = {6436927642569553426ull};
  std::set<uint64_t> has_fids = {6436927642569553426ull};
  std::set<uint64_t> select_fids = {6436927642569553426ull,
                                    6457882839108881377ull};

  // 1626537600 -> 2021-07-18 00:00:00
  int64_t req_time_min = 1626537600;
  EXPECT_TRUE(IsInstanceOfInterest(instance, {}, {}, {}, {}, req_time_min, {}));
  EXPECT_TRUE(IsInstanceOfInterest(instance, {}, {}, {}, {}, now, {}));
  EXPECT_TRUE(!IsInstanceOfInterest(instance, {}, {}, {}, {}, now + 1, {}));
  EXPECT_TRUE(!IsInstanceOfInterest(instance, filter_fids, {}, {}, {},
                                    req_time_min, {}));
  EXPECT_TRUE(
      IsInstanceOfInterest(instance, {}, has_fids, {}, {}, req_time_min, {}));
  EXPECT_TRUE(IsInstanceOfInterest(instance, {}, {}, select_fids, {},
                                   req_time_min, {}));
  EXPECT_TRUE(!IsInstanceOfInterest(instance, filter_fids, has_fids,
                                    select_fids, {}, req_time_min, {}));
  EXPECT_TRUE(
      IsInstanceOfInterest(instance, {}, {}, {}, {-1}, req_time_min, {}));
  EXPECT_TRUE(
      IsInstanceOfInterest(instance, {}, {}, {}, {1, 5}, req_time_min, {}));
  EXPECT_TRUE(IsInstanceOfInterest(instance, {}, has_fids, select_fids, {1, 5},
                                   req_time_min, {}));
}

TEST(CollectFidIntoSet, Instance) {
  parser::proto::Instance instance;
  instance.mutable_fid()->Add(GetFidV1(2, 200));
  instance.mutable_fid()->Add(GetFidV1(3, 300));

  auto f1 = instance.mutable_feature()->Add();
  f1->mutable_fid()->Add(GetFidV2(1024, 102400));
  auto f2 = instance.mutable_feature()->Add();
  f2->mutable_fid()->Add(GetFidV2(4096, 409600));

  std::set<uint32_t> slots, slots_expected{2, 3, 1024, 4096}, intersection;
  CollectSlotIntoSet(instance, &slots);
  std::set_intersection(slots.begin(), slots.end(), slots_expected.begin(),
                        slots_expected.end(),
                        std::inserter(intersection, intersection.begin()));
  EXPECT_EQ(intersection.size(), slots_expected.size());

  std::set<uint32_t> select_slots1 = {2, 3, 1024, 4096},
                     select_slots2 = {2, 10};
  EXPECT_TRUE(IsInstanceOfInterest(instance, {}, {}, {}, {}, 0, select_slots1));
  EXPECT_TRUE(
      !IsInstanceOfInterest(instance, {}, {}, {}, {}, 0, select_slots2));
}

TEST(CollectFidIntoSet, Example) {
  monolith::io::proto::Example example;

  auto f1 = example.mutable_named_feature()->Add();
  f1->set_name("user_id");
  f1->mutable_feature()->mutable_fid_v1_list()->mutable_value()->Add(
      GetFidV1(2, 200));
  auto f2 = example.mutable_named_feature()->Add();
  f2->set_name("item_id");
  f2->mutable_feature()->mutable_fid_v1_list()->mutable_value()->Add(
      GetFidV1(3, 300));
  auto f3 = example.mutable_named_feature()->Add();
  f3->set_name("gender");
  f3->mutable_feature()->mutable_fid_v2_list()->mutable_value()->Add(
      GetFidV2(1024, 102400));
  auto f4 = example.mutable_named_feature()->Add();
  f4->set_name("age");
  f4->mutable_feature()->mutable_fid_v2_list()->mutable_value()->Add(
      GetFidV2(4096, 409600));

  std::set<uint32_t> slots, slots_expected{2, 3, 1024, 4096}, intersection;
  CollectSlotIntoSet(example, &slots);
  std::set_intersection(slots.begin(), slots.end(), slots_expected.begin(),
                        slots_expected.end(),
                        std::inserter(intersection, intersection.begin()));
  EXPECT_EQ(intersection.size(), slots_expected.size());

  std::set<uint32_t> select_slots1 = {2, 3, 1024, 4096},
                     select_slots2 = {2, 10};
  EXPECT_TRUE(IsInstanceOfInterest(example, {}, {}, {}, {}, 0, select_slots1));
  EXPECT_TRUE(!IsInstanceOfInterest(example, {}, {}, {}, {}, 0, select_slots2));
}

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
