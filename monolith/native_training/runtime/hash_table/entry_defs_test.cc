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

#include "monolith/native_training/runtime/hash_table/entry_defs.h"

#include "gtest/gtest.h"

#include "gmock/gmock.h"

namespace monolith {
namespace hash_table {

TEST(InlineEntryTest, Basic) {
  InlineEntry<8> entry;
  *reinterpret_cast<float*>(entry.get()) = 1.0;
  EXPECT_THAT(entry.capacity(), 4);
  entry.SetTimestamp(1234);
  EXPECT_THAT(entry.GetTimestamp(), 1234);
  EXPECT_THAT(*reinterpret_cast<float*>(entry.get()), 1.0);
}

}  // namespace hash_table
}  // namespace monolith