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

#include "monolith/native_training/data/kernels/internal/datasource_utils.h"

#include <memory>
#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {
namespace {

TEST(DatasourceUtils, JavaHashCode) {
  int32_t code = java_hash_code("datasource_inst");
  EXPECT_EQ(code, -1487072768);
}

}  // namespace
}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
