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

#include "monolith/native_training/data/kernels/internal/value_filter_by_line_id.h"
#include "monolith/native_training/data/kernels/internal/value_filter_by_feature.h"

#include <memory>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "idl/matrix/proto/line_id.pb.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {
namespace {

using ::idl::matrix::proto::LineId;

TEST(LineIdValueFilter, Int) {
  LineId line_id;
  line_id.set_uid(2);

  tensorflow::Env* env = tensorflow::Env::Default();
  LineIdValueFilter filter_eq("uid", "eq", {}, {2}, {}, "", false);
  EXPECT_TRUE(filter_eq.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_neq("uid", "neq", {}, {2}, {}, "", false);
  EXPECT_FALSE(filter_neq.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_le("uid", "le", {}, {3}, {}, "", false);
  EXPECT_TRUE(filter_le.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_ge("uid", "ge", {}, {1}, {}, "", false);
  EXPECT_TRUE(filter_ge.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_between("uid", "between", {}, {1, 3}, {}, "", false);
  EXPECT_TRUE(filter_between.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_in("uid", "in", {}, {1, 2, 3}, {}, "", false);
  EXPECT_TRUE(filter_in.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_notin("uid", "not-in", {}, {1, 2, 3}, {}, "", false);
  EXPECT_FALSE(filter_notin.IsInstanceOfInterest(env, line_id));
}

TEST(LineIdValueFilter, IntArray) {
  LineId line_id;
  line_id.mutable_actions()->Add(2);
  line_id.mutable_actions()->Add(3);

  tensorflow::Env* env = tensorflow::Env::Default();
  LineIdValueFilter filter_any1("actions", "any", {}, {1, 2}, {}, "", false);
  EXPECT_TRUE(filter_any1.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_any2("actions", "any", {}, {1, 4}, {}, "", false);
  EXPECT_FALSE(filter_any2.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_all1("actions", "all", {}, {2, 3}, {}, "", false);
  EXPECT_TRUE(filter_all1.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_all2("actions", "all", {}, {2, 3, 4}, {}, "", false);
  EXPECT_FALSE(filter_all2.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_diff1("actions", "diff", {}, {1, 4}, {}, "", false);
  EXPECT_TRUE(filter_diff1.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_diff2("actions", "diff", {}, {1, 2, 4}, {}, "",
                                 false);
  EXPECT_FALSE(filter_diff2.IsInstanceOfInterest(env, line_id));
}

TEST(LineIdValueFilter, Float) {
  LineId line_id;
  line_id.set_q_pred(2.0f);

  tensorflow::Env* env = tensorflow::Env::Default();
  LineIdValueFilter filter_eq("q_pred", "eq", {2.0f}, {}, {}, "", false);
  EXPECT_TRUE(filter_eq.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_neq("q_pred", "neq", {2.0f}, {}, {}, "", false);
  EXPECT_FALSE(filter_neq.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_le("q_pred", "le", {3.0f}, {}, {}, "", false);
  EXPECT_TRUE(filter_le.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_ge("q_pred", "ge", {1.0f}, {}, {}, "", false);
  EXPECT_TRUE(filter_ge.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_between("q_pred", "between", {1.0f, 3.0f}, {}, {},
                                   "", false);
  EXPECT_TRUE(filter_between.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_in("q_pred", "in", {1.0f, 2.0f, 3.0f}, {}, {}, "",
                              false);
  EXPECT_TRUE(filter_in.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_notin("q_pred", "not-in", {1.0f, 2.0f, 3.0f}, {}, {},
                                 "", false);
  EXPECT_FALSE(filter_notin.IsInstanceOfInterest(env, line_id));
}

TEST(LineIdValueFilter, String) {
  LineId line_id;
  line_id.set_vid("hello");

  tensorflow::Env* env = tensorflow::Env::Default();
  LineIdValueFilter filter_eq("vid", "eq", {}, {}, {"hello"}, "", false);
  EXPECT_TRUE(filter_eq.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_neq("vid", "neq", {}, {}, {"hello"}, "", false);
  EXPECT_FALSE(filter_neq.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_le("vid", "le", {}, {}, {"hello1"}, "", false);
  EXPECT_TRUE(filter_le.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_ge("vid", "ge", {}, {}, {"hell"}, "", false);
  EXPECT_TRUE(filter_ge.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_between("vid", "between", {}, {}, {"hell", "hello1"},
                                   "", false);
  EXPECT_TRUE(filter_between.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_in("vid", "in", {}, {}, {"hello", "world"}, "",
                              false);
  EXPECT_TRUE(filter_in.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_notin("vid", "not-in", {}, {}, {"hello", "world"},
                                 "", false);
  EXPECT_FALSE(filter_notin.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_startswith("vid", "startswith", {}, {}, {"hell"}, "",
                                      false);
  EXPECT_TRUE(filter_startswith.IsInstanceOfInterest(env, line_id));

  LineIdValueFilter filter_endswith("vid", "endswith", {}, {}, {"llo"}, "",
                                    false);
  EXPECT_TRUE(filter_endswith.IsInstanceOfInterest(env, line_id));
}

}  // namespace
}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
