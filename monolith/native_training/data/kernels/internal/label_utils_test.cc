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

#include "monolith/native_training/data/kernels/internal/label_utils.h"

#include <memory>
#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {
namespace {
using LabelConf = ::monolith::native_training::data::config::LabelConf;

TEST(LabelUtils, HasIntersection) {
  std::set<int32_t> lhs = {1, 2, 3}, rhs1 = {3, 4, 5}, rhs2 = {};
  EXPECT_TRUE(HasIntersection(lhs, rhs1));
  EXPECT_FALSE(HasIntersection(lhs, rhs2));
}

TEST(LabelUtils, ParseTaskConfigBasic) {
  LabelConf label_conf;

  auto *task_conf = label_conf.add_conf();
  task_conf->add_pos_actions(-7);
  task_conf->add_pos_actions(-9);
  task_conf->add_neg_actions(-41);
  task_conf->set_sample_rate(0.5f);

  task_conf = label_conf.add_conf();
  task_conf->add_pos_actions(75);
  task_conf->add_pos_actions(-103);
  task_conf->add_pos_actions(74);
  task_conf->add_neg_actions(-41);
  task_conf->set_sample_rate(1.0f);

  task_conf = label_conf.add_conf();
  task_conf->add_pos_actions(101);
  task_conf->add_pos_actions(102);
  task_conf->set_sample_rate(1.0f);

  std::string config;
  label_conf.SerializeToString(&config);
  std::vector<TaskConfig> task_configs;
  ParseTaskConfig(config, &task_configs);

  EXPECT_EQ(task_configs.size(), 3);
  std::set<int32_t> pos_actions0 = {-7, -9}, pos_actions1 = {-103, 74, 75},
                    pos_actions2 = {101, 102};
  std::set<int32_t> neg_actions0 = {-41}, neg_actions1 = {-41},
                    neg_actions2 = {};
  EXPECT_EQ(task_configs[0].pos_actions, pos_actions0);
  EXPECT_EQ(task_configs[1].pos_actions, pos_actions1);
  EXPECT_EQ(task_configs[2].pos_actions, pos_actions2);

  EXPECT_EQ(task_configs[0].neg_actions, neg_actions0);
  EXPECT_EQ(task_configs[1].neg_actions, neg_actions1);
  EXPECT_EQ(task_configs[2].neg_actions, neg_actions2);

  EXPECT_EQ(task_configs[0].sample_rate, 0.5);
  EXPECT_EQ(task_configs[1].sample_rate, 1.0);
  EXPECT_EQ(task_configs[2].sample_rate, 1.0);
}

}  // namespace
}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
