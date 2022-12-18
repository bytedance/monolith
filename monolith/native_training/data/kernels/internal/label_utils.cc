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

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {
using LabelConf = ::monolith::native_training::data::config::LabelConf;

bool HasIntersection(const std::set<int32_t> &lhs,
                     const std::set<int32_t> &rhs) {
  std::set<uint64_t> intersection;
  std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::inserter(intersection, intersection.begin()));
  return !intersection.empty();
}

bool ParseTaskConfig(const std::string &config,
                     std::vector<TaskConfig> *task_configs) {
  task_configs->clear();
  LabelConf label_conf;
  if (!label_conf.ParseFromString(config)) {
    LOG(FATAL) << "Parse label config error: " << config;
  }

  CHECK_GT(label_conf.conf_size(), 0);
  task_configs->reserve(label_conf.conf_size());

  for (const auto &t : label_conf.conf()) {
    // pos_actions : neg_actions : sample_rate

    std::set<int32_t> pos_actions, neg_actions;
    CHECK(!t.pos_actions().empty());
    pos_actions.insert(t.pos_actions().begin(), t.pos_actions().end());

    if (!t.neg_actions().empty()) {
      neg_actions.insert(t.neg_actions().begin(), t.neg_actions().end());
    }

    CHECK(!HasIntersection(pos_actions, neg_actions));

    float sample_rate = t.sample_rate();
    CHECK_GE(sample_rate, 0);
    CHECK_LE(sample_rate, 1.0);

    task_configs->push_back({pos_actions, neg_actions, sample_rate});
  }

  return true;
}

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow
