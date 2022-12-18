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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_LABEL_UTILS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_LABEL_UTILS_H_

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "monolith/native_training/data/data_op_config.pb.h"
#include "third_party/nlohmann/json.hpp"

namespace tensorflow {
namespace monolith_tf {
namespace internal {

constexpr float INVALID_LABEL = std::numeric_limits<float>::lowest();
constexpr float POSITIVE_LABEL = 1.0;

struct TaskConfig {
  std::set<int32_t> pos_actions;
  std::set<int32_t> neg_actions;
  float sample_rate;

  std::string ToString() const {
    nlohmann::json j;
    j["pos_actions"] = pos_actions;
    j["neg_actions"] = neg_actions;
    j["sample_rate"] = sample_rate;
    return j.dump(2);
  }
};

bool HasIntersection(const std::set<int32_t> &lhs,
                     const std::set<int32_t> &rhs);

bool ParseTaskConfig(const std::string &config,
                     std::vector<TaskConfig> *task_configs);

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
IsAlmostEqual(T x, T y, int ulp = 2) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
         // unless the result is subnormal
         || std::abs(x - y) < std::numeric_limits<T>::min();
}

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_ADD_LABEL_UTILS_H_
