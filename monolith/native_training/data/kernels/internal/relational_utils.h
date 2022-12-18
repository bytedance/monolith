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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_RELATIONAL_UTILS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_RELATIONAL_UTILS_H_

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {

static const std::string GT = "gt";
static const std::string GE = "ge";
static const std::string EQ = "eq";
static const std::string LT = "lt";
static const std::string LE = "le";
static const std::string NEQ = "neq";
static const std::string BETWEEN = "between";
static const std::string IN = "in";
static const std::string NOT_IN = "not-in";

static const std::unordered_set<std::string> VALID_OPS = {
    GT, GE, EQ, LT, LE, NEQ, BETWEEN, IN, NOT_IN};

static const std::unordered_set<std::string> COMPARE_OPS = {GT, GE,  EQ,     LT,
                                                            LE, NEQ, BETWEEN};

template <typename T1, typename T2 = T1>
bool compare(const std::string& op, const T1& value,
             const std::vector<T2>& operands) {
  if (op == GT) {
    return value > operands[0];
  } else if (op == GE) {
    return value >= operands[0];
  } else if (op == EQ) {
    return value == operands[0];
  } else if (op == LT) {
    return value < operands[0];
  } else if (op == LE) {
    return value <= operands[0];
  } else if (op == NEQ) {
    return value != operands[0];
  } else if (op == BETWEEN) {
    return value >= operands[0] && value < operands[1];
  } else {
    LOG(FATAL) << "Invalid op: " << op;
    return false;
  }
}

template <typename T1, typename T2 = T1>
bool contains(const std::string& op, const T1& value,
              const std::unordered_set<T2>& operand_set) {
  if (op == IN) {
    return operand_set.count(value);
  } else if (op == NOT_IN) {
    return !operand_set.count(value);
  } else {
    LOG(FATAL) << "Invalid op: " << op;
    return false;
  }
}

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_RELATIONAL_UTILS_H_
