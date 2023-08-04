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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_LINE_ID_VALUE_FILTER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_LINE_ID_VALUE_FILTER_H_

#include <set>
#include "absl/synchronization/mutex.h"
#include "idl/matrix/proto/line_id.pb.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace monolith_tf {
namespace internal {

class LineIdValueFilter {
 public:
  LineIdValueFilter(std::string field_name, std::string op,
                    std::vector<float> float_operand,
                    std::vector<int64> int_operand,
                    std::vector<std::string> string_operand,
                    std::string operand_filepath, bool keep_empty);

  bool IsInstanceOfInterest(tensorflow::Env* env,
                            const ::idl::matrix::proto::LineId& line_id);

 private:
  Status EnsureLoadFilterValues(tensorflow::Env* env);

  bool cmp(const std::vector<int64>& values) {
    std::set<int64> intersection;
    std::set_intersection(values.begin(), values.end(), int_operand_.begin(),
                          int_operand_.end(),
                          std::inserter(intersection, intersection.begin()));
    if (op_ == "any") {
      return intersection.size() > 0;
    } else if (op_ == "all") {
      return intersection.size() == int_operand_.size();
    } else if (op_ == "diff") {
      return intersection.size() == 0;
    } else {
      LOG(FATAL) << "Invalid op: " << op_;
      return false;
    }
  }

 private:
  mutable absl::Mutex mu_;
  bool load_filter_values_finished_ ABSL_GUARDED_BY(mu_) = false;
  const google::protobuf::FieldDescriptor* field_;
  std::string field_name_;
  std::string op_;  // gt, ge, eq, lt, le, neq, between
  bool keep_empty_ = false;
  std::string operand_filepath_;

  std::vector<float> float_operand_;
  std::vector<int64> int_operand_;
  std::vector<uint64> uint_operand_;
  std::vector<std::string> string_operand_;

  std::unordered_set<float> float_operand_set_;
  std::unordered_set<int64> int_operand_set_;
  std::unordered_set<uint64> uint_operand_set_;
  std::unordered_set<std::string> string_operand_set_;
};

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_LINE_ID_VALUE_FILTER_H_
