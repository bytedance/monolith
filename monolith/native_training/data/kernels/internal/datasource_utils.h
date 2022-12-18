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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_DATASOURCE_UTILS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_DATASOURCE_UTILS_H_

#include <string>

namespace tensorflow {
namespace monolith_tf {
namespace internal {

int32_t java_hash_code(const std::string &data_flow_name);

}  // namespace internal
}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_DATASOURCE_UTILS_H_
