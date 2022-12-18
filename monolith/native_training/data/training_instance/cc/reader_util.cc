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

#include "monolith/native_training/data/training_instance/cc/reader_util.h"

const size_t FALLBACK_RESERVE_VALUE = 0xfefefefe;

namespace tensorflow {
namespace monolith_tf {

void to_json(nlohmann::json& j, const FeatureNameMapperIdInfo& info) {
  j["id"] = info.id;
  j["sorted_id"] = info.sorted_id;
}

}  // namespace monolith_tf
}  // namespace tensorflow
