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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_INITIALIZER_CONSTANTS_INITIALIZER
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_INITIALIZER_CONSTANTS_INITIALIZER
#include <memory>

#include "monolith/native_training/runtime/hash_table/initializer/initializer_config.pb.h"
#include "monolith/native_training/runtime/hash_table/initializer/initializer_interface.h"

namespace monolith {
namespace hash_table {

std::unique_ptr<InitializerInterface> NewZerosInitializer(
    ZerosInitializerConfig config);

std::unique_ptr<InitializerInterface> NewZerosInitializer(int dim_size);

std::unique_ptr<InitializerInterface> NewOnesInitializer(
    OnesInitializerConfig config);

std::unique_ptr<InitializerInterface> NewConstantsInitializer(
    ConstantsInitializerConfig config);

std::unique_ptr<InitializerInterface> NewConstantsInitializer(int dim_size,
                                                              float constant);

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_INITIALIZER_CONSTANTS_INITIALIZER
