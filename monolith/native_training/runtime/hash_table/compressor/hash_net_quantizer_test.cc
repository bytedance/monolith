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

#include "monolith/native_training/runtime/hash_table/compressor/hash_net_quantizer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace monolith {
namespace hash_table {
namespace {

TEST(HashNetQuantizer, ForwardAndBackward) {
  FloatCompressorConfig_OneBit config;
  config.set_step_size(1000);
  HashNetQuantizer model(config);

  EXPECT_FLOAT_EQ(model.GetScale(), 1.0f);
  EXPECT_FLOAT_EQ(model.Forward(1.0f), 0.07615941f);
  EXPECT_FLOAT_EQ(model.Forward(2.0f), 0.09640275f);

  float grad = 1.0f, grad2 = 2.0f;
  model.Backward(2.0f, &grad, 0);
  model.Backward(2.0f, &grad2, 999);
  EXPECT_FLOAT_EQ(grad, 0.00706508f);
  EXPECT_FLOAT_EQ(grad2, 0.01413016f);

  grad = 100.0f, grad2 = 200.0f;
  model.Backward(2.0f, &grad, 1000);
  model.Backward(2.0f, &grad2, 1001);
  EXPECT_FLOAT_EQ(grad, 0.005442613f);
  EXPECT_FLOAT_EQ(grad2, 0.01088523f);
  EXPECT_FLOAT_EQ(model.Forward(2.0f), 0.09998888f);
  EXPECT_FLOAT_EQ(model.GetScale(), 2.44948974f);
}

}  // namespace
}  // namespace hash_table
}  // namespace monolith
