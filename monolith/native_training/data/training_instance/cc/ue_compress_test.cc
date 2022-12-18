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

#include "ue_compress.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace monolith_tf {

using ::testing::Pointwise;
using ::testing::FloatNear;

TEST(UECompressTest, Basic) {
  std::shared_ptr<UECompress> ue_compress_ = std::make_shared<UECompress>();
  std::vector<float> float_values;
  float_values.push_back(1.1);
  float_values.push_back(0.1);
  float_values.push_back(3.1);
  float_values.push_back(5.1);
  float_values.push_back(2.2);
  float_values.push_back(3.3);
  float_values.push_back(4.3);
  ::idl::matrix::proto::Feature feature;
  feature.set_name("fc_test");
  feature.mutable_float_value()->Reserve(float_values.size());
  for (auto& v : float_values) {
    feature.add_float_value(v);
  }

  ue_compress_->compress_embeddings(&feature, UECompressMethod::COMPRESS_QTZ8);
  for (auto& values : feature.bytes_value()) {
    std::cout << "values " << values << std::endl;
  }

  feature.clear_float_value();
  std::vector<float> embedding;
  bool ret = ue_compress_->decompress_embeddings(
      feature, &embedding, UECompressMethod::COMPRESS_QTZ8);
  ASSERT_TRUE(ret);

  ASSERT_EQ(float_values.size(), embedding.size());
  for (int i = 0; i < embedding.size(); i++) {
    ASSERT_THAT(embedding, Pointwise(FloatNear(1e-2), float_values));
  }
}

}  // namespace monolith_tf
}  // namespace tensorflow
