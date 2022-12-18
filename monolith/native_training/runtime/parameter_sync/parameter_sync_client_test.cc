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

#include "monolith/native_training/runtime/parameter_sync/parameter_sync_client.h"

#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "grpcpp/grpcpp.h"
#include "gtest/gtest.h"

namespace monolith {
namespace parameter_sync {
namespace {

using ::tensorflow::serving::PredictRequest;

TEST(MultiHashTableTest, Basic) {
  PushRequest req;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(R"(
    model_name: "test_model"
    signature_name: "table/raw_assign"
    delta_multi_hash_tables: [
      {
        fids: [1, 2]
        embeddings: [1.0, 2.0, 3.0, 4.0]
      },
      {
        fids: [3]
        embeddings: [5.0]
      }
    ]
  )",
                                                            &req));
  auto result = ParameterSyncClient::Convert(req);
  PredictRequest expected_predict_req;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(R"(
    model_spec {
      name: "test_model"
      signature_name: "table/raw_assign"
    }
    inputs {
      key: "flat_value"
      value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 5
          }
        }
        float_val: [1.0, 2.0, 3.0, 4.0, 5.0]
      }
    }
    inputs {
      key: "id"
      value {
        dtype: DT_INT64
        tensor_shape {
          dim {
            size: 3
          }
        }
        int64_val: [1, 2, 3]
      }
    }
    inputs {
      key: "id_split"
      value {
        dtype: DT_INT64
        tensor_shape {
          dim {
            size: 3
          }
        }
        int64_val: [0, 2, 3]
      }
    }
  )",
                                                    &expected_predict_req));
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      result.req, expected_predict_req));
}

TEST(HashTableTest, Basic) {
  PushRequest req;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(R"(
    model_name: "test_model"
    signature_name: "hashtable_assign"
    delta_hash_tables: [
      {
        unique_id: "table1"
        dim_size: 2
        fids: [1, 2]
        embeddings: [1.0, 2.0, 3.0, 4.0]
      },
      {
        unique_id: "table2"
        dim_size: 1
        fids: [3]
        embeddings: [5.0]
      }
    ]
  )",
                                                            &req));
  auto result = ParameterSyncClient::Convert(req);
  PredictRequest expected_predict_req;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(R"(
    model_spec {
      name: "test_model"
      signature_name: "hashtable_assign"
    }
    inputs {
      key: "table1_id"
      value {
        dtype: DT_INT64
        tensor_shape {
          dim {
            size: 2
          }
        }
        int64_val: [1, 2]
      }
    }
    inputs {
      key: "table1_value"
      value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        float_val: [1.0, 2.0, 3.0, 4.0]
      }
    }
    inputs {
      key: "table2_id"
      value {
        dtype: DT_INT64
        tensor_shape {
          dim {
            size: 1
          }
        }
        int64_val: [3]
      }
    }
    inputs {
      key: "table2_value"
      value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 1
          }
        }
        float_val: [5.0]
      }
    }
  )",
                                                    &expected_predict_req));
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      result.req, expected_predict_req));
}

}  // namespace
}  // namespace parameter_sync
}  // namespace monolith