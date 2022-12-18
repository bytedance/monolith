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

#include "monolith/native_training/runtime/parameter_sync/request_splitter.h"

#include <numeric>

#include "glog/logging.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"

namespace monolith {
namespace parameter_sync {
namespace {

using ::google::protobuf::util::MessageDifferencer;

PushRequest_DeltaEmbeddingHashTable SetUpOneDeltaHashTable(
    const std::string& unique_id, size_t fid_num, size_t dim,
    int fid_value_offset = 0) {
  PushRequest_DeltaEmbeddingHashTable table;
  table.set_unique_id(unique_id);
  table.set_dim_size(dim);

  std::vector<int64_t> fids(fid_num);
  std::iota(fids.begin(), fids.end(), fid_value_offset);
  std::vector<float> embeddings(fid_num * dim);
  for (size_t i = 0; i < fid_num; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      embeddings[i * dim + j] = static_cast<float>(i + fid_value_offset);
    }
  }

  table.mutable_fids()->Add(fids.begin(), fids.end());
  table.mutable_embeddings()->Add(embeddings.begin(), embeddings.end());
  return table;
}

TEST(RequestSplitter, NoSplit) {
  PushRequest request;
  request.set_model_name("hello");
  request.set_signature_name("hashtable_assign");
  auto t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  auto t1 = SetUpOneDeltaHashTable("table1", 2, 1);
  auto t2 = SetUpOneDeltaHashTable("table2", 3, 2);
  request.mutable_delta_hash_tables()->Add(std::move(t0));
  request.mutable_delta_hash_tables()->Add(std::move(t1));
  request.mutable_delta_hash_tables()->Add(std::move(t2));

  RequestSplitter splitter;
  std::vector<PushRequest> requests = splitter.Split(request, 4 * 1024 * 1024);
  EXPECT_EQ(requests.size(), 1);
  EXPECT_TRUE(MessageDifferencer::Equals(request, requests.front()));
}

// Test case(byte size = 111)
TEST(RequestSplitter, SplitIntoTwo) {
  PushRequest request, request1, request2;
  request.set_model_name("hello");
  request.set_signature_name("hashtable_assign");
  auto t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  auto t1 = SetUpOneDeltaHashTable("table1", 2, 1);
  auto t2 = SetUpOneDeltaHashTable("table2", 3, 2);
  request.mutable_delta_hash_tables()->Add(std::move(t0));
  request.mutable_delta_hash_tables()->Add(std::move(t1));
  request.mutable_delta_hash_tables()->Add(std::move(t2));
  request.set_timeout_in_ms(100);

  RequestSplitter splitter;
  std::vector<PushRequest> requests = splitter.Split(request, 60);
  EXPECT_EQ(requests.size(), 2);

  // part 1
  request1.set_model_name("hello");
  request1.set_signature_name("hashtable_assign");
  t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  t1 = SetUpOneDeltaHashTable("table1", 1, 1);
  t2 = SetUpOneDeltaHashTable("table2", 1, 2);
  request1.mutable_delta_hash_tables()->Add(std::move(t0));
  request1.mutable_delta_hash_tables()->Add(std::move(t1));
  request1.mutable_delta_hash_tables()->Add(std::move(t2));
  request1.set_timeout_in_ms(100);
  EXPECT_TRUE(MessageDifferencer::Equals(requests.front(), request1));

  // part 2
  request2.set_model_name("hello");
  request2.set_signature_name("hashtable_assign");
  t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  t1 = SetUpOneDeltaHashTable("table1", 1, 1, 1.0f);
  t2 = SetUpOneDeltaHashTable("table2", 2, 2, 1.0f);
  request2.mutable_delta_hash_tables()->Add(std::move(t0));
  request2.mutable_delta_hash_tables()->Add(std::move(t1));
  request2.mutable_delta_hash_tables()->Add(std::move(t2));
  request2.set_timeout_in_ms(100);
  EXPECT_TRUE(MessageDifferencer::Equals(requests.back(), request2));
}

// Test case(byte size = 113)
TEST(RequestSplitter, MultiHashTableSplitIntoTwo) {
  PushRequest request, request1, request2;
  request.set_model_name("hello");
  request.set_signature_name("table/raw_assign");
  auto t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  auto t1 = SetUpOneDeltaHashTable("table1", 2, 1);
  auto t2 = SetUpOneDeltaHashTable("table2", 3, 2);
  request.mutable_delta_multi_hash_tables()->Add(std::move(t0));
  request.mutable_delta_multi_hash_tables()->Add(std::move(t1));
  request.mutable_delta_multi_hash_tables()->Add(std::move(t2));
  request.set_timeout_in_ms(100);

  RequestSplitter splitter;
  std::vector<PushRequest> requests = splitter.Split(request, 60);
  EXPECT_EQ(requests.size(), 2);

  // part 1
  request1.set_model_name("hello");
  request1.set_signature_name("table/raw_assign");
  t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  t1 = SetUpOneDeltaHashTable("table1", 1, 1);
  t2 = SetUpOneDeltaHashTable("table2", 1, 2);
  request1.mutable_delta_multi_hash_tables()->Add(std::move(t0));
  request1.mutable_delta_multi_hash_tables()->Add(std::move(t1));
  request1.mutable_delta_multi_hash_tables()->Add(std::move(t2));
  request1.set_timeout_in_ms(100);
  EXPECT_TRUE(MessageDifferencer::Equals(requests[0], request1));

  // part 2
  request2.set_model_name("hello");
  request2.set_signature_name("table/raw_assign");
  t0 = SetUpOneDeltaHashTable("table0", 0, 1);
  t1 = SetUpOneDeltaHashTable("table1", 1, 1, 1.0f);
  t2 = SetUpOneDeltaHashTable("table2", 2, 2, 1.0f);
  request2.mutable_delta_multi_hash_tables()->Add(std::move(t0));
  request2.mutable_delta_multi_hash_tables()->Add(std::move(t1));
  request2.mutable_delta_multi_hash_tables()->Add(std::move(t2));
  request2.set_timeout_in_ms(100);
  EXPECT_TRUE(MessageDifferencer::Equals(requests[1], request2));
}

}  // namespace
}  // namespace parameter_sync
}  // namespace monolith
