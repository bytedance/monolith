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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_EMBEDDING_HASH_TABLE_TEST_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_EMBEDDING_HASH_TABLE_TEST_H_

#include <memory>
#include <thread>
#include <tuple>

#include "absl/synchronization/mutex.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/hash_table/embedding_hash_table_factory.h"
#include "monolith/native_training/runtime/hash_table/entry_accessor.h"

namespace monolith {
namespace hash_table {

namespace proto2 = google::protobuf;
constexpr int64_t kSecondsPerDay = 24 * 60 * 60;

// The tests assume dim size == 1, sgd optimizer and zero initializer
// Please see the config in flat_embedding_hash_table_test.cc as a reference
class ReadWriteEmbeddingHashTableTest
    : public ::testing::TestWithParam<
          std::tuple<EmbeddingHashTableConfig, std::vector<float>>> {};

TEST_P(ReadWriteEmbeddingHashTableTest, SingleThread) {
  auto p = GetParam();
  EmbeddingHashTableConfig config = std::get<0>(p);
  const auto& learning_rates = std::get<1>(p);
  std::unique_ptr<EmbeddingHashTableInterface> table =
      NewEmbeddingHashTableFromConfig(config);
  std::vector<float> num_buffer(1);
  absl::Span<float> num = absl::MakeSpan(num_buffer);
  table->Lookup(5, num);
  EXPECT_THAT(num, ::testing::ElementsAre(0));
  table->AssignAdd(-10, {2.5}, 100LL);
  table->Lookup(-10, num);
  EXPECT_THAT(num, ::testing::ElementsAre(2.5));
  if (config.entry_config().entry_type() == EntryConfig::TRAINING) {
    table->Optimize(13, {1.0f}, learning_rates, 0, 0);
    table->Lookup(13, num);
    EXPECT_THAT(num, ::testing::ElementsAre(-0.01));
  }

  std::vector<int64_t> ids{-10, 13, 100};
  std::vector<float> update1 = {1.0f}, update2 = {2.0f}, update3 = {3.0f};
  std::vector<absl::Span<const float>> updates = {absl::MakeSpan(update1),
                                                  absl::MakeSpan(update2),
                                                  absl::MakeSpan(update3)};
  table->Assign(absl::MakeSpan(ids), absl::MakeSpan(updates), 0);

  std::vector<int64_t> lookup_ids{5, -10, 13, 100};
  std::vector<float> emb1 = {0}, emb2 = {0}, emb3 = {0}, emb4 = {0};
  std::vector<absl::Span<float>> embeddings = {
      absl::MakeSpan(emb1), absl::MakeSpan(emb2), absl::MakeSpan(emb3),
      absl::MakeSpan(emb4)};
  table->BatchLookup(absl::MakeSpan(lookup_ids), absl::MakeSpan(embeddings));
  EXPECT_THAT(embeddings[0], ::testing::ElementsAre(0));
  EXPECT_THAT(embeddings[1], ::testing::ElementsAre(1));
  EXPECT_THAT(embeddings[2], ::testing::ElementsAre(2));
  EXPECT_THAT(embeddings[3], ::testing::ElementsAre(3));

  std::vector<EntryDump> entries(4);
  table->BatchLookupEntry(absl::MakeSpan(lookup_ids), absl::MakeSpan(entries));
  EXPECT_EQ(entries[0].SerializeAsString(), "");
  EntryDump expect;
  proto2::TextFormat::ParseFromString(R"(
    num: 1
    opt {
      dump {
        sgd {
        }
      }
    }
    last_update_ts_sec: 0
  )",
                                      &expect);
  EXPECT_EQ(entries[1].SerializeAsString(), expect.SerializeAsString());
}

TEST_P(ReadWriteEmbeddingHashTableTest, MultiThread) {
  auto p = GetParam();
  EmbeddingHashTableConfig config = std::get<0>(p);
  const auto& learning_rates = std::get<1>(p);
  std::unique_ptr<EmbeddingHashTableInterface> table =
      NewEmbeddingHashTableFromConfig(config);

  auto func = [&table](int id) {
    table->AssignAdd(id, {static_cast<float>(id)}, 0);
  };
  auto func2 = [&table](int id, const std::vector<float>& learning_rates) {
    table->Optimize(id, {static_cast<float>(-200 * id)}, learning_rates, 0, 0);
  };
  const int kNumThread = 100;
  std::vector<std::unique_ptr<std::thread>> threads;
  for (int i = 0; i < kNumThread; ++i) {
    threads.emplace_back(std::make_unique<std::thread>(func, i));
    if (config.entry_config().entry_type() == EntryConfig::TRAINING) {
      threads.emplace_back(
          std::make_unique<std::thread>(func2, i, learning_rates));
    }
  }
  for (auto& thread : threads) {
    thread->join();
  }
  for (int i = 0; i < kNumThread; ++i) {
    std::vector<float> num(1);
    table->Lookup(i, absl::MakeSpan(num));
    if (config.entry_config().entry_type() == EntryConfig::TRAINING) {
      EXPECT_THAT(num, ::testing::ElementsAre(3 * i));
    } else {
      EXPECT_THAT(num, ::testing::ElementsAre(i));
    }
  }
}

TEST_P(ReadWriteEmbeddingHashTableTest, Clear) {
  auto p = GetParam();
  EmbeddingHashTableConfig config = std::get<0>(p);
  std::unique_ptr<EmbeddingHashTableHelper> table =
      std::make_unique<EmbeddingHashTableHelper>(
          NewEmbeddingHashTableFromConfig(config));
  table->Assign({1}, {{2.0f}});
  std::vector<float> emb(1);
  table->Lookup(1, absl::MakeSpan(emb));
  EXPECT_THAT(emb, testing::ElementsAre(2.0f));
  table->Clear();
  table->Lookup(1, absl::MakeSpan(emb));
  EXPECT_THAT(emb, testing::ElementsAre(0.0f));
}

class SaveRestoreEmbeddingHashTestTest
    : public ::testing::TestWithParam<
          std::tuple<EmbeddingHashTableConfig, std::vector<float>>> {};

TEST_P(SaveRestoreEmbeddingHashTestTest, SaveRestore) {
  auto p = GetParam();
  auto table = std::make_unique<EmbeddingHashTableHelper>(
      NewEmbeddingHashTableFromConfig(std::get<0>(p)));
  table->AssignAdd(5, {2.5}, 0);
  table->AssignAdd(-3, {-0.5}, 0);
  std::vector<EntryDump> dumps;
  auto write_fn = [&dumps](EntryDump dump) {
    dumps.push_back(dump);
    return true;
  };
  const EmbeddingHashTableInterface::DumpShard kSingleShard{0, 1};
  table->Save(kSingleShard, write_fn);

  std::unique_ptr<EmbeddingHashTableInterface> table2 =
      NewEmbeddingHashTableFromConfig(std::get<0>(p));
  int idx = 0;
  auto get_fn = [&idx, &dumps](EntryDump* dump, int64_t* max_update_ts) {
    if (idx == static_cast<int>(dumps.size())) return false;
    *dump = dumps[idx++];
    return true;
  };
  table2->Restore(kSingleShard, get_fn);
  std::vector<float> num(1);
  table2->Lookup(5, absl::MakeSpan(num));
  EXPECT_THAT(num, ::testing::ElementsAre(2.5));
  table2->Lookup(-3, absl::MakeSpan(num));
  EXPECT_THAT(num, ::testing::ElementsAre(-0.5));
}

TEST_P(SaveRestoreEmbeddingHashTestTest, SaveWithOffset) {
  auto p = GetParam();
  std::unique_ptr<EmbeddingHashTableInterface> table =
      NewEmbeddingHashTableFromConfig(std::get<0>(p));
  table->AssignAdd(5, {2.5}, 0);
  table->AssignAdd(-3, {-0.5}, 0);
  std::vector<EntryDump> dumps;
  auto write_fn = [&dumps](EntryDump dump) {
    dumps.push_back(dump);
    return true;
  };
  EmbeddingHashTableInterface::DumpShard shard{0, 1};
  shard.limit = 1;
  EmbeddingHashTableInterface::DumpIterator iter;
  table->Save(shard, write_fn, &iter);
  EXPECT_THAT(dumps.size(), 1);
  table->Save(shard, write_fn, &iter);
  std::vector<int64_t> ids;
  for (int i = 0; i < dumps.size(); ++i) {
    ids.push_back(dumps[i].id());
  }
  EXPECT_THAT(ids, testing::UnorderedElementsAre(5, -3));
}

TEST_P(SaveRestoreEmbeddingHashTestTest, SaveRestoreMultithreaded) {
  auto p = GetParam();
  auto table =
      EmbeddingHashTableHelper(NewEmbeddingHashTableFromConfig(std::get<0>(p)));
  const int kNumThreads = 10;
  const int kPerThreadIds = 2600;
  for (int64_t i = 0; i < kNumThreads * kPerThreadIds; ++i) {
    table.AssignOne(i - kNumThreads * kPerThreadIds / 2, {float(i)});
  }
  std::vector<EntryDump> dumps;
  absl::Mutex mu;
  auto write_fn = [&dumps, &mu](EntryDump dump) {
    absl::MutexLock l(&mu);
    dumps.push_back(dump);
    return true;
  };
  std::vector<std::unique_ptr<std::thread>> save_threads;
  for (int i = 0; i < kNumThreads; ++i) {
    auto save_func = [kNumThreads, &table, &write_fn](int i) {
      EmbeddingHashTableInterface::DumpShard shard{i, kNumThreads};
      table.Save(shard, write_fn);
      return true;
    };
    save_threads.push_back(std::make_unique<std::thread>(save_func, i));
  }
  for (int i = 0; i < kNumThreads; ++i) {
    save_threads[i]->join();
  }
  ASSERT_THAT(dumps.size(), kNumThreads * kPerThreadIds);

  std::unique_ptr<EmbeddingHashTableInterface> table2 =
      NewEmbeddingHashTableFromConfig(std::get<0>(p));

  std::vector<std::unique_ptr<std::thread>> restore_threads;
  for (int i = 0; i < kNumThreads; ++i) {
    auto restore_fn = [&table2, &dumps, kNumThreads, kPerThreadIds](int i) {
      int idx = i * kPerThreadIds;
      int end_idx = (i + 1) * kPerThreadIds;
      auto get_fn = [&dumps, &idx, &end_idx](EntryDump* dump, int64_t*) {
        if (idx == end_idx) return false;
        *dump = dumps[idx++];
        return true;
      };
      table2->Restore({i, kNumThreads}, get_fn);
    };
    restore_threads.push_back(std::make_unique<std::thread>(restore_fn, i));
  }
  for (int i = 0; i < kNumThreads; ++i) {
    restore_threads[i]->join();
  }
  for (int64_t i = 0; i < kNumThreads * kPerThreadIds; ++i) {
    std::vector<float> nums(1);
    table.Lookup(i - kNumThreads * kPerThreadIds / 2, absl::MakeSpan(nums));
    ASSERT_THAT(nums[0], i);
  }
}

TEST_P(SaveRestoreEmbeddingHashTestTest, SaveWithStopEarly) {
  auto p = GetParam();
  auto table =
      EmbeddingHashTableHelper(NewEmbeddingHashTableFromConfig(std::get<0>(p)));
  table.Assign({0, 1}, {{0.0}, {0.0}});
  int called = 0;
  auto write_fn = [&called](EntryDump dump) {
    ++called;
    return false;
  };
  EmbeddingHashTableInterface::DumpShard shard{0, 1};
  table.Save(shard, write_fn);
  // Should stop early
  EXPECT_THAT(called, 1);
}

class EmbeddingHashTableEvictTest
    : public ::testing::TestWithParam<
          std::tuple<EmbeddingHashTableConfig, std::vector<float>>> {};

TEST_P(EmbeddingHashTableEvictTest, OneTimeEvict) {
  auto p = GetParam();
  auto embedding_hash_table_config = std::get<0>(p);
  auto* slot_expire_time_config =
      embedding_hash_table_config.mutable_slot_expire_time_config();
  slot_expire_time_config->set_default_expire_time(14);
  const std::vector<int64_t> slot_to_expire_time = {0, 5, 6};
  for (int i = 0; i < slot_to_expire_time.size(); ++i) {
    auto* expire_time = slot_expire_time_config->add_slot_expire_times();
    expire_time->set_slot(i);
    expire_time->set_expire_time(slot_to_expire_time[i]);
  }
  auto table = NewEmbeddingHashTableFromConfig(embedding_hash_table_config);

  const int64_t kFidUpdateTime = 1234;
  const int64_t kSlot1Fid = ((1LL << 48) | (123));
  const int64_t kSlot2Fid = ((2LL << 48) | (234));
  const int64_t kSlot3Fid = ((3LL << 48) | (456));

  table->Assign({kSlot1Fid}, {{2.0f}}, kFidUpdateTime);
  table->Assign({kSlot2Fid}, {{5.0f}}, kFidUpdateTime);
  table->Assign({kSlot3Fid}, {{7.0f}}, kFidUpdateTime);
  std::vector<float> emb(1);
  table->Lookup(kSlot1Fid, absl::MakeSpan(emb));
  EXPECT_THAT(emb, testing::ElementsAre(2.0f));
  table->Lookup(kSlot2Fid, absl::MakeSpan(emb));
  EXPECT_THAT(emb, testing::ElementsAre(5.0f));
  table->Lookup(kSlot3Fid, absl::MakeSpan(emb));
  EXPECT_THAT(emb, testing::ElementsAre(7.0f));

  const int64_t current_time = kFidUpdateTime + 5 * kSecondsPerDay + 60;
  table->Evict(current_time);
  table->Lookup(kSlot1Fid, absl::MakeSpan(emb));
  // Slot 1 expire time is 5 days, the time gap is 5 days + 60 seconds, so
  // should be evited.
  EXPECT_THAT(emb, testing::ElementsAre(0.0f));
  table->Lookup(kSlot2Fid, absl::MakeSpan(emb));
  // Slot 1 expire time is 6 days, the time gap is 5 days + 60 seconds, so
  // should NOT be evited.
  EXPECT_THAT(emb, testing::ElementsAre(5.0f));
  table->Lookup(kSlot3Fid, absl::MakeSpan(emb));
  // Slot 3 expire time should use default 14 days, the time gap is 5 days
  // + 60 seconds, so should NOT be evited.
  EXPECT_THAT(emb, testing::ElementsAre(7.0f));
}

// Testing evict would work during the hash table rehashing.
TEST_P(EmbeddingHashTableEvictTest, EvictWhileRehash) {
  auto p = GetParam();
  auto embedding_hash_table_config = std::get<0>(p);
  // We keep the initial capacity very small so that inserting will trigger
  // rehash.
  const int64_t kInitialHashTableCapacity = 1;
  const int64_t kNumInsertThreads = 20;
  const int64_t kIdPerThread = 50;
  const float kDefaultValue = 123.0f;
  embedding_hash_table_config.set_initial_capacity(kInitialHashTableCapacity);
  auto* slot_expire_time_config =
      embedding_hash_table_config.mutable_slot_expire_time_config();
  slot_expire_time_config->set_default_expire_time(0);
  for (int i = 0; i < kNumInsertThreads; ++i) {
    auto* expire_time = slot_expire_time_config->add_slot_expire_times();
    expire_time->set_slot(i);
    expire_time->set_expire_time(i);
  }
  auto table = NewEmbeddingHashTableFromConfig(embedding_hash_table_config);

  const int64_t kFidUpdateTime = 1234;
  std::vector<std::unique_ptr<std::thread>> insert_threads;
  for (int i = 0; i < kNumInsertThreads; ++i) {
    auto insert_func = [&table, kDefaultValue](int i) {
      for (int id = 0; id < kIdPerThread; ++id) {
        const int64_t slot_id = i;
        const int64_t fid = ((slot_id << 48) | id);
        table->Assign({fid}, {{kDefaultValue}}, kFidUpdateTime + id);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    };
    insert_threads.push_back(std::make_unique<std::thread>(insert_func, i));
  }
  const int64_t kCurrentTime =
      kFidUpdateTime + 5 * kSecondsPerDay + kSecondsPerDay / 2;
  // Run Evict every 2 seconds 3 times.
  std::unique_ptr<std::thread> evict_thread;
  auto evict_func = [&table, kCurrentTime]() {
    for (int i = 0; i < 3; ++i) {
      table->Evict(kCurrentTime);
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  };
  evict_thread = std::make_unique<std::thread>(evict_func);
  for (int i = 0; i < kNumInsertThreads; ++i) {
    insert_threads[i]->join();
  }
  evict_thread->join();
  // Have a final evict to make sure the test is not flaky.
  table->Evict(kCurrentTime);

  for (int i = 0; i < kNumInsertThreads; ++i) {
    std::vector<float> num(1);
    for (int id = 0; id < kIdPerThread; ++id) {
      const int64_t slot_id = i;
      const int64_t fid = ((slot_id << 48) | id);
      table->Lookup(fid, absl::MakeSpan(num));
      if (i <= 5) {
        EXPECT_THAT(num, ::testing::ElementsAre(0));
      } else {
        EXPECT_THAT(num, ::testing::ElementsAre(kDefaultValue));
      }
    }
  }
}

}  // namespace hash_table
}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_EMBEDDING_HASH_TABLE_TEST_H_
