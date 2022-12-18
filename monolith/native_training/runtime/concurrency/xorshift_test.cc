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

#include "monolith/native_training/runtime/concurrency/xorshift.h"
#include <random>
#include <thread>

#include "gtest/gtest.h"

namespace monolith {
namespace concurrency {
namespace {

const int64_t NUM = 1000000, RADIUS = std::numeric_limits<uint32_t>::max() / 2;
const int64_t RADIUS_SQUARE = RADIUS * RADIUS;
const float PI = 3.14f, eps = 0.01f;

float EstimatingPiWithMonteCarlo(const std::function<uint32_t()>& generator) {
  int count = 0;
  for (int i = 0; i < NUM; ++i) {
    auto x = generator() % RADIUS;
    auto y = generator() % RADIUS;
    if (x * x + y * y < RADIUS_SQUARE) {
      ++count;
    }
  }

  return static_cast<float>(count * 4) / NUM;
}

TEST(XorShift, SingleThread) {
  std::random_device random_device;
  std::mt19937 engine{random_device()};
  std::uniform_int_distribution<> dist(0, RADIUS);
  float pi1 = EstimatingPiWithMonteCarlo([&]() { return dist(engine); });
  EXPECT_NEAR(pi1, PI, eps);

  XorShift generator;
  float pi2 = EstimatingPiWithMonteCarlo([&]() { return generator.Rand32(); });
  EXPECT_NEAR(pi2, PI, eps);
}

TEST(XorShift, MultiThread) {
  std::random_device random_device;
  std::mt19937 engine{random_device()};
  std::uniform_int_distribution<> dist(0, RADIUS);
  float pi = EstimatingPiWithMonteCarlo([&]() { return dist(engine); });
  EXPECT_NEAR(pi, PI, eps);

  int thread_num = 10;
  std::vector<float> pi_array(thread_num);
  std::vector<std::thread> threads;
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back([&]() {
      float pi = EstimatingPiWithMonteCarlo(
          [&]() { return XorShift::Rand32ThreadSafe(); });
      EXPECT_NEAR(pi, PI, eps);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace
}  // namespace concurrency
}  // namespace monolith
