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

#include <stdlib.h>
#include <unistd.h>

#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "monolith/native_training/runtime/common/metrics.h"

namespace monolith {

TEST(MetricsTest, Default) {
  putenv(const_cast<char *>("TCE_PSM=data.tob.test"));
  static cpputil::metrics2::MetricCollector *metrics1 = monolith::GetMetrics();
  static cpputil::metrics2::MetricCollector *metrics2 = monolith::GetMetrics();
  EXPECT_EQ(metrics1, metrics2);
}

}  // namespace monolith
