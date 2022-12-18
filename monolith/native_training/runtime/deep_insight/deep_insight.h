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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_DEEP_INSIGHT
#define MONOLITH_NATIVE_TRAINING_RUNTIME_DEEP_INSIGHT
#include <ctime>
#include <iostream>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "third_party/nlohmann/json.hpp"

namespace monolith {
namespace deep_insight {

class ExtraField {
 public:
  explicit ExtraField(const std::string& k) : key_(k) {}
  virtual void add_to(nlohmann::json* j) = 0;
  const std::string& key() { return key_; }

 private:
  std::string key_;
};

class FloatExtraField : public ExtraField {
 public:
  explicit FloatExtraField(const std::string& k, const float& v)
      : ExtraField(k), value_(v) {}
  void add_to(nlohmann::json* j) { (*j)["extra_float"][key()] = value_; }

 private:
  float value_;
};

class Int64ExtraField : public ExtraField {
 public:
  explicit Int64ExtraField(const std::string& k, const int64_t& v)
      : ExtraField(k), value_(v) {}
  void add_to(nlohmann::json* j) { (*j)["extra_int"][key()] = value_; }

 private:
  int64_t value_;
};

class StringExtraField : public ExtraField {
 public:
  explicit StringExtraField(const std::string& k, const std::string& v)
      : ExtraField(k), value_(v) {}
  void add_to(nlohmann::json* j) { (*j)["extra_str"][key()] = value_; }

 private:
  std::string value_;
};
class DeepInsight {
 public:
  template <typename... Args>
  explicit DeepInsight(Args...) {}

  template <typename... Args>
  std::string SendV2(Args...) {
    return "";
  }

  template <typename... Args>
  bool HitSampleRatio(Args...) { return false; }

  int64_t GenerateTrainingTime() { return 0; }

  uint64_t GetTotalSendCounter() { return 0; }
};
}  // namespace deep_insight
}  // namespace monolith
#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_DEEP_INSIGHT
