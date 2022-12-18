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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_COMMON_METRICS_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_COMMON_METRICS_H_
#include <string>
#include <vector>

namespace cpputil {
namespace metrics2 {

// This is a dummy implementation
// Will be replaced by a unified interface
class MetricCollector {
 public:
  typedef std::vector<std::pair<std::string, std::string>> TagkvList;

  MetricCollector() = default;
  virtual ~MetricCollector() = default;

  template <class T>
  int init(const T& conf) {
    return 0;
  }

  int define_tagk(const std::string& tagk) { return 0; }

  int define_tagkv(const std::string& tagk,
                   const std::vector<std::string>& tagv_list) {
    return 0;
  }

  int define_counter(const std::string& name) { return 0; }

  int define_counter(const std::string& name, const std::string& ) {
    return 0;
  }

  int define_rate_counter(const std::string& name) { return 0; }

  int define_rate_counter(const std::string& name,
                          const std::string& ) {
    return 0;
  }

  int define_meter(const std::string& name) { return 0; }

  int define_meter(const std::string& name, const std::string& ) {
    return 0;
  }

  int define_timer(const std::string& name) { return 0; }

  int define_timer(const std::string& name, const std::string& ) {
    return 0;
  }

  int define_store(const std::string& name) { return 0; }

  int define_store(const std::string& name, const std::string& ) {
    return 0;
  }

  int define_ts_store(const std::string& name) { return 0; }

  int define_ts_store(const std::string& name, const std::string& ) {
    return 0;
  }

  int emit_counter(const std::string& name, double value) const { return 0; }

  int emit_counter(const std::string& name, double value,
                   std::string tagkv) const {
    return 0;
  }

  int emit_counter(const std::string& name, double value,
                   const TagkvList& tagkv_list) const {
    return 0;
  }

  int emit_rate_counter(const std::string& name, double value) const {
    return 0;
  }

  int emit_rate_counter(const std::string& name, double value,
                        const std::string& tagkv) const {
    return 0;
  }

  int emit_rate_counter(const std::string& name, double value,
                        const TagkvList& tagkv_list) {
    return 0;
  }

  int emit_meter(const std::string& name, double value) const { return 0; }

  int emit_meter(const std::string& name, double value,
                 const std::string& tagkv) const {
    return 0;
  }

  int emit_meter(const std::string& name, double value,
                 const TagkvList& tagkv_list) {
    return 0;
  }

  int emit_timer(const std::string& name, double value) const { return 0; }

  int emit_timer(const std::string& name, double value,
                 std::string tagkv) const {
    return 0;
  }

  int emit_timer(const std::string& name, double value,
                 const TagkvList& tagkv_list) const {
    return 0;
  }

  int emit_store(const std::string& name, double value) const { return 0; }

  int emit_store(const std::string& name, double value,
                 std::string tagkv) const {
    return 0;
  }

  int emit_store(const std::string& name, double value,
                 const TagkvList& tagkv_list) const {
    return 0;
  }

  int emit_ts_store(const std::string& name, double value, time_t ts) const {
    return 0;
  }

  int emit_ts_store(const std::string& name, double value, time_t ts,
                    std::string tagkv) const {
    return 0;
  }

  int emit_ts_store(const std::string& name, double value, time_t ts,
                    const TagkvList& tagkv_list) const {
    return 0;
  }

  int reset_counter(const std::string& name) const { return 0; }

  int reset_counter(const std::string& name, std::string tagkv) const {
    return 0;
  }

  int reset_counter(const std::string& name,
                    const TagkvList& tagkv_list) const {
    return 0;
  }

  int reset_rate_counter(const std::string& name) const { return 0; }

  int reset_rate_counter(const std::string& name, const std::string& tagkv) {
    return 0;
  }

  int reset_rate_counter(const std::string& name, const TagkvList& tagkv_list) {
    return 0;
  }

  int reset_timer(const std::string& name) const { return 0; }

  int reset_timer(const std::string& name, std::string tagkv) const {
    return 0;
  }

  int reset_timer(const std::string& name, const TagkvList& tagkv_list) const {
    return 0;
  }

  int reset_store(const std::string& name) const { return 0; }

  int reset_store(const std::string& name, std::string tagkv) const {
    return 0;
  }

  int reset_store(const std::string& name, const TagkvList& tagkv_list) const {
    return 0;
  }

  int reset_ts_store(const std::string& name) const { return 0; }

  int reset_ts_store(const std::string& name, std::string tagkv) const {
    return 0;
  }

  int reset_ts_store(const std::string& name,
                     const TagkvList& tagkv_list) const {
    return 0;
  }

  // deprecated
  static int start_flush_thread() { return 1; }

  // deprecated
  static int start_listening_thread() { return 1; }

  static std::string make_tagkv(const TagkvList& tagkv_list) { return ""; }
};

}  // namespace metrics2
}  // namespace cpputil
namespace monolith {

cpputil::metrics2::MetricCollector *GetMetrics();

}  // namespace monolith

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_COMMON_METRICS_H_
