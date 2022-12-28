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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_DEEP_INSIGHT_FILE_METRIC_WRITER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_DEEP_INSIGHT_FILE_METRIC_WRITER_H_

#include <atomic>
#include <memory>
#include <string>

#include "tensorflow/core/platform/file_system.h"

#include "monolith/native_training/runtime/concurrency/queue.h"
#include "monolith/native_training/runtime/concurrency/thread_pool.h"

namespace monolith {
namespace deep_insight {

class FileMetricWriter {
 public:
  explicit FileMetricWriter(std::string filename);

  ~FileMetricWriter();

  void Write(const std::string& msg);

 private:
  std::string filename_;
  std::atomic_bool finished_;
  int64_t total_produce_;
  int64_t total_consume_;
  int64_t total_dump_;
  std::unique_ptr<tensorflow::WritableFile> fp_;
  std::unique_ptr<monolith::concurrency::Queue<std::string>> queue_;
  std::unique_ptr<monolith::concurrency::ThreadPool> thread_pool_;
};

}  // namespace deep_insight
}  // namespace monolith

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_DEEP_INSIGHT_FILE_METRIC_WRITER_H_
