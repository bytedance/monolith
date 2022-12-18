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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_FILE_MATCH_SPLIT_PROVIDER_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_FILE_MATCH_SPLIT_PROVIDER_H_

#include <atomic>
#include "monolith/native_training/runtime/concurrency/queue.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {
namespace monolith_tf {

// SplitProvider which reads splits from a tf.data service dispatcher over RPC.
class FileMatchSplitProvider : public SplitProvider {
 public:
  explicit FileMatchSplitProvider(const std::vector<std::string>& patterns,
                                  int queue_size = 1024)
      : canceled_(false),
        finished_feed_(false),
        patterns_(patterns),
        results_(queue_size) {}

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  mutex mu_;
  std::atomic<bool> canceled_;
  std::atomic<bool> finished_feed_;

  std::string current_pat_ = "";
  std::string current_file_ = "";
  const std::vector<std::string> patterns_;
  ::monolith::concurrency::Queue<std::string> results_;
  std::unique_ptr<Thread> feeder_;

  Status EnsureFeederInitialized();
  void FeederThread();
};

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_KERNELS_INTERNAL_FILE_MATCH_SPLIT_PROVIDER_H_
