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

#ifndef MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_MULTI_HASH_TABLE_H_
#define MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_MULTI_HASH_TABLE_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "monolith/native_training/runtime/ops/embedding_hash_table_tf_bridge.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
namespace monolith_tf {

class MultiHashTable : public ResourceBase {
 public:
  explicit MultiHashTable(absl::string_view shared_name)
      : shared_name_(std::string(shared_name)) {}

  void add_table(absl::string_view name,
                 core::RefCountPtr<EmbeddingHashTableTfBridge> table) {
    names_.push_back(std::string(name));
    tables_.push_back(std::move(table));
  }

  EmbeddingHashTableTfBridge* table(int i) const { return tables_[i].get(); }

  const std::vector<std::string>& names() const { return names_; }

  const std::string& name(int i) { return names_[i]; }

  int size() const { return names_.size(); }

  const std::string& shared_name() const { return shared_name_; }

  std::string DebugString() const override {
    std::string ret;
    for (int i = 0; i < size(); ++i) {
      ret += absl::StrCat("name: ", names_[i], ":", tables_[i]->DebugString(),
                          ";");
    }
    return ret;
  }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    for (const auto& table : tables_) {
      ret += table->MemoryUsed();
    }
    return ret;
  }

 private:
  std::string shared_name_;
  std::vector<core::RefCountPtr<EmbeddingHashTableTfBridge>> tables_;
  std::vector<std::string> names_;
};

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // MONOLITH_NATIVE_TRAINING_RUNTIME_OPS_MULTI_HASH_TABLE_H_
