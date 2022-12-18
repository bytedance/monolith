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

#include "monolith/native_training/runtime/parameter_sync/sync_client_manager.h"

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "glog/logging.h"
#include "tensorflow/core/platform/default/logging.h"

#include "monolith/native_training/runtime/common/metrics.h"

namespace monolith {
namespace parameter_sync {

// 4M
const int MAX_MESSAGE_LENGTH = 4 * 1024 * 1024;

SyncClientManager::SyncClientManager(
    std::function<std::unique_ptr<SyncClientInterface>(const std::string&)>
        client_factory)
    : client_factory_(std::move(client_factory)) {}

std::string SyncClientManager::PushRequestDebugString(
    const PushRequest& request, int index, int total) const {
  std::vector<std::string> delta_hash_table_info;
  delta_hash_table_info.reserve(request.delta_hash_tables_size());
  std::string prefix = "MonolithHashTable_";
  for (const auto& table : request.delta_hash_tables()) {
    std::string simple_id = table.unique_id();
    if (absl::StartsWith(simple_id, prefix)) {
      simple_id = simple_id.substr(prefix.length());
    }
    delta_hash_table_info.push_back(absl::StrFormat(
        "(unique_id: %s, fid_num: %d)", simple_id, table.fids().size()));
  }

  for (const auto& table : request.delta_multi_hash_tables()) {
    std::string simple_id = table.unique_id();
    delta_hash_table_info.push_back(absl::StrFormat(
        "(unique_id: %s, fid_num: %d)", simple_id, table.fids().size()));
  }

  return absl::StrFormat(
      "PushRequest[%d/%d]: model_name = %s, signature_name = %s, "
      "delta_hash_table = [%s]",
      index, total, request.model_name(), request.signature_name(),
      absl::StrJoin(delta_hash_table_info, ", "));
}

PushResult SyncClientManager::Push(const PushRequest& request,
                                   const std::string& model_name,
                                   const std::string& signature_name) const {
  LOG_EVERY_N_SEC(INFO, 60) << PushRequestDebugString(request, -1, -1);
  std::vector<PushRequest> requests =
      request_splitter_.Split(request, MAX_MESSAGE_LENGTH);
  int total = static_cast<int>(requests.size());
  std::vector<std::string> debug_string(total);
  auto split_log = [&]() {
    for (int i = 0; i < total; ++i) {
      debug_string[i] = PushRequestDebugString(requests[i], i, total);
    }
    return absl::StrJoin(debug_string, "\n");
  };
  LOG_EVERY_N_SEC(INFO, 60) << split_log();

  std::vector<int64_t> request_fid_count;
  request_fid_count.reserve(requests.size());
  std::transform(requests.begin(), requests.end(),
                 std::back_inserter(request_fid_count),
                 [](const PushRequest& req) {
                   int64_t count = 0;
                   for (const auto& t : req.delta_hash_tables()) {
                     count += t.fids_size();
                   }
                   for (const auto& t : req.delta_multi_hash_tables()) {
                     count += t.fids_size();
                   }
                   return count;
                 });

  auto MakeTagKV = [&](const std::string& target, const std::string& status) {
    return absl::StrFormat(
        "model_name=%s|signature_name=%s|target=%s|status=%s", model_name,
        signature_name, target, status);
  };

  PushResult result;
  {
    absl::ReaderMutexLock l(&mu_);
    for (const auto& kv : clients_) {
      std::unordered_map<std::string, int64_t> fid_count = {{"OK", 0},
                                                            {"KO", 0}};
      std::unordered_map<std::string, int64_t> byte_size = {{"OK", 0},
                                                            {"KO", 0}};
      for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];
        auto response = result.add_responses();
        if (!req.delta_hash_tables().empty() ||
            !req.delta_multi_hash_tables().empty()) {
          int64_t start = absl::ToUnixMicros(absl::Now());
          auto status = kv.second->Push(req, response);
          int64_t end = absl::ToUnixMicros(absl::Now());
          std::string status_key = status.ok() ? "OK" : "KO";
          std::string tag_kv = MakeTagKV(kv.first, status_key);
          monolith::GetMetrics()->emit_timer("parameter_sync_latency",
                                             end - start, tag_kv);
          fid_count[status_key] += request_fid_count[i];
          byte_size[status_key] += static_cast<int64_t>(req.ByteSizeLong());
        }
        response->set_target(kv.first);
      }

      for (const auto& p : fid_count) {
        if (p.second) {
          std::string tag_kv = MakeTagKV(kv.first, p.first);
          monolith::GetMetrics()->emit_counter("parameter_sync_fid_count",
                                               p.second, tag_kv);
        }
      }
      for (const auto& p : byte_size) {
        if (p.second) {
          std::string tag_kv = MakeTagKV(kv.first, p.first);
          monolith::GetMetrics()->emit_counter("parameter_sync_byte_size",
                                               p.second, tag_kv);
        }
      }
    }
  }

  return result;
}

bool SyncClientManager::TryReplace(
    const google::protobuf::RepeatedPtrField<std::string>& targets) {
  absl::WriterMutexLock l(&mu_);
  std::unordered_set<std::string> unique_targets;
  for (const auto& target : targets) {
    unique_targets.insert(target);
  }

  // Remove invalid targets
  std::vector<std::string> invalid_targets;
  for (const auto& kv : clients_) {
    if (!unique_targets.count(kv.first)) {
      invalid_targets.emplace_back(kv.first);
    }
  }
  for (const auto& target : invalid_targets) {
    clients_.erase(target);
  }

  // Add new targets
  for (const auto& target : unique_targets) {
    if (!clients_.count(target)) {
      auto client = client_factory_(target);
      clients_[target] = std::move(client);
    }
  }

  return true;
}

}  // namespace parameter_sync
}  // namespace monolith
