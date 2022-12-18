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

#include "monolith/native_training/data/kernels/internal/file_match_split_provider.h"

#include <chrono>
#include <memory>
#include <thread>
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"

static constexpr char kCurrentPat[] = "current_pattern";
static constexpr char kCurrentFile[] = "current_file";
static constexpr char kQueueContent[] = "queue_content";
using std::chrono::milliseconds;

namespace tensorflow {
namespace data {
namespace monolith_tf {

Status FileMatchSplitProvider::GetNext(Tensor *split, bool *end_of_splits) {
  mutex_lock l(mu_);
  if (!feeder_) {
    TF_RETURN_IF_ERROR(EnsureFeederInitialized());
    LOG(INFO) << "EnsureFeederInitialized Done!";
  }

  *end_of_splits = false;
  *split = Tensor(DT_STRING, TensorShape{});
  if (canceled_) {
    *end_of_splits = true;
    return errors::Cancelled(
        "FileMatchSplitProvider canceled, get an end_of_splits!");
  }

  std::string item;
  while (!results_.try_pop(item, milliseconds(10))) {
    if (finished_feed_ && results_.empty()) {
      *end_of_splits = true;
      std::string info = absl::StrCat(
          "finished_feed is ", finished_feed_.load(), ", and results empty is ",
          results_.empty(), ", get an end_of_splits!");
      return errors::OutOfRange(info);
    }
  }

  split->scalar<tstring>()() = item;
  return Status::OK();
}

Status FileMatchSplitProvider::Reset() {
  mutex_lock l(mu_);
  // ensure feeder thread join
  canceled_ = true;
  finished_feed_ = true;
  feeder_ = nullptr;

  // clear queue
  std::string item;
  while (!results_.empty()) {
    results_.try_pop(item, milliseconds(1));
  }

  canceled_ = false;
  finished_feed_ = false;
  current_pat_ = "";
  current_file_ = "";
  TF_RETURN_IF_ERROR(EnsureFeederInitialized());
  return Status::OK();
}

Status FileMatchSplitProvider::Save(
    std::function<std::string(std::string)> key_name_fn,
    IteratorStateWriter *writer) {
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_name_fn(kCurrentPat), current_pat_));
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_name_fn(kCurrentFile), current_file_));

  std::vector<std::string> content;
  while (!results_.empty()) {
    std::string item;
    results_.pop(item);
    content.push_back(item);
  }
  TF_RETURN_IF_ERROR(
      writer->WriteScalar(key_name_fn(kQueueContent),
                          absl::StrJoin(content.begin(), content.end(), ",")));
  return Status::OK();
}

Status FileMatchSplitProvider::Restore(
    std::function<std::string(std::string)> key_name_fn,
    IteratorStateReader *reader) {
  canceled_ = false;
  finished_feed_ = false;
  tstring current_pat, current_file, content_str;
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(key_name_fn(kCurrentPat), &current_pat));
  current_pat_ = std::string(current_pat);
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(key_name_fn(kCurrentFile), &current_file));
  current_file_ = std::string(current_file);
  TF_RETURN_IF_ERROR(
      reader->ReadScalar(key_name_fn(kQueueContent), &content_str));
  std::vector<std::string> content_list =
      absl::StrSplit(absl::string_view(content_str), ',');
  for (const std::string &item : content_list) {
    results_.push(item);
  }
  return Status::OK();
}

Status FileMatchSplitProvider::EnsureFeederInitialized() {
  finished_feed_ = false;
  feeder_ = absl::WrapUnique(Env::Default()->StartThread(
      {}, "file-match-split-provider-feeder", [this]() { FeederThread(); }));
  return Status::OK();
}

void FileMatchSplitProvider::FeederThread() {
  LOG(INFO) << "thread file-match-split-provider-feeder started!";
  int max_retry = 5, current_try = 0;
  const auto timeout = milliseconds(10);
  Env *env = Env::Default();

  // find the start point
  int start = 0;
  if (!current_pat_.empty()) {
    for (const std::string &pattern : patterns_) {
      start++;
      if (pattern == current_pat_) {
        break;
      }
    }
    LOG(INFO) << "current_pat is " << current_pat_ << ", start at "
              << start - 1;
  }

  if (start >= patterns_.size() && patterns_.back() != current_pat_) {
    LOG(WARNING) << "Cannot find " << current_pat_ << " in patterns, skip!";
    current_pat_ = "";
    start = 0;
  }

  // finish the files in current_pat_ if any
  std::vector<std::string> matched_files;
  if (!current_pat_.empty()) {
    current_try = 0;
    while (!env->GetMatchingPaths(current_pat_, &matched_files).ok()) {
      if (canceled_) return;
      current_try++;
      matched_files.clear();
      std::this_thread::sleep_for(milliseconds(1000));
      if (current_try >= max_retry) {
        LOG(INFO) << "GetMatchingPaths for pattern " << current_pat_
                  << " fail, retry!";
        break;
      }
    }

    int idx = 0;
    if (!current_file_.empty()) {
      for (const std::string &file : matched_files) {
        if (file != current_file_) {
          idx++;
        } else {
          break;
        }
      }
      LOG(INFO) << "current_file is " << current_file_ << ", start at " << idx;
    }

    int num_files = 0;
    for (size_t i = idx; i < matched_files.size(); ++i) {
      current_file_ = matched_files[i];
      while (!results_.try_push(current_file_, timeout)) {
        if (canceled_) return;
        std::this_thread::sleep_for(timeout);
      }
      num_files++;
    }
    LOG(INFO) << "Pattern " << current_pat_ << " has matched " << num_files
              << "/" << matched_files.size() << " files";
  }

  // for the patterns after current_pat_
  for (size_t i = start; i < patterns_.size(); ++i) {
    if (canceled_) return;
    matched_files.clear();
    current_pat_ = patterns_[i];
    current_try = 0;
    while (!env->GetMatchingPaths(current_pat_, &matched_files).ok()) {
      if (canceled_) return;
      current_try++;
      matched_files.clear();
      std::this_thread::sleep_for(milliseconds(1000));
      if (current_try >= max_retry) {
        LOG(INFO) << "GetMatchingPaths for pattern " << current_pat_
                  << " fail, retry!";
      }
    }

    int num_files = 0;
    for (const std::string &file : matched_files) {
      current_file_ = file;
      while (!results_.try_push(current_file_, timeout)) {
        if (canceled_) return;
        std::this_thread::sleep_for(timeout);
      }
      num_files++;
    }
    LOG(INFO) << "Pattern " << current_pat_ << " has matched " << num_files
              << "/" << matched_files.size() << " files";
  }

  finished_feed_ = true;
  LOG(INFO) << "thread file-match-split-provider-feeder finished!";
}

}  // namespace monolith_tf
}  // namespace data
}  // namespace tensorflow
