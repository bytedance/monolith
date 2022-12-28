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

#include "monolith/native_training/runtime/ops/file_metric_writer.h"

#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/path.h"

namespace monolith {
namespace deep_insight {

using tensorflow::Env;
using tensorflow::Status;
using tensorflow::io::RecordWriter;
using tensorflow::io::RecordWriterOptions;

FileMetricWriter::FileMetricWriter(std::string filename)
    : filename_(std::move(filename)),
      finished_(false),
      total_produce_(0),
      total_consume_(0),
      total_dump_(0) {
  LOG(INFO) << "deepinsight dump filename: " << filename_;

  if (!filename_.empty()) {
    Env* env = Env::Default();
    std::string dirname(tensorflow::io::Dirname(filename_));
    TF_CHECK_OK(env->RecursivelyCreateDir(dirname));
    TF_CHECK_OK(env->NewWritableFile(filename_, &fp_));

    queue_ = std::make_unique<monolith::concurrency::Queue<std::string>>(8192);
    thread_pool_ = std::make_unique<monolith::concurrency::ThreadPool>(1);
    thread_pool_->Schedule([this]() {
      RecordWriterOptions options;
      RecordWriter writer(fp_.get(), options);

      while (!finished_ || !queue_->empty()) {
        std::string msg;
        bool ok = queue_->try_pop(msg, std::chrono::milliseconds(10));
        if (ok) {
          ++total_consume_;
          LOG_EVERY_N_SEC(INFO, 300)
              << absl::StrFormat("Consume %ld records", total_consume_);

          Status s = writer.WriteRecord(msg);
          if (s.ok()) {
            ++total_dump_;
            LOG_EVERY_N_SEC(INFO, 300)
                << absl::StrFormat("Dump %ld records", total_dump_);
          } else {
            LOG(ERROR) << absl::StrFormat(
                "Failed to write record: %s, status=%s", msg,
                s.error_message());
          }
        } else {
          LOG_EVERY_N_SEC(INFO, 300) << "Failed to try pop, queue maybe empty!";
        }
      }

      LOG(INFO) << absl::StrFormat("Totally produce %ld records",
                                   total_produce_);
      LOG(INFO) << absl::StrFormat("Totally consume %ld records",
                                   total_consume_);
      LOG(INFO) << absl::StrFormat("Totally dump %ld records", total_dump_);
      TF_CHECK_OK(writer.Close());
      TF_CHECK_OK(fp_->Close());
    });
  }
}

FileMetricWriter::~FileMetricWriter() { finished_ = true; }

void FileMetricWriter::Write(const std::string& msg) {
  if (fp_) {
    while (true) {
      bool ok = queue_->try_push(msg, std::chrono::milliseconds(10));
      if (ok) {
        ++total_produce_;
        LOG_EVERY_N_SEC(INFO, 300)
            << absl::StrFormat("Produce %ld records", total_produce_);
        break;
      } else {
        LOG_EVERY_N_SEC(INFO, 60) << "Failed to try push, queue maybe full!";
      }
    }
  }
}

}  // namespace deep_insight
}  // namespace monolith
