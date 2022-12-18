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

#include <iostream>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "gflags/gflags.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "third_party/nlohmann/json.hpp"

DEFINE_bool(kafka_dump, false, "kafka_dump");
DEFINE_bool(kafka_dump_prefix, false, "kafka_dump_prefix");
DEFINE_bool(has_sort_id, true, "has_sort_id");
DEFINE_string(has_fids, "",
              "The instance of interest should contain at least one of the "
              "given fids, or it will be dropped");
DEFINE_string(has_actions, "",
              "The instance of interest should contain at least one of the "
              "given actions, or it will be dropped");
DEFINE_string(
    filter_fids, "",
    "The instance will be dropped if it contains any one of the given fids.");
DEFINE_string(
    select_fids, "",
    "The instance of interest should contain all of the given fids, or it "
    "will be dropped.");
DEFINE_int64(
    req_time_min, 0,
    "The instance of interest should satisfy line_id.req_time >= req_time_min");
DEFINE_int32(buffer_size, 32, "The buffer number of instance");

using tensorflow::Status;
using tensorflow::tstring;
using tensorflow::uint64;
using tensorflow::monolith_tf::IsInstanceOfInterest;
using ::tensorflow::monolith_tf::PBIterator;
using ::tensorflow::monolith_tf::DataFormatOptions;
using ::tensorflow::monolith_tf::StdinStreamReader;
using ::tensorflow::monolith_tf::StrToIntegerSet;

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto has_fids = StrToIntegerSet<uint64_t>(FLAGS_has_fids);
  auto filter_fids = StrToIntegerSet<uint64_t>(FLAGS_filter_fids);
  auto select_fids = StrToIntegerSet<uint64_t>(FLAGS_select_fids);
  auto has_actions = StrToIntegerSet<int32_t>(FLAGS_has_actions);
  absl::Time t = absl::FromUnixSeconds(FLAGS_req_time_min);

  nlohmann::json json;
  json["kafka_dump"] = FLAGS_kafka_dump;
  json["kafka_dump_prefix"] = FLAGS_kafka_dump_prefix;
  json["has_sort_id"] = FLAGS_has_sort_id;
  json["has_fids"] = has_fids;
  json["filter_fids"] = filter_fids;
  json["select_fids"] = select_fids;
  json["has_actions"] = has_actions;
  json["req_time_min"] = FLAGS_req_time_min;
  json["req_time_min_human_readable"] = absl::FormatTime(t);
  std::cerr << absl::StrFormat("%s Instance processor config:\n%s",
                               absl::FormatTime(absl::Now()), json.dump(2))
            << std::endl;

  DataFormatOptions options;
  options.kafka_dump = FLAGS_kafka_dump;
  options.kafka_dump_prefix = FLAGS_kafka_dump_prefix;
  options.has_sort_id = FLAGS_has_sort_id;
  PBIterator reader(std::make_unique<StdinStreamReader>(options),
                    tensorflow::monolith_tf::PRUNING_RAW_FEATURE);

  uint64 offset = 0, count = 0, total = 0;
  tstring sort_id, serialized_instance;
  std::stringstream ss;
  uint32_t data_source_key;
  while (reader.next(&offset, &data_source_key, &serialized_instance) ==
         Status::OK()) {
    offset = reader.GetOffset();
    parser::proto::Instance instance;
    instance.ParseFromArray(serialized_instance.data(),
                            serialized_instance.size());
    ++total;
    if (IsInstanceOfInterest(instance, filter_fids, has_fids, select_fids,
                             has_actions, FLAGS_req_time_min, {})) {
      std::string serialized_instance = instance.SerializeAsString();
      uint64_t size_of_sort_id = sort_id.length();
      uint64_t size_of_pb = serialized_instance.length();
      ss.write(reinterpret_cast<char*>(&size_of_sort_id),
               sizeof(size_of_sort_id));
      ss.write(sort_id.data(), sort_id.length());
      ss.write(reinterpret_cast<char*>(&size_of_pb), sizeof(size_of_pb));
      ss.write(const_cast<char*>(serialized_instance.data()),
               serialized_instance.length());
      ++count;

      if (count % FLAGS_buffer_size == 0) {
        std::string output = ss.str();
        ss.str("");
        std::cout.write(output.data(), output.length());
        std::cout.flush();
      }
    }

    if (total % 1000000 == 0) {
      std::cerr
          << absl::StrFormat(
                 "%s Instance processor input_num = %ld, output_num = %ld.",
                 absl::FormatTime(absl::Now()), total, count)
          << std::endl;
    }
  }

  if (count % FLAGS_buffer_size) {
    std::string output = ss.str();
    std::cout.write(output.data(), output.length());
    std::cout.flush();
  }

  std::cerr << absl::StrFormat(
                   "%s Instance processor input_num = %ld, output_num = %ld. "
                   "Successfully finished!",
                   absl::FormatTime(absl::Now()), total, count)
            << std::endl;

  return 0;
}
