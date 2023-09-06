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
#include <queue>
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/fid.h"
#include "monolith/native_training/data/transform/cc/transforms.h"
#include "monolith/native_training/data/transform/transform_config.pb.h"
#include "third_party/cli11/CLI11.hpp"
#include "third_party/nlohmann/json.hpp"

namespace tf = tensorflow;
using monolith::io::proto::Example;
using monolith::io::proto::ExampleBatch;
using monolith::native_training::data::TransformConfig;
using parser::proto::Instance;
using tf::monolith_tf::FeatureNameMapper;
using tf::monolith_tf::FeaturePruningType;
using tf::monolith_tf::FileStreamReader;
using tf::monolith_tf::InputCompressType;
using tf::monolith_tf::PBIterator;
using tf::monolith_tf::StdinStreamReader;
using tf::monolith_tf::TransformInterface;

struct Options {
  int verbose_level = 0;
  std::string filepath;
  std::string dtype = "instance";
  std::string compression_type = "none";
  std::string config;
  bool lagrangex_header = false;
  bool kafka_dump = false;
  bool kafka_dump_prefix = false;
  bool has_sort_id = true;
  int64_t limit = std::numeric_limits<int64_t>::max();
  std::string output_format = "json";
  bool silent = false;
};

void AddOptions(CLI::App& app, Options* options) {
  app.add_option("-v,--verbose", options->verbose_level,
                 "Verbose level, default: 0");
  app.add_option("-i,--input", options->filepath,
                 "Input filepath, read from stdin if empty!");
  app.add_option("-d,--dtype", options->dtype,
                 "Data type, default: instance, choices = [instance, example, "
                 "example_batch]")
      ->check([](std::string choice) {
        std::unordered_set<std::string> choices = {"instance", "example",
                                                   "example_batch"};
        if (!choices.count(choice)) {
          return absl::StrFormat("Invalid dtype: %s", choice);
        }
        return std::string();
      });
  app.add_option("-c,--compression_type", options->compression_type,
                 "Compression type, default: none, choices = [none, snappy]")
      ->check([](std::string choice) {
        std::unordered_set<std::string> choices = {"none", "snappy"};
        if (!choices.count(choice)) {
          return absl::StrFormat("Invalid compression_type: %s", choice);
        }
        return std::string();
      });
  app.add_option("--lagrangex_header", options->lagrangex_header,
                 "default: false");
  app.add_option("-k,--kafka_dump", options->kafka_dump, "default: false");
  app.add_option("--kafka_dump_prefix", options->kafka_dump_prefix,
                 "default: false");
  app.add_option("--has_sort_id", options->has_sort_id, "default: true");
  app.add_option("--config", options->config,
                 "Transform config, plain text, e.g. configs { basic_config { "
                 "filter_by_fid { select_fids: 18428264561369945341 } } }");
  app.add_option("-l,--limit", options->limit,
                 "Output limit number records, default: inf");
  app.add_option("-f,--format", options->output_format,
                 "Output format, default: json, choices = [json, pbtxt]")
      ->check([](std::string choice) {
        std::unordered_set<std::string> choices = {"json", "pbtxt"};
        if (!choices.count(choice)) {
          return absl::StrFormat("Invalid output format: %s", choice);
        }
        return std::string();
      });
  app.add_option("--silent", options->silent,
                 "Output nothing but statistics information.");
}

class InputReader {
 public:
  explicit InputReader(const Options& options, TransformConfig config)
      : config_(std::move(config)),
        offset_(0),
        total_(0),
        count_(0),
        end_of_sequence_(false) {
    tf::monolith_tf::DataFormatOptions ds_options{
        options.lagrangex_header, options.kafka_dump_prefix,
        options.has_sort_id, options.kafka_dump};

    tf::Env* env = tf::Env::Default();
    std::unique_ptr<tf::monolith_tf::BaseStreamReader> stream_reader;
    if (options.filepath.empty()) {
      stream_reader = std::make_unique<StdinStreamReader>(ds_options);
    } else {
      std::unique_ptr<tf::RandomAccessFile> f;
      TF_CHECK_OK(env->NewRandomAccessFile(options.filepath, &f));
      stream_reader = std::make_unique<FileStreamReader>(
          ds_options, std::move(f),
          options.compression_type == "none" ? InputCompressType::NO
                                             : InputCompressType::SNAPPY);
    }

    if (options.dtype == "instance" || options.dtype == "example") {
      reader_ = absl::make_unique<tf::monolith_tf::PBIterator>(
          std::move(stream_reader), FeaturePruningType::AS_IS);
    } else {
      mapper_ = std::make_unique<FeatureNameMapper>();
      reader_ = absl::make_unique<tf::monolith_tf::ExampleBatchIterator>(
          std::move(stream_reader), FeaturePruningType::AS_IS, mapper_.get());
    }

    transform_ = tf::monolith_tf::NewTransformFromConfig(config_);
  }

  template <typename T>
  bool ReadOne(T* output) {
    if (IsBufferEmpty<T>()) {
      tf::tstring serialized_instance;
      try {
        uint32_t data_source_key = 0;
        while (!end_of_sequence_ &&
               reader_->next(&offset_, &data_source_key, &serialized_instance)
                   .ok()) {
          offset_ = reader_->GetOffset();
          std::shared_ptr<T> sample = std::make_shared<T>();
          if (!sample->ParseFromArray(serialized_instance.data(),
                                      serialized_instance.size())) {
            LOG(ERROR) << "Unable to parse data. Data might be corrupted";
            return false;
          }

          ++total_;
          std::vector<std::shared_ptr<T>> outputs;
          Transform(sample, &outputs);
          count_ += outputs.size();
          for (const auto& sample : outputs) {
            PushIntoBuffer<T>(sample);
          }

          if (!outputs.empty()) {
            break;
          }
        }
      } catch (const std::out_of_range& e) {
        end_of_sequence_ = true;
        LOG(INFO) << e.what();
      } catch (const std::exception& e) {
        end_of_sequence_ = true;
        LOG(ERROR) << e.what();
      }
    }

    if (!IsBufferEmpty<T>()) {
      std::shared_ptr<T> front;
      PopFromBuffer(&front);
      front->Swap(output);
      return true;
    }

    return false;
  }

  template <typename T>
  bool ReadOneSerialized(std::string* serialized) {
    T t;
    bool success = ReadOne(&t);
    if (success) {
      *serialized = t.SerializeAsString();
    }
    return success;
  }

 private:
  template <typename T>
  typename std::enable_if<std::is_same<T, Instance>::value, void>::type
  Transform(std::shared_ptr<T> input, std::vector<std::shared_ptr<T>>* output) {
    transform_->Transform(input, output);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Example>::value, void>::type
  Transform(std::shared_ptr<T> input, std::vector<std::shared_ptr<T>>* output) {
    transform_->Transform(input, output);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, ExampleBatch>::value, void>::type
  Transform(std::shared_ptr<T> input, std::vector<std::shared_ptr<T>>* output) {
    output->push_back(input);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Instance>::value, bool>::type
  IsBufferEmpty() {
    return instance_buffer_.empty();
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Example>::value, bool>::type
  IsBufferEmpty() {
    return example_buffer_.empty();
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, ExampleBatch>::value, bool>::type
  IsBufferEmpty() {
    return example_batch_buffer_.empty();
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Instance>::value, void>::type
  PushIntoBuffer(std::shared_ptr<T> t) {
    instance_buffer_.push(t);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Example>::value, void>::type
  PushIntoBuffer(std::shared_ptr<T> t) {
    example_buffer_.push(t);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, ExampleBatch>::value, void>::type
  PushIntoBuffer(std::shared_ptr<T> t) {
    example_batch_buffer_.push(t);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Instance>::value, void>::type
  PopFromBuffer(std::shared_ptr<T>* t) {
    *t = instance_buffer_.front();
    instance_buffer_.pop();
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, Example>::value, void>::type
  PopFromBuffer(std::shared_ptr<T>* t) {
    *t = example_buffer_.front();
    example_buffer_.pop();
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, ExampleBatch>::value, void>::type
  PopFromBuffer(std::shared_ptr<T>* t) {
    *t = example_batch_buffer_.front();
    example_batch_buffer_.pop();
  }

  TransformConfig config_;
  std::unique_ptr<TransformInterface> transform_;
  std::unique_ptr<PBIterator> reader_;
  std::unique_ptr<FeatureNameMapper> mapper_;
  std::queue<std::shared_ptr<Instance>> instance_buffer_;
  std::queue<std::shared_ptr<monolith::io::proto::Example>> example_buffer_;
  std::queue<std::shared_ptr<monolith::io::proto::ExampleBatch>>
      example_batch_buffer_;
  tf::uint64 offset_;
  uint64_t total_;
  uint64_t count_;
  bool end_of_sequence_;
};

template <typename T>
void ReadAndSerialize(
    InputReader& reader, T* t, const Options& options,
    const std::function<std::string(const std::string& serialized)>&
        callback_fn) {
  auto json_options = google::protobuf::util::JsonOptions();
  json_options.add_whitespace = true;
  json_options.preserve_proto_field_names = true;
  for (int64_t i = 0; i < options.limit; ++i) {
    if (reader.ReadOne(t)) {
      if (!options.silent) {
        std::string output;
        if (options.output_format == "json") {
          google::protobuf::util::MessageToJsonString(*t, &output,
                                                      json_options);
          output = callback_fn(output);
        } else {
          output = t->DebugString();
        }

        std::cout.write(output.data(), output.length());
        std::cout.flush();
      }
    } else {
      break;
    }
  }
}

void to_json(nlohmann::json& j, const FIDV1& fid) { j = fid.DebugString(); }

void to_json(nlohmann::json& j, const FIDV2& fid) { j = fid.DebugString(); }

int main(int argc, char* argv[]) {
  CLI::App app("instance_reader");
  app.set_version_flag("--version", "0.0.1");

  Options options;
  AddOptions(app, &options);
  CLI11_PARSE(app, argc, argv)

  if (options.dtype == "example_batch" && !options.config.empty()) {
    LOG(FATAL) << "Transform cannot process ExampleBatch!";
  }
  TransformConfig config;
  CHECK(google::protobuf::TextFormat::ParseFromString(options.config, &config));
  std::cerr << config.DebugString();

  InputReader reader(options, config);
  std::string output;

  auto json_callback_fn = [](const std::string& serialized) {
    nlohmann::json json;
    try {
      json = nlohmann::json::parse(serialized);
    } catch (const std::exception& e) {
      LOG(FATAL) << e.what() << "\nserialized:\n" << serialized;
    }

    auto CollectFID = [](const nlohmann::json& j, std::vector<uint64_t>* fids) {
      CHECK(j.is_array());
      fids->reserve(j.size());
      for (absl::string_view fid_str : j) {
        uint64_t fid = 0;
        CHECK(absl::SimpleAtoi(fid_str, &fid));
        fids->emplace_back(fid);
      }
      std::sort(fids->begin(), fids->end());
    };

    // instance fid(v1)
    if (json.contains("fid") && json["fid"].is_array()) {
      std::vector<uint64_t> fids;
      CollectFID(json["fid"], &fids);
      std::vector<FIDV1> fids_v1(fids.begin(), fids.end());
      json["fid"] = fids_v1;
    }

    // instance feature(v2)
    if (json.contains("feature") && json["feature"].is_array()) {
      for (nlohmann::json& element : json["feature"]) {
        if (element.contains("fid") && element["fid"].is_array()) {
          std::vector<uint64_t> fids;
          CollectFID(element["fid"], &fids);
          std::vector<FIDV2> fids_v2(fids.begin(), fids.end());
          element["fid"] = fids_v2;
        }
      }
    }

    auto ReplaceFeatureFn = [&](nlohmann::json& feature) {
      if (feature.contains("fid_v1_list")) {
        nlohmann::json& fid_v1_list = feature["fid_v1_list"];
        if (fid_v1_list.contains("value") && fid_v1_list["value"].is_array()) {
          std::vector<uint64_t> fids;
          CollectFID(fid_v1_list["value"], &fids);
          std::vector<FIDV1> fids_v1(fids.begin(), fids.end());
          fid_v1_list["value"] = fids_v1;
        }
      } else if (feature.contains("fid_v2_list")) {
        nlohmann::json& fid_v2_list = feature["fid_v2_list"];
        if (fid_v2_list.contains("value") && fid_v2_list["value"].is_array()) {
          std::vector<uint64_t> fids;
          CollectFID(fid_v2_list["value"], &fids);
          std::vector<FIDV2> fids_v2(fids.begin(), fids.end());
          fid_v2_list["value"] = fids_v2;
        }
      }
    };

    // example named_feature
    if (json.contains("named_feature") && json["named_feature"].is_array()) {
      for (nlohmann::json& element : json["named_feature"]) {
        if (element.contains("feature")) {
          ReplaceFeatureFn(element["feature"]);
        }
      }
    }

    // ExampleBatch named_feature_list
    if (json.contains("named_feature_list") &&
        json["named_feature_list"].is_array()) {
      for (nlohmann::json& element : json["named_feature_list"]) {
        if (element.contains("feature") && element["feature"].is_array()) {
          for (nlohmann::json& feature : element["feature"]) {
            ReplaceFeatureFn(feature);
          }
        }
      }
    }

    return json.dump(2);
  };

  if (options.dtype == "instance") {
    Instance instance;
    ReadAndSerialize(reader, &instance, options, json_callback_fn);
  } else if (options.dtype == "example") {
    Example example;
    ReadAndSerialize(reader, &example, options, json_callback_fn);
  } else if (options.dtype == "example_batch") {
    ExampleBatch example_batch;
    ReadAndSerialize(reader, &example_batch, options, json_callback_fn);
  } else {
    throw std::invalid_argument(
        absl::StrFormat("Invalid dtype=%s", options.dtype));
  }

  return 0;
}
