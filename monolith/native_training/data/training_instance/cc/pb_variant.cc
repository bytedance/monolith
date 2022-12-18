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

#include <unordered_map>
#include "absl/strings/str_cat.h"

#include "monolith/native_training/data/training_instance/cc/pb_variant.h"

namespace tensorflow {
using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using Instance = ::parser::proto::Instance;

template <>
std::string TypeNameVariant<Example>(const Example &value) {
  return "Example";
}

template <>
std::string DebugStringVariant<Example>(const Example &value) {
  return "Example DebugString";
}

template <>
bool DecodeVariant<Example>(std::string *buf, Example *value) {
  std::cout << "DecodeVariant<EFeature> - 1" << std::endl;
  value->ParseFromArray(buf->data(), buf->size());
  return true;
}

template <>
void EncodeVariant<Example>(const Example &value, std::string *buf) {
  value.SerializeToString(buf);
}

template <>
bool DecodeVariant<Example>(VariantTensorData *data, Example *value) {
  return false;
}

template <>
void EncodeVariant<Example>(const Example &value, VariantTensorData *data) {}

template <>
std::string TypeNameVariant<ExampleBatch>(const ExampleBatch &value) {
  return "ExampleBatch";
}

template <>
std::string DebugStringVariant<ExampleBatch>(const ExampleBatch &value) {
  return "ExampleBatch DebugString";
}

template <>
bool DecodeVariant<ExampleBatch>(std::string *buf, ExampleBatch *value) {
  std::cout << "DecodeVariant<EFeature> - 1" << std::endl;
  value->ParseFromArray(buf->data(), buf->size());
  return true;
}

template <>
void EncodeVariant<ExampleBatch>(const ExampleBatch &value, std::string *buf) {
  value.SerializeToString(buf);
}

template <>
bool DecodeVariant<ExampleBatch>(VariantTensorData *data, ExampleBatch *value) {
  return false;
}

template <>
void EncodeVariant<ExampleBatch>(const ExampleBatch &value,
                                 VariantTensorData *data) {}

template <>
std::string TypeNameVariant<Instance>(const Instance &value) {
  return "Instance";
}

template <>
std::string DebugStringVariant<Instance>(const Instance &value) {
  return "Instance DebugString";
}

template <>
bool DecodeVariant<Instance>(std::string *buf, Instance *value) {
  std::cout << "DecodeVariant<EFeature> - 1" << std::endl;
  value->ParseFromArray(buf->data(), buf->size());
  return true;
}

template <>
void EncodeVariant<Instance>(const Instance &value, std::string *buf) {
  value.SerializeToString(buf);
}

template <>
bool DecodeVariant<Instance>(VariantTensorData *data, Instance *value) {
  return false;
}

template <>
void EncodeVariant<Instance>(const Instance &value, VariantTensorData *data) {}

}  // namespace tensorflow
