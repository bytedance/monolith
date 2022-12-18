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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_PB_VARIANT_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_PB_VARIANT_H_

#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "tensorflow/core/framework/variant.h"

namespace tensorflow {
template <>
std::string TypeNameVariant<::monolith::io::proto::Example>(
    const ::monolith::io::proto::Example &value);

template <>
std::string DebugStringVariant<::monolith::io::proto::Example>(
    const ::monolith::io::proto::Example &value);

template <>
bool DecodeVariant<::monolith::io::proto::Example>(
    std::string *buf, ::monolith::io::proto::Example *value);

template <>
void EncodeVariant<::monolith::io::proto::Example>(
    const ::monolith::io::proto::Example &value, std::string *buf);

template <>
bool DecodeVariant<::monolith::io::proto::Example>(
    VariantTensorData *data, ::monolith::io::proto::Example *value);

template <>
void EncodeVariant<::monolith::io::proto::Example>(
    const ::monolith::io::proto::Example &value, VariantTensorData *data);

template <>
std::string TypeNameVariant<::monolith::io::proto::ExampleBatch>(
    const ::monolith::io::proto::ExampleBatch &value);

template <>
std::string DebugStringVariant<::monolith::io::proto::ExampleBatch>(
    const ::monolith::io::proto::ExampleBatch &value);

template <>
bool DecodeVariant<::monolith::io::proto::ExampleBatch>(
    std::string *buf, ::monolith::io::proto::ExampleBatch *value);

template <>
void EncodeVariant<::monolith::io::proto::ExampleBatch>(
    const ::monolith::io::proto::ExampleBatch &value, std::string *buf);

template <>
bool DecodeVariant<::monolith::io::proto::ExampleBatch>(
    VariantTensorData *data, ::monolith::io::proto::ExampleBatch *value);

template <>
void EncodeVariant<::monolith::io::proto::ExampleBatch>(
    const ::monolith::io::proto::ExampleBatch &value, VariantTensorData *data);

template <>
std::string TypeNameVariant<::parser::proto::Instance>(
    const ::parser::proto::Instance &value);

template <>
std::string DebugStringVariant<::parser::proto::Instance>(
    const ::parser::proto::Instance &value);

template <>
bool DecodeVariant<::parser::proto::Instance>(std::string *buf,
                                              ::parser::proto::Instance *value);

template <>
void EncodeVariant<::parser::proto::Instance>(
    const ::parser::proto::Instance &value, std::string *buf);

template <>
bool DecodeVariant<::parser::proto::Instance>(VariantTensorData *data,
                                              ::parser::proto::Instance *value);

template <>
void EncodeVariant<::parser::proto::Instance>(
    const ::parser::proto::Instance &value, VariantTensorData *data);

}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_PB_VARIANT_H_
