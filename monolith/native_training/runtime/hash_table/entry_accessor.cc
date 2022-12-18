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

#include "monolith/native_training/runtime/hash_table/entry_accessor.h"

#include <cstdint>
#include <exception>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "monolith/native_training/runtime/hash_table/compressor/float_compressor.h"
#include "monolith/native_training/runtime/hash_table/initializer/initializer_combination.h"
#include "monolith/native_training/runtime/hash_table/initializer/initializer_config.pb.h"
#include "monolith/native_training/runtime/hash_table/initializer/initializer_factory.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer.pb.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_combination.h"
#include "monolith/native_training/runtime/hash_table/optimizer/optimizer_factory.h"
#include "monolith/native_training/runtime/hash_table/retriever/fake_quant_retriever.h"
#include "monolith/native_training/runtime/hash_table/retriever/hash_net_retriever.h"
#include "monolith/native_training/runtime/hash_table/retriever/raw_retriever.h"
#include "monolith/native_training/runtime/hash_table/retriever/retriever_combination.h"

namespace monolith {
namespace hash_table {
namespace {

namespace proto2 = google::protobuf;

class ServingEntryAccessor final : public EntryAccessorInterface {
 public:
  explicit ServingEntryAccessor(
      std::unique_ptr<FloatCompressorInterface> compressor)
      : compressor_(std::move(compressor)),
        size_bytes_(compressor_->SizeBytes()) {}

  int64_t SizeBytes() const override { return size_bytes_; }

  int DimSize() const override { return compressor_->DimSize(); }

  int SliceSize() const override {
    throw std::runtime_error("ServingEntryAccessor doesn't support SliceSize");
  }

  void Init(void* ctx) const override {
    // No need to initialize serving entry
  }

  void Fill(const void* ctx, absl::Span<float> num) const override {
    compressor_->Decode(ctx, num);
  }

  void Assign(absl::Span<const float> num, void* ctx) const override {
    compressor_->Encode(num, ctx);
  }

  void AssignAdd(absl::Span<const float> num, void* ctx) const override {
    std::vector<float> embedding(num.size());
    compressor_->Decode(ctx, absl::MakeSpan(embedding));
    for (int i = 0; i < num.size(); ++i) {
      embedding[i] += num[i];
    }
    compressor_->Encode(embedding, ctx);
  }

  void Optimize(void* ctx, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    throw std::runtime_error("ServingEntryAccessor doesn't support Optimize");
  }

  EntryDump Save(const void* ctx, uint32_t timestamp_sec) const override {
    throw std::runtime_error("ServingEntryAccessor doesn't support Save");
  }

  void Restore(void* ctx, uint32_t* timestamp_sec, EntryDump dump) const {
    (void)timestamp_sec;
    std::vector<float> num(dump.num_size());
    absl::c_copy(dump.num(), num.begin());
    compressor_->Encode(num, ctx);
  }

 private:
  std::unique_ptr<FloatCompressorInterface> compressor_;
  int size_bytes_;
};

// The layout of ctx is:
// float * dim_size_ | Info | optimizer_data
class EntryAccessor final : public EntryAccessorInterface {
 public:
  EntryAccessor(std::unique_ptr<InitializerInterface> initializer,
                std::unique_ptr<OptimizerInterface> optimizer,
                std::unique_ptr<RetrieverInterface> retriever)
      : initializer_(std::move(initializer)),
        optimizer_(std::move(optimizer)),
        retriever_(std::move(retriever)),
        optimizer_bytes_(optimizer_->SizeBytes()),
        dim_size_(initializer_->DimSize()),
        slice_size_(optimizer_->SliceSize()),
        num_bytes_(retriever_->SizeBytes()) {
    if (initializer_->DimSize() != optimizer_->DimSize() ||
        initializer_->DimSize() != retriever_->DimSize()) {
      throw std::invalid_argument(
          absl::StrFormat("Initializer/Optimizer/Retriever dim size should "
                          "match. But got %d vs %d vs %d",
                          initializer_->DimSize(), optimizer_->DimSize(),
                          retriever_->DimSize()));
    }
  }

  EntryAccessor(EntryAccessor&&) = default;
  EntryAccessor& operator=(EntryAccessor&&) = default;

  int64_t SizeBytes() const override { return num_bytes_ + optimizer_bytes_; }

  int DimSize() const override { return dim_size_; }

  int SliceSize() const override { return slice_size_; }

  absl::Span<float> GetMutableNum(void* ctx) const {
    float* ctx_float = static_cast<float*>(ctx);
    return absl::MakeSpan(ctx_float, ctx_float + dim_size_);
  }

  void Init(void* ctx) const override {
    auto num_span = GetMutableNum(ctx);
    initializer_->Initialize(num_span);
    optimizer_->Init(GetMutableOptimizerCtx(ctx));
  }

  void Fill(const void* ctx, absl::Span<float> num) const override {
    retriever_->Retrieve(ctx, num);
  }

  void Assign(absl::Span<const float> num, void* ctx) const override {
    auto ctx_float = static_cast<float*>(ctx);
    auto embedding = absl::MakeSpan(ctx_float, ctx_float + dim_size_);
    std::memcpy(embedding.data(), num.data(), sizeof(float) * num.size());
  }

  void AssignAdd(absl::Span<const float> num, void* ctx) const override {
    auto ctx_float = static_cast<float*>(ctx);
    auto embedding = absl::MakeSpan(ctx_float, ctx_float + dim_size_);
    for (int i = 0; i < num.size(); ++i) {
      embedding[i] += num[i];
    }
  }

  void Optimize(void* ctx, absl::Span<const float> grad,
                absl::Span<const float> learning_rates,
                const int64_t global_step) const override {
    auto* mutable_grad = const_cast<float*>(grad.data());
    retriever_->Backward(GetNum(ctx), absl::MakeSpan(mutable_grad, grad.size()),
                         global_step);
    optimizer_->Optimize(GetMutableOptimizerCtx(ctx), GetMutableNum(ctx), grad,
                         learning_rates, global_step);
  }

  EntryDump Save(const void* ctx, uint32_t timestamp) const override;
  void Restore(void* ctx, uint32_t* timestamp_sec,
               EntryDump dump) const override;

 private:
  absl::Span<const float> GetNum(const void* ctx) const {
    const float* ctx_float = static_cast<const float*>(ctx);
    return absl::MakeConstSpan(ctx_float, ctx_float + dim_size_);
  }

  void* GetMutableOptimizerCtx(void* ctx) const {
    return AddOffset(ctx, num_bytes_);
  }

  const void* GetOptimizerCtx(const void* ctx) const {
    return AddOffset(ctx, num_bytes_);
  }

  std::unique_ptr<InitializerInterface> initializer_;
  std::unique_ptr<OptimizerInterface> optimizer_;
  std::unique_ptr<RetrieverInterface> retriever_;
  const int64_t optimizer_bytes_ = 0;
  const int dim_size_ = 0;
  const int slice_size_ = 0;
  const int64_t num_bytes_ = 0;
};

EntryDump EntryAccessor::Save(const void* ctx, uint32_t timestamp_sec) const {
  EntryDump dump;
  absl::c_copy(GetNum(ctx),
               proto2::RepeatedFieldBackInserter(dump.mutable_num()));
  *dump.mutable_opt() = optimizer_->Save(GetOptimizerCtx(ctx));
  dump.set_last_update_ts_sec(timestamp_sec);
  return dump;
}

void EntryAccessor::Restore(void* ctx, uint32_t* timestamp_sec,
                            EntryDump dump) const {
  auto num = GetMutableNum(ctx);
  for (int i = 0; i < dump.num_size(); ++i) {
    num[i] = dump.num(i);
  }

  *timestamp_sec = dump.last_update_ts_sec();
  optimizer_->Restore(GetMutableOptimizerCtx(ctx),
                      std::move(*dump.mutable_opt()));
}

// Write dim_size into sub field of T (T can be OptimizerConfig,
// InitializerConfig or FloatCompressorConfig).
template <class T>
void WriteDimSize(T* conf, int dim_size) {
  const proto2::Descriptor* descriptor = conf->GetDescriptor();
  const proto2::Reflection* reflection = conf->GetReflection();
  const proto2::OneofDescriptor* type = descriptor->FindOneofByName("type");
  const proto2::FieldDescriptor* type_field =
      reflection->GetOneofFieldDescriptor(*conf, type);
  if (type_field == nullptr ||
      type_field->type() != proto2::FieldDescriptor::TYPE_MESSAGE) {
    throw std::invalid_argument(absl::StrFormat("%s must be set type. Got %s",
                                                descriptor->name(),
                                                conf->ShortDebugString()));
  }
  proto2::Message* type_msg = reflection->MutableMessage(conf, type_field);
  const proto2::FieldDescriptor* dim_size_field =
      type_msg->GetDescriptor()->FindFieldByName("dim_size");
  type_msg->GetReflection()->SetInt32(type_msg, dim_size_field, dim_size);
}

struct Objects {
  std::unique_ptr<InitializerInterface> init;
  std::unique_ptr<OptimizerInterface> opt;
  std::unique_ptr<FloatCompressorInterface> comp;
  std::unique_ptr<RetrieverInterface> retriever;
};

template <class T, class F>
void AssignOrCombine(T* t1, T t2, F combine_fn) {
  if (*t1 == nullptr) {
    *t1 = std::move(t2);
    return;
  }
  *t1 = combine_fn(std::move(*t1), std::move(t2));
}

Objects GenerateObjFromSegments(
    proto2::RepeatedPtrField<EntryConfig::Segment>* segments) {
  Objects obj;
  for (EntryConfig::Segment& segment : *segments) {
    if (segment.has_comp_config() && segment.comp_config().has_fixed_r8()) {
      auto retriever = NewFakeQuantRetriever(
          segment.dim_size(),
          FakeQuantizer(segment.comp_config().fixed_r8().r()));
      AssignOrCombine(&obj.retriever, std::move(retriever), CombineRetrievers);
    } else if (segment.has_comp_config() &&
               segment.comp_config().has_one_bit()) {
      auto hash_net_quantizer =
          std::make_unique<HashNetQuantizer>(segment.comp_config().one_bit());
      auto retriever = NewHashNetRetriever(segment.dim_size(),
                                           std::move(hash_net_quantizer));
      AssignOrCombine(&obj.retriever, std::move(retriever), CombineRetrievers);
    } else {
      auto retriever = NewRawRetriever(segment.dim_size());
      AssignOrCombine(&obj.retriever, std::move(retriever), CombineRetrievers);
    }
    if (segment.has_opt_config()) {
      WriteDimSize(segment.mutable_opt_config(), segment.dim_size());
      auto new_opt = NewOptimizerFromConfig(segment.opt_config());
      AssignOrCombine(&obj.opt, std::move(new_opt), CombineOptimizers);
    }
    if (segment.has_init_config()) {
      WriteDimSize(segment.mutable_init_config(), segment.dim_size());
      auto new_init = NewInitializerFromConfig(segment.init_config());
      AssignOrCombine(&obj.init, std::move(new_init), CombineInitializers);
    }
    if (segment.has_comp_config()) {
      WriteDimSize(segment.mutable_comp_config(), segment.dim_size());
      auto new_comp = NewFloatCompressor(segment.comp_config());
      AssignOrCombine(&obj.comp, std::move(new_comp), CombineFloatCompressor);
    }
  }
  return obj;
}

}  // namespace

std::unique_ptr<EntryAccessorInterface> NewEntryAccessor(EntryConfig config) {
  LOG(INFO) << "EntryConfig: " << config.DebugString();
  Objects obj = GenerateObjFromSegments(config.mutable_segments());
  switch (config.entry_type()) {
    case EntryConfig::TRAINING:
      if (obj.init == nullptr || obj.opt == nullptr) {
        throw std::invalid_argument(absl::StrFormat(
            "init or opt config is missing from entry config : %s",
            config.ShortDebugString()));
      }

      return std::make_unique<EntryAccessor>(
          std::move(obj.init), std::move(obj.opt), std::move(obj.retriever));
    case EntryConfig::SERVING:
      if (obj.comp == nullptr) {
        throw std::invalid_argument(
            absl::StrFormat("comp config is missing form entry config: %s",
                            config.ShortDebugString()));
      }
      return std::make_unique<ServingEntryAccessor>(std::move(obj.comp));
    default:
      throw std::invalid_argument(
          absl::StrFormat("Unknown entry type: %s", config.ShortDebugString()));
  }
}

}  // namespace hash_table
}  // namespace monolith
