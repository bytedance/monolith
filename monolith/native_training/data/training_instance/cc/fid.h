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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_FID_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_FID_H_

#include <iostream>
#include <sstream>

union FIDV2;

union FIDV1 {
  struct Underlying {
    uint64_t signature : 54;
    uint64_t slot : 10;
    Underlying(uint64_t slot, uint64_t signature)
        : slot(slot), signature(signature) {}
  };

  Underlying underlying;
  uint64_t value;

  FIDV1() : underlying(0, 0) {}
  FIDV1(uint64_t slot, int64_t signature) : underlying(slot, signature) {
    if (slot >= 1024) {
      throw std::invalid_argument("slot should be less than 1024, while got " +
                                  std::to_string(slot));
    }
  }
  FIDV1(uint64_t fid_v1_value) : value(fid_v1_value) {}

  operator uint64_t() const { return this->value; }

  [[nodiscard]] uint64_t slot() const { return this->underlying.slot; }

  [[nodiscard]] uint64_t signature() const {
    return this->underlying.signature;
  }

  [[nodiscard]] std::string DebugString() const {
    std::stringstream ss;
    ss << value << "(v1|slot=" << underlying.slot
       << "|sig=" << underlying.signature << ")";
    return ss.str();
  }

  [[nodiscard]] FIDV2 ConvertAsV2() const;
};

union FIDV2 {
  struct Underlying {
    uint64_t signature : 48;
    uint64_t slot : 15;
    uint64_t reserved : 1;

    Underlying(uint64_t slot, uint64_t signature)
        : reserved(0), slot(slot), signature(signature) {}
  };

  Underlying underlying;
  uint64_t value;

  FIDV2() : underlying(0, 0) {}
  FIDV2(uint64_t slot, uint64_t signature) : underlying(slot, signature) {
    if (slot >= 32768) {
      throw std::invalid_argument("slot should be less than 32768, while got " +
                                  std::to_string(slot));
    }
  }
  FIDV2(uint64_t fid_v2_value) : value(fid_v2_value) {
    if (this->underlying.reserved == 1) {
      throw std::invalid_argument("slot should be less than 32768, while got " +
                                  std::to_string(this->slot() + 32768));
    }
  }

  operator uint64_t() const { return value; }

  [[nodiscard]] uint64_t slot() const { return this->underlying.slot; }

  [[nodiscard]] uint64_t signature() const {
    return this->underlying.signature;
  }

  [[nodiscard]] std::string DebugString() const {
    std::stringstream ss;
    ss << value << "(v2|slot=" << underlying.slot
       << "|sig=" << underlying.signature << ")";
    return ss.str();
  }
};

FIDV2 FIDV1::ConvertAsV2() const {
  return {this->underlying.slot, this->underlying.signature};
}

namespace std {

template <>
struct hash<FIDV1> {
  std::size_t operator()(FIDV1 fid) const { return std::hash<uint64_t>()(fid); }
};

template <>
struct hash<FIDV2> {
  std::size_t operator()(FIDV2 fid) const { return std::hash<uint64_t>()(fid); }
};

}  // namespace std

#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_FID_H_
