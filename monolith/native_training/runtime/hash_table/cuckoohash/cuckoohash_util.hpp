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

/** \file */

#ifndef _MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_CUCKOOHASH_CUCKOOHASH_UTIL_HPP
#define _MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_CUCKOOHASH_CUCKOOHASH_UTIL_HPP

#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include "monolith/native_training/runtime/hash_table/cuckoohash/cuckoohash_config.hpp"  // for LIBCUCKOO_DEBUG

namespace libcuckoo {

#if LIBCUCKOO_DEBUG
// When \ref LIBCUCKOO_DEBUG is 0, LIBCUCKOO_DBG will printing out status
// messages in various situations
#define LIBCUCKOO_DBG(fmt, ...)                                     \
  fprintf(stderr,                                                   \
          "\x1b[32m"                                                \
          "[libcuckoo:%s:%d:%lu] " fmt                              \
          ""                                                        \
          "\x1b[0m",                                                \
          __FILE__, __LINE__,                                       \
          std::hash<std::thread::id>()(std::this_thread::get_id()), \
          __VA_ARGS__)
#else
// When \ref LIBCUCKOO_DEBUG is 0, LIBCUCKOO_DBG does nothing
#define LIBCUCKOO_DBG(fmt, ...) \
  do {                          \
  } while (0)
#endif

/**
 * alignas() requires GCC >= 4.9, so we stick with the alignment attribute for
 * GCC.
 */
#ifdef __GNUC__
#define LIBCUCKOO_ALIGNAS(x) __attribute__((aligned(x)))
#else
#define LIBCUCKOO_ALIGNAS(x) alignas(x)
#endif

/**
 * At higher warning levels, MSVC produces an annoying warning that alignment
 * may cause wasted space: "structure was padded due to __declspec(align())".
 */
#ifdef _MSC_VER
#define LIBCUCKOO_SQUELCH_PADDING_WARNING __pragma(warning(suppress : 4324))
#else
#define LIBCUCKOO_SQUELCH_PADDING_WARNING
#endif

/**
 * At higher warning levels, MSVC may issue a deadcode warning which depends on
 * the template arguments given. For certain other template arguments, the code
 * is not really "dead".
 */
#ifdef _MSC_VER
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_BEGIN \
  do {                                           \
    __pragma(warning(push));                     \
    __pragma(warning(disable : 4702))            \
  } while (0)
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_END __pragma(warning(pop))
#else
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_BEGIN
#define LIBCUCKOO_SQUELCH_DEADCODE_WARNING_END
#endif

/**
 * Thrown when an automatic expansion is triggered, but the load factor of the
 * table is below a minimum threshold, which can be set by the \ref
 * cuckoohash_map::minimum_load_factor method. This can happen if the hash
 * function does not properly distribute keys, or for certain adversarial
 * workloads.
 */
class load_factor_too_low : public std::exception {
 public:
  /**
   * Constructor
   *
   * @param lf the load factor of the table when the exception was thrown
   */
  load_factor_too_low(const double lf) noexcept : load_factor_(lf) {}  // NOLINT

  /**
   * @return a descriptive error message
   */
  virtual const char *what() const noexcept override {  // NOLINT
    return "Automatic expansion triggered when load factor was below "
           "minimum threshold";
  }

  /**
   * @return the load factor of the table when the exception was thrown
   */
  double load_factor() const noexcept { return load_factor_; }

 private:
  const double load_factor_;
};

/**
 * Thrown when an expansion is triggered, but the hashpower specified is greater
 * than the maximum, which can be set with the \ref
 * cuckoohash_map::maximum_hashpower method.
 */
class maximum_hashpower_exceeded : public std::exception {
 public:
  /**
   * Constructor
   *
   * @param hp the hash power we were trying to expand to
   */
  maximum_hashpower_exceeded(const size_t hp) noexcept : hashpower_(hp) {}  // NOLINT

  /**
   * @return a descriptive error message
   */
  virtual const char *what() const noexcept override {  // NOLINT
    return "Expansion beyond maximum hashpower";
  }

  /**
   * @return the hashpower we were trying to expand to
   */
  size_t hashpower() const noexcept { return hashpower_; }

 private:
  const size_t hashpower_;
};

}  // namespace libcuckoo

#endif  // _MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_CUCKOOHASH_CUCKOOHASH_UTIL_HPP
