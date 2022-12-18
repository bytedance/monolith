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

#ifndef _MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_CUCKOOHASH_CUCKOOHASH_CONFIG_HPP
#define _MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_CUCKOOHASH_CUCKOOHASH_CONFIG_HPP

#include <cstddef>
#include <limits>

namespace libcuckoo {

// The default maximum number of keys per bucket
constexpr size_t DEFAULT_SLOT_PER_BUCKET = 4;

// The default number of elements in an empty hash table
constexpr size_t DEFAULT_SIZE = (1U << 16) * DEFAULT_SLOT_PER_BUCKET;

// The default minimum load factor that the table allows for automatic
// expansion. It must be a number between 0.0 and 1.0. The table will throw
// load_factor_too_low if the load factor falls below this value
// during an automatic expansion.
constexpr double DEFAULT_MINIMUM_LOAD_FACTOR = 0.05;

// An alias for the value that sets no limit on the maximum hashpower. If this
// value is set as the maximum hashpower limit, there will be no limit. This
// is also the default initial value for the maximum hashpower in a table.
constexpr size_t NO_MAXIMUM_HASHPOWER = std::numeric_limits<size_t>::max();

// set LIBCUCKOO_DEBUG to 1 to enable debug output
#define LIBCUCKOO_DEBUG 0

}  // namespace libcuckoo

#endif  // _MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_CUCKOOHASH_CUCKOOHASH_CONFIG_HPP
