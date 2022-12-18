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

//
// Created by david on 2020-11-27.
//

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_DYNAMIC_WD_AVX_UTILS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_DYNAMIC_WD_AVX_UTILS_H_

#if defined(_ENABLE_AVX) && defined(__AVX__)
#include <immintrin.h>
#endif

namespace monolith {
namespace hash_table {

inline void BaselineDynamicWdAdagradOptimize(float* num, float* norm, const float* grad,
                                    size_t len, float lr, float w_decay,
                                    bool decouple_wd) {
  for (size_t i = 0; i < len; ++i) {
    float g = grad[i];
    if (!decouple_wd) {
      g += w_decay * num[i];
    }
    norm[i] += g * g;
    float effective_lr = lr / std::sqrt(norm[i]);
    float grad_update = effective_lr * g;
    if (decouple_wd) {
      grad_update += lr * w_decay * num[i];
    }

    num[i] -= grad_update;
  }
}

inline void BaseReduceSum(const float* a, const float* b, float* output,
                          size_t len) {
  for (size_t i = 0; i < len; ++i) {
    output[i] = a[i] + b[i];
  }
}

#if defined(_ENABLE_AVX) && defined(__AVX__)
inline void Avx256DynamicWdAdagradOptimize(float* num, float* norm, const float* grad,
                                  size_t len, float lr, float w_decay) {
  const __m256 lamda = _mm256_set1_ps(w_decay);
  float lrs[8] = {lr, lr, lr, lr, lr, lr, lr, lr};
  const __m256 _lr = _mm256_loadu_ps(lrs);

  // OPTIMIZE: Loads floating-point vector from an aligned memory address
  for (; len > 7; len -= 8, num += 8, norm += 8, grad += 8) {
    const __m256 _num = _mm256_loadu_ps(num);
    const __m256 _norm = _mm256_loadu_ps(norm);
    const __m256 _grad = _mm256_loadu_ps(grad);
    const __m256 updated_grad = _mm256_fmadd_ps(lamda, _num, _grad);
    __m256 _norm_new = _mm256_fmadd_ps(updated_grad, updated_grad, _norm);
    _mm256_storeu_ps(norm, _norm_new);

    const __m256 _norm_new_sqrt = _mm256_sqrt_ps(_norm_new);
    const __m256 effective_lr = _mm256_div_ps(_lr, _norm_new_sqrt);
    __m256 _num_new = _mm256_fnmadd_ps(effective_lr, _grad, _num);
    _mm256_storeu_ps(num, _num_new);
  }

  if (len) {
    BaselineDynamicWdAdagradOptimize(num, norm, grad, len, lr, w_decay, false);
  }
}

inline void Avx256DynamicWdAdagradOptimizeDecoupleWd(float* num, float* norm, const float* grad,
                                  size_t len, float lr, float w_decay) {
  const __m256 lamda = _mm256_set1_ps(w_decay);
  float lrs[8] = {lr, lr, lr, lr, lr, lr, lr, lr};
  const __m256 _lr = _mm256_loadu_ps(lrs);

  // OPTIMIZE: Loads floating-point vector from an aligned memory address
  for (; len > 7; len -= 8, num += 8, norm += 8, grad += 8) {
    const __m256 _num = _mm256_loadu_ps(num);
    const __m256 _norm = _mm256_loadu_ps(norm);
    const __m256 _grad = _mm256_loadu_ps(grad);
    __m256 _norm_new = _mm256_fmadd_ps(_grad, _grad, _norm);
    _mm256_storeu_ps(norm, _norm_new);

    const __m256 _norm_new_sqrt = _mm256_sqrt_ps(_norm_new);
    const __m256 effective_lr = _mm256_div_ps(_lr, _norm_new_sqrt);
    __m256 _num_new = _mm256_fnmadd_ps(effective_lr, _grad, _num);
    const __m256 effective_wd = _mm256_mul_ps(_lr, lamda);
    __m256 _num_after_wd = _mm256_fnmadd_ps(effective_wd, _num, _num_new);
    _mm256_storeu_ps(num, _num_after_wd);
  }

  if (len) {
    BaselineDynamicWdAdagradOptimize(num, norm, grad, len, lr, w_decay, true);
  }
}

inline void Avx256ReduceSum(const float* a, const float* b, float* output,
                            size_t len) {
  for (; len > 7; len -= 8, a += 8, b += 8, output += 8) {
    const __m256 _a = _mm256_loadu_ps(a);
    const __m256 _b = _mm256_loadu_ps(b);
    const __m256 _output = _mm256_add_ps(_a, _b);
    _mm256_storeu_ps(output, _output);
  }
  if (len) {
    BaseReduceSum(a, b, output, len);
  }
}
#endif

inline void DynamicWdAdagradOptimize(float* num, float* norm, const float* grad,
                            size_t len, float lr, float w_decay,
                            bool decouple_wd = false) {
#if defined(_ENABLE_AVX) && defined(__AVX__)
  if (decouple_wd) {
    Avx256DynamicWdAdagradOptimizeDecoupleWd(num, norm, grad, len, lr, w_decay);
  } else {
    Avx256DynamicWdAdagradOptimize(num, norm, grad, len, lr, w_decay);
  }
#else
  BaselineDynamicWdAdagradOptimize(
    num, norm, grad, len, lr, w_decay, decouple_wd);
#endif
}

inline void ReduceSum(const float* a, const float* b, float* output,
                      size_t len) {
#if defined(_ENABLE_AVX) && defined(__AVX__)
  Avx256ReduceSum(a, b, output, len);
#else
  BaseReduceSum(a, b, output, len);
#endif
}

}  // namespace hash_table
}  // namespace monolith
#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_DYNAMIC_WD_AVX_UTILS_H_
