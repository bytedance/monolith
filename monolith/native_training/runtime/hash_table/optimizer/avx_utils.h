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

#ifndef MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_AVX_UTILS_H_
#define MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_AVX_UTILS_H_

#if defined(_ENABLE_AVX) && defined(__AVX__)
#include <immintrin.h>
#endif

namespace monolith {
namespace hash_table {

inline void BaselineAdagradOptimize(float* num, float* norm, const float* grad,
                                    size_t len, float lr, float w_decay) {
  for (size_t i = 0; i < len; ++i) {
    float g = grad[i] + w_decay * num[i];

    norm[i] += g * g;
    float effective_lr = lr / std::sqrt(norm[i]);
    num[i] -= effective_lr * g;
  }
}

inline float BaselineGetGroupNorm(float* num, float* norm, const float* grad,
                                  float* zero, size_t len, float effective_lr) {
  float group_zt_norm = 0;
  for (size_t i = 0; i < len; ++i) {
    auto norm_new = norm[i] + grad[i] * grad[i];
    auto sigma = (std::sqrt(norm_new) - std::sqrt(norm[i])) / effective_lr;
    zero[i] += (grad[i] - sigma * num[i]);
    norm[i] = norm_new;
    group_zt_norm += zero[i] * zero[i];
  }
  return group_zt_norm;
}

inline void BaselineSetWeightsWithGroupNorm(float group_zt_norm, float* num,
                                            float* norm, float* zero,
                                            size_t len, float effective_lr,
                                            float l1_regularization_strength,
                                            float l2_regularization_strength,
                                            float beta) {
  if (group_zt_norm < l1_regularization_strength) {
    for (size_t i = 0; i < len; ++i) {
      num[i] = 0;
    }
  } else {
    float normwise =
        (l1_regularization_strength - group_zt_norm) / group_zt_norm;
    for (size_t i = 0; i < len; ++i) {
      num[i] = effective_lr * zero[i] * normwise /
               (beta + std::sqrt(norm[i]) +
                l2_regularization_strength * effective_lr);
    }
  }
}

inline void BaselineGroupFTRLOptimize(float* num, float* norm,
                                      const float* grad, float* zero,
                                      size_t len, float effective_lr,
                                      float l1_regularization_strength,
                                      float l2_regularization_strength,
                                      float beta) {
  float group_zt_norm =
      BaselineGetGroupNorm(num, norm, grad, zero, len, effective_lr);
  group_zt_norm = std::abs(std::sqrt(group_zt_norm));
  BaselineSetWeightsWithGroupNorm(group_zt_norm, num, norm, zero, len,
                                  effective_lr, l1_regularization_strength,
                                  l2_regularization_strength, beta);
}

inline void BaseReduceSum(const float* a, const float* b, float* output,
                          size_t len) {
  for (size_t i = 0; i < len; ++i) {
    output[i] = a[i] + b[i];
  }
}

#if defined(_ENABLE_AVX) && defined(__AVX__)
inline void Avx256AdagradOptimize(float* num, float* norm, const float* grad,
                                  size_t len, float lr, float w_decay) {
  const __m256 lamda = _mm256_set1_ps(w_decay);
  const __m256 _lr = _mm256_set1_ps(lr);

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
    BaselineAdagradOptimize(num, norm, grad, len, lr, w_decay);
  }
}

// horizontal sum of mm256
inline float sum8(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);

  return _mm_cvtss_f32(sum);
}

inline void Avx256GroupFTRLOptimize(float* num, float* norm, const float* grad,
                                    float* zero, size_t len, float effective_lr,
                                    float l1_regularization_strength,
                                    float l2_regularization_strength,
                                    float beta) {
  const __m256 _lr = _mm256_set1_ps(effective_lr);
  float group_zt_norm = 0.0;

  float* numCopy = num;
  size_t lenCopy = len;

  float* normCopy = norm;
  float* zeroCopy = zero;

  for (; len > 7; len -= 8, num += 8, norm += 8, grad += 8, zero += 8) {
    const __m256 _group_zt_norm = _mm256_set1_ps(0.0);

    const __m256 _num = _mm256_loadu_ps(num);
    const __m256 _norm = _mm256_loadu_ps(norm);
    const __m256 _grad = _mm256_loadu_ps(grad);
    const __m256 _zero = _mm256_loadu_ps(zero);

    const __m256 _new_norm = _mm256_fmadd_ps(_grad, _grad, _norm);
    const __m256 _norm_new_sqrt = _mm256_sqrt_ps(_new_norm);
    const __m256 _norm_sqrt = _mm256_sqrt_ps(_norm);
    __m256 _sigma = _mm256_sub_ps(_norm_new_sqrt, _norm_sqrt);
    _sigma = _mm256_div_ps(_sigma, _lr);

    const __m256 _add_zero = _mm256_fnmadd_ps(_sigma, _num, _grad);
    const __m256 _new_zero = _mm256_add_ps(_zero, _add_zero);

    _mm256_storeu_ps(zero, _new_zero);
    _mm256_storeu_ps(norm, _new_norm);

    group_zt_norm +=
        sum8(_mm256_fmadd_ps(_new_zero, _new_zero, _group_zt_norm));
  }

  if (len) {
    group_zt_norm +=
        BaselineGetGroupNorm(num, norm, grad, zero, len, effective_lr);
  }

  group_zt_norm = std::abs(std::sqrt(group_zt_norm));

  if (group_zt_norm < l1_regularization_strength) {
    for (; lenCopy > 7; lenCopy -= 8, numCopy += 8) {
      _mm256_storeu_ps(numCopy, _mm256_set1_ps(0.0));
    }
  } else {
    const __m256 _normwise = _mm256_set1_ps(
        (l1_regularization_strength - group_zt_norm) / group_zt_norm);
    const __m256 _l2_regularization =
        _mm256_set1_ps(l2_regularization_strength);
    const __m256 _beta = _mm256_set1_ps(beta);

    for (; lenCopy > 7;
         lenCopy -= 8, numCopy += 8, normCopy += 8, zeroCopy += 8) {
      const __m256 _norm = _mm256_loadu_ps(normCopy);
      const __m256 _zero = _mm256_loadu_ps(zeroCopy);
      const __m256 _sqrt_norm = _mm256_sqrt_ps(_norm);

      __m256 _denom = _mm256_fnmadd_ps(_l2_regularization, _lr, _sqrt_norm);
      _denom = _mm256_add_ps(_denom, _beta);

      __m256 _numer = _mm256_mul_ps(_lr, _zero);
      _numer = _mm256_mul_ps(_numer, _normwise);
      __m256 _new_num = _mm256_div_ps(_numer, _denom);
      _mm256_storeu_ps(numCopy, _new_num);
    }
  }

  if (lenCopy) {
    BaselineSetWeightsWithGroupNorm(
        group_zt_norm, numCopy, normCopy, zeroCopy, lenCopy, effective_lr,
        l1_regularization_strength, l2_regularization_strength, beta);
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

inline void AdagradOptimize(float* num, float* norm, const float* grad,
                            size_t len, float lr, float w_decay) {
#if defined(_ENABLE_AVX) && defined(__AVX__)
  Avx256AdagradOptimize(num, norm, grad, len, lr, w_decay);
#else
  BaselineAdagradOptimize(num, norm, grad, len, lr, w_decay);
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
#endif  // MONOLITH_MONOLITH_NATIVE_TRAINING_RUNTIME_HASH_TABLE_OPTIMIZER_AVX_UTILS_H_
