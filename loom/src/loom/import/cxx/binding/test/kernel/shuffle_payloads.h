// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_TEST_KERNEL_SHUFFLE_PAYLOADS_H_
#define LOOM_IMPORT_CXX_BINDING_TEST_KERNEL_SHUFFLE_PAYLOADS_H_

#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

using Words4 = unsigned __attribute__((ext_vector_type(4)));
using Bytes32 = unsigned char __attribute__((ext_vector_type(32)));
using BFloats16 = __bf16 __attribute__((ext_vector_type(16)));

// Logical component counts for payloads spanning multiple native packets.
template <class T>
struct WidePayload;
template <>
struct WidePayload<Bytes32> {
  using Element = unsigned char;
  // Number of logical components carrying independent source fingerprints.
  static constexpr unsigned count = 32;
};
template <>
struct WidePayload<BFloats16> {
  using Element = __bf16;
  // Number of logical components carrying independent source fingerprints.
  static constexpr unsigned count = 16;
};

// Use object bits, including floating NaNs, without numerical
// conversions. Each wide component carries a distinct source-lane fingerprint.
template <class T>
LOOM_INLINE T payload(unsigned long long bits) {
  if constexpr (sizeof(T) > 16) {
    T values{};
    for (unsigned component = 0; component < WidePayload<T>::count;
         ++component) {
      values[component] = payload<typename WidePayload<T>::Element>(
          bits ^ (0x9325ull * component));
    }
    return values;
  } else if constexpr (__is_same(T, bool)) {
    return (bits & 1) != 0;
  } else if constexpr (sizeof(T) == 1) {
    return __builtin_bit_cast(T, static_cast<unsigned char>(bits));
  } else if constexpr (sizeof(T) == 2) {
    return __builtin_bit_cast(T, static_cast<unsigned short>(bits));
  } else if constexpr (sizeof(T) == 4) {
    return __builtin_bit_cast(T, static_cast<unsigned>(bits));
  } else if constexpr (sizeof(T) == 8) {
    return __builtin_bit_cast(T, bits);
  } else {
    static_assert(sizeof(T) == 16);
    Words4 words{static_cast<unsigned>(bits), static_cast<unsigned>(bits >> 32),
                 static_cast<unsigned>(bits ^ 0x73412915u),
                 static_cast<unsigned>((bits >> 32) ^ 0xd439e7b3u)};
    return __builtin_bit_cast(T, words);
  }
}

template <class T>
LOOM_INLINE bool same_bits(T left, T right) {
  if constexpr (sizeof(T) > 16) {
    bool equal = true;
    for (unsigned component = 0; component < WidePayload<T>::count;
         ++component) {
      equal &= same_bits<typename WidePayload<T>::Element>(left[component],
                                                           right[component]);
    }
    return equal;
  } else if constexpr (__is_same(T, bool)) {
    return left == right;
  } else if constexpr (sizeof(T) == 1) {
    return __builtin_bit_cast(unsigned char, left) ==
           __builtin_bit_cast(unsigned char, right);
  } else if constexpr (sizeof(T) == 2) {
    return __builtin_bit_cast(unsigned short, left) ==
           __builtin_bit_cast(unsigned short, right);
  } else if constexpr (sizeof(T) == 4) {
    return __builtin_bit_cast(unsigned, left) ==
           __builtin_bit_cast(unsigned, right);
  } else if constexpr (sizeof(T) == 8) {
    return __builtin_bit_cast(unsigned long long, left) ==
           __builtin_bit_cast(unsigned long long, right);
  } else {
    Words4 a = __builtin_bit_cast(Words4, left);
    Words4 b = __builtin_bit_cast(Words4, right);
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
  }
}

#endif  // LOOM_IMPORT_CXX_BINDING_TEST_KERNEL_SHUFFLE_PAYLOADS_H_
