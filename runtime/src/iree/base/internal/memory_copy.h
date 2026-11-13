// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_MEMORY_COPY_H_
#define IREE_BASE_INTERNAL_MEMORY_COPY_H_

#include <assert.h>

#include "iree/base/alignment.h"
#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#if defined(IREE_ARCH_X86_64)
#include <emmintrin.h>
#endif  // IREE_ARCH_X86_64

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Copies |length| bytes between disjoint byte-addressable memory ranges.
// Each destination byte is written once, without destination loads or accesses
// outside either range. Access widths are unspecified and the copy is not
// atomic. Zero length permits null pointers. Ordinary stores preserve the
// mapping's cache policy; cache-line allocation and native device visibility
// still depend on that mapping and the caller's publication protocol.
static inline void iree_memcpy_write_only(void* IREE_RESTRICT target,
                                          const void* IREE_RESTRICT source,
                                          iree_host_size_t length) {
  uint8_t* target_bytes = (uint8_t*)target;
  const uint8_t* source_bytes = (const uint8_t*)source;
#if defined(IREE_ARCH_X86_64) && defined(IREE_COMPILER_GCC_COMPAT)
  // Volatile vector writes preserve each exact extent through optimization.
  for (; length >= 64; length -= 64, target_bytes += 64, source_bytes += 64) {
    const __m128i value0 = _mm_loadu_si128((const __m128i*)(source_bytes + 0));
    const __m128i value1 = _mm_loadu_si128((const __m128i*)(source_bytes + 16));
    const __m128i value2 = _mm_loadu_si128((const __m128i*)(source_bytes + 32));
    const __m128i value3 = _mm_loadu_si128((const __m128i*)(source_bytes + 48));
    *(volatile __m128i_u*)(target_bytes + 0) = value0;
    *(volatile __m128i_u*)(target_bytes + 16) = value1;
    *(volatile __m128i_u*)(target_bytes + 32) = value2;
    *(volatile __m128i_u*)(target_bytes + 48) = value3;
  }
  for (; length >= 16; length -= 16, target_bytes += 16, source_bytes += 16) {
    *(volatile __m128i_u*)target_bytes =
        _mm_loadu_si128((const __m128i*)source_bytes);
  }
  typedef uint64_t unaligned_uint64_t __attribute__((aligned(1), may_alias));
  typedef uint32_t unaligned_uint32_t __attribute__((aligned(1), may_alias));
  typedef uint16_t unaligned_uint16_t __attribute__((aligned(1), may_alias));
  if (length & 8) {
    *(volatile unaligned_uint64_t*)target_bytes =
        iree_unaligned_load_le_u64(source_bytes);
    target_bytes += 8;
    source_bytes += 8;
  }
  if (length & 4) {
    *(volatile unaligned_uint32_t*)target_bytes =
        iree_unaligned_load_le_u32(source_bytes);
    target_bytes += 4;
    source_bytes += 4;
  }
  if (length & 2) {
    *(volatile unaligned_uint16_t*)target_bytes =
        iree_unaligned_load_le_u16(source_bytes);
    target_bytes += 2;
    source_bytes += 2;
  }
  if (length & 1) {
    *(volatile uint8_t*)target_bytes = *source_bytes;
  }
#else
  for (iree_host_size_t i = 0; i < length; ++i) {
    ((volatile uint8_t*)target_bytes)[i] = source_bytes[i];
  }
#endif  // IREE_ARCH_X86_64 && IREE_COMPILER_GCC_COMPAT
}

#if defined(IREE_ARCH_X86_64)
// Non-temporal intrinsics may lower to advisory IR metadata that optimization
// can discard. GNU-compatible compilers use exact instructions because cache
// bypass is part of the publication contract, not a performance hint.
static inline void iree_memory_stream_store_u32(void* target, uint32_t value) {
#if defined(IREE_COMPILER_GCC_COMPAT) || defined(IREE_COMPILER_CLANG)
  __asm__ __volatile__("movnti %1, %0" : "=m"(*(uint32_t*)target) : "r"(value));
#else
  _mm_stream_si32((int*)target, (int)value);
#endif  // IREE_COMPILER_GCC_COMPAT || IREE_COMPILER_CLANG
}

static inline void iree_memory_stream_store_128(void* target, __m128i value) {
#if defined(IREE_COMPILER_GCC_COMPAT) || defined(IREE_COMPILER_CLANG)
  __asm__ __volatile__("movntdq %1, %0" : "=m"(*(__m128i*)target) : "x"(value));
#else
  _mm_stream_si128((__m128i*)target, value);
#endif  // IREE_COMPILER_GCC_COMPAT || IREE_COMPILER_CLANG
}
#endif  // IREE_ARCH_X86_64

// Copies an aligned DWORD extent between disjoint memory ranges. The target
// is four-byte aligned and length is a multiple of four; the source may have
// any alignment. Each destination byte is written once without destination
// reads or accesses outside either extent. On x86-64 every destination store
// is non-temporal. Other hosts use exact ordinary stores. Zero length permits
// null pointers. The caller owns cache visibility and the store fence before
// publishing a consumer; this routine performs neither operation.
static inline void iree_memcpy_stream_dst(void* IREE_RESTRICT target,
                                          const void* IREE_RESTRICT source,
                                          iree_host_size_t length) {
  assert((length & 3) == 0);
  uint8_t* target_bytes = (uint8_t*)target;
  const uint8_t* source_bytes = (const uint8_t*)source;
  IREE_BUILTIN_ASSUME_ALIGNED(target_bytes, 4);
#if defined(IREE_ARCH_X86_64)
  for (; ((uintptr_t)target_bytes & 15) && length;
       length -= 4, target_bytes += 4, source_bytes += 4) {
    iree_memory_stream_store_u32(target_bytes,
                                 iree_unaligned_load_le_u32(source_bytes));
  }
  for (; length >= 64; length -= 64, target_bytes += 64, source_bytes += 64) {
    const __m128i value0 = _mm_loadu_si128((const __m128i*)(source_bytes + 0));
    const __m128i value1 = _mm_loadu_si128((const __m128i*)(source_bytes + 16));
    const __m128i value2 = _mm_loadu_si128((const __m128i*)(source_bytes + 32));
    const __m128i value3 = _mm_loadu_si128((const __m128i*)(source_bytes + 48));
    iree_memory_stream_store_128(target_bytes + 0, value0);
    iree_memory_stream_store_128(target_bytes + 16, value1);
    iree_memory_stream_store_128(target_bytes + 32, value2);
    iree_memory_stream_store_128(target_bytes + 48, value3);
  }
  for (; length >= 16; length -= 16, target_bytes += 16, source_bytes += 16) {
    iree_memory_stream_store_128(target_bytes,
                                 _mm_loadu_si128((const __m128i*)source_bytes));
  }
  for (; length; length -= 4, target_bytes += 4, source_bytes += 4) {
    iree_memory_stream_store_u32(target_bytes,
                                 iree_unaligned_load_le_u32(source_bytes));
  }
#else
  iree_memcpy_write_only(target_bytes, source_bytes, length);
#endif  // IREE_ARCH_X86_64
}

// Fills an aligned DWORD extent with the same exact non-temporal destination
// and caller-owned publication contract as iree_memcpy_stream_dst.
static inline void iree_memset_stream_dst(void* target, uint8_t value,
                                          iree_host_size_t length) {
  assert((length & 3) == 0);
  uint8_t* target_bytes = (uint8_t*)target;
  IREE_BUILTIN_ASSUME_ALIGNED(target_bytes, 4);
#if defined(IREE_ARCH_X86_64)
  const uint32_t scalar = UINT32_C(0x01010101) * value;
  for (; ((uintptr_t)target_bytes & 15) && length;
       length -= 4, target_bytes += 4) {
    iree_memory_stream_store_u32(target_bytes, scalar);
  }
  const __m128i vector = _mm_set1_epi8((char)value);
  for (; length >= 16; length -= 16, target_bytes += 16) {
    iree_memory_stream_store_128(target_bytes, vector);
  }
  for (; length; length -= 4, target_bytes += 4) {
    iree_memory_stream_store_u32(target_bytes, scalar);
  }
#else
  for (iree_host_size_t i = 0; i < length; ++i) {
    ((volatile uint8_t*)target_bytes)[i] = value;
  }
#endif  // IREE_ARCH_X86_64
}

// Fills |length| bytes with |value| using the same exact destination-access and
// caller-owned publication contract as iree_memcpy_write_only.
static inline void iree_memset_write_only(void* target, uint8_t value,
                                          iree_host_size_t length) {
  uint8_t* target_bytes = (uint8_t*)target;
#if defined(IREE_ARCH_X86_64) && defined(IREE_COMPILER_GCC_COMPAT)
  const __m128i vector = _mm_set1_epi8((char)value);
  for (; length >= 64; length -= 64, target_bytes += 64) {
    *(volatile __m128i_u*)(target_bytes + 0) = vector;
    *(volatile __m128i_u*)(target_bytes + 16) = vector;
    *(volatile __m128i_u*)(target_bytes + 32) = vector;
    *(volatile __m128i_u*)(target_bytes + 48) = vector;
  }
  for (; length >= 16; length -= 16, target_bytes += 16) {
    *(volatile __m128i_u*)target_bytes = vector;
  }
  const uint64_t scalar = UINT64_C(0x0101010101010101) * value;
  typedef uint64_t unaligned_uint64_t __attribute__((aligned(1), may_alias));
  typedef uint32_t unaligned_uint32_t __attribute__((aligned(1), may_alias));
  typedef uint16_t unaligned_uint16_t __attribute__((aligned(1), may_alias));
  if (length & 8) {
    *(volatile unaligned_uint64_t*)target_bytes = scalar;
    target_bytes += 8;
  }
  if (length & 4) {
    *(volatile unaligned_uint32_t*)target_bytes = (uint32_t)scalar;
    target_bytes += 4;
  }
  if (length & 2) {
    *(volatile unaligned_uint16_t*)target_bytes = (uint16_t)scalar;
    target_bytes += 2;
  }
  if (length & 1) {
    *(volatile uint8_t*)target_bytes = value;
  }
#else
  for (iree_host_size_t i = 0; i < length; ++i) {
    ((volatile uint8_t*)target_bytes)[i] = value;
  }
#endif  // IREE_ARCH_X86_64 && IREE_COMPILER_GCC_COMPAT
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_MEMORY_COPY_H_
