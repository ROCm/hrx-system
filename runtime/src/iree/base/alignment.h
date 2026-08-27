// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation of the primitives from stdalign.h used for cross-target
// value alignment specification and queries.

#ifndef IREE_BASE_ALIGNMENT_H_
#define IREE_BASE_ALIGNMENT_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// IREE_PTR_SIZE_*
//==============================================================================

// Verify that the pointer size of the machine matches the expectation that
// uintptr_t can round-trip the value. This isn't a common issue unless the
// toolchain is doing weird things.
// See https://stackoverflow.com/q/51616057.
static_assert(sizeof(void*) == sizeof(uintptr_t),
              "can't determine pointer size");

#if UINTPTR_MAX == 0xFFFFFFFF
#define IREE_PTR_SIZE_32
#define IREE_PTR_SIZE 4
#elif UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFu
#define IREE_PTR_SIZE_64
#define IREE_PTR_SIZE 8
#else
#error "can't determine pointer size"
#endif

//===----------------------------------------------------------------------===//
// Alignment utilities
//===----------------------------------------------------------------------===//

// Returns the number of elements in an array as a compile-time constant, which
// can be used in defining new arrays. Fails at compile-time if |arr| is not a
// static array (such as if used on a pointer type). Similar to `countof()`.
//
// Example:
//  uint8_t kConstantArray[512];
//  assert(IREE_ARRAYSIZE(kConstantArray) == 512);
#define IREE_ARRAYSIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define iree_min(lhs, rhs) ((lhs) <= (rhs) ? (lhs) : (rhs))
#define iree_max(lhs, rhs) ((lhs) <= (rhs) ? (rhs) : (lhs))

// https://en.cppreference.com/w/c/types/max_align_t
#if defined(IREE_PLATFORM_WINDOWS)
// NOTE: 16 is a specified Microsoft API requirement for some functions.
#define iree_max_align_t 16
#else
#define iree_max_align_t sizeof(long double)
#endif  // IREE_PLATFORM_*

// https://en.cppreference.com/w/c/language/_Alignas
// https://en.cppreference.com/w/c/language/_Alignof
#if defined(IREE_COMPILER_MSVC)
#define iree_alignas(x) __declspec(align(x))
#define iree_alignof(x) __alignof(x)
#else
#define iree_alignas(x) __attribute__((__aligned__(x)))
#define iree_alignof(x) __alignof__(x)
#endif  // IREE_COMPILER_*

// Container-of macro for getting struct from embedded member.
#define iree_containerof(ptr, type, member) \
  ((type*)((char*)(ptr) - offsetof(type, member)))

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline iree_host_size_t iree_host_align(iree_host_size_t value,
                                               iree_host_size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns true if |value| is a power-of-two.
static inline bool iree_host_size_is_power_of_two(iree_host_size_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// Returns true if |alignment| is valid for aligned allocation functions.
// Valid alignments are either 0 (use default) or a power of two.
static inline bool iree_host_size_is_valid_alignment(
    iree_host_size_t alignment) {
  return alignment == 0 || iree_host_size_is_power_of_two(alignment);
}

// Returns true if |value| matches the given minimum |alignment|.
static inline bool iree_host_size_has_alignment(iree_host_size_t value,
                                                iree_host_size_t alignment) {
  return iree_host_align(value, alignment) == value;
}

// Returns true if |ptr| meets the given minimum |alignment|.
static inline bool iree_host_ptr_has_alignment(const void* ptr,
                                               iree_host_size_t alignment) {
  return iree_host_size_has_alignment((iree_host_size_t)(uintptr_t)ptr,
                                      alignment);
}

// Returns the smallest power of two >= |value|.
// Returns 1 for |value| of 0 or 1.
// Saturates to maximum representable value on overflow.
static inline iree_host_size_t iree_host_size_next_power_of_two(
    iree_host_size_t value) {
  if (value <= 1) return 1;
  // Check for overflow: max representable power-of-two is SIZE_MAX/2 + 1.
  // Any value larger than that cannot have a next power-of-two in this type.
  if (value > ((~(iree_host_size_t)0) >> 1) + 1) {
    return ~(iree_host_size_t)0;  // Saturate to maximum value.
  }
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  if (sizeof(iree_host_size_t) == 8) {
    value |= value >> 32;
  }
  return value + 1;
}

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline iree_device_size_t iree_device_align(
    iree_device_size_t value, iree_device_size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns true if |value| is a power-of-two.
static inline bool iree_device_size_is_power_of_two(iree_device_size_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// Returns true if |alignment| is valid for aligned allocation functions.
// Valid alignments are either 0 (use default) or a power of two.
static inline bool iree_device_size_is_valid_alignment(
    iree_device_size_t alignment) {
  return alignment == 0 || iree_device_size_is_power_of_two(alignment);
}

// Returns true if |value| matches the given minimum |alignment|.
static inline bool iree_device_size_has_alignment(
    iree_device_size_t value, iree_device_size_t alignment) {
  return iree_device_align(value, alignment) == value;
}

// Returns the smallest power of two >= |value|.
// Returns 1 for |value| of 0 or 1.
// Saturates to maximum representable value on overflow.
static inline iree_device_size_t iree_device_size_next_power_of_two(
    iree_device_size_t value) {
  if (value <= 1) return 1;
  // Check for overflow: max representable power-of-two is SIZE_MAX/2 + 1.
  // Any value larger than that cannot have a next power-of-two in this type.
  if (value > ((~(iree_device_size_t)0) >> 1) + 1) {
    return ~(iree_device_size_t)0;  // Saturate to maximum value.
  }
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  if (sizeof(iree_device_size_t) == 8) {
    value |= value >> 32;
  }
  return value + 1;
}

// Returns true if |value| is a power-of-two.
static inline bool iree_is_power_of_two_uint64(uint64_t value) {
  return (value != 0) && ((value & (value - 1)) == 0);
}

// TODO(benvanik): when C23 is fully adopted we can make a single generic
// version of the alignment functions that uses typeof to cast back to the
// expected result. For now we explicitly spell out the variants we want.

// Aligns |value| up to the given power-of-two |alignment| if required.
// https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
static inline uint64_t iree_align_uint64(uint64_t value, uint64_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// Returns the size of a struct padded out to iree_max_align_t.
// This must be used when performing manual trailing allocation packing to
// ensure the alignment requirements of the trailing data are satisfied.
//
// NOTE: do not use this if using VLAs (`struct { int trailing[]; }`) - those
// must precisely follow the normal sizeof(t) as the compiler does the padding
// for you.
//
// Example:
//  some_buffer_ptr_t* p = NULL;
//  iree_host_size_t total_size = iree_sizeof_struct(*buffer) + extra_data_size;
//  IREE_CHECK_OK(iree_allocator_malloc(allocator, total_size, (void**)&p));
#define iree_sizeof_struct(t) iree_host_align(sizeof(t), iree_max_align_t)

// Returns the ceil-divide of |lhs| by non-zero |rhs|.
static inline iree_host_size_t iree_host_size_ceil_div(iree_host_size_t lhs,
                                                       iree_host_size_t rhs) {
  return ((lhs != 0) && (lhs > 0) == (rhs > 0))
             ? ((lhs + ((rhs > 0) ? -1 : 1)) / rhs) + 1
             : -(-lhs / rhs);
}

// Returns the floor-divide of |lhs| by non-zero |rhs|.
static inline iree_host_size_t iree_host_size_floor_div(iree_host_size_t lhs,
                                                        iree_host_size_t rhs) {
  return ((lhs != 0) && ((lhs < 0) != (rhs < 0)))
             ? -((-lhs + ((rhs < 0) ? 1 : -1)) / rhs) - 1
             : lhs / rhs;
}

// Returns the ceil-divide of |lhs| by non-zero |rhs|.
static inline iree_device_size_t iree_device_size_ceil_div(
    iree_device_size_t lhs, iree_device_size_t rhs) {
  return ((lhs != 0) && (lhs > 0) == (rhs > 0))
             ? ((lhs + ((rhs > 0) ? -1 : 1)) / rhs) + 1
             : -(-lhs / rhs);
}

// Returns the floor-divide of |lhs| by non-zero |rhs|.
static inline iree_device_size_t iree_device_size_floor_div(
    iree_device_size_t lhs, iree_device_size_t rhs) {
  return ((lhs != 0) && ((lhs < 0) != (rhs < 0)))
             ? -((-lhs + ((rhs < 0) ? 1 : -1)) / rhs) - 1
             : lhs / rhs;
}

// Returns the greatest common divisor between two values.
//
// See: https://en.wikipedia.org/wiki/Greatest_common_divisor
//
// Examples:
//  gcd(8, 16) = 8
//  gcd(3, 5) = 1
static inline iree_device_size_t iree_device_size_gcd(iree_device_size_t a,
                                                      iree_device_size_t b) {
  if (b == 0) return a;
  return iree_device_size_gcd(b, a % b);
}

// Returns the least common multiple between two values, often used for
// finding a common alignment.
//
// See: https://en.wikipedia.org/wiki/Least_common_multiple
//
// Examples:
//  lcm(8, 16) = 16
//  lcm(3, 5) = 15 (15 % 3 == 0, 15 % 5 == 0)
static inline iree_device_size_t iree_device_size_lcm(iree_device_size_t a,
                                                      iree_device_size_t b) {
  return a * (b / iree_device_size_gcd(a, b));
}

//===----------------------------------------------------------------------===//
// Byte and page range manipulation
//===----------------------------------------------------------------------===//

// Defines a range of bytes with any arbitrary alignment.
// Most operations will adjust this range by the allocation granularity, meaning
// that a range that straddles a page boundary will be specifying multiple pages
// (such as offset=1, length=4096 with a page size of 4096 indicating 2 pages).
typedef struct iree_byte_range_t {
  iree_host_size_t offset;
  iree_host_size_t length;
} iree_byte_range_t;

// Defines a range of bytes with page-appropriate alignment.
// Any operation taking page ranges requires that the offset and length respect
// the size and granularity requirements of the page mode the memory was defined
// with. For example, if an allocation is using large pages then both offset and
// length must be multiples of the iree_memory_info_t::large_page_granularity.
typedef struct iree_page_range_t {
  iree_host_size_t offset;
  iree_host_size_t length;
} iree_page_range_t;

// Aligns |addr| up to |page_alignment|.
static inline uintptr_t iree_page_align_start(uintptr_t addr,
                                              iree_host_size_t page_alignment) {
  return addr & (~(page_alignment - 1));
}

// Aligns |addr| down to |page_alignment|.
static inline uintptr_t iree_page_align_end(uintptr_t addr,
                                            iree_host_size_t page_alignment) {
  return iree_page_align_start(addr + (page_alignment - 1), page_alignment);
}

// Unions two page ranges to create the min/max extents of both.
static inline iree_page_range_t iree_page_range_union(
    const iree_page_range_t a, const iree_page_range_t b) {
  iree_host_size_t start = iree_min(a.offset, b.offset);
  iree_host_size_t end = iree_max(a.offset + a.length, b.offset + b.length);
  iree_page_range_t page_range = {0};
  page_range.offset = start;
  page_range.length = end - start;
  return page_range;
}

// Aligns a byte range to page boundaries defined by |page_alignment|.
static inline iree_page_range_t iree_align_byte_range_to_pages(
    const iree_byte_range_t byte_range, iree_host_size_t page_alignment) {
  iree_page_range_t page_range = {0};
  page_range.offset = iree_host_align(byte_range.offset, page_alignment);
  page_range.length = iree_host_align(byte_range.length, page_alignment);
  return page_range;
}

// Computes a page-aligned range base and total length from a range.
// This will produce a starting address <= the range offset and a length >=
// the range length.
static inline void iree_page_align_range(void* base_address,
                                         iree_byte_range_t range,
                                         iree_host_size_t page_alignment,
                                         void** out_start_address,
                                         iree_host_size_t* out_aligned_length) {
  void* range_start = (void*)iree_page_align_start(
      (uintptr_t)base_address + range.offset, page_alignment);
  void* range_end = (void*)iree_page_align_end(
      (uintptr_t)base_address + range.offset + range.length, page_alignment);
  *out_start_address = range_start;
  *out_aligned_length =
      (iree_host_size_t)range_end - (iree_host_size_t)range_start;
}

//===----------------------------------------------------------------------===//
// Alignment intrinsics
//===----------------------------------------------------------------------===//

#if IREE_HAVE_BUILTIN(__builtin_unreachable) || defined(__GNUC__)
#define IREE_BUILTIN_UNREACHABLE() __builtin_unreachable()
#elif defined(IREE_COMPILER_MSVC)
#define IREE_BUILTIN_UNREACHABLE() __assume(false)
#else
#define IREE_BUILTIN_UNREACHABLE() ((void)0)
#endif  // IREE_HAVE_BUILTIN(__builtin_unreachable) || defined(__GNUC__)

#if !defined(__cplusplus)
#define IREE_DECLTYPE(v) __typeof__(v)
#else
#define IREE_DECLTYPE(v) decltype(v)
#endif  // __cplusplus

#if IREE_HAVE_BUILTIN(__builtin_assume_aligned) || defined(__GNUC__)
// NOTE: gcc only assumes on the result so we have to reset ptr.
#define IREE_BUILTIN_ASSUME_ALIGNED_IMPL(ptr, size) \
  (ptr = (IREE_DECLTYPE(ptr))(__builtin_assume_aligned((void*)(ptr), (size))))
#elif 0  // defined(IREE_COMPILER_MSVC)
#define IREE_BUILTIN_ASSUME_ALIGNED_IMPL(ptr, size) \
  (__assume((((uintptr_t)(ptr)) & ((1 << (size))) - 1)) == 0)
#else
#define IREE_BUILTIN_ASSUME_ALIGNED_IMPL(ptr, size) \
  ((((uintptr_t)(ptr) % (size)) == 0) ? (ptr)       \
                                      : (IREE_BUILTIN_UNREACHABLE(), (ptr)))
#endif  // IREE_HAVE_BUILTIN(__builtin_assume_aligned) || defined(__GNUC__)

#define IREE_BUILTIN_ASSUME_ALIGNED(ptr, size)   \
  do {                                           \
    assert((uintptr_t)(ptr) % (size) == 0);      \
    IREE_BUILTIN_ASSUME_ALIGNED_IMPL(ptr, size); \
  } while (false)

//===----------------------------------------------------------------------===//
// Alignment-safe memory accesses
//===----------------------------------------------------------------------===//

static_assert(sizeof(float) == sizeof(uint32_t),
              "f32 accesses require a 32-bit float");
static_assert(sizeof(double) == sizeof(uint64_t),
              "f64 accesses require a 64-bit double");

// Converts between host and little-endian byte order. The conversion is its
// own inverse and is used by both loads and stores.
static inline uint16_t iree_unaligned_convert_native_le_u16(uint16_t value) {
#if defined(IREE_ENDIANNESS_BIG)
  return (uint16_t)((value >> 8) | (value << 8));
#else
  return value;
#endif  // IREE_ENDIANNESS_*
}

static inline uint32_t iree_unaligned_convert_native_le_u32(uint32_t value) {
#if defined(IREE_ENDIANNESS_BIG)
  return ((value & UINT32_C(0x000000FF)) << 24) |
         ((value & UINT32_C(0x0000FF00)) << 8) |
         ((value & UINT32_C(0x00FF0000)) >> 8) |
         ((value & UINT32_C(0xFF000000)) >> 24);
#else
  return value;
#endif  // IREE_ENDIANNESS_*
}

static inline uint64_t iree_unaligned_convert_native_le_u64(uint64_t value) {
#if defined(IREE_ENDIANNESS_BIG)
  value = ((value & UINT64_C(0x00000000FFFFFFFF)) << 32) |
          ((value & UINT64_C(0xFFFFFFFF00000000)) >> 32);
  value = ((value & UINT64_C(0x0000FFFF0000FFFF)) << 16) |
          ((value & UINT64_C(0xFFFF0000FFFF0000)) >> 16);
  return ((value & UINT64_C(0x00FF00FF00FF00FF)) << 8) |
         ((value & UINT64_C(0xFF00FF00FF00FF00)) >> 8);
#else
  return value;
#endif  // IREE_ENDIANNESS_*
}

// Loads little-endian scalar values from byte addresses that may have any
// alignment. Fixed-size memcpy calls lower to native unaligned accesses when
// the target supports them.
static inline uint8_t iree_unaligned_load_le_u8(const void* ptr) {
  uint8_t value;
  memcpy(&value, ptr, sizeof(value));
  return value;
}

static inline uint16_t iree_unaligned_load_le_u16(const void* ptr) {
  uint16_t value;
  memcpy(&value, ptr, sizeof(value));
  return iree_unaligned_convert_native_le_u16(value);
}

static inline uint32_t iree_unaligned_load_le_u32(const void* ptr) {
  uint32_t value;
  memcpy(&value, ptr, sizeof(value));
  return iree_unaligned_convert_native_le_u32(value);
}

static inline uint64_t iree_unaligned_load_le_u64(const void* ptr) {
  uint64_t value;
  memcpy(&value, ptr, sizeof(value));
  return iree_unaligned_convert_native_le_u64(value);
}

static inline float iree_unaligned_load_le_f32(const void* ptr) {
  const uint32_t value_bits = iree_unaligned_load_le_u32(ptr);
  float value;
  memcpy(&value, &value_bits, sizeof(value));
  return value;
}

static inline double iree_unaligned_load_le_f64(const void* ptr) {
  const uint64_t value_bits = iree_unaligned_load_le_u64(ptr);
  double value;
  memcpy(&value, &value_bits, sizeof(value));
  return value;
}

// Stores little-endian scalar values to byte addresses that may have any
// alignment. Fixed-size memcpy calls lower to native unaligned accesses when
// the target supports them.
static inline void iree_unaligned_store_le_u8(void* ptr, uint8_t value) {
  memcpy(ptr, &value, sizeof(value));
}

static inline void iree_unaligned_store_le_u16(void* ptr, uint16_t value) {
  value = iree_unaligned_convert_native_le_u16(value);
  memcpy(ptr, &value, sizeof(value));
}

static inline void iree_unaligned_store_le_u32(void* ptr, uint32_t value) {
  value = iree_unaligned_convert_native_le_u32(value);
  memcpy(ptr, &value, sizeof(value));
}

static inline void iree_unaligned_store_le_u64(void* ptr, uint64_t value) {
  value = iree_unaligned_convert_native_le_u64(value);
  memcpy(ptr, &value, sizeof(value));
}

static inline void iree_unaligned_store_le_f32(void* ptr, float value) {
  uint32_t value_bits;
  memcpy(&value_bits, &value, sizeof(value_bits));
  iree_unaligned_store_le_u32(ptr, value_bits);
}

static inline void iree_unaligned_store_le_f64(void* ptr, double value) {
  uint64_t value_bits;
  memcpy(&value_bits, &value, sizeof(value_bits));
  iree_unaligned_store_le_u64(ptr, value_bits);
}

// clang-format off

// Dereferences |ptr| and returns the value.
// Automatically handles unaligned accesses on architectures that may not
// support them natively (or efficiently). Memory is treated as little-endian.
#if defined(__cplusplus)

extern "C++" {

static inline uint8_t iree_unaligned_load_le(const int8_t* ptr) {
  return iree_unaligned_load_le_u8(ptr);
}
static inline uint8_t iree_unaligned_load_le(const uint8_t* ptr) {
  return iree_unaligned_load_le_u8(ptr);
}
static inline uint16_t iree_unaligned_load_le(const int16_t* ptr) {
  return iree_unaligned_load_le_u16(ptr);
}
static inline uint16_t iree_unaligned_load_le(const uint16_t* ptr) {
  return iree_unaligned_load_le_u16(ptr);
}
static inline uint32_t iree_unaligned_load_le(const int32_t* ptr) {
  return iree_unaligned_load_le_u32(ptr);
}
static inline uint32_t iree_unaligned_load_le(const uint32_t* ptr) {
  return iree_unaligned_load_le_u32(ptr);
}
static inline uint64_t iree_unaligned_load_le(const int64_t* ptr) {
  return iree_unaligned_load_le_u64(ptr);
}
static inline uint64_t iree_unaligned_load_le(const uint64_t* ptr) {
  return iree_unaligned_load_le_u64(ptr);
}
static inline float iree_unaligned_load_le(const float* ptr) {
  return iree_unaligned_load_le_f32(ptr);
}
static inline double iree_unaligned_load_le(const double* ptr) {
  return iree_unaligned_load_le_f64(ptr);
}

}  // extern "C++"

#else

#define iree_unaligned_load_le(ptr)                                  \
  _Generic((ptr),                                                    \
        int8_t*: iree_unaligned_load_le_u8((ptr)),                  \
       uint8_t*: iree_unaligned_load_le_u8((ptr)),                  \
       int16_t*: iree_unaligned_load_le_u16((ptr)),                 \
      uint16_t*: iree_unaligned_load_le_u16((ptr)),                 \
       int32_t*: iree_unaligned_load_le_u32((ptr)),                 \
      uint32_t*: iree_unaligned_load_le_u32((ptr)),                 \
       int64_t*: iree_unaligned_load_le_u64((ptr)),                 \
      uint64_t*: iree_unaligned_load_le_u64((ptr)),                 \
         float*: iree_unaligned_load_le_f32((ptr)),                 \
        double*: iree_unaligned_load_le_f64((ptr)),                 \
  const int8_t*: iree_unaligned_load_le_u8((ptr)),                  \
 const uint8_t*: iree_unaligned_load_le_u8((ptr)),                  \
 const int16_t*: iree_unaligned_load_le_u16((ptr)),                 \
const uint16_t*: iree_unaligned_load_le_u16((ptr)),                 \
 const int32_t*: iree_unaligned_load_le_u32((ptr)),                 \
const uint32_t*: iree_unaligned_load_le_u32((ptr)),                 \
 const int64_t*: iree_unaligned_load_le_u64((ptr)),                 \
const uint64_t*: iree_unaligned_load_le_u64((ptr)),                 \
   const float*: iree_unaligned_load_le_f32((ptr)),                 \
  const double*: iree_unaligned_load_le_f64((ptr))                  \
  )

#endif  // defined(__cplusplus)

// Dereferences |ptr| and writes the given |value|.
// Automatically handles unaligned accesses on architectures that may not
// support them natively (or efficiently). Memory is treated as little-endian.
#if defined(__cplusplus)

extern "C++" {

static inline void iree_unaligned_store_le(int8_t* ptr, uint8_t value) {
  iree_unaligned_store_le_u8(ptr, value);
}
static inline void iree_unaligned_store_le(uint8_t* ptr, uint8_t value) {
  iree_unaligned_store_le_u8(ptr, value);
}
static inline void iree_unaligned_store_le(int16_t* ptr, uint16_t value) {
  iree_unaligned_store_le_u16(ptr, value);
}
static inline void iree_unaligned_store_le(uint16_t* ptr, uint16_t value) {
  iree_unaligned_store_le_u16(ptr, value);
}
static inline void iree_unaligned_store_le(int32_t* ptr, uint32_t value) {
  iree_unaligned_store_le_u32(ptr, value);
}
static inline void iree_unaligned_store_le(uint32_t* ptr, uint32_t value) {
  iree_unaligned_store_le_u32(ptr, value);
}
static inline void iree_unaligned_store_le(int64_t* ptr, uint64_t value) {
  iree_unaligned_store_le_u64(ptr, value);
}
static inline void iree_unaligned_store_le(uint64_t* ptr, uint64_t value) {
  iree_unaligned_store_le_u64(ptr, value);
}
static inline void iree_unaligned_store_le(float* ptr, float value) {
  iree_unaligned_store_le_f32(ptr, value);
}
static inline void iree_unaligned_store_le(double* ptr, double value) {
  iree_unaligned_store_le_f64(ptr, value);
}

}  // extern "C++"

#else

#define iree_unaligned_store_le(ptr, value)                         \
  _Generic((ptr),                                                   \
        int8_t*: iree_unaligned_store_le_u8((ptr), (value)),       \
       uint8_t*: iree_unaligned_store_le_u8((ptr), (value)),       \
       int16_t*: iree_unaligned_store_le_u16((ptr), (value)),      \
      uint16_t*: iree_unaligned_store_le_u16((ptr), (value)),      \
       int32_t*: iree_unaligned_store_le_u32((ptr), (value)),      \
      uint32_t*: iree_unaligned_store_le_u32((ptr), (value)),      \
       int64_t*: iree_unaligned_store_le_u64((ptr), (value)),      \
      uint64_t*: iree_unaligned_store_le_u64((ptr), (value)),      \
         float*: iree_unaligned_store_le_f32((ptr), (value)),      \
        double*: iree_unaligned_store_le_f64((ptr), (value))       \
  )

#endif  // defined(__cplusplus)

// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_ALIGNMENT_H_
