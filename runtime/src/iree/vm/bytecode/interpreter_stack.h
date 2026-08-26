// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_STACK_H_
#define IREE_VM_BYTECODE_INTERPRETER_STACK_H_

#include <stdint.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core/selectors.h"

// Resolves one verified indexed local-byte access. Static verification has
// already proven |base| plus |access_length| fits the local byte array and
// |scale| is nonzero. Division folds the architectural u16 index ceiling and
// scaled-range requirement into one failure branch before any mutation.
static inline iree_status_t iree_vm_bytecode_stack_resolve_index(
    uint16_t local_byte_length, uint16_t base, uint8_t access_length,
    uint64_t index, uint8_t scale, uint16_t* out_effective_base) {
  const uint32_t available = (uint32_t)local_byte_length - access_length - base;
  if (IREE_UNLIKELY(index > available / scale)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "indexed stack access is out of range");
  }
  *out_effective_base = (uint16_t)(base + (uint32_t)index * scale);
  return iree_ok_status();
}

// Repeats the low |pattern_width| bytes of |pattern| across |target|. The
// caller handles an empty range before forming |target|.
static inline void iree_vm_bytecode_fill_pattern(uint8_t* target,
                                                 iree_host_size_t length,
                                                 uint64_t pattern,
                                                 uint8_t pattern_width) {
  if (pattern_width == 1) {
    memset(target, (uint8_t)pattern, length);
    return;
  }
  uint64_t expanded_pattern = pattern;
  if (pattern_width == 2) {
    expanded_pattern &= UINT64_C(0xFFFF);
    expanded_pattern |= expanded_pattern << 16;
    expanded_pattern |= expanded_pattern << 32;
  } else if (pattern_width == 4) {
    expanded_pattern &= UINT64_C(0xFFFFFFFF);
    expanded_pattern |= expanded_pattern << 32;
  }
  while (length >= sizeof(expanded_pattern)) {
    iree_unaligned_store_le_u64(target, expanded_pattern);
    target += sizeof(expanded_pattern);
    length -= sizeof(expanded_pattern);
  }
  if (length != 0) {
    uint8_t tail[sizeof(expanded_pattern)];
    iree_unaligned_store_le_u64(tail, expanded_pattern);
    memcpy(target, tail, length);
  }
}

// Fills |count| i32 stack cells from one sign-extended s16 immediate.
static inline void iree_vm_bytecode_stack_const_s16_i32(uint8_t* target,
                                                        uint16_t count,
                                                        int16_t immediate) {
  const uint32_t value = (uint32_t)(int32_t)immediate;
  for (uint16_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u32(target + i * sizeof(value), value);
  }
}

// Fills |count| i64 stack cells from one sign-extended s16 immediate.
static inline void iree_vm_bytecode_stack_const_s16_i64(uint8_t* target,
                                                        uint16_t count,
                                                        int16_t immediate) {
  const uint64_t value = (uint64_t)(int64_t)immediate;
  for (uint16_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u64(target + i * sizeof(value), value);
  }
}

// Packs |count| zero-extended u16 immediates into i32 stack cells.
static inline void iree_vm_bytecode_stack_pack_i32(uint8_t* target,
                                                   const uint16_t* immediates,
                                                   uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u32(target + i * sizeof(uint32_t), immediates[i]);
  }
}

// Packs |count| zero-extended u32 immediates into i64 stack cells.
static inline void iree_vm_bytecode_stack_pack_i64(uint8_t* target,
                                                   const uint32_t* immediates,
                                                   uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u64(target + i * sizeof(uint64_t), immediates[i]);
  }
}

// Loads one verified memory.format lane group from |source|. The caller has
// already proven the complete byte and register ranges. Keeping every format
// as a fixed-width leaf lets compilers emit straight-line unaligned loads after
// the one selector dispatch.
static inline void iree_vm_bytecode_stack_load_lanes(uint8_t format,
                                                     const uint8_t* source,
                                                     uint64_t* target_values) {
#define IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, byte_offset, value_offset) \
  target_values[value_offset] = load_fn(source + byte_offset)
#define IREE_VM_BYTECODE_STACK_LOAD_X1(load_fn, element_bytes) \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 0, 0)
#define IREE_VM_BYTECODE_STACK_LOAD_X2(load_fn, element_bytes) \
  IREE_VM_BYTECODE_STACK_LOAD_X1(load_fn, element_bytes);      \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, element_bytes, 1)
#define IREE_VM_BYTECODE_STACK_LOAD_X4(load_fn, element_bytes)     \
  IREE_VM_BYTECODE_STACK_LOAD_X2(load_fn, element_bytes);          \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 2 * element_bytes, 2); \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 3 * element_bytes, 3)
#define IREE_VM_BYTECODE_STACK_LOAD_X8(load_fn, element_bytes)     \
  IREE_VM_BYTECODE_STACK_LOAD_X4(load_fn, element_bytes);          \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 4 * element_bytes, 4); \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 5 * element_bytes, 5); \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 6 * element_bytes, 6); \
  IREE_VM_BYTECODE_STACK_LOAD_LANE(load_fn, 7 * element_bytes, 7)
  switch (format) {
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X1:
      IREE_VM_BYTECODE_STACK_LOAD_X1(iree_unaligned_load_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X2:
      IREE_VM_BYTECODE_STACK_LOAD_X2(iree_unaligned_load_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X4:
      IREE_VM_BYTECODE_STACK_LOAD_X4(iree_unaligned_load_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X8:
      IREE_VM_BYTECODE_STACK_LOAD_X8(iree_unaligned_load_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X1:
      IREE_VM_BYTECODE_STACK_LOAD_X1(iree_unaligned_load_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X2:
      IREE_VM_BYTECODE_STACK_LOAD_X2(iree_unaligned_load_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X4:
      IREE_VM_BYTECODE_STACK_LOAD_X4(iree_unaligned_load_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X8:
      IREE_VM_BYTECODE_STACK_LOAD_X8(iree_unaligned_load_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X1:
      IREE_VM_BYTECODE_STACK_LOAD_X1(iree_unaligned_load_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X2:
      IREE_VM_BYTECODE_STACK_LOAD_X2(iree_unaligned_load_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X4:
      IREE_VM_BYTECODE_STACK_LOAD_X4(iree_unaligned_load_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X8:
      IREE_VM_BYTECODE_STACK_LOAD_X8(iree_unaligned_load_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I64_X1:
      IREE_VM_BYTECODE_STACK_LOAD_X1(iree_unaligned_load_le_u64, 8);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I64_X2:
      IREE_VM_BYTECODE_STACK_LOAD_X2(iree_unaligned_load_le_u64, 8);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I64_X4:
      IREE_VM_BYTECODE_STACK_LOAD_X4(iree_unaligned_load_le_u64, 8);
      return;
    default:
      IREE_VM_BYTECODE_STACK_LOAD_X8(iree_unaligned_load_le_u64, 8);
      return;
  }
#undef IREE_VM_BYTECODE_STACK_LOAD_X8
#undef IREE_VM_BYTECODE_STACK_LOAD_X4
#undef IREE_VM_BYTECODE_STACK_LOAD_X2
#undef IREE_VM_BYTECODE_STACK_LOAD_X1
#undef IREE_VM_BYTECODE_STACK_LOAD_LANE
}

// Stores one verified memory.format lane group to |target|. The caller has
// already proven the complete byte and register ranges.
static inline void iree_vm_bytecode_stack_store_lanes(
    uint8_t format, const uint64_t* source_values, uint8_t* target) {
#define IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, byte_offset, value_offset) \
  store_fn(target + byte_offset, source_values[value_offset])
#define IREE_VM_BYTECODE_STACK_STORE_X1(store_fn, element_bytes) \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 0, 0)
#define IREE_VM_BYTECODE_STACK_STORE_X2(store_fn, element_bytes) \
  IREE_VM_BYTECODE_STACK_STORE_X1(store_fn, element_bytes);      \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, element_bytes, 1)
#define IREE_VM_BYTECODE_STACK_STORE_X4(store_fn, element_bytes)     \
  IREE_VM_BYTECODE_STACK_STORE_X2(store_fn, element_bytes);          \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 2 * element_bytes, 2); \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 3 * element_bytes, 3)
#define IREE_VM_BYTECODE_STACK_STORE_X8(store_fn, element_bytes)     \
  IREE_VM_BYTECODE_STACK_STORE_X4(store_fn, element_bytes);          \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 4 * element_bytes, 4); \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 5 * element_bytes, 5); \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 6 * element_bytes, 6); \
  IREE_VM_BYTECODE_STACK_STORE_LANE(store_fn, 7 * element_bytes, 7)
  switch (format) {
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X1:
      IREE_VM_BYTECODE_STACK_STORE_X1(iree_unaligned_store_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X2:
      IREE_VM_BYTECODE_STACK_STORE_X2(iree_unaligned_store_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X4:
      IREE_VM_BYTECODE_STACK_STORE_X4(iree_unaligned_store_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I8_X8:
      IREE_VM_BYTECODE_STACK_STORE_X8(iree_unaligned_store_le_u8, 1);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X1:
      IREE_VM_BYTECODE_STACK_STORE_X1(iree_unaligned_store_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X2:
      IREE_VM_BYTECODE_STACK_STORE_X2(iree_unaligned_store_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X4:
      IREE_VM_BYTECODE_STACK_STORE_X4(iree_unaligned_store_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I16_X8:
      IREE_VM_BYTECODE_STACK_STORE_X8(iree_unaligned_store_le_u16, 2);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X1:
      IREE_VM_BYTECODE_STACK_STORE_X1(iree_unaligned_store_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X2:
      IREE_VM_BYTECODE_STACK_STORE_X2(iree_unaligned_store_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X4:
      IREE_VM_BYTECODE_STACK_STORE_X4(iree_unaligned_store_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I32_X8:
      IREE_VM_BYTECODE_STACK_STORE_X8(iree_unaligned_store_le_u32, 4);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I64_X1:
      IREE_VM_BYTECODE_STACK_STORE_X1(iree_unaligned_store_le_u64, 8);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I64_X2:
      IREE_VM_BYTECODE_STACK_STORE_X2(iree_unaligned_store_le_u64, 8);
      return;
    case IREE_VM_ISA_MEMORY_FORMAT_I64_X4:
      IREE_VM_BYTECODE_STACK_STORE_X4(iree_unaligned_store_le_u64, 8);
      return;
    default:
      IREE_VM_BYTECODE_STACK_STORE_X8(iree_unaligned_store_le_u64, 8);
      return;
  }
#undef IREE_VM_BYTECODE_STACK_STORE_X8
#undef IREE_VM_BYTECODE_STACK_STORE_X4
#undef IREE_VM_BYTECODE_STACK_STORE_X2
#undef IREE_VM_BYTECODE_STACK_STORE_X1
#undef IREE_VM_BYTECODE_STACK_STORE_LANE
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_STACK_H_
