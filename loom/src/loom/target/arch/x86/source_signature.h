// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86-64 SysV classification for complete verified C source objects.

#ifndef LOOM_TARGET_ARCH_X86_SOURCE_SIGNATURE_H_
#define LOOM_TARGET_ARCH_X86_SOURCE_SIGNATURE_H_

#include "loom/target/abi/c/signature_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of eightbytes carried by one native vector register.
#define LOOM_X86_SOURCE_SIGNATURE_MAX_EIGHTBYTE_COUNT 8

// SysV AMD64 classes used by the supported C object family. MEMORY is retained
// explicitly while x87 and complex classes are rejected by the source
// signature.
typedef uint8_t loom_x86_source_signature_class_t;
enum loom_x86_source_signature_class_e {
  LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS = 0,
  LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER = 1,
  LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE = 2,
  LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP = 3,
  LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY = 4,
};
static_assert(LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY <= UINT8_MAX,
              "x86 source signature classes must fit in one byte");

// Complete classification of one root C source object.
typedef struct loom_x86_source_signature_classification_t {
  // Exact source object footprint in bytes.
  uint32_t byte_length;
  // Required source object alignment in bytes.
  uint32_t byte_alignment;
  // Number of flattened Loom values projected from this object.
  uint16_t component_count;
  // Number of populated entries in |eightbytes| for a direct object. Memory
  // objects have zero direct eightbytes regardless of their footprint.
  uint8_t eightbyte_count;
  // True when the complete object crosses the ABI through memory.
  bool is_memory;
  // Direct object classes in increasing byte-offset order.
  loom_x86_source_signature_class_t
      eightbytes[LOOM_X86_SOURCE_SIGNATURE_MAX_EIGHTBYTE_COUNT];
} loom_x86_source_signature_classification_t;
static_assert(sizeof(loom_x86_source_signature_classification_t) == 20,
              "x86 source signature classification must be exactly 20 bytes");

// Classifies one normalized root C object. |maximum_vector_bytes| is 16, 32,
// or 64 when the selected profile can transport the corresponding native
// vector class, and zero when it exposes no SIMD boundary registers. Returns
// false only for source families deliberately outside the supported x86
// interop contract. ABI-mandated stack objects return true with is_memory.
bool loom_x86_source_signature_classify(
    const loom_c_abi_signature_layout_t* layout,
    const loom_c_abi_object_t* object, const loom_c_abi_leaf_t* object_leaves,
    uint16_t maximum_vector_bytes,
    loom_x86_source_signature_classification_t* out_classification);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_SOURCE_SIGNATURE_H_
