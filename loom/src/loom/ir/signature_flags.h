// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IR_SIGNATURE_FLAGS_H_
#define LOOM_IR_SIGNATURE_FLAGS_H_

#include "loom/ir/argument_access.h"
#include "loom/ir/attribute.h"

// One source signature value's explicit boundary contract. Signature storage
// contains arguments followed by results, matching the function type. Zero
// leaves every property unspecified. This is authored semantics, not inferred
// facts. The current access subfield applies only to buffer/view arguments;
// result words and all other bits must be zero.
typedef uint8_t loom_signature_flags_t;

enum loom_signature_flags_mask_e {
  LOOM_SIGNATURE_FLAGS_ACCESS_MASK = 0x07,
};

static inline loom_argument_access_t loom_signature_flags_access(
    loom_signature_flags_t flags) {
  return (loom_argument_access_t)(flags & LOOM_SIGNATURE_FLAGS_ACCESS_MASK);
}

// Public signature admission checks the word before analysis consumes it.
static inline bool loom_signature_flags_argument_is_valid(
    loom_signature_flags_t flags) {
  return flags <= LOOM_ARGUMENT_ACCESS_NONE;
}

static inline bool loom_signature_flags_result_is_valid(
    loom_signature_flags_t flags) {
  return flags == 0;
}

// Combines the uses of two arguments that the signature owner has established
// always receive the same value. An unspecified bound permits escape; otherwise
// the combined argument must permit every access from either original.
static inline loom_signature_flags_t loom_signature_flags_join_arguments(
    loom_signature_flags_t lhs, loom_signature_flags_t rhs) {
  if (!lhs || !rhs) {
    return 0;
  }
  const uint8_t access = (lhs | rhs) & LOOM_ARGUMENT_ACCESS_READ_WRITE;
  return access ? access : LOOM_ARGUMENT_ACCESS_NONE;
}

// Signature flags use the byte attribute's 32-bit length: the sum of two
// independently valid 16-bit arities can exceed UINT16_MAX. Payloads are
// immutable module storage and may be shared by cloned operations.
static inline loom_attribute_t loom_attr_signature_flags(
    const loom_signature_flags_t* values, uint32_t count) {
  return loom_attr_bytes(values, count);
}

#endif  // LOOM_IR_SIGNATURE_FLAGS_H_
