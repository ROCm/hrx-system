// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target register payload helpers.
//
// Core IR owns only the by-value type storage and structural equality. The
// target-low layer owns the interpretation of LOOM_TYPE_REGISTER payload
// fields:
//   dims[0]: low descriptor-set stable ID.
//   dims[1]: descriptor-local register class ID, contiguous unit count.

#ifndef LOOM_TARGET_REGISTERS_H_
#define LOOM_TARGET_REGISTERS_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for an absent descriptor-local register-class ID.
#define LOOM_LOW_REGISTER_CLASS_ID_INVALID UINT16_MAX

// Returns true when |type| carries a target-low register payload.
static inline bool loom_low_type_is_register(loom_type_t type) {
  return loom_type_is_register(type);
}

// Packs a descriptor-local register class and unit count into a payload word.
static inline uint64_t loom_low_register_type_pack_payload1(
    uint16_t register_class_id, uint32_t unit_count) {
  IREE_ASSERT(unit_count > 0, "register unit count must be non-zero");
  return (uint64_t)register_class_id | ((uint64_t)unit_count << 16);
}

// Creates a target-low register type from compact descriptor identity.
// |register_class_id| is descriptor-set-local. |unit_count| is the number of
// contiguous register-class units carried by the value.
static inline loom_type_t loom_low_register_type(
    uint64_t descriptor_set_stable_id, uint16_t register_class_id,
    uint32_t unit_count) {
  return loom_type_register_payload(
      descriptor_set_stable_id,
      loom_low_register_type_pack_payload1(register_class_id, unit_count));
}

// Returns the low descriptor-set stable ID stored in |type|.
static inline uint64_t loom_low_register_type_descriptor_set_stable_id(
    loom_type_t type) {
  return loom_type_register_payload0(type);
}

// Returns the descriptor-set-local register-class ID stored in |type|.
static inline uint16_t loom_low_register_type_class_id(loom_type_t type) {
  return (uint16_t)(loom_type_register_payload1(type) & 0xFFFFu);
}

// Returns the number of register-class units carried by |type|.
static inline uint32_t loom_low_register_type_unit_count(loom_type_t type) {
  return (uint32_t)((loom_type_register_payload1(type) >> 16) & 0xFFFFFFFFu);
}

// Returns a register type with the same descriptor/class identity as |type|
// and a different unit count.
static inline loom_type_t loom_low_register_type_with_unit_count(
    loom_type_t type, uint32_t unit_count) {
  return loom_low_register_type(
      loom_low_register_type_descriptor_set_stable_id(type),
      loom_low_register_type_class_id(type), unit_count);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REGISTERS_H_
