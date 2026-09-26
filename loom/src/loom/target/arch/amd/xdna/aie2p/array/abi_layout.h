// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dense external binding-table shape for AIE2P array programs.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ABI_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ABI_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/schemas/xdna_executable.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Maximum binding-table cardinality representable by the XDNA image ABI.
  LOOM_AIE2P_ARRAY_MAX_BINDING_SLOT_COUNT =
      IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT,
};

// Typed interpretation of the array-program low.func.def abi_layout.
typedef struct loom_aie2p_array_abi_layout_t {
  // Exact dense binding-table cardinality when explicitly authored.
  uint32_t binding_slot_count;
  // Whether binding_slot_count is explicit instead of topology-derived.
  bool has_binding_slot_count;
} loom_aie2p_array_abi_layout_t;

// Kind of malformed authored array-program ABI layout.
typedef enum loom_aie2p_array_abi_layout_issue_kind_e {
  LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_NONE = 0,
  LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_UNEXPECTED_FIELD = 1,
  LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_BINDING_COUNT_KIND = 2,
  LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_BINDING_COUNT_RANGE = 3,
} loom_aie2p_array_abi_layout_issue_kind_t;

// Exact authored field rejected by array-program ABI verification.
typedef struct loom_aie2p_array_abi_layout_issue_t {
  // Classification selecting the correcting diagnostic.
  loom_aie2p_array_abi_layout_issue_kind_t kind;
  // Borrowed spelling of the rejected dictionary field.
  iree_string_view_t field_name;
  // Rejected field value, or absent for an unexpected field name.
  loom_attribute_t attribute;
} loom_aie2p_array_abi_layout_issue_t;

// Validates an untrusted array-program ABI layout dictionary.
//
// Empty dictionaries are valid and request the minimal table containing every
// active binding ordinal. A non-empty dictionary contains exactly one i64
// binding_count in the XDNA image-table range. Returns false and records the
// first correcting issue for malformed authored input.
bool loom_aie2p_array_abi_layout_validate(
    const loom_module_t* module, loom_named_attr_slice_t attributes,
    loom_aie2p_array_abi_layout_issue_t* out_issue);

// Interprets the ABI layout of a verified AIE2P array low.func.def.
//
// The Low verifier owns schema admission. Consumers receive the typed result
// directly and do not repeat recoverable validation.
loom_aie2p_array_abi_layout_t loom_aie2p_array_abi_layout_from_verified(
    const loom_op_t* function_op);

// Builds the canonical explicit array-program ABI layout attribute.
iree_status_t loom_aie2p_array_abi_layout_make_attr(loom_module_t* module,
                                                    uint32_t binding_slot_count,
                                                    loom_attribute_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ABI_LAYOUT_H_
