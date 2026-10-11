// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_NATIVE_CALL_PLAN_H_
#define LOOM_TARGET_ARCH_VM_NATIVE_CALL_PLAN_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/vm/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static logical-to-record layout for a vm_native import declaration.
//
// abi_layout contains four fields: record is a u32 word prototype; arguments
// and results contain one word offset per logical value, in signature order;
// version is the provider export version. Selector words in record are zero
// placeholders. Every patch offset is distinct. Range counts and offsets are
// provider-specific literal words, verified against its schema at VM linking.
// This representation names no physical register and introduces no exports for
// individual list cardinalities.
typedef struct loom_vm_native_call_template_t {
  // Immutable prototype words with zero selector placeholders.
  const uint32_t* words;
  // Word count of the complete root and nested range payloads.
  uint32_t word_count;
  // Record word offsets, arguments followed by results in logical order.
  const uint32_t* operand_offsets;
  // Number of entries in |operand_offsets|.
  uint32_t operand_count;
  // Version required of the named native export.
  uint32_t version;
  // Final module-name string ordinal assigned during metadata planning.
  uint16_t module_name;
  // Final export-name string ordinal assigned during metadata planning.
  uint16_t export_name;
} loom_vm_native_call_template_t;

// Validates the user-authored ABI layout once and retains a typed template in
// |arena|. Calls consuming the template use its indexed operand offsets.
iree_status_t loom_vm_native_call_template_build(
    const loom_module_t* module, loom_named_attr_slice_t layout,
    uint16_t argument_count, uint16_t result_count,
    iree_arena_allocator_t* arena,
    loom_vm_native_call_template_t** out_template);

// Append-only native callsite state shared across function emission.
typedef struct loom_vm_native_call_plan_t {
  // Program arena, surviving resets of per-function allocation scratch.
  iree_arena_allocator_t* arena;
  // First physical callsite, or NULL while empty.
  loom_vm_native_callsite_plan_t* head;
  // Last physical callsite, or NULL while empty.
  loom_vm_native_callsite_plan_t* tail;
  // Number of published callsites.
  uint32_t count;
  // Total word extent of the published callsites.
  uint32_t word_count;
} loom_vm_native_call_plan_t;

// Appends one callsite and copies its literal prototype. The caller writes
// selectors through the template's operand offsets before publishing the plan.
iree_status_t loom_vm_native_call_plan_append(
    loom_vm_native_call_plan_t* plan,
    const loom_vm_native_call_template_t* call_template,
    uint16_t function_ordinal, uint32_t* out_ordinal, uint32_t** out_words);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_NATIVE_CALL_PLAN_H_
