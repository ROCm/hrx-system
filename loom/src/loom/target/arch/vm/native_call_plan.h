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
#include "loom/target/arch/vm/reference_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static logical-to-record layout for a vm_native import declaration.
//
// abi_layout contains a vm.record signature, a u32 record word prototype,
// argument/result word offsets in logical order, and an exact export version.
// The signature owns leaf types, directions, and range element shapes,
// including empty ranges. Selector words are zero placeholders at distinct
// patch offsets. This representation names no physical register and introduces
// no exports for individual list cardinalities.
typedef struct loom_vm_native_call_template_t {
  // Canonical vm.record type retaining all leaf types and range structure.
  loom_type_id_t signature;
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
  // Final native import ordinal assigned during metadata planning.
  uint16_t import_ordinal;
  // Root layout ordinal in the program's native declaration table.
  uint16_t layout_ordinal;
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
  // Typed layouts in child-before-parent order.
  loom_vm_native_layout_plan_t* layouts;
  // Last typed layout, or NULL while empty.
  loom_vm_native_layout_plan_t* layout_tail;
  // Number of serialized layout rows.
  uint32_t layout_count;
  // Total field extent across all layouts.
  uint32_t field_count;
} loom_vm_native_call_plan_t;

// Projects a canonical vm.record declaration into typed wire layouts, sharing
// repeated range types through |layouts_by_type|. The zero-initialized map has
// one pointer per canonical module type and is owned by metadata scratch.
// Ref ordinals are provisional until the shared reference plan is finalized.
iree_status_t loom_vm_native_call_layout_bind(
    loom_vm_native_call_plan_t* plan, const loom_module_t* module,
    loom_vm_native_call_template_t* call_template,
    loom_vm_native_layout_plan_t** layouts_by_type,
    loom_vm_reference_plan_t* references,
    iree_arena_allocator_t* scratch_arena);

// Checks the authored record's canonical geometry and matches every logical
// operand to an exact typed leaf, including its result direction. Allocation
// later patches only selectors; no schema discovery occurs during emission.
iree_status_t loom_vm_native_call_template_validate(
    const loom_vm_native_call_template_t* call_template,
    const loom_vm_native_layout_plan_t* const* layouts,
    const iree_vm_bytecode_v0_signature_descriptor_row_t* operands,
    uint16_t argument_count, iree_arena_allocator_t* scratch_arena);

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
