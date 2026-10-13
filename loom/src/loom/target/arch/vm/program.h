// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared physical VM program consumed by module binary emission.

#ifndef LOOM_TARGET_ARCH_VM_PROGRAM_H_
#define LOOM_TARGET_ARCH_VM_PROGRAM_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/vm/bytecode/wire/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable read-only data block selected by program planning.
typedef struct loom_vm_rodata_plan_t {
  // Borrowed immutable contents retained by the planned source module.
  iree_const_byte_span_t contents;
  // Required power-of-two alignment relative to the rodata section.
  uint32_t alignment;
} loom_vm_rodata_plan_t;

// One typed native record. Lists are in child-before-parent ordinal order.
typedef struct loom_vm_native_layout_plan_t {
  // Next layout in serialization order, or NULL at the end.
  struct loom_vm_native_layout_plan_t* next;
  // Program-local ordinal assigned when the layout is published.
  uint16_t ordinal;
  // Final field slice, stride and depth.
  iree_vm_bytecode_v0_native_layout_row_t row;
  // Ordered leaf and range descriptors for this fixed record.
  iree_vm_bytecode_v0_native_field_row_t* fields;
} loom_vm_native_layout_plan_t;

// One finalized structured native callsite. Rows and payloads are emitted in
// list order; the retained offsets already partition the complete word tail.
typedef struct loom_vm_native_callsite_plan_t {
  // Next callsite in function/instruction order, or NULL at the end.
  struct loom_vm_native_callsite_plan_t* next;
  // Final caller, import identity, version, and word extent.
  iree_vm_bytecode_v0_native_callsite_row_t row;
  // Immutable physical selector/count/offset words.
  const uint32_t* words;
} loom_vm_native_callsite_plan_t;

// Immutable physical VM program produced by target planning.
//
// Every ordinal, row, bytecode displacement, physical extent, and format
// limit has been finalized. The binary writer performs no target resolution,
// semantic validation, scheduling, allocation, or mutable IR access.
typedef struct loom_vm_program_plan_t {
  // Strings in final wire ordinal order.
  const iree_string_view_t* strings;
  // Number of entries in |strings|.
  uint32_t string_count;

  // Reference type namespace groups in final wire order.
  const iree_vm_bytecode_v0_ref_type_group_row_t* reference_groups;
  // Number of entries in |reference_groups|.
  uint32_t reference_group_count;
  // Reference type entries in final wire order.
  const iree_vm_bytecode_v0_ref_type_entry_row_t* reference_entries;
  // Number of entries in |reference_entries|.
  uint32_t reference_count;

  // Interned signature rows in callable ordinal order.
  const iree_vm_bytecode_v0_signature_row_t* signatures;
  // Number of entries in |signatures| and the implicit callable-type table.
  uint32_t signature_count;
  // Concatenated signature descriptors addressed by each signature row.
  const iree_vm_bytecode_v0_signature_descriptor_row_t* signature_descriptors;
  // Number of entries in |signature_descriptors|.
  uint32_t signature_descriptor_count;

  // Native import namespace groups in final wire order.
  const iree_vm_bytecode_v0_import_group_row_t* import_groups;
  // Number of entries in |import_groups|.
  uint32_t import_group_count;
  // Unique native import entries in final wire order.
  const iree_vm_bytecode_v0_import_entry_row_t* imports;
  // Number of entries in |imports|.
  uint32_t import_count;

  // Structured native callsites in instruction emission order.
  const loom_vm_native_callsite_plan_t* native_calls;
  // Number of entries in |native_calls|.
  uint32_t native_call_count;
  // Typed native layouts in child-before-parent order.
  const loom_vm_native_layout_plan_t* native_layouts;
  // Number of native layout rows.
  uint32_t native_layout_count;
  // Total field count across native layouts.
  uint32_t native_field_count;
  // Exact-version native imports in ordinal order.
  const iree_vm_bytecode_v0_native_import_row_t* native_imports;
  // Number of native imports.
  uint32_t native_import_count;

  // Public export rows sorted by export name.
  const iree_vm_bytecode_v0_export_row_t* exports;
  // Number of entries in |exports|.
  uint32_t export_count;

  // Final local-function rows in function ordinal order.
  const iree_vm_bytecode_v0_function_row_t* functions;
  // Number of entries in |functions|.
  uint32_t function_count;
  // Owned immutable function instruction streams addressed by the bytecode
  // offsets and lengths in |functions|.
  iree_byte_sequence_t* function_bytecode;
  // Maximum block count across |functions|.
  uint32_t maximum_block_count;

  // Selected read-only data blocks in first-use order.
  const loom_vm_rodata_plan_t* rodata;
  // Number of entries in |rodata|.
  uint32_t rodata_count;
  // Maximum alignment of the rodata section and its blocks.
  uint32_t rodata_alignment;
  // True when the source declared rodata and the image retains its section,
  // including when no block was selected by a VM function.
  bool has_rodata_section;
} loom_vm_program_plan_t;

// Releases resources retained by |plan|. Arena-owned rows remain owned by the
// arena supplied while building the plan.
void loom_vm_program_plan_deinitialize(loom_vm_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_PROGRAM_H_
