// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared native assembly-fragment formatting over low packet tables.
//
// This layer is intentionally small: it owns native-fragment contract
// validation, scheduled packet iteration, block labels, and
// descriptor/attribute lookup helpers.
// Target packages own instruction syntax, register spelling, ABI/prologue
// policy, and any future handoff to assemblers or external validators.

#ifndef LOOM_TARGET_EMIT_NATIVE_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_native_assembly_packet_context_t {
  // Schedule table being formatted.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying physical locations.
  const loom_low_allocation_table_t* allocation;
  // Scheduled packet currently being formatted.
  const loom_low_packet_view_t* packet;
  // Text destination for the fragment.
  iree_string_builder_t* builder;
  // Reusable scratch for structural parallel-copy sequencing.
  loom_low_move_sequence_scratch_t* move_scratch;
} loom_native_assembly_packet_context_t;

typedef iree_status_t (*loom_native_assembly_append_packet_fn_t)(
    void* user_data, const loom_native_assembly_packet_context_t* context);

typedef struct loom_native_assembly_append_packet_callback_t {
  // Function that appends one packet to the output builder. It may append no
  // text when a structural packet has been coalesced away.
  loom_native_assembly_append_packet_fn_t fn;
  // Opaque callback state forwarded to |fn|.
  void* user_data;
} loom_native_assembly_append_packet_callback_t;

typedef struct loom_native_assembly_structural_packet_callback_t {
  // Structural low op handled by this callback row.
  loom_op_kind_t op_kind;
  // Formatter for packets with |op_kind|.
  loom_native_assembly_append_packet_callback_t append_packet;
} loom_native_assembly_structural_packet_callback_t;

typedef struct loom_native_assembly_format_options_t {
  // Optional hook that appends zero or more complete assembly lines before the
  // current scheduled packet. Targets use this for table-materialized packets
  // such as waits or delays without teaching the shared formatter target facts.
  loom_native_assembly_append_packet_callback_t append_before_packet;
  // Required formatter for descriptor-backed low.op and low.const packets.
  loom_native_assembly_append_packet_callback_t append_descriptor_packet;
  // Target-owned structural low op formatter rows.
  const loom_native_assembly_structural_packet_callback_t*
      structural_packet_callbacks;
  // Number of rows in |structural_packet_callbacks|.
  iree_host_size_t structural_packet_callback_count;
} loom_native_assembly_format_options_t;

// Formats a scheduled and allocated low.func.def as a native assembly fragment.
// The output is target-owned assembly text, not a complete object/kernel ABI
// envelope. Unsupported structural packets fail loud through the target-owned
// callback table instead of being printed as comments or pseudo-ops.
iree_status_t loom_native_assembly_format_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_native_assembly_format_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

// Appends the canonical local block label for |block|, such as ".Lbb0".
iree_status_t loom_native_assembly_append_block_label(
    const loom_low_schedule_table_t* schedule, const loom_block_t* block,
    iree_string_builder_t* builder);

// Returns the interned module string for |string_id|, or the empty string when
// |string_id| is invalid.
iree_string_view_t loom_native_assembly_module_string(
    const loom_module_t* module, loom_string_id_t string_id);

// Resolves a required string from |descriptor_set|.
iree_status_t loom_native_assembly_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string);

// Finds a named attribute by textual key in |attrs|.
const loom_named_attr_t* loom_native_assembly_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name);

// Reads a required I64 attribute from |attrs|.
iree_status_t loom_native_assembly_read_i64_attr(const loom_module_t* module,
                                                 loom_named_attr_slice_t attrs,
                                                 iree_string_view_t name,
                                                 int64_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_ASSEMBLY_H_
