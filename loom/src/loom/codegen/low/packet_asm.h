// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scheduled packet assembly text over low schedule/allocation tables.
//
// This formatter is an emitter-facing diagnostic view, not a target artifact
// format. The shared low layer owns packet order, descriptor asm-form mapping,
// and structural control-flow spelling. Target packages own physical register
// or target-id spelling through the value formatting callback.

#ifndef LOOM_CODEGEN_LOW_PACKET_ASM_H_
#define LOOM_CODEGEN_LOW_PACKET_ASM_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/packet.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_low_packet_asm_format_value_fn_t)(
    void* user_data, const loom_low_allocation_table_t* allocation,
    loom_value_id_t value_id,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, iree_string_builder_t* builder);

// Callback used to spell one allocated SSA value in packet assembly output.
typedef struct loom_low_packet_asm_format_value_callback_t {
  // Function that appends the value spelling to the output builder.
  loom_low_packet_asm_format_value_fn_t fn;
  // Opaque callback state forwarded to |fn|.
  void* user_data;
} loom_low_packet_asm_format_value_callback_t;

// Options controlling scheduled packet assembly text emission.
typedef struct loom_low_packet_asm_options_t {
  // Optional selected asm-form table. When present, descriptor-backed packets
  // use explicit per-packet form choices and fall back to descriptor canonical
  // forms for entries set to LOOM_LOW_ASM_FORM_ORDINAL_NONE.
  const loom_low_packet_asm_form_table_t* selected_asm_forms;
  // Target-owned value formatter for physical registers, target IDs, and spill
  // locations. The callback is required because core low has no target register
  // spelling contract.
  loom_low_packet_asm_format_value_callback_t format_value;
} loom_low_packet_asm_options_t;

// Formats |schedule| and |allocation| as scheduled packet assembly text.
// The tables must describe the same target-low function and target descriptor
// set.
iree_status_t loom_low_packet_asm_format(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_asm_options_t* options,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_ASM_H_
