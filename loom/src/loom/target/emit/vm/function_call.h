// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM call-packet layout and encoding.

#ifndef LOOM_TARGET_EMIT_VM_FUNCTION_CALL_H_
#define LOOM_TARGET_EMIT_VM_FUNCTION_CALL_H_

#include "iree/base/api.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/packet.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/emit/vm/module_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kind of structural Low call represented by a packet.
typedef uint8_t loom_vm_function_call_kind_t;
enum loom_vm_function_call_kind_e {
  // The operation is not a structural function call.
  LOOM_VM_FUNCTION_CALL_KIND_NONE = 0,
  // The operation names a module function or import directly.
  LOOM_VM_FUNCTION_CALL_KIND_DIRECT = 1,
  // The operation calls a first-class function value.
  LOOM_VM_FUNCTION_CALL_KIND_INDIRECT = 2,
};

// Logical callable fields carried by one structural Low call operation.
typedef struct loom_vm_function_call_view_t {
  // Direct or indirect operation kind.
  loom_vm_function_call_kind_t kind;
  // Packet operand ordinal at which logical arguments begin.
  uint16_t argument_operand_base;
  // Logical call arguments in source order.
  loom_value_slice_t arguments;
  // Logical call results in source order.
  loom_value_slice_t results;
} loom_vm_function_call_view_t;

// Returns true and populates |out_call| when |op| is a structural Low call.
bool loom_vm_function_call_try_view(const loom_op_t* op,
                                    loom_vm_function_call_view_t* out_call);

// Builds the canonical caller-local packet layout for |call|.
iree_status_t loom_vm_function_call_layout_build(
    const loom_module_t* module, const loom_vm_function_call_view_t* call,
    loom_vm_call_abi_packet_layout_t* out_layout);

// Returns the exact record-stream byte length required by |layout|.
uint32_t loom_vm_function_call_record_byte_length(
    const loom_vm_call_abi_packet_layout_t* layout);

// Encodes one structural call and its caller-local overflow transfers.
iree_status_t loom_vm_function_call_encode(
    const loom_low_emission_frame_t* frame,
    const loom_vm_module_layout_t* module_layout,
    const loom_low_packet_view_t* packet,
    const loom_vm_function_call_view_t* call,
    const loom_vm_call_abi_packet_layout_t* layout,
    loom_bytecode_page_writer_t* writer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_FUNCTION_CALL_H_
