// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V atomic instruction emission for target-low descriptor packets.

#ifndef LOOM_TARGET_EMIT_SPIRV_FUNCTION_ATOMIC_H_
#define LOOM_TARGET_EMIT_SPIRV_FUNCTION_ATOMIC_H_

#include "iree/base/api.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/target/arch/spirv/packet_rows.h"
#include "loom/target/emit/spirv/function_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits an OpAtomic* instruction with one value operand.
iree_status_t loom_spirv_emit_atomic_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row);

// Emits an OpAtomicCompareExchange instruction.
iree_status_t loom_spirv_emit_atomic_compare_exchange_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row);

// Emits an integer OpAtomicExchange over bitcast floating-point operands.
iree_status_t loom_spirv_emit_atomic_float_bitcast_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row);

// Emits an exact floating-point atomic operation with an integer CAS loop.
iree_status_t loom_spirv_emit_atomic_float_cas_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row);

// Emits integer compare-exchange over bitcast floating-point operands.
iree_status_t loom_spirv_emit_atomic_float_compare_exchange_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_FUNCTION_ATOMIC_H_
