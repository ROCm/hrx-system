// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL-kernel ABI materialization for target-low kernels.
//
// Function-local low.resource imports describe the HAL-facing binding ABI, and
// remaining entry block arguments describe direct dispatch constants. This
// layer lowers both forms into AMDGPU packet-level sequences that load from the
// kernarg segment. Descriptor-consuming operations use an explicit
// amdgpu.hal.buffer_descriptor pseudo, which this layer expands into the
// four-SGPR buffer descriptor construction sequence.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_MATERIALIZATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_hal_binding_materialization_result_t {
  // ABI layout captured before low.resource ops are rewritten.
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout;
  // Number of low.resource<hal_binding> ops expanded into pointer loads.
  iree_host_size_t materialized_binding_count;
  // Number of direct entry block arguments expanded into scalar loads.
  iree_host_size_t materialized_direct_arg_count;
  // Number of amdgpu.hal.buffer_descriptor pseudos expanded into low IR.
  iree_host_size_t materialized_descriptor_count;
  // True when the pass inserted the kernarg segment pointer live-in.
  bool inserted_kernarg_segment_ptr_live_in;
} loom_amdgpu_hal_binding_materialization_result_t;

// Expands AMDGPU HAL low.resource imports, direct arguments, and descriptor
// pseudos in
// |function_op|.
//
// |descriptor_set| must be the target-low descriptor set selected for
// |target_bundle|. Materialized packets are selected by stable descriptor ref
// and verified against that concrete set.
//
// The expansion inserts or reuses a low.live_in<amdgpu.kernarg_segment_ptr>
// value, loads one 64-bit binding pointer from each kernarg slot assigned by
// low.resource<hal_binding>, loads direct entry block arguments from the HAL
// constant segment, removes those direct arguments, and replaces resource
// imports with reg<amdgpu.sgpr x2>. One semantic direct argument may consume
// multiple 32-bit HAL constants; adjacent 32-bit arguments may be packed into
// wider SMEM loads when the target-low descriptor set supports the packet.
// Separate amdgpu.hal.buffer_descriptor pseudos materialize range, flags, and
// pointer-high descriptor bits only for selected descriptor-consuming packets.
iree_status_t loom_amdgpu_hal_binding_materialize(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_binding_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena);

// Expands only amdgpu.hal.buffer_descriptor pseudos in |function_op|.
//
// This is the source-low artifact boundary for descriptors that are compiler
// internals rather than authorable low asm. Unlike
// loom_amdgpu_hal_binding_materialize, this does not materialize
// low.resource<hal_binding> imports, direct entry arguments, or the kernarg
// segment pointer.
iree_status_t loom_amdgpu_hal_binding_materialize_buffer_descriptors(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_materialized_count,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_MATERIALIZATION_H_
