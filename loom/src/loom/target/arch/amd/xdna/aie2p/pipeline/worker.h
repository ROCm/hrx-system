// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_WORKER_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_WORKER_H_

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/transforms/pipeline/realization.h"

// Typed emitter shared by the selected channel and DMA helper bodies. Helpers
// remain ordinary Low callables and use the normal inlining and codegen path.
typedef struct loom_aie2p_worker_builder_t {
  // Source module receiving generated ordinary callables.
  loom_module_t* module;
  // Scratch storage for collision-free symbols and descriptors.
  iree_arena_allocator_t* arena;
  // Owning compilation occurrence for helper target-context registration.
  const loom_pipeline_realization_t* realization;
  // Calling worker in the realization's retained context array.
  iree_host_size_t worker_index;
  // Compiler diagnostic sink used by helper function-contract refinement.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Module symbol inventory extended as helpers are emitted.
  loom_symbol_map_t names;
  // Next candidate suffix for a helper name.
  uint32_t next_name;
  // Canonical core descriptor vocabulary.
  const loom_low_descriptor_set_t* descriptors;
  // Core representation contract for every helper.
  loom_string_id_t contract;
  // Scalar carrier selected by the core descriptor family.
  loom_type_t scalar_type;
  // Tile-local pointer carrier selected by the core descriptor family.
  loom_type_t address_type;
  // Interned ordinary constant immediate name.
  loom_string_id_t integer_name;
  // Interned indexed-memory displacement name.
  loom_string_id_t displacement_name;
  // Interned lock selector name.
  loom_string_id_t lock_name;
} loom_aie2p_worker_builder_t;

iree_status_t loom_aie2p_worker_builder_initialize(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_aie2p_worker_builder_t* out_builder);

iree_status_t loom_aie2p_worker_symbol(loom_aie2p_worker_builder_t* context,
                                       loom_symbol_ref_t* out_symbol);

// Creates a helper and positions builder at its entry. Its signature is the
// selected physical ABI, independently of the semantic low.invoke callsite.
iree_status_t loom_aie2p_worker_helper(
    loom_aie2p_worker_builder_t* context, const loom_type_t* arguments,
    iree_host_size_t argument_count, const loom_type_t* results,
    iree_host_size_t result_count, loom_builder_t* builder,
    loom_symbol_ref_t* out_symbol, loom_op_t** out_function);

iree_status_t loom_aie2p_worker_constant(loom_aie2p_worker_builder_t* context,
                                         loom_builder_t* builder, int32_t value,
                                         loom_value_id_t* out_value);
iree_status_t loom_aie2p_worker_op(loom_aie2p_worker_builder_t* context,
                                   loom_builder_t* builder, uint32_t descriptor,
                                   const loom_value_id_t* operands,
                                   iree_host_size_t operand_count,
                                   loom_named_attr_slice_t attributes,
                                   const loom_type_t* result_type,
                                   loom_value_id_t* out_value);
iree_status_t loom_aie2p_worker_binary(loom_aie2p_worker_builder_t* context,
                                       loom_builder_t* builder,
                                       uint32_t descriptor, loom_value_id_t lhs,
                                       loom_value_id_t rhs,
                                       loom_value_id_t* out_value);
iree_status_t loom_aie2p_worker_return(loom_builder_t* builder,
                                       const loom_value_id_t* values,
                                       iree_host_size_t count);
iree_status_t loom_aie2p_worker_lock(loom_aie2p_worker_builder_t* context,
                                     loom_builder_t* builder,
                                     uint32_t descriptor, uint16_t selector,
                                     int32_t delta);
// Emits a native control packet. The admitted register offset fits 20 bits;
// count is 1..4 words and the payload's last word carries TLAST.
iree_status_t loom_aie2p_worker_control_write(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint8_t packet_id, uint32_t register_offset, const loom_value_id_t* words,
    iree_host_size_t count);

// Encodes the register address, payload length, and odd parity of a write
// command. The admitted register offset fits 20 bits and count is 1..4 words.
uint32_t loom_aie2p_worker_control_write_command(uint32_t register_offset,
                                                 iree_host_size_t count);

// Emits a write whose encoded command is a runtime operand. Shared DMA helpers
// receive this operand from their callers instead of specializing the body for
// every descriptor address. count matches the encoded command's payload length.
iree_status_t loom_aie2p_worker_control_write_encoded(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint8_t packet_id, loom_value_id_t command, const loom_value_id_t* words,
    iree_host_size_t count);

// Reads 1..4 consecutive registers from a 16-byte-aligned address through a
// selected return route. The scalar stream has been drained before the request:
// its packet header and values cannot interleave with outstanding task tokens.
iree_status_t loom_aie2p_worker_control_read(
    loom_aie2p_worker_builder_t* context, loom_builder_t* builder,
    uint8_t packet_id, uint8_t reply_id, uint32_t register_offset,
    iree_host_size_t count, loom_value_id_t* out_values);

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_WORKER_H_
