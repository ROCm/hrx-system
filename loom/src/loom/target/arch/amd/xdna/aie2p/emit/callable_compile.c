// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/callable_compile.h"

#include "loom/ir/module.h"
#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/call_clobbers.h"

typedef struct loom_aie2p_callable_builder_t {
  // ABI carriers for arguments followed by results.
  loom_low_allocation_abi_location_t* locations;
  // Number of argument carriers in |locations|.
  uint16_t argument_count;
  // Number of result carriers following the arguments.
  uint16_t result_count;
  // Immutable call contract borrowing arena-owned carriers and clobbers.
  loom_low_call_contract_t call_contract;
  // Native object paired with this exact register contract.
  loom_aie2p_leaf_contribution_t contribution;
} loom_aie2p_callable_builder_t;

static iree_status_t loom_aie2p_callable_prepare_abi(
    const loom_module_t* module, loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_aie2p_callable_builder_t* callable) {
  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &callable->argument_count);
  callable->result_count = function_op->result_count;
  const iree_host_size_t location_count =
      (iree_host_size_t)callable->argument_count + callable->result_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, location_count,
                                                 sizeof(*callable->locations),
                                                 (void**)&callable->locations));
  for (uint16_t side = 0; side < 2; ++side) {
    loom_aie2p_call_abi_requirements_t requirements = {0};
    const uint16_t count =
        side == 0 ? callable->argument_count : callable->result_count;
    for (uint16_t i = 0; i < count; ++i) {
      const loom_value_id_t value =
          side == 0 ? arguments[i] : loom_op_results(function_op)[i];
      loom_low_allocation_abi_location_t* location =
          &callable->locations[(side == 0 ? 0 : callable->argument_count) + i];
      IREE_RETURN_IF_ERROR(loom_aie2p_call_abi_select_location(
          loom_module_value_type(module, value), &requirements, location));
    }
  }
  callable->call_contract = (loom_low_call_contract_t){
      .arguments = callable->locations,
      .argument_count = callable->argument_count,
      .results = callable->result_count != 0
                     ? callable->locations + callable->argument_count
                     : NULL,
      .result_count = callable->result_count,
  };
  return iree_ok_status();
}

iree_status_t loom_aie2p_callable_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    loom_symbol_ref_array_t direct_callees, iree_arena_allocator_t* arena,
    bool* out_compiled, loom_aie2p_callable_t* out_callable) {
  *out_compiled = false;
  *out_callable = (loom_aie2p_callable_t){0};
  loom_aie2p_callable_builder_t builder = {0};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_callable_prepare_abi(module, function_op, arena, &builder));
  loom_aie2p_leaf_compile_options_t leaf_options = *options;
  leaf_options.is_helper = true;
  leaf_options.allocation_entry_locations = builder.call_contract.arguments;
  leaf_options.allocation_entry_location_count =
      builder.call_contract.argument_count;
  leaf_options.allocation_exit_locations = builder.call_contract.results;
  leaf_options.allocation_exit_location_count =
      builder.call_contract.result_count;
  bool compiled = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(module, function_op,
                                               &leaf_options, arena, &compiled,
                                               &builder.contribution));
  if (!compiled) {
    return iree_ok_status();
  }
  const loom_aie2p_physical_register_id_t restored_register =
      options->return_address.kind == LOOM_AIE2P_RETURN_ADDRESS_FRAME
          ? options->return_address.shadow_register
          : LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_call_clobbers_build(
      builder.contribution.realization.register_writes, restored_register,
      &builder.call_contract, direct_callees, options->call_contracts, arena,
      &builder.call_contract.clobbers));
  *out_callable = (loom_aie2p_callable_t){
      .call_contract = builder.call_contract,
      .stack_byte_length =
          builder.contribution.realization.stack_frame_byte_length +
          options->callee_stack_byte_length,
      .contribution = builder.contribution,
  };
  *out_compiled = true;
  return iree_ok_status();
}
