// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/callable_compile.h"

#include <stdlib.h>

#include "loom/ir/module.h"
#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/registers.h"

typedef struct loom_aie2p_callable_builder_t {
  // ABI carriers for arguments followed by results.
  loom_low_allocation_abi_location_t* locations;
  // Number of argument carriers in |locations|.
  uint16_t argument_count;
  // Number of result carriers following the arguments.
  uint16_t result_count;
  // Physical registers overwritten by the helper and its ABI transport.
  loom_low_call_clobber_t* clobbers;
  // Number of distinct physical-register clobber records.
  iree_host_size_t clobber_count;
  // Immutable call contract borrowing the carriers and clobbers above.
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
      .results = callable->locations + callable->argument_count,
      .result_count = callable->result_count,
  };
  return iree_ok_status();
}

static iree_host_size_t loom_aie2p_callable_clobber_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_aie2p_leaf_realization_t* realization) {
  iree_host_size_t count = 0;
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    if (!loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
      continue;
    }
    for (uint16_t i = 0; i < reg_class->allocatable_count; ++i) {
      const uint16_t physical_register =
          loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                              class_id, i);
      count +=
          loom_aie2p_leaf_may_write_register(realization, physical_register);
    }
  }
  return count;
}

static iree_host_size_t loom_aie2p_callable_transport_clobber_count(
    const loom_aie2p_callable_builder_t* callable) {
  iree_host_size_t count = 0;
  const iree_host_size_t location_count =
      (iree_host_size_t)callable->argument_count + callable->result_count;
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    count += callable->locations[i].location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  }
  return count;
}

static int loom_aie2p_callable_compare_clobbers(const void* lhs_ptr,
                                                const void* rhs_ptr) {
  const loom_low_call_clobber_t* lhs = lhs_ptr;
  const loom_low_call_clobber_t* rhs = rhs_ptr;
  if (lhs->location != rhs->location) {
    return lhs->location < rhs->location ? -1 : 1;
  }
  if (lhs->register_class != rhs->register_class) {
    return lhs->register_class < rhs->register_class ? -1 : 1;
  }
  return 0;
}

static iree_status_t loom_aie2p_callable_prepare_clobbers(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_aie2p_callable_builder_t* callable) {
  const iree_host_size_t clobber_capacity =
      loom_aie2p_callable_clobber_count(descriptor_set,
                                        &callable->contribution.realization) +
      loom_aie2p_callable_transport_clobber_count(callable);
  if (clobber_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, clobber_capacity, sizeof(*callable->clobbers),
        (void**)&callable->clobbers));
  }
  iree_host_size_t clobber_index = 0;
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    if (!loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
      continue;
    }
    for (uint16_t i = 0; i < reg_class->allocatable_count; ++i) {
      const uint16_t physical_register =
          loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                              class_id, i);
      if (!loom_aie2p_leaf_may_write_register(
              &callable->contribution.realization, physical_register)) {
        continue;
      }
      callable->clobbers[clobber_index++] = (loom_low_call_clobber_t){
          .register_class = class_id,
          .location = physical_register,
          .count = 1,
      };
    }
  }
  // Argument transport overwrites ABI carriers before the helper starts, and
  // result transport overwrites them before caller values resume. These writes
  // belong to the call boundary rather than the helper body.
  const iree_host_size_t location_count =
      (iree_host_size_t)callable->argument_count + callable->result_count;
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    const loom_low_allocation_abi_location_t* location =
        &callable->locations[i];
    if (location->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    callable->clobbers[clobber_index++] = (loom_low_call_clobber_t){
        .register_class = location->descriptor_reg_class_id,
        .location = location->location_base,
        .count = 1,
    };
  }
  IREE_ASSERT_EQ(clobber_index, clobber_capacity);
  if (clobber_index != 0) {
    qsort(callable->clobbers, clobber_index, sizeof(*callable->clobbers),
          loom_aie2p_callable_compare_clobbers);
  }
  callable->clobber_count = 0;
  for (iree_host_size_t i = 0; i < clobber_index; ++i) {
    if (callable->clobber_count != 0 &&
        loom_aie2p_callable_compare_clobbers(
            &callable->clobbers[callable->clobber_count - 1],
            &callable->clobbers[i]) == 0) {
      continue;
    }
    callable->clobbers[callable->clobber_count++] = callable->clobbers[i];
  }
  callable->call_contract.clobbers = (loom_low_call_clobber_list_t){
      .values = callable->clobbers,
      .count = callable->clobber_count,
  };
  return iree_ok_status();
}

iree_status_t loom_aie2p_callable_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_aie2p_callable_t* out_callable) {
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
  IREE_RETURN_IF_ERROR(loom_aie2p_callable_prepare_clobbers(
      loom_aie2p_core_descriptor_set(), arena, &builder));
  *out_callable = (loom_aie2p_callable_t){
      .call_contract = builder.call_contract,
      .contribution = builder.contribution,
  };
  *out_compiled = true;
  return iree_ok_status();
}
