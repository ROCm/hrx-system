// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_compile.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"
#include "loom/target/registers.h"
#include "loom/util/call_graph.h"

typedef struct loom_aie2p_program_callable_t {
  loom_symbol_id_t symbol_id;
  loom_op_t* function_op;
  loom_low_allocation_abi_location_t* locations;
  uint16_t argument_count;
  uint16_t result_count;
  loom_low_call_clobber_t* clobbers;
  iree_host_size_t clobber_count;
  loom_low_call_contract_t call_contract;
  loom_aie2p_leaf_contribution_t contribution;
} loom_aie2p_program_callable_t;

typedef struct loom_aie2p_program_call_contract_context_t {
  const loom_module_t* module;
  loom_aie2p_program_callable_t* const* bindings_by_symbol;
} loom_aie2p_program_call_contract_context_t;

static iree_status_t loom_aie2p_program_call_contract_validate(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_aie2p_program_call_contract_context_t* context = user_data;
  if (callee.module_id == 0 &&
      callee.symbol_id < context->module->symbols.count &&
      context->bindings_by_symbol[callee.symbol_id] != NULL) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "AIE2P direct call has no compiled local helper");
}

static const loom_low_call_contract_t* loom_aie2p_program_call_contract_query(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_aie2p_program_call_contract_context_t* context = user_data;
  return &context->bindings_by_symbol[callee.symbol_id]->call_contract;
}

static iree_status_t loom_aie2p_program_select_abi_register(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t* er_count, uint16_t* ep_count,
    loom_low_allocation_abi_location_t* out_location) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          AIE2P_CORE_DESCRIPTOR_SET_ID ||
      loom_low_register_type_unit_count(type) != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P direct helper ABI requires one-unit register values");
  }
  const uint16_t register_class = loom_low_register_type_class_id(type);
  uint16_t* count = NULL;
  if (register_class == AIE2P_CORE_REG_CLASS_ID_AIE2P_ER) {
    count = er_count;
    if (*count >= 16) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "AIE2P direct helper ABI supports at most 16 scalar values per side");
    }
  } else if (register_class == AIE2P_CORE_REG_CLASS_ID_AIE2P_EP) {
    count = ep_count;
  } else {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P direct helper ABI supports only aie2p.er and aie2p.ep values");
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[register_class];
  if (*count >= reg_class->allocatable_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P direct helper ABI register bank overflow");
  }
  *out_location = (loom_low_allocation_abi_location_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = register_class,
      .location_base = loom_low_descriptor_set_physical_register_candidate(
          descriptor_set, register_class, (*count)++),
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_prepare_abi(
    const loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_aie2p_program_callable_t* callable) {
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
    uint16_t er_count = 0;
    uint16_t ep_count = 0;
    const uint16_t count =
        side == 0 ? callable->argument_count : callable->result_count;
    for (uint16_t i = 0; i < count; ++i) {
      const loom_value_id_t value =
          side == 0 ? arguments[i] : loom_op_results(function_op)[i];
      loom_low_allocation_abi_location_t* location =
          &callable->locations[(side == 0 ? 0 : callable->argument_count) + i];
      IREE_RETURN_IF_ERROR(loom_aie2p_program_select_abi_register(
          descriptor_set, loom_module_value_type(module, value), &er_count,
          &ep_count, location));
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

static iree_host_size_t loom_aie2p_program_clobber_count(
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

static int loom_aie2p_program_compare_clobbers(const void* lhs_ptr,
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

static iree_status_t loom_aie2p_program_prepare_clobbers(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_aie2p_program_callable_t* callable) {
  callable->clobber_count = loom_aie2p_program_clobber_count(
      descriptor_set, &callable->contribution.realization);
  if (callable->clobber_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, callable->clobber_count, sizeof(*callable->clobbers),
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
  IREE_ASSERT_EQ(clobber_index, callable->clobber_count);
  if (callable->clobber_count != 0) {
    qsort(callable->clobbers, callable->clobber_count,
          sizeof(*callable->clobbers), loom_aie2p_program_compare_clobbers);
  }
  callable->call_contract.clobbers = (loom_low_call_clobber_list_t){
      .values = callable->clobbers,
      .count = callable->clobber_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_compile_impl(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_aie2p_leaf_compile_options_t* entry_options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    bool* out_compiled, loom_aie2p_leaf_contribution_t* out_contribution) {
  loom_call_graph_t call_graph;
  IREE_RETURN_IF_ERROR(
      loom_call_graph_build(module, scratch_arena, &call_graph));
  const loom_symbol_ref_t entry_ref =
      loom_func_like_callee(loom_func_like_cast(module, entry_function_op));
  const loom_call_graph_node_t* entry_node =
      loom_call_graph_node(&call_graph, entry_ref.symbol_id);
  if (entry_node == NULL || entry_node->callee_count == 0) {
    return loom_aie2p_leaf_compile(module, entry_function_op, entry_options,
                                   arena, out_compiled, out_contribution);
  }
  if (loom_call_graph_is_recursive(&call_graph, entry_ref.symbol_id)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AIE2P native calls cannot be recursive");
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const loom_aie2p_physical_register_id_t link_shadow_register =
      loom_aie2p_machine_find_physical_register(IREE_SV("r25"));
  IREE_ASSERT(link_shadow_register != LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
  const loom_low_allocation_reserved_range_t link_reservation = {
      .register_class = IREE_SV("aie2p.er"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = link_shadow_register,
      .location_count = 1,
  };

  loom_aie2p_program_callable_t* helpers = NULL;
  loom_aie2p_program_callable_t** bindings_by_symbol = NULL;
  loom_aie2p_leaf_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, entry_node->callee_count,
                                sizeof(*helpers), (void**)&helpers));
  memset(helpers, 0, entry_node->callee_count * sizeof(*helpers));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, module->symbols.count, sizeof(*bindings_by_symbol),
      (void**)&bindings_by_symbol));
  memset(bindings_by_symbol, 0,
         module->symbols.count * sizeof(*bindings_by_symbol));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, entry_node->callee_count + 1, sizeof(*contributions),
      (void**)&contributions));

  for (uint16_t i = 0; i < entry_node->callee_count; ++i) {
    const loom_symbol_id_t symbol_id = entry_node->callees[i];
    const loom_call_graph_node_t* helper_node =
        loom_call_graph_node(&call_graph, symbol_id);
    if (helper_node == NULL || helper_node->callee_count != 0 ||
        loom_call_graph_is_recursive(&call_graph, symbol_id)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P native helpers must be direct, call-free leaves");
    }
    loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (symbol->defining_op == NULL ||
        !loom_low_func_def_isa(symbol->defining_op) ||
        !loom_func_like_is_module_internal(
            loom_func_like_cast(module, symbol->defining_op))) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AIE2P native helper must be a module-internal Low definition");
    }
    loom_aie2p_program_callable_t* helper = &helpers[i];
    helper->symbol_id = symbol_id;
    helper->function_op = symbol->defining_op;
    bindings_by_symbol[symbol_id] = helper;
    IREE_RETURN_IF_ERROR(loom_aie2p_program_prepare_abi(
        module, helper->function_op, descriptor_set, scratch_arena, helper));
    const loom_aie2p_leaf_compile_options_t helper_options = {
        .function_target_facts = entry_options->function_target_facts,
        .descriptor_registry = entry_options->descriptor_registry,
        .allocation_entry_locations = helper->call_contract.arguments,
        .allocation_entry_location_count = helper->call_contract.argument_count,
        .allocation_exit_locations = helper->call_contract.results,
        .allocation_exit_location_count = helper->call_contract.result_count,
        .allocation_reserved_ranges = &link_reservation,
        .allocation_reserved_range_count = 1,
        .diagnostic_emitter = entry_options->diagnostic_emitter,
    };
    bool helper_compiled = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(
        module, helper->function_op, &helper_options, arena, &helper_compiled,
        &helper->contribution));
    if (!helper_compiled) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_program_prepare_clobbers(
        descriptor_set, scratch_arena, helper));
  }

  loom_low_allocation_reserved_range_t* entry_reservations = NULL;
  const iree_host_size_t entry_reservation_count =
      entry_options->allocation_reserved_range_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, entry_reservation_count, sizeof(*entry_reservations),
      (void**)&entry_reservations));
  if (entry_options->allocation_reserved_range_count != 0) {
    memcpy(entry_reservations, entry_options->allocation_reserved_ranges,
           entry_options->allocation_reserved_range_count *
               sizeof(*entry_reservations));
  }
  entry_reservations[entry_reservation_count - 1] = link_reservation;
  loom_aie2p_program_call_contract_context_t call_context = {
      .module = module,
      .bindings_by_symbol = bindings_by_symbol,
  };
  loom_aie2p_leaf_compile_options_t resolved_entry_options = *entry_options;
  resolved_entry_options.call_contracts = (loom_low_call_contract_provider_t){
      .validate = loom_aie2p_program_call_contract_validate,
      .query = loom_aie2p_program_call_contract_query,
      .user_data = &call_context,
  };
  resolved_entry_options.allocation_reserved_ranges = entry_reservations;
  resolved_entry_options.allocation_reserved_range_count =
      entry_reservation_count;
  resolved_entry_options.preserve_link_register = true;
  resolved_entry_options.link_shadow_register = link_shadow_register;
  bool entry_compiled = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(
      module, entry_function_op, &resolved_entry_options, arena,
      &entry_compiled, &contributions[0]));
  if (!entry_compiled) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < entry_node->callee_count; ++i) {
    contributions[i + 1] = helpers[i].contribution;
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_object_merge(
      contributions, entry_node->callee_count + 1, arena, out_contribution));
  *out_compiled = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_compile(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_aie2p_leaf_compile_options_t* entry_options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_aie2p_leaf_contribution_t* out_contribution) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(entry_function_op);
  IREE_ASSERT_ARGUMENT(entry_options);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_compiled);
  IREE_ASSERT_ARGUMENT(out_contribution);
  *out_compiled = false;
  *out_contribution = (loom_aie2p_leaf_contribution_t){0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_aie2p_program_compile_impl(
      module, entry_function_op, entry_options, arena, &scratch_arena,
      out_compiled, out_contribution);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
