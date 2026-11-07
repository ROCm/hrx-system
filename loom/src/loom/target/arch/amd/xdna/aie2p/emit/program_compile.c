// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_compile.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_capacity.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"
#include "loom/target/registers.h"

typedef struct loom_aie2p_program_callable_t {
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
} loom_aie2p_program_callable_t;

struct loom_aie2p_program_compiler_t {
  // Prepared module whose helper target versions remain fixed.
  loom_module_t* module;
  // Owner of retained helper objects, contracts, and complete programs.
  iree_arena_allocator_t* arena;
  // Whether every retained helper carries its function-local compile report.
  bool retains_compile_reports;
  // Requested report detail fixed for every program compiled by this instance.
  loom_target_compile_report_detail_flags_t requested_detail_flags;
  // Compiled helpers indexed by source symbol; allocated on first native call.
  loom_aie2p_program_callable_t** helpers_by_symbol;
  // Optional helper reports indexed in parallel with |helpers_by_symbol|.
  loom_target_compile_report_t** helper_reports_by_symbol;
};

iree_status_t loom_aie2p_program_compiler_allocate(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_target_compile_report_t* report_template,
    loom_aie2p_program_compiler_t** out_compiler) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(**out_compiler), (void**)out_compiler));
  **out_compiler = (loom_aie2p_program_compiler_t){
      .module = module,
      .arena = arena,
      .retains_compile_reports = report_template != NULL,
      .requested_detail_flags =
          report_template != NULL ? report_template->requested_detail_flags : 0,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_call_contract_validate(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_aie2p_program_compiler_t* context = user_data;
  if (callee.module_id == 0 &&
      callee.symbol_id < context->module->symbols.count &&
      context->helpers_by_symbol[callee.symbol_id] != NULL) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "AIE2P direct call has no compiled local helper");
}

static const loom_low_call_contract_t* loom_aie2p_program_call_contract_query(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_aie2p_program_compiler_t* context = user_data;
  return &context->helpers_by_symbol[callee.symbol_id]->call_contract;
}

static iree_status_t loom_aie2p_program_prepare_abi(
    const loom_module_t* module, loom_op_t* function_op,
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

static iree_host_size_t loom_aie2p_program_transport_clobber_count(
    const loom_aie2p_program_callable_t* callable) {
  iree_host_size_t count = 0;
  const iree_host_size_t location_count =
      (iree_host_size_t)callable->argument_count + callable->result_count;
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    count += callable->locations[i].location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
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
  const iree_host_size_t clobber_capacity =
      loom_aie2p_program_clobber_count(descriptor_set,
                                       &callable->contribution.realization) +
      loom_aie2p_program_transport_clobber_count(callable);
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
          loom_aie2p_program_compare_clobbers);
  }
  callable->clobber_count = 0;
  for (iree_host_size_t i = 0; i < clobber_index; ++i) {
    if (callable->clobber_count != 0 &&
        loom_aie2p_program_compare_clobbers(
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

// Composes function-local reports into one final physical tile program. Native
// code is resident once per helper, while its dynamic mix scales by the exact
// number of calls made by one entry invocation. Final padding remains a target
// layout fact taken from the merged program object.
static iree_status_t loom_aie2p_program_compose_report(
    const loom_aie2p_program_compiler_t* compiler,
    const loom_op_t* entry_function_op,
    loom_aie2p_program_helper_list_t direct_helpers,
    const loom_aie2p_program_object_t* program,
    loom_target_compile_report_t* report) {
  const loom_symbol_ref_t entry_ref =
      loom_low_func_def_callee(entry_function_op);
  uint64_t emitted_instruction_count = report->emitted_instruction_count;
  uint64_t emitted_code_byte_count = report->emitted_code_byte_count;
  loom_target_compile_report_emission_breakdown_t emission_breakdown =
      report->emission_breakdown;
  loom_target_compile_report_static_instruction_mix_t static_mix =
      report->static_instruction_mix;
  for (iree_host_size_t i = 0; i < direct_helpers.count; ++i) {
    const loom_symbol_ref_t helper_ref = direct_helpers.values[i].function;
    const loom_aie2p_program_callable_t* helper =
        compiler->helpers_by_symbol[helper_ref.symbol_id];
    IREE_ASSERT(helper != NULL);
    const loom_target_compile_report_t* helper_report =
        compiler->helper_reports_by_symbol[helper_ref.symbol_id];
    IREE_ASSERT(helper_report != NULL);
    const loom_target_compile_report_call_edge_t* call_edge =
        loom_target_compile_report_find_call_edge(report, entry_ref,
                                                  helper_ref);
    IREE_ASSERT(call_edge != NULL);
    const loom_target_compile_report_function_merge_options_t merge_options = {
        .invocation_count = call_edge->dynamic_call_count,
        .has_exact_invocation_count = iree_all_bits_set(
            call_edge->flags,
            LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT),
    };
    IREE_RETURN_IF_ERROR(loom_target_compile_report_merge_function_evidence(
        report, helper_report, &merge_options));
    if (program->canonical_indices[i + 1] != i + 1) {
      continue;
    }
    loom_target_compile_report_accumulate_static_mix(
        &static_mix, &helper_report->static_instruction_mix);
    emitted_instruction_count += helper_report->emitted_instruction_count;
    emitted_code_byte_count += helper_report->emitted_code_byte_count;
    emission_breakdown.body_instruction_count +=
        helper_report->emission_breakdown.body_instruction_count;
    emission_breakdown.entry_instruction_count +=
        helper_report->emission_breakdown.entry_instruction_count;
    emission_breakdown.coissued_instruction_count +=
        helper_report->emission_breakdown.coissued_instruction_count;
    emission_breakdown.coissued_component_count +=
        helper_report->emission_breakdown.coissued_component_count;
  }
  loom_target_compile_report_record_emission(
      report, emitted_instruction_count, emitted_code_byte_count,
      program->contribution.realization.code.byte_length);
  loom_target_compile_report_record_emission_breakdown(report,
                                                       &emission_breakdown);
  // Logical helpers retain their compilation evidence and dynamic calls, but
  // physical instruction counts include each canonical body only once.
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    loom_target_compile_report_record_static_instruction_mix(report,
                                                             &static_mix);
  }
  return iree_ok_status();
}

// Native contributions retain module-unique names. Project the already admitted
// inventory once so composition can bind those names and preserve constraints
// between definitions used by different functions.
static iree_status_t loom_aie2p_program_bind_data(
    const loom_module_t* module, const loom_low_read_only_data_set_t* inventory,
    iree_arena_allocator_t* arena, loom_aie2p_program_data_t** out_data) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, inventory->count, sizeof(**out_data), (void**)out_data));
  for (iree_host_size_t i = 0; i < inventory->count; ++i) {
    const loom_low_read_only_data_requirement_t* source = &inventory->values[i];
    const loom_symbol_t* symbol =
        &module->symbols.entries[source->symbol.symbol_id];
    uint32_t* conflicts = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, source->bank_conflicts.count,
                                  sizeof(*conflicts), (void**)&conflicts));
    iree_host_size_t conflict_count = 0;
    for (iree_host_size_t j = 0; j < source->bank_conflicts.count; ++j) {
      const uint32_t ordinal = loom_low_read_only_data_ordinal(
          inventory, source->bank_conflicts.values[j]);
      if (ordinal != LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID) {
        conflicts[conflict_count++] = ordinal;
      }
    }
    (*out_data)[i] = (loom_aie2p_program_data_t){
        .name = loom_string_table_get(&module->strings, symbol->name_id),
        .bank_conflicts = conflicts,
        .bank_conflict_count = conflict_count,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_compile_impl(
    loom_aie2p_program_compiler_t* compiler, loom_op_t* entry_function_op,
    bool may_return, loom_aie2p_program_helper_list_t direct_helpers,
    const loom_low_read_only_data_set_t* program_data,
    const loom_aie2p_leaf_compile_options_t* entry_options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    bool* out_compiled, loom_aie2p_leaf_contribution_t* out_contribution) {
  loom_module_t* module = compiler->module;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  // r24 and r25 are the position registers in the two load-FIFO tuples. Keep
  // the link shadow out of both tuples so native calls do not reduce the
  // worker from two simultaneous input streams to one.
  const loom_aie2p_physical_register_id_t link_shadow_register =
      loom_aie2p_machine_find_physical_register(IREE_SV("r23"));
  IREE_ASSERT(link_shadow_register != LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
  const loom_low_allocation_reserved_range_t link_reservation = {
      .register_class = IREE_SV("aie2p.er"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = link_shadow_register,
      .location_count = 1,
  };
  enum {
    LOOM_AIE2P_PROGRAM_HELPER_RESERVATION_COUNT =
        1 + LOOM_AIE2P_CALL_ABI_PRESERVED_SCALAR_REGISTER_COUNT,
  };
  loom_low_allocation_reserved_range_t
      helper_reservations[LOOM_AIE2P_PROGRAM_HELPER_RESERVATION_COUNT];
  for (uint16_t i = 0; i < LOOM_AIE2P_CALL_ABI_PRESERVED_SCALAR_REGISTER_COUNT;
       ++i) {
    helper_reservations[i] = (loom_low_allocation_reserved_range_t){
        .register_class = IREE_SV("aie2p.er"),
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = loom_aie2p_call_abi_preserved_scalar_register(i),
        .location_count = 1,
    };
  }
  helper_reservations[LOOM_AIE2P_PROGRAM_HELPER_RESERVATION_COUNT - 1] =
      link_reservation;

  loom_aie2p_leaf_contribution_t* contributions = NULL;
  if (compiler->helpers_by_symbol == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*compiler->helpers_by_symbol),
        (void**)&compiler->helpers_by_symbol));
    memset(compiler->helpers_by_symbol, 0,
           module->symbols.count * sizeof(*compiler->helpers_by_symbol));
    if (compiler->retains_compile_reports) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, module->symbols.count,
          sizeof(*compiler->helper_reports_by_symbol),
          (void**)&compiler->helper_reports_by_symbol));
      memset(
          compiler->helper_reports_by_symbol, 0,
          module->symbols.count * sizeof(*compiler->helper_reports_by_symbol));
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, direct_helpers.count + 1, sizeof(*contributions),
      (void**)&contributions));

  for (iree_host_size_t i = 0; i < direct_helpers.count; ++i) {
    const loom_aie2p_program_helper_t* helper_input = &direct_helpers.values[i];
    const loom_symbol_ref_t helper_ref = helper_input->function;
    IREE_ASSERT_EQ(helper_ref.module_id, 0u);
    IREE_ASSERT_LT(helper_ref.symbol_id, module->symbols.count);
    const loom_symbol_id_t symbol_id = helper_ref.symbol_id;
    const loom_aie2p_program_callable_t* retained_helper =
        compiler->helpers_by_symbol[symbol_id];
    if (retained_helper != NULL) {
      contributions[i + 1] = retained_helper->contribution;
      continue;
    }
    loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    IREE_ASSERT(symbol->defining_op != NULL);
    IREE_ASSERT(loom_low_func_def_isa(symbol->defining_op));
    IREE_ASSERT(loom_func_like_is_module_internal(
        loom_func_like_cast(module, symbol->defining_op)));
    loom_aie2p_program_callable_t* helper = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*helper), (void**)&helper));
    *helper = (loom_aie2p_program_callable_t){0};
    IREE_RETURN_IF_ERROR(loom_aie2p_program_prepare_abi(
        module, symbol->defining_op, arena, helper));
    loom_target_compile_report_t* helper_report_ptr = NULL;
    if (compiler->retains_compile_reports) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          arena, sizeof(*helper_report_ptr), (void**)&helper_report_ptr));
      loom_target_compile_report_initialize(iree_arena_allocator(arena),
                                            helper_report_ptr);
      helper_report_ptr->requested_detail_flags =
          compiler->requested_detail_flags;
    }
    const loom_aie2p_leaf_compile_options_t helper_options = {
        .function_target_facts = helper_input->function_target_facts,
        .memory_accesses = helper_input->memory_accesses,
        .loop_execution = helper_input->loop_execution,
        .descriptor_registry = entry_options->descriptor_registry,
        .allocation_entry_locations = helper->call_contract.arguments,
        .allocation_entry_location_count = helper->call_contract.argument_count,
        .allocation_exit_locations = helper->call_contract.results,
        .allocation_exit_location_count = helper->call_contract.result_count,
        .allocation_reserved_ranges = helper_reservations,
        .allocation_reserved_range_count =
            LOOM_AIE2P_PROGRAM_HELPER_RESERVATION_COUNT,
        .diagnostic_emitter = entry_options->diagnostic_emitter,
        .compile_report = helper_report_ptr,
    };
    bool helper_compiled = false;
    iree_status_t status =
        loom_aie2p_leaf_compile(module, symbol->defining_op, &helper_options,
                                arena, &helper_compiled, &helper->contribution);
    if (helper_report_ptr != NULL) {
      if (!helper_compiled) {
        status = iree_status_join(
            status, loom_target_compile_report_merge_evidence(
                        entry_options->compile_report, helper_report_ptr));
      }
    }
    IREE_RETURN_IF_ERROR(status);
    if (!helper_compiled) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_prepare_clobbers(descriptor_set, arena, helper));
    compiler->helpers_by_symbol[symbol_id] = helper;
    if (helper_report_ptr != NULL) {
      compiler->helper_reports_by_symbol[symbol_id] = helper_report_ptr;
    }
    contributions[i + 1] = helper->contribution;
  }

  loom_aie2p_leaf_compile_options_t resolved_entry_options = *entry_options;
  resolved_entry_options.call_contracts = (loom_low_call_contract_provider_t){
      .validate = loom_aie2p_program_call_contract_validate,
      .query = loom_aie2p_program_call_contract_query,
      .user_data = compiler,
  };
  if (may_return) {
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
    resolved_entry_options.allocation_reserved_ranges = entry_reservations;
    resolved_entry_options.allocation_reserved_range_count =
        entry_reservation_count;
  }
  resolved_entry_options.preserve_link_register = may_return;
  resolved_entry_options.link_shadow_register = link_shadow_register;
  bool entry_compiled = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(
      module, entry_function_op, &resolved_entry_options, arena,
      &entry_compiled, &contributions[0]));
  if (!entry_compiled) {
    return iree_ok_status();
  }
  loom_aie2p_program_data_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_bind_data(module, program_data, scratch_arena, &data));
  loom_aie2p_program_object_t program = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_program_object_merge(
      contributions, direct_helpers.count + 1, data, program_data->count, arena,
      scratch_arena, &program));
  if (entry_options->compile_report != NULL) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_compose_report(
        compiler, entry_function_op, direct_helpers, &program,
        entry_options->compile_report));
  }
  bool capacity_admitted = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_capacity_admit(
      module, entry_function_op,
      program.contribution.realization.code.byte_length,
      entry_options->diagnostic_emitter, &capacity_admitted));
  if (!capacity_admitted) {
    return iree_ok_status();
  }
  *out_contribution = program.contribution;
  *out_compiled = true;
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_compile(
    loom_aie2p_program_compiler_t* compiler, loom_op_t* entry_function_op,
    bool may_return, loom_aie2p_program_helper_list_t direct_helpers,
    const loom_low_read_only_data_set_t* program_data,
    const loom_aie2p_leaf_compile_options_t* entry_options, bool* out_compiled,
    loom_aie2p_leaf_contribution_t* out_contribution) {
  IREE_ASSERT_ARGUMENT(compiler);
  IREE_ASSERT_ARGUMENT(entry_function_op);
  IREE_ASSERT_ARGUMENT(entry_options);
  IREE_ASSERT_ARGUMENT(out_compiled);
  IREE_ASSERT_ARGUMENT(out_contribution);
  IREE_ASSERT_EQ(compiler->retains_compile_reports,
                 entry_options->compile_report != NULL);
  if (entry_options->compile_report != NULL) {
    IREE_ASSERT_EQ(compiler->requested_detail_flags,
                   entry_options->compile_report->requested_detail_flags);
  }
  *out_compiled = false;
  *out_contribution = (loom_aie2p_leaf_contribution_t){0};
  iree_arena_allocator_t* arena = compiler->arena;
  if (direct_helpers.count == 0) {
    return loom_aie2p_leaf_compile(compiler->module, entry_function_op,
                                   entry_options, arena, out_compiled,
                                   out_contribution);
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_status_t status = loom_aie2p_program_compile_impl(
      compiler, entry_function_op, may_return, direct_helpers, program_data,
      entry_options, arena, &scratch_arena, out_compiled, out_contribution);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
