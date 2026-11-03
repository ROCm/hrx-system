// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_call.h"

#include <stdlib.h>
#include <string.h>

#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/parameter.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_program_ordered_clobber_t {
  loom_low_call_clobber_t clobber;
  uint64_t storage_space;
} loom_amdgpu_program_ordered_clobber_t;

static uint64_t loom_amdgpu_program_clobber_storage_space(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t register_class) {
  IREE_ASSERT_LT(register_class, descriptor_set->reg_class_count);
  return ((uint64_t)loom_low_allocation_storage_reg_class_location_kind(
              &descriptor_set->reg_classes[register_class])
          << 32) |
         loom_low_reg_class_storage_key(descriptor_set, register_class);
}

static int loom_amdgpu_program_compare_ordered_clobbers(const void* lhs_ptr,
                                                        const void* rhs_ptr) {
  const loom_amdgpu_program_ordered_clobber_t* lhs = lhs_ptr;
  const loom_amdgpu_program_ordered_clobber_t* rhs = rhs_ptr;
  if (lhs->storage_space != rhs->storage_space) {
    return lhs->storage_space < rhs->storage_space ? -1 : 1;
  }
  if (lhs->clobber.location != rhs->clobber.location) {
    return lhs->clobber.location < rhs->clobber.location ? -1 : 1;
  }
  return 0;
}

static iree_status_t loom_amdgpu_program_publish_ordered_clobbers(
    loom_amdgpu_program_ordered_clobber_t* ordered,
    iree_host_size_t ordered_count, iree_arena_allocator_t* arena,
    loom_low_call_clobber_t** out_clobbers,
    iree_host_size_t* out_clobber_count) {
  *out_clobbers = NULL;
  *out_clobber_count = 0;
  if (ordered_count == 0) {
    return iree_ok_status();
  }
  qsort(ordered, ordered_count, sizeof(*ordered),
        loom_amdgpu_program_compare_ordered_clobbers);
  loom_low_call_clobber_t* clobbers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, ordered_count, sizeof(*clobbers), (void**)&clobbers));
  iree_host_size_t clobber_count = 0;
  uint64_t prior_storage_space = 0;
  for (iree_host_size_t i = 0; i < ordered_count; ++i) {
    const loom_amdgpu_program_ordered_clobber_t* row = &ordered[i];
    if (row->clobber.count == 0) {
      continue;
    }
    if (clobber_count != 0) {
      loom_low_call_clobber_t* prior = &clobbers[clobber_count - 1];
      const uint64_t prior_end = (uint64_t)prior->location + prior->count;
      const uint64_t row_end =
          (uint64_t)row->clobber.location + row->clobber.count;
      if (prior_storage_space == row->storage_space &&
          row->clobber.location <= prior_end) {
        const uint64_t merged_end = iree_max(prior_end, row_end);
        IREE_ASSERT_LE(merged_end, (uint64_t)UINT32_MAX + 1u);
        prior->count = (uint32_t)(merged_end - prior->location);
        continue;
      }
    }
    clobbers[clobber_count++] = row->clobber;
    prior_storage_space = row->storage_space;
  }
  *out_clobbers = clobbers;
  *out_clobber_count = clobber_count;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_call_table_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t symbol_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_program_call_table_t* out_table) {
  *out_table = (loom_amdgpu_program_call_table_t){
      .descriptor_set = descriptor_set,
      .symbol_count = symbol_count,
  };
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, symbol_count, sizeof(*out_table->callables_by_symbol),
        (void**)&out_table->callables_by_symbol));
    memset(out_table->callables_by_symbol, 0,
           symbol_count * sizeof(*out_table->callables_by_symbol));
  }

  iree_host_size_t clobber_capacity = 1;
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    if (class_id != LOOM_AMDGPU_REG_CLASS_ID_EXEC &&
        loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
      clobber_capacity += reg_class->allocatable_count;
    }
  }
  loom_amdgpu_program_ordered_clobber_t* ordered = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, clobber_capacity, sizeof(*ordered), (void**)&ordered));
  iree_host_size_t clobber_count = 0;
  ordered[clobber_count++] = (loom_amdgpu_program_ordered_clobber_t){
      .clobber =
          {
              .register_class = LOOM_AMDGPU_REG_CLASS_ID_SGPR,
              .location = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE,
              .count = LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT,
          },
      .storage_space = loom_amdgpu_program_clobber_storage_space(
          descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR),
  };
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    if (class_id == LOOM_AMDGPU_REG_CLASS_ID_EXEC ||
        !loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
      continue;
    }
    const uint64_t storage_space =
        loom_amdgpu_program_clobber_storage_space(descriptor_set, class_id);
    for (uint16_t i = 0; i < reg_class->allocatable_count; ++i) {
      ordered[clobber_count++] = (loom_amdgpu_program_ordered_clobber_t){
          .clobber =
              {
                  .register_class = class_id,
                  .location =
                      loom_low_descriptor_set_physical_register_candidate(
                          descriptor_set, class_id, i),
                  .count = 1,
              },
          .storage_space = storage_space,
      };
    }
  }
  IREE_ASSERT_EQ(clobber_count, clobber_capacity);
  loom_low_call_clobber_t* common_clobbers = NULL;
  iree_host_size_t common_clobber_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_publish_ordered_clobbers(
      ordered, clobber_count, arena, &common_clobbers, &common_clobber_count));
  out_table->common_clobbers = (loom_low_call_clobber_list_t){
      .values = common_clobbers,
      .count = common_clobber_count,
  };
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_program_callable_emit_unsupported_abi(
    const loom_module_t* module, loom_op_t* function_op,
    iree_string_view_t side, uint32_t index, loom_type_t type,
    loom_amdgpu_call_abi_requirements_t requirements, iree_string_view_t reason,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
      loom_param_string(side),
      loom_param_u32(index),
      loom_param_type(type),
      loom_param_u32(requirements.sgpr_count),
      loom_param_u32(requirements.vgpr_count),
      loom_param_string(reason),
  };
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = function_op,
      .error = LOOM_ERR_AMDGPU_055,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_amdgpu_program_callable_prepare_abi_side(
    const loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_id_t* values, uint16_t value_count,
    iree_string_view_t side, iree_host_size_t location_offset,
    iree_diagnostic_emitter_t emitter, bool* out_prepared,
    loom_amdgpu_program_callable_t* callable,
    loom_amdgpu_call_abi_requirements_t* out_requirements) {
  *out_prepared = false;
  loom_amdgpu_call_abi_requirements_t requirements = {0};
  for (uint16_t i = 0; i < value_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, values[i]);
    loom_amdgpu_call_abi_requirements_t next_requirements = requirements;
    if (!loom_amdgpu_call_abi_add_low_type(descriptor_set, type,
                                           &next_requirements)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_emit_unsupported_abi(
          module, function_op, side, i, type, requirements,
          IREE_SV("unsupported-carrier"), emitter));
      return iree_ok_status();
    }
    if (!loom_amdgpu_call_abi_requirements_fit(next_requirements)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_emit_unsupported_abi(
          module, function_op, side, i, type, next_requirements,
          IREE_SV("carrier-capacity"), emitter));
      return iree_ok_status();
    }
    loom_low_allocation_abi_location_t* location =
        &callable->locations[location_offset + i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_call_abi_select_location(
        descriptor_set, type, &requirements, location));
    callable->transport_clobbers[location_offset + i] =
        (loom_low_call_clobber_t){
            .register_class = location->descriptor_reg_class_id,
            .location = location->location_base,
            .count = loom_low_register_type_unit_count(type),
        };
  }
  *out_requirements = requirements;
  *out_prepared = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_callable_prepare_abi(
    const loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_prepared, loom_amdgpu_program_callable_t* callable) {
  *out_prepared = false;
  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &callable->argument_count);
  callable->result_count = function_op->result_count;
  const iree_host_size_t location_count =
      (iree_host_size_t)callable->argument_count + callable->result_count;
  if (location_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, location_count, sizeof(*callable->locations),
        (void**)&callable->locations));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, location_count, sizeof(*callable->transport_clobbers),
        (void**)&callable->transport_clobbers));
  }
  callable->transport_clobber_count = location_count;

  bool side_prepared = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_prepare_abi_side(
      module, function_op, descriptor_set, arguments, callable->argument_count,
      IREE_SV("argument"), 0, emitter, &side_prepared, callable,
      &callable->argument_requirements));
  if (!side_prepared) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_callable_prepare_abi_side(
      module, function_op, descriptor_set, loom_op_results(function_op),
      callable->result_count, IREE_SV("result"), callable->argument_count,
      emitter, &side_prepared, callable, &callable->result_requirements));
  if (!side_prepared) {
    return iree_ok_status();
  }
  callable->call_contract = (loom_low_call_contract_t){
      .arguments = callable->locations,
      .argument_count = callable->argument_count,
      .results = callable->locations != NULL
                     ? callable->locations + callable->argument_count
                     : NULL,
      .result_count = callable->result_count,
  };
  *out_prepared = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_program_callable_prepare_clobbers(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_program_callable_t* callable) {
  const loom_low_allocation_table_t* allocation =
      &callable->compilation.frame.allocation;
  IREE_ASSERT_EQ(allocation->physical_extents.count,
                 descriptor_set->reg_class_count);
  iree_host_size_t clobber_capacity = callable->transport_clobber_count;
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    if (!loom_low_reg_class_uses_explicit_physical_registers(reg_class) &&
        allocation->physical_extents.ends_by_reg_class[class_id] != 0) {
      ++clobber_capacity;
    }
  }
  loom_amdgpu_program_ordered_clobber_t* ordered = NULL;
  if (clobber_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, clobber_capacity, sizeof(*ordered), (void**)&ordered));
  }
  iree_host_size_t clobber_count = 0;
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    const uint32_t extent =
        allocation->physical_extents.ends_by_reg_class[class_id];
    if (extent == 0 ||
        loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
      continue;
    }
    ordered[clobber_count++] = (loom_amdgpu_program_ordered_clobber_t){
        .clobber =
            {
                .register_class = class_id,
                .location = 0,
                .count = extent,
            },
        .storage_space =
            loom_amdgpu_program_clobber_storage_space(descriptor_set, class_id),
    };
  }
  for (iree_host_size_t i = 0; i < callable->transport_clobber_count; ++i) {
    const loom_low_call_clobber_t clobber = callable->transport_clobbers[i];
    ordered[clobber_count++] = (loom_amdgpu_program_ordered_clobber_t){
        .clobber = clobber,
        .storage_space = loom_amdgpu_program_clobber_storage_space(
            descriptor_set, clobber.register_class),
    };
  }
  IREE_ASSERT_EQ(clobber_count, clobber_capacity);
  IREE_RETURN_IF_ERROR(loom_amdgpu_program_publish_ordered_clobbers(
      ordered, clobber_count, arena, &callable->clobbers,
      &callable->clobber_count));
  callable->call_contract.clobbers = (loom_low_call_clobber_list_t){
      .values = callable->clobbers,
      .count = callable->clobber_count,
  };
  return iree_ok_status();
}

void loom_amdgpu_program_call_table_publish(
    loom_amdgpu_program_call_table_t* table,
    loom_amdgpu_program_callable_t* callable) {
  IREE_ASSERT_EQ(callable->function.module_id, 0u);
  IREE_ASSERT_LT(callable->function.symbol_id, table->symbol_count);
  IREE_ASSERT(table->callables_by_symbol[callable->function.symbol_id] == NULL);
  table->callables_by_symbol[callable->function.symbol_id] = callable;
}

static const loom_low_call_contract_t* loom_amdgpu_program_call_table_query(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_amdgpu_program_call_table_t* table = user_data;
  IREE_ASSERT_EQ(callee.module_id, 0u);
  IREE_ASSERT_LT(callee.symbol_id, table->symbol_count);
  IREE_ASSERT(table->callables_by_symbol[callee.symbol_id] != NULL);
  return &table->callables_by_symbol[callee.symbol_id]->call_contract;
}

static loom_low_call_clobber_list_t
loom_amdgpu_program_call_table_query_common_clobbers(
    void* user_data, const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_program_call_table_t* table = user_data;
  IREE_ASSERT_EQ(descriptor_set, table->descriptor_set);
  return table->common_clobbers;
}

loom_low_call_contract_provider_t loom_amdgpu_program_call_table_provider(
    loom_amdgpu_program_call_table_t* table) {
  return (loom_low_call_contract_provider_t){
      .query = loom_amdgpu_program_call_table_query,
      .query_common_clobbers =
          loom_amdgpu_program_call_table_query_common_clobbers,
      .user_data = table,
  };
}

static const loom_amdgpu_call_state_t*
loom_amdgpu_program_call_table_query_state(void* user_data,
                                           loom_symbol_ref_t callee) {
  const loom_amdgpu_program_call_table_t* table = user_data;
  IREE_ASSERT_EQ(callee.module_id, 0u);
  IREE_ASSERT_LT(callee.symbol_id, table->symbol_count);
  IREE_ASSERT(table->callables_by_symbol[callee.symbol_id] != NULL);
  return &table->callables_by_symbol[callee.symbol_id]->call_state;
}

loom_amdgpu_call_state_provider_t loom_amdgpu_program_call_table_state_provider(
    loom_amdgpu_program_call_table_t* table) {
  return (loom_amdgpu_call_state_provider_t){
      .query = loom_amdgpu_program_call_table_query_state,
      .user_data = table,
  };
}
