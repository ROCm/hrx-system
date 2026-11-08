// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/type_refs.h"
#include "loom/ops/cache.h"
#include "loom/ops/callable_effects.h"
#include "loom/ops/combining.h"
#include "loom/ops/command/composition.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/kernel/launch_config_contract.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/template/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

static iree_status_t loom_kernel_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_kernel_emit_integer_field_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  return loom_kernel_emit_integer_field_constraint(
      emitter, op, attr_name, actual_value, expected_constraint);
}

static iree_status_t loom_kernel_verify_positive_u32_attr(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_attribute_t attribute, iree_string_view_t attr_name) {
  if (loom_attr_is_absent(attribute)) {
    return iree_ok_status();
  }
  const int64_t value = loom_attr_as_i64(attribute);
  if (value > 0 && value <= UINT32_MAX) {
    return iree_ok_status();
  }
  return loom_kernel_emit_attribute_value_constraint(
      emitter, op, attr_name, value, IREE_SV("positive u32"));
}

static iree_string_view_t loom_kernel_launch_config_effect_kind(
    loom_trait_flags_t traits) {
  if (iree_any_bit_set(traits, LOOM_TRAIT_WRITES_MEMORY)) {
    return IREE_SV("memory writes");
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_NON_DETERMINISTIC)) {
    return IREE_SV("non-deterministic behavior");
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    return IREE_SV("unknown effects");
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_UNIQUE_IDENTITY)) {
    return IREE_SV("unique-identity behavior");
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_CONVERGENT)) {
    return IREE_SV("convergent behavior");
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_MEMORY_FENCE)) {
    return IREE_SV("memory-ordering effects");
  } else if (iree_any_bit_set(traits, LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return IREE_SV("independently observable effects");
  }
  return IREE_SV("unsupported effects");
}

static iree_string_view_t loom_kernel_launch_config_field_name(
    loom_kernel_launch_config_contract_failure_t failure, uint8_t dimension) {
  static const iree_string_view_t workgroup_count_names[] = {
      IREE_SVL("workgroup_count[x]"),
      IREE_SVL("workgroup_count[y]"),
      IREE_SVL("workgroup_count[z]"),
  };
  static const iree_string_view_t workgroup_size_names[] = {
      IREE_SVL("workgroup_size[x]"),
      IREE_SVL("workgroup_size[y]"),
      IREE_SVL("workgroup_size[z]"),
  };
  static const iree_string_view_t cluster_size_names[] = {
      IREE_SVL("cluster_size[x]"),
      IREE_SVL("cluster_size[y]"),
      IREE_SVL("cluster_size[z]"),
  };
  IREE_ASSERT_LT(dimension, 3);
  switch (failure) {
    case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_WORKGROUP_COUNT:
      return workgroup_count_names[dimension];
    case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_WORKGROUP_SIZE:
      return workgroup_size_names[dimension];
    case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_CLUSTER_SIZE:
      return cluster_size_names[dimension];
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_kernel_verify_launch_config_contract(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_kernel_launch_config_contract_t contract = {0};
  iree_status_t status =
      loom_kernel_launch_config_contract_analyze(module, op, &arena, &contract);
  if (iree_status_is_ok(status)) {
    switch (contract.failure) {
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE:
        break;
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_EFFECT: {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_op_name(module, contract.failure_op)),
            loom_param_string(
                loom_kernel_launch_config_effect_kind(contract.failure_traits)),
        };
        status = loom_kernel_emit(emitter, contract.failure_op,
                                  LOOM_ERR_STRUCTURE_058, params,
                                  IREE_ARRAYSIZE(params));
        break;
      }
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_OPERATION: {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_op_name(module, contract.failure_op)),
        };
        status = loom_kernel_emit(emitter, contract.failure_op,
                                  LOOM_ERR_STRUCTURE_059, params,
                                  IREE_ARRAYSIZE(params));
        break;
      }
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_ORIGIN: {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_op_name(module, contract.failure_op)),
        };
        status = loom_kernel_emit(emitter, contract.failure_op,
                                  LOOM_ERR_STRUCTURE_060, params,
                                  IREE_ARRAYSIZE(params));
        break;
      }
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_WORKGROUP_SIZE:
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_CLUSTER_SIZE: {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_kernel_launch_config_field_name(
                contract.failure, contract.failure_dimension)),
        };
        status = loom_kernel_emit(emitter, contract.failure_op,
                                  LOOM_ERR_STRUCTURE_061, params,
                                  IREE_ARRAYSIZE(params));
        break;
      }
      case LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_WORKGROUP_COUNT: {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(loom_kernel_launch_config_field_name(
                contract.failure, contract.failure_dimension)),
            loom_param_i64(contract.failure_range_lower),
            loom_param_i64(contract.failure_range_upper),
        };
        status = loom_kernel_emit(emitter, contract.failure_op,
                                  LOOM_ERR_STRUCTURE_062, params,
                                  IREE_ARRAYSIZE(params));
        break;
      }
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}

static iree_status_t loom_kernel_emit_entry_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* use_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = use_op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_kernel_verify_entry_operand_group(
    const loom_module_t* module, const loom_op_t* launch_op,
    const loom_op_t* definition_op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t actual_field_name,
    iree_string_view_t expected_field_name, loom_value_slice_t actual_values,
    const loom_value_id_t* expected_values, uint16_t expected_value_count,
    uint16_t flat_operand_offset,
    const loom_type_value_remap_t* enclosing_remap) {
  if (actual_values.count != expected_value_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(actual_field_name),
        loom_param_u32(actual_values.count),
        loom_param_string(expected_field_name),
        loom_param_u32(expected_value_count),
    };
    return loom_kernel_emit_entry_related(emitter, launch_op, definition_op,
                                          LOOM_ERR_STRUCTURE_013, params,
                                          IREE_ARRAYSIZE(params));
  }

  loom_type_value_remap_t value_remap = {
      .source_values = expected_values,
      .target_values = actual_values.values,
      .count = actual_values.count,
      .flags = actual_values.count > 2
                   ? LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE
                   : 0,
      .next = enclosing_remap,
  };
  for (uint16_t i = 0; i < actual_values.count; ++i) {
    loom_type_t actual_type =
        loom_module_value_type(module, actual_values.values[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, expected_values[i]);
    bool matches = false;
    IREE_RETURN_IF_ERROR(loom_function_call_argument_type_matches(
        module, actual_type, expected_type, &value_remap,
        LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION,
        &matches));
    if (matches) {
      continue;
    }

    char actual_name[32];
    char expected_name[32];
    iree_snprintf(actual_name, sizeof(actual_name), "%.*s %u",
                  (int)actual_field_name.size, actual_field_name.data, i);
    iree_snprintf(expected_name, sizeof(expected_name), "%.*s %u",
                  (int)expected_field_name.size, expected_field_name.data, i);
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(
            loom_param_string(iree_make_cstring_view(actual_name)),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                      flat_operand_offset + i)),
        loom_param_type(actual_type),
        loom_param_string(iree_make_cstring_view(expected_name)),
        loom_param_type(expected_type),
    };
    return loom_kernel_emit_entry_related(emitter, launch_op, definition_op,
                                          LOOM_ERR_TYPE_001, params,
                                          IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_emit_launch_placement_error(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t expected_parent) {
  iree_string_view_t actual_parent =
      op->parent_op ? loom_op_name(module, op->parent_op) : IREE_SV("none");
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(IREE_SV("direct")),
      loom_param_string(expected_parent),
      loom_param_string(actual_parent),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_029, params,
                          IREE_ARRAYSIZE(params));
}

iree_status_t loom_kernel_launch_config_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  (void)module;
  const uint32_t cluster_dimension_count =
      (uint32_t)loom_kernel_launch_config_workgroup_cluster_size_x_is_present(
          op) +
      (uint32_t)loom_kernel_launch_config_workgroup_cluster_size_y_is_present(
          op) +
      (uint32_t)loom_kernel_launch_config_workgroup_cluster_size_z_is_present(
          op);
  if (cluster_dimension_count == 0 || cluster_dimension_count == 3) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("kernel.launch.config")),
      loom_param_u32(op->operand_count),
      loom_param_u32(9),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_kernel_value_name(const loom_module_t* module,
                                                 loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return IREE_SV("<invalid>");
  }
  loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return loom_string_table_get(&module->strings, name_id);
}

static iree_string_view_t loom_kernel_op_name(const loom_module_t* module,
                                              const loom_op_t* op) {
  if (!op) {
    return IREE_SV("<null>");
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) {
    return IREE_SV("<unknown>");
  }
  return loom_op_vtable_name(vtable);
}

static iree_status_t loom_kernel_emit_value_use_count_constraint(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t value_id, uint32_t actual_count,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_value_name(module, value_id)),
      loom_param_u32(actual_count),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_DOMINANCE_009, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_value_user_constraint(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t value_id, const loom_op_t* user_op,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_value_name(module, value_id)),
      loom_param_string(loom_kernel_op_name(module, user_op)),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_DOMINANCE_010, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_kernel_type_is_opaque_dialect(const loom_module_t* module,
                                               loom_type_t type,
                                               iree_string_view_t name) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  loom_string_id_t name_id = loom_type_dialect_name_id(type);
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(
      loom_string_table_get(&module->strings, name_id), name);
}

static bool loom_kernel_type_is_async_token(const loom_module_t* module,
                                            loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(module, type,
                                            IREE_SV("kernel.async.token"));
}

static bool loom_kernel_type_is_async_group(const loom_module_t* module,
                                            loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(module, type,
                                            IREE_SV("kernel.async.group"));
}

static bool loom_kernel_type_is_barrier_phase(const loom_module_t* module,
                                              loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(module, type,
                                            IREE_SV("kernel.barrier.phase"));
}

static bool loom_kernel_type_is_tensor_lds_descriptor(
    const loom_module_t* module, loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(
      module, type, IREE_SV("kernel.tensor.lds.descriptor"));
}

static iree_status_t loom_kernel_verify_result_async_token(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name,
    loom_value_id_t result_id) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_kernel_type_is_async_token(module, result_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(
      emitter, op, result_name, result_type, IREE_SV("kernel.async.token"));
}

static iree_status_t loom_kernel_verify_result_async_group(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name,
    loom_value_id_t result_id) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_kernel_type_is_async_group(module, result_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(
      emitter, op, result_name, result_type, IREE_SV("kernel.async.group"));
}

static iree_status_t loom_kernel_verify_result_tensor_lds_descriptor(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name,
    loom_value_id_t result_id) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_kernel_type_is_tensor_lds_descriptor(module, result_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(
      emitter, op, result_name, result_type,
      IREE_SV("kernel.tensor.lds.descriptor"));
}

static iree_status_t loom_kernel_verify_operand_async_group(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_kernel_type_is_async_group(module, operand_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, operand_type, IREE_SV("kernel.async.group"));
}

static iree_status_t loom_kernel_verify_operand_tensor_lds_descriptor(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_kernel_type_is_tensor_lds_descriptor(module, operand_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, operand_type,
      IREE_SV("kernel.tensor.lds.descriptor"));
}

static iree_status_t loom_kernel_verify_operand_i32(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_type_is_scalar(operand_type) &&
      loom_type_element_type(operand_type) == LOOM_SCALAR_TYPE_I32) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(emitter, op, operand_name,
                                             operand_type, IREE_SV("i32"));
}

static bool loom_kernel_type_is_collective_value(loom_type_t type) {
  return loom_type_is_scalar(type) ||
         (loom_type_is_vector(type) && loom_type_rank(type) == 1);
}

static iree_status_t loom_kernel_verify_collective_operand(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_kernel_type_is_collective_value(operand_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, operand_type,
      IREE_SV("scalar or rank-1 vector"));
}

static iree_status_t loom_kernel_verify_first_result_i32_or_i64(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name) {
  if (op->result_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t result_id = loom_op_const_results(op)[0];
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_type_is_scalar(result_type)) {
    loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
    if (scalar_type == LOOM_SCALAR_TYPE_I32 ||
        scalar_type == LOOM_SCALAR_TYPE_I64) {
      return iree_ok_status();
    }
  }
  return loom_kernel_emit_result_constraint(emitter, op, result_name,
                                            result_type, IREE_SV("i32 or i64"));
}

static iree_status_t loom_kernel_verify_result_i1(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, uint16_t result_index,
    iree_string_view_t result_name) {
  if (result_index >= op->result_count) {
    return iree_ok_status();
  }
  loom_value_id_t result_id = loom_op_const_results(op)[result_index];
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_type_is_scalar(result_type) &&
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(emitter, op, result_name,
                                            result_type, IREE_SV("i1"));
}

static iree_status_t loom_kernel_verify_combining_kind_for_value(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t value_id, loom_combining_kind_t kind) {
  if (!loom_combining_kind_is_valid(kind)) {
    return iree_ok_status();
  }
  loom_type_t value_type = loom_module_value_type(module, value_id);
  if (!loom_kernel_type_is_collective_value(value_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (op->instance_flags != 0 && !loom_scalar_type_is_float(element_type)) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("math"), op->instance_flags,
        IREE_SV("preservation requirements apply only to floating-point "
                "collectives"));
  }
  if (loom_scalar_type_is_integer(element_type) &&
      loom_combining_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_combining_kind_accepts_float(kind)) {
    return iree_ok_status();
  }
  iree_string_view_t expected_constraint =
      loom_combining_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for combining kind")
          : IREE_SV("floating-point element type for combining kind");
  return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("value"),
                                             value_type, expected_constraint);
}

static iree_status_t loom_kernel_verify_cluster_attrs(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_attribute_t cluster_size, loom_attribute_t cluster_stride) {
  if (!loom_attr_is_absent(cluster_stride) &&
      loom_attr_is_absent(cluster_size)) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cluster_size"), 0,
        IREE_SV("present when cluster_stride is present"));
  }
  IREE_RETURN_IF_ERROR(loom_kernel_verify_positive_u32_attr(
      emitter, op, cluster_size, IREE_SV("cluster_size")));
  return loom_kernel_verify_positive_u32_attr(emitter, op, cluster_stride,
                                              IREE_SV("cluster_stride"));
}

static bool loom_kernel_type_static_element_byte_count(
    loom_type_t type, int64_t* out_byte_count) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) {
    return false;
  }
  *out_byte_count = bit_count / 8;
  return true;
}

// An unresolved extent leaves the footprint symbolic until specialization.
// Element addressability and the known factors remain source constraints.
static bool loom_kernel_type_byte_count_from_axis(loom_type_t type,
                                                  uint8_t first_axis,
                                                  int64_t* out_byte_count) {
  if (!out_byte_count || !loom_type_is_view(type) ||
      first_axis > loom_type_rank(type)) {
    return false;
  }

  int64_t byte_count = 0;
  if (!loom_kernel_type_static_element_byte_count(type, &byte_count)) {
    return false;
  }
  bool is_static = true;
  for (uint8_t axis = first_axis; axis < loom_type_rank(type); ++axis) {
    if (loom_type_dim_is_dynamic_at(type, axis)) {
      is_static = false;
      continue;
    }
    int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 0 ||
        !iree_checked_mul_i64(byte_count, dimension_size, &byte_count)) {
      return false;
    }
  }

  *out_byte_count = is_static ? byte_count : -1;
  return true;
}

static bool loom_kernel_type_static_byte_count_from_axis(
    loom_type_t type, uint8_t first_axis, int64_t* out_byte_count) {
  return loom_kernel_type_byte_count_from_axis(type, first_axis,
                                               out_byte_count) &&
         *out_byte_count >= 0;
}

static bool loom_kernel_type_static_byte_count(loom_type_t type,
                                               int64_t* out_byte_count) {
  return loom_kernel_type_static_byte_count_from_axis(type, /*first_axis=*/0,
                                                      out_byte_count);
}

static iree_status_t loom_kernel_verify_static_byte_count(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t view_id, int64_t* out_byte_count) {
  loom_type_t view_type = loom_module_value_type(module, view_id);
  if (loom_kernel_type_static_byte_count(view_type, out_byte_count)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, view_type,
      IREE_SV("view with a static byte-addressable footprint"));
}

static iree_status_t loom_kernel_verify_same_byte_count(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  int64_t source_byte_count = 0;
  loom_type_t source_type = loom_module_value_type(module, source_id);
  if (!loom_kernel_type_byte_count_from_axis(source_type, 0,
                                             &source_byte_count)) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("view with a byte-addressable footprint"));
  }

  int64_t dest_byte_count = 0;
  loom_type_t dest_type = loom_module_value_type(module, dest_id);
  if (!loom_kernel_type_byte_count_from_axis(dest_type, 0, &dest_byte_count)) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("view with a byte-addressable footprint"));
  }

  // Movement analysis requires equal concrete payload sizes before selecting a
  // target transfer. Source verification can only reject a known mismatch.
  if (source_byte_count < 0 || dest_byte_count < 0 ||
      source_byte_count == dest_byte_count) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, IREE_SV("dest"), dest_type,
      IREE_SV("same static byte footprint as source"));
}

static bool loom_kernel_byte_count_is_cluster_load_supported(
    int64_t byte_count) {
  return byte_count == 1 || byte_count == 4 || byte_count == 8 ||
         byte_count == 16;
}

static iree_status_t loom_kernel_verify_cluster_static_byte_count(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  int64_t source_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("source"), source_id, &source_byte_count));

  int64_t dest_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("dest"), dest_id, &dest_byte_count));

  if (source_byte_count != dest_byte_count) {
    loom_type_t dest_type = loom_module_value_type(module, dest_id);
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("same static byte footprint as source"));
  }

  if (loom_kernel_byte_count_is_cluster_load_supported(source_byte_count)) {
    return iree_ok_status();
  }
  loom_type_t source_type = loom_module_value_type(module, source_id);
  return loom_kernel_emit_operand_constraint(
      emitter, op, IREE_SV("source"), source_type,
      IREE_SV("view with static 1, 4, 8, or 16 byte footprint"));
}

static iree_status_t loom_kernel_verify_transfer_token_group_use(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t token_id) {
  if (token_id >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* token = loom_module_value(module, token_id);
  if (token->use_count != 1) {
    return loom_kernel_emit_value_use_count_constraint(
        module, emitter, op, token_id, token->use_count,
        IREE_SV("exactly one kernel.async.group use"));
  }

  const loom_use_t use = loom_value_uses(token)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  if (loom_kernel_async_group_isa(user_op)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, token_id, user_op,
      IREE_SV("kernel.async.group token operand"));
}

static iree_status_t loom_kernel_verify_token_defined_by_transfer(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t token_id) {
  if (token_id >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* token = loom_module_value(module, token_id);
  const loom_op_t* defining_op =
      loom_value_is_block_arg(token) ? NULL : loom_value_def_op(token);
  const loom_op_semantics_t semantics =
      defining_op ? loom_op_semantics(module, defining_op)
                  : loom_op_semantics_empty();
  if (loom_contract_family_set_has_any(semantics.contract_families,
                                       LOOM_CONTRACT_KERNEL_ASYNC) &&
      loom_cache_policy_isa(loom_cache_policy_cast(module, defining_op))) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, token_id, defining_op,
      IREE_SV("kernel async transfer result"));
}

static iree_status_t loom_kernel_verify_group_has_uses(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t group_id) {
  if (group_id >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* group = loom_module_value(module, group_id);
  if (group->use_count != 0) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_use_count_constraint(
      module, emitter, op, group_id, 0,
      IREE_SV("at least one wait or carried group use"));
}

static iree_status_t loom_kernel_verify_group_origin_if_local(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t group_id) {
  if (group_id >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* group = loom_module_value(module, group_id);
  if (loom_value_is_block_arg(group)) {
    return iree_ok_status();
  }
  const loom_op_t* defining_op = loom_value_def_op(group);
  // Template applications are selected and inlined before whole-function async
  // lifetime verification. Runtime calls remain unsupported ownership
  // boundaries and are deliberately rejected here.
  if (defining_op && (loom_kernel_async_group_isa(defining_op) ||
                      loom_template_apply_isa(defining_op))) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, group_id, defining_op,
      IREE_SV("kernel.async.group result"));
}

static iree_status_t loom_kernel_verify_gather_destination(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  loom_type_t source_type = loom_module_value_type(module, source_id);
  loom_type_t dest_type = loom_module_value_type(module, dest_id);
  if (!loom_type_is_view(source_type) || !loom_type_is_view(dest_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t dest_rank = loom_type_rank(dest_type);
  if (dest_rank != source_rank + 1) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("view with one leading subgroup-lane axis"));
  }

  int64_t source_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("source"), source_id, &source_byte_count));

  int64_t dest_lane_byte_count = 0;
  if (!loom_kernel_type_static_byte_count_from_axis(dest_type, /*first_axis=*/1,
                                                    &dest_lane_byte_count)) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("view with static byte-addressable trailing lane footprint"));
  }
  if (source_byte_count <= dest_lane_byte_count) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, IREE_SV("dest"), dest_type,
      IREE_SV("trailing lane byte footprint at least source footprint"));
}

static iree_status_t loom_kernel_verify_async_copy_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    uint8_t direction, loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_same_byte_count(module, emitter, op,
                                                          source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_cache_policy_verify(
      module, op,
      direction == LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL
          ? LOOM_CACHE_POLICY_ACCESS_STORE
          : LOOM_CACHE_POLICY_ACCESS_LOAD,
      emitter));
  return loom_kernel_verify_transfer_token_group_use(module, emitter, op,
                                                     token_id);
}

static iree_status_t loom_kernel_verify_async_gather_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_gather_destination(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_cache_policy_verify(
      module, op, LOOM_CACHE_POLICY_ACCESS_LOAD, emitter));
  return loom_kernel_verify_transfer_token_group_use(module, emitter, op,
                                                     token_id);
}

static iree_status_t loom_kernel_verify_async_cluster_gather_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    loom_value_id_t cluster_mask_id, loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_cluster_static_byte_count(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_operand_i32(
      module, emitter, op, IREE_SV("cluster_mask"), cluster_mask_id));
  IREE_RETURN_IF_ERROR(loom_cache_policy_verify(
      module, op, LOOM_CACHE_POLICY_ACCESS_LOAD, emitter));
  return loom_kernel_verify_transfer_token_group_use(module, emitter, op,
                                                     token_id);
}

static bool loom_kernel_type_is_static_i32_vector(loom_type_t type,
                                                  int64_t lane_count) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         !loom_type_dim_is_dynamic_at(type, 0) &&
         loom_type_dim_static_size_at(type, 0) == lane_count &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static iree_status_t loom_kernel_verify_tensor_endpoint_types(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  loom_type_t source_type = loom_module_value_type(module, source_id);
  loom_type_t dest_type = loom_module_value_type(module, dest_id);
  if (!loom_type_is_view(source_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("source"),
                                               source_type, IREE_SV("view"));
  }
  if (!loom_type_is_view(dest_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("dest"),
                                               dest_type, IREE_SV("view"));
  }

  uint8_t source_rank = loom_type_rank(source_type);
  if (source_rank == 0 || source_rank > 5) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("source"),
                                               source_type,
                                               IREE_SV("view rank in [1, 5]"));
  }
  if (source_rank != loom_type_rank(dest_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("dest"),
                                               dest_type,
                                               IREE_SV("same rank as source"));
  }

  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  if (source_element_type != loom_type_element_type(dest_type)) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("same element type as source"));
  }

  int32_t element_bit_count = loom_scalar_type_bitwidth(source_element_type);
  if (element_bit_count != 8 && element_bit_count != 16 &&
      element_bit_count != 32 && element_bit_count != 64) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("view with 1, 2, 4, or 8 byte element type"));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_verify_async_tensor_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    loom_value_id_t descriptor_id, uint8_t direction,
    loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_operand_tensor_lds_descriptor(
      module, emitter, op, IREE_SV("descriptor"), descriptor_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_tensor_endpoint_types(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_cache_policy_verify(
      module, op,
      direction == LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL
          ? LOOM_CACHE_POLICY_ACCESS_STORE
          : LOOM_CACHE_POLICY_ACCESS_LOAD,
      emitter));
  return loom_kernel_verify_transfer_token_group_use(module, emitter, op,
                                                     token_id);
}

iree_status_t loom_kernel_def_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  IREE_RETURN_IF_ERROR(
      loom_kernel_verify_launch_config_contract(module, op, emitter));
  return iree_ok_status();
}

iree_status_t loom_kernel_decl_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  return loom_function_contract_verify(module, op, emitter);
}

static bool loom_kernel_is_indirect_workgroup_count_type(loom_type_t type) {
  return loom_type_is_view(type) && loom_type_rank(type) == 1 &&
         !loom_type_dim_is_dynamic_at(type, 0) &&
         loom_type_dim_static_size_at(type, 0) == 3 &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         !loom_type_has_encoding(type);
}

static iree_status_t loom_kernel_verify_dispatch_workgroup_counts(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_slice_t counts = loom_kernel_dispatch_workgroup_counts(op);
  if (counts.count == 0 || counts.count > 3) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("workgroup count operand count"), counts.count,
        IREE_SV("one to three"));
  }

  const loom_type_t first_type =
      loom_module_value_type(module, counts.values[0]);
  if (counts.count == 1 &&
      loom_kernel_is_indirect_workgroup_count_type(first_type)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < counts.count; ++i) {
    const loom_type_t type = loom_module_value_type(module, counts.values[i]);
    if (loom_type_is_scalar(type) &&
        loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX) {
      continue;
    }
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("workgroup_counts"), type,
        counts.count == 1 ? IREE_SV("index or dense view<3xi32>")
                          : IREE_SV("index"));
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_kernel_launch_callee(op);
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  const loom_op_t* definition_op = symbol->defining_op;

  const loom_func_like_t entry =
      loom_func_like_const_cast(module, definition_op);
  uint16_t entry_argument_count = 0;
  const loom_value_id_t* entry_argument_ids =
      loom_func_like_arg_ids(entry, &entry_argument_count);
  const bool is_pipeline =
      loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE);

  const loom_value_id_t* workload_arg_ids = NULL;
  uint16_t workload_arg_count = 0;
  const loom_value_id_t* argument_ids = entry_argument_ids;
  uint16_t argument_count = entry_argument_count;
  iree_string_view_t workload_name = IREE_SV("kernel workload");
  iree_string_view_t argument_name = IREE_SV("kernel ABI argument");
  if (is_pipeline) {
    const loom_symbol_product_carrier_t scope =
        loom_symbol_definition_product_carrier(
            loom_op_vtable(module, definition_op)->symbol_def, definition_op);
    if (scope != LOOM_PIPELINE_SCOPE_KERNEL) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("pipeline scope")),
          loom_param_i64(scope),
          loom_param_string(IREE_SV("kernel")),
      };
      return loom_kernel_emit_entry_related(emitter, op, definition_op,
                                            LOOM_ERR_STRUCTURE_014, params,
                                            IREE_ARRAYSIZE(params));
    }

    const int64_t specialization_count =
        loom_func_like_specialization_count(entry);
    if (specialization_count < 0 ||
        specialization_count > entry_argument_count) {
      // The pipeline definition owns this malformed signature diagnostic.
      return iree_ok_status();
    }
    workload_arg_ids = entry_argument_ids;
    workload_arg_count = (uint16_t)specialization_count;
    if (specialization_count != 0) {
      argument_ids += specialization_count;
    }
    argument_count -= (uint16_t)specialization_count;
    workload_name = IREE_SV("pipeline specialization");
    argument_name = IREE_SV("pipeline launch binding");
  } else {
    const loom_value_slice_t kernel_workload_args =
        loom_kernel_workload_arg_ids(module, definition_op);
    workload_arg_ids = kernel_workload_args.values;
    workload_arg_count = kernel_workload_args.count;
  }

  const loom_value_slice_t workloads = loom_kernel_launch_workloads(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_entry_operand_group(
      module, op, definition_op, emitter, IREE_SV("workload"), workload_name,
      workloads, workload_arg_ids, workload_arg_count,
      /*flat_operand_offset=*/0, /*enclosing_remap=*/NULL));

  // Pipeline run types may reference the leading staged arguments. Both
  // groups belong to one signature, even though the launch spells them apart.
  const loom_type_value_remap_t specialization_remap = {
      .source_values = workload_arg_ids,
      .target_values = workloads.values,
      .count = iree_min(workloads.count, workload_arg_count),
      .flags = workload_arg_count > 2
                   ? LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE
                   : 0,
  };
  const loom_value_slice_t arguments = loom_kernel_launch_arguments(op);
  return loom_kernel_verify_entry_operand_group(
      module, op, definition_op, emitter, IREE_SV("argument"), argument_name,
      arguments, argument_ids, argument_count, workloads.count,
      is_pipeline ? &specialization_remap : NULL);
}

iree_status_t loom_kernel_dispatch_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_kernel_verify_dispatch_workgroup_counts(module, op, emitter));
  const loom_value_slice_t workgroup_size =
      loom_kernel_dispatch_workgroup_size(op);
  if (workgroup_size.count != 0 && workgroup_size.count != 3) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("workgroup size operand count"),
        workgroup_size.count, IREE_SV("zero or three"));
  }

  const loom_symbol_ref_t callee = loom_kernel_dispatch_callee(op);
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  loom_op_t* definition_op = symbol->defining_op;
  if (!loom_kernel_entry_decl_isa(definition_op)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(
            loom_string_table_get(&module->strings, symbol->name_id)),
    };
    return loom_kernel_emit_entry_related(emitter, op, definition_op,
                                          LOOM_ERR_STRUCTURE_063, params,
                                          IREE_ARRAYSIZE(params));
  }
  const loom_func_like_t entry = loom_func_like_cast(module, definition_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(entry, &argument_count);
  const loom_value_slice_t arguments = loom_kernel_dispatch_arguments(op);
  const loom_value_slice_t workgroup_counts =
      loom_kernel_dispatch_workgroup_counts(op);
  return loom_kernel_verify_entry_operand_group(
      module, op, definition_op, emitter, IREE_SV("argument"),
      IREE_SV("kernel entry ABI argument"), arguments, argument_ids,
      argument_count, workgroup_counts.count, /*enclosing_remap=*/NULL);
}

iree_status_t loom_kernel_launch_yield_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (loom_command_composition_is_kernel_launch_schedule(op->parent_op)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_launch_placement_error(
      module, op, emitter,
      IREE_SV("kernel.launch.serial or kernel.launch.concurrent"));
}

typedef struct loom_kernel_launch_schedule_verification_t {
  // Module containing the schedule being verified.
  const loom_module_t* module;
  // Destination for invalid preparation diagnostics.
  iree_diagnostic_emitter_t emitter;
} loom_kernel_launch_schedule_verification_t;

static iree_status_t loom_kernel_verify_launch_preparation(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  const loom_kernel_launch_schedule_verification_t* verification = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  const loom_trait_flags_t traits =
      loom_op_effective_traits(verification->module, op);
  const loom_call_like_t call = loom_call_like_cast(verification->module, op);
  const bool is_kernel_issue =
      iree_any_bit_set(traits, LOOM_TRAIT_COMMAND_EFFECT) &&
      loom_call_like_kind(call) == LOOM_CALL_LIKE_KIND_SEMANTIC;
  const bool is_launch_schedule =
      loom_command_composition_is_kernel_launch_schedule(op);
  if (is_kernel_issue || is_launch_schedule) {
    // Each nested schedule verifies its own preparation. Pruning it keeps
    // every preparation operation owned by exactly one schedule check.
    *out_result = LOOM_WALK_SKIP;
    return iree_ok_status();
  }
  if (!iree_any_bit_set(traits, LOOM_TRAIT_COMMAND_EFFECT) &&
      loom_callable_effects_classify_traits(traits) !=
          LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    return iree_ok_status();
  }
  *out_result = LOOM_WALK_ABORT;
  return loom_kernel_emit_launch_placement_error(
      verification->module, op, verification->emitter,
      IREE_SV("kernel launch schedule"));
}

iree_status_t loom_kernel_launch_schedule_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_region_t* body = loom_op_regions(op)[0];
  loom_kernel_launch_schedule_verification_t verification = {
      .module = module,
      .emitter = emitter,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_region(
      module, body, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_kernel_verify_launch_preparation,
                             &verification},
      &walk_result);
}

static iree_status_t loom_kernel_verify_barrier_attributes(
    const loom_op_t* op, loom_value_fact_memory_space_t memory_space,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope,
    iree_diagnostic_emitter_t emitter) {
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("memory_space"), memory_space,
        IREE_SV("workgroup or global memory space"));
  }

  const bool ordering_supported =
      memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? ordering == LOOM_ATOMIC_ORDERING_ACQ_REL
          : ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
                ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
                ordering == LOOM_ATOMIC_ORDERING_ACQ_REL;
  if (!ordering_supported) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("ordering"), ordering,
        memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? IREE_SV("acq_rel ordering for workgroup memory")
            : IREE_SV(
                  "acquire, release, or acq_rel ordering for global memory"));
  }

  if (scope != LOOM_ATOMIC_SCOPE_SUBGROUP &&
      scope != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("scope"), scope,
        IREE_SV("subgroup or workgroup execution scope"));
  }

  return iree_ok_status();
}

iree_status_t loom_kernel_barrier_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  (void)module;
  return loom_kernel_verify_barrier_attributes(
      op, loom_kernel_barrier_memory_space(op),
      loom_kernel_barrier_ordering(op), loom_kernel_barrier_scope(op), emitter);
}

iree_status_t loom_kernel_barrier_arrive_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_barrier_attributes(
      op, loom_kernel_barrier_arrive_memory_space(op),
      loom_kernel_barrier_arrive_ordering(op),
      loom_kernel_barrier_arrive_scope(op), emitter));

  const loom_value_id_t phase_id = loom_kernel_barrier_arrive_phase(op);
  const loom_type_t phase_type = loom_module_value_type(module, phase_id);
  if (!loom_kernel_type_is_barrier_phase(module, phase_type)) {
    return loom_kernel_emit_result_constraint(emitter, op, IREE_SV("phase"),
                                              phase_type,
                                              IREE_SV("kernel.barrier.phase"));
  }
  if (phase_id >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* phase = loom_module_value(module, phase_id);
  if (phase->use_count != 1) {
    return loom_kernel_emit_value_use_count_constraint(
        module, emitter, op, phase_id, phase->use_count,
        IREE_SV("exactly one kernel.barrier.wait use"));
  }
  const loom_op_t* user_op = loom_use_user_op(loom_value_uses(phase)[0]);
  if (!loom_kernel_barrier_wait_isa(user_op)) {
    return loom_kernel_emit_value_user_constraint(
        module, emitter, op, phase_id, user_op,
        IREE_SV("kernel.barrier.wait phase operand"));
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_barrier_wait_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t phase_id = loom_kernel_barrier_wait_phase(op);
  const loom_type_t phase_type = loom_module_value_type(module, phase_id);
  if (!loom_kernel_type_is_barrier_phase(module, phase_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("phase"),
                                               phase_type,
                                               IREE_SV("kernel.barrier.phase"));
  }
  if (phase_id >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* phase = loom_module_value(module, phase_id);
  const loom_op_t* defining_op =
      loom_value_is_block_arg(phase) ? NULL : loom_value_def_op(phase);
  if (defining_op && loom_kernel_barrier_arrive_isa(defining_op)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, phase_id, defining_op,
      IREE_SV("kernel.barrier.arrive result"));
}

iree_status_t loom_kernel_tensor_lds_descriptor_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t dgroups = loom_kernel_tensor_lds_descriptor_dgroups(op);
  if (dgroups.count != 2 && dgroups.count != 4) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("dgroups"), dgroups.count,
        IREE_SV("two or four AMDGPU tensor dgroups"));
  }

  static const int64_t expected_lanes[] = {4, 8, 4, 4};
  for (uint16_t i = 0; i < dgroups.count; ++i) {
    loom_value_id_t dgroup_id = loom_value_slice_get(dgroups, i);
    loom_type_t dgroup_type = loom_module_value_type(module, dgroup_id);
    if (!loom_kernel_type_is_static_i32_vector(dgroup_type,
                                               expected_lanes[i])) {
      return loom_kernel_emit_operand_constraint(
          emitter, op, IREE_SV("dgroups"), dgroup_type,
          i == 1 ? IREE_SV("D1 vector<8xi32>")
                 : IREE_SV("D0/D2/D3 vector<4xi32>"));
    }
  }

  loom_value_id_t descriptor_id =
      loom_kernel_tensor_lds_descriptor_descriptor(op);
  return loom_kernel_verify_result_tensor_lds_descriptor(
      module, emitter, op, IREE_SV("descriptor"), descriptor_id);
}

iree_status_t loom_kernel_subgroup_shuffle_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"),
      loom_kernel_subgroup_shuffle_value(op)));
  IREE_RETURN_IF_ERROR(
      loom_kernel_verify_operand_i32(module, emitter, op, IREE_SV("offset"),
                                     loom_kernel_subgroup_shuffle_offset(op)));
  return loom_kernel_verify_operand_i32(module, emitter, op, IREE_SV("width"),
                                        loom_kernel_subgroup_shuffle_width(op));
}

iree_status_t loom_kernel_subgroup_broadcast_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"),
      loom_kernel_subgroup_broadcast_value(op)));
  return loom_kernel_verify_operand_i32(
      module, emitter, op, IREE_SV("lane"),
      loom_kernel_subgroup_broadcast_lane(op));
}

iree_status_t loom_kernel_subgroup_value_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (op->operand_count == 0) {
    return iree_ok_status();
  }
  return loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), loom_op_const_operands(op)[0]);
}

iree_status_t loom_kernel_subgroup_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_subgroup_reduce_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_subgroup_reduce_kind(op)));
  return loom_kernel_verify_cluster_attrs(
      emitter, op, loom_kernel_subgroup_reduce_cluster_size_attr(op),
      loom_kernel_subgroup_reduce_cluster_stride_attr(op));
}

iree_status_t loom_kernel_subgroup_scan_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_subgroup_scan_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_subgroup_scan_kind(op)));
  return loom_kernel_verify_cluster_attrs(
      emitter, op, loom_kernel_subgroup_scan_cluster_size_attr(op),
      loom_kernel_subgroup_scan_cluster_stride_attr(op));
}

iree_status_t loom_kernel_subgroup_mask_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_first_result_i32_or_i64(module, emitter, op,
                                                    IREE_SV("mask"));
}

iree_status_t loom_kernel_subgroup_match_all_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_first_result_i32_or_i64(
      module, emitter, op, IREE_SV("mask")));
  return loom_kernel_verify_result_i1(module, emitter, op, /*result_index=*/1,
                                      IREE_SV("all_equal"));
}

iree_status_t loom_kernel_workgroup_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_slice_t values = loom_kernel_workgroup_reduce_values(op);
  if (values.count == 0) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("values"), 0,
        IREE_SV("at least one reduction payload"));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < values.count && iree_status_is_ok(status);
       ++i) {
    status = loom_kernel_verify_collective_operand(
        module, emitter, op, IREE_SV("values"), values.values[i]);
    if (iree_status_is_ok(status)) {
      status = loom_kernel_verify_combining_kind_for_value(
          module, emitter, op, values.values[i],
          loom_kernel_workgroup_reduce_kind(op));
    }
  }
  return status;
}

iree_status_t loom_kernel_workgroup_scan_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_workgroup_scan_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  return loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_workgroup_scan_kind(op));
}

iree_status_t loom_kernel_workgroup_vote_count_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_first_result_i32_or_i64(module, emitter, op,
                                                    IREE_SV("result"));
}

iree_status_t loom_kernel_async_copy_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_copy_like(
      module, emitter, op, loom_kernel_async_copy_source(op),
      loom_kernel_async_copy_dest(op), loom_kernel_async_copy_direction(op),
      loom_kernel_async_copy_token(op));
}

iree_status_t loom_kernel_async_copy_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_copy_like(
      module, emitter, op, loom_kernel_async_copy_mask_source(op),
      loom_kernel_async_copy_mask_dest(op),
      loom_kernel_async_copy_mask_direction(op),
      loom_kernel_async_copy_mask_token(op));
}

iree_status_t loom_kernel_async_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_gather_like(
      module, emitter, op, loom_kernel_async_gather_source(op),
      loom_kernel_async_gather_dest(op), loom_kernel_async_gather_token(op));
}

iree_status_t loom_kernel_async_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_gather_like(
      module, emitter, op, loom_kernel_async_gather_mask_source(op),
      loom_kernel_async_gather_mask_dest(op),
      loom_kernel_async_gather_mask_token(op));
}

iree_status_t loom_kernel_async_cluster_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_cluster_gather_like(
      module, emitter, op, loom_kernel_async_cluster_gather_source(op),
      loom_kernel_async_cluster_gather_dest(op),
      loom_kernel_async_cluster_gather_cluster_mask(op),
      loom_kernel_async_cluster_gather_token(op));
}

iree_status_t loom_kernel_async_cluster_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_cluster_gather_like(
      module, emitter, op, loom_kernel_async_cluster_gather_mask_source(op),
      loom_kernel_async_cluster_gather_mask_dest(op),
      loom_kernel_async_cluster_gather_mask_cluster_mask(op),
      loom_kernel_async_cluster_gather_mask_token(op));
}

iree_status_t loom_kernel_async_tensor_load_to_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_tensor_like(
      module, emitter, op, loom_kernel_async_tensor_load_to_lds_source(op),
      loom_kernel_async_tensor_load_to_lds_dest(op),
      loom_kernel_async_tensor_load_to_lds_descriptor(op),
      LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP,
      loom_kernel_async_tensor_load_to_lds_token(op));
}

iree_status_t loom_kernel_async_tensor_store_from_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_tensor_like(
      module, emitter, op, loom_kernel_async_tensor_store_from_lds_source(op),
      loom_kernel_async_tensor_store_from_lds_dest(op),
      loom_kernel_async_tensor_store_from_lds_descriptor(op),
      LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL,
      loom_kernel_async_tensor_store_from_lds_token(op));
}

iree_status_t loom_kernel_async_group_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t tokens = loom_kernel_async_group_tokens(op);
  for (uint16_t i = 0; i < tokens.count; ++i) {
    loom_value_id_t token_id = loom_value_slice_get(tokens, i);
    loom_type_t token_type = loom_module_value_type(module, token_id);
    if (!loom_kernel_type_is_async_token(module, token_type)) {
      IREE_RETURN_IF_ERROR(loom_kernel_emit_operand_constraint(
          emitter, op, IREE_SV("tokens"), token_type,
          IREE_SV("kernel.async.token")));
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_kernel_verify_token_defined_by_transfer(
        module, emitter, op, token_id));
  }

  loom_value_id_t group_id = loom_kernel_async_group_group(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_group(
      module, emitter, op, IREE_SV("group"), group_id));
  return loom_kernel_verify_group_has_uses(module, emitter, op, group_id);
}

iree_status_t loom_kernel_async_wait_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_value_id_t group_id = loom_kernel_async_wait_group(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_operand_async_group(
      module, emitter, op, IREE_SV("group"), group_id));
  int64_t newer_groups = loom_kernel_async_wait_newer_groups(op);
  if (newer_groups < 0 || newer_groups > UINT16_MAX) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("newer_groups"), newer_groups,
        IREE_SV("nonnegative i16 wait count"));
  }
  return loom_kernel_verify_group_origin_if_local(module, emitter, op,
                                                  group_id);
}
