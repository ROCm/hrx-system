// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_launch_config.h"

#include <inttypes.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_value_constraints.h"
#include "loom/ir/facts.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/types.h"
#include "loom/ops/func/ops.h"
#include "loom/pass/value_facts.h"

loom_kernel_launch_config_function_t loom_kernel_launch_config_function_bind(
    const loom_module_t* module, loom_func_like_t function) {
  const loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  const loom_block_t* block =
      loom_region_const_entry_block(loom_func_like_body(function));
  const loom_value_slice_t results = loom_func_return_operands(block->last_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  return (loom_kernel_launch_config_function_t){
      .function = function,
      .name = loom_string_table_get(&module->strings, symbol->name_id),
      .argument_ids = argument_ids,
      .result_ids = results.values,
      .argument_count = argument_count,
  };
}

static bool loom_kernel_launch_config_argument_facts(
    loom_scalar_type_t scalar_type, uint64_t bits,
    loom_value_facts_t* out_facts) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_I64:
      *out_facts = loom_value_facts_make_signed_raw_bits(bits, 64);
      return true;
    case LOOM_SCALAR_TYPE_OFFSET:
      return loom_value_facts_make_unsigned_raw_bits(bits, 64, out_facts);
    case LOOM_SCALAR_TYPE_I1:
      *out_facts = loom_value_facts_exact_i64((bits & 1) != 0 ? 1 : 0);
      return true;
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
      *out_facts = loom_value_facts_make_signed_raw_bits(
          bits, loom_scalar_type_bitwidth(scalar_type));
      return true;
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return loom_value_facts_from_float_bits(scalar_type, bits, out_facts);
    default:
      return false;
  }
}

static iree_status_t loom_kernel_launch_config_check_argument(
    const loom_module_t* module,
    const loom_kernel_launch_config_function_t* function,
    uint16_t argument_ordinal, loom_value_facts_t facts) {
  const loom_value_id_t value_id = function->argument_ids[argument_ordinal];
  const loom_type_t type = loom_module_value_type(module, value_id);
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (loom_scalar_type_is_float(scalar_type)) {
    return iree_ok_status();
  }

  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &exact_value)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "launch config function '@%.*s' argument %u is outside the exact "
        "fact domain",
        (int)function->name.size, function->name.data,
        (unsigned)argument_ordinal);
  }
  const loom_attribute_t exact_attribute =
      scalar_type == LOOM_SCALAR_TYPE_I1 ? loom_attr_bool(exact_value != 0)
                                         : loom_attr_i64(exact_value);
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function->function, &predicate_count);
  return loom_symbol_value_constraints_check_exact(
      function->name, type, value_id, exact_attribute,
      loom_attr_predicate_list((loom_predicate_t*)predicates, predicate_count));
}

static iree_status_t loom_kernel_launch_config_exact_u32(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    iree_string_view_t function_name, const char* field_name,
    bool require_nonzero, uint32_t* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "launch config function '@%.*s' %s is not exact",
                            (int)function_name.size, function_name.data,
                            field_name);
  }
  if (value < (require_nonzero ? 1 : 0) || value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "launch config function '@%.*s' %s value %" PRId64
                            " is outside its u32 domain",
                            (int)function_name.size, function_name.data,
                            field_name, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_exact_u64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    iree_string_view_t function_name, const char* field_name,
    uint64_t* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "launch config function '@%.*s' %s is not exact",
                            (int)function_name.size, function_name.data,
                            field_name);
  }
  if (value < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "launch config function '@%.*s' %s value %" PRId64
                            " is outside its u64 domain",
                            (int)function_name.size, function_name.data,
                            field_name, value);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_function_evaluate(
    const loom_module_t* module,
    const loom_kernel_launch_config_function_t* function,
    const uint64_t* workload_argument_bits,
    iree_host_size_t workload_argument_count,
    loom_pass_value_fact_owner_t* fact_owner,
    loom_kernel_launch_config_t* out_config) {
  if (workload_argument_count != function->argument_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' expects %u arguments but received "
        "%" PRIhsz,
        (int)function->name.size, function->name.data,
        (unsigned)function->argument_count, workload_argument_count);
  }

  loom_value_fact_table_t* fact_table = NULL;
  iree_status_t status = loom_pass_value_fact_owner_prepare(
      fact_owner, module,
      loom_pass_value_fact_scope_function(function->function), &fact_table);
  for (uint16_t i = 0;
       iree_status_is_ok(status) && i < function->argument_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(module, function->argument_ids[i]);
    loom_value_facts_t facts = loom_value_facts_unknown();
    if (!loom_kernel_launch_config_argument_facts(
            loom_type_element_type(type), workload_argument_bits[i], &facts)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "launch config function '@%.*s' argument %u bit pattern cannot be "
          "represented as exact facts",
          (int)function->name.size, function->name.data, (unsigned)i);
      break;
    }
    status =
        loom_kernel_launch_config_check_argument(module, function, i, facts);
    if (iree_status_is_ok(status)) {
      status = loom_value_fact_table_define(fact_table,
                                            function->argument_ids[i], facts);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_value_fact_table_compute(fact_table, module, function->function);
  }

  loom_kernel_launch_config_t config = {
      .fields = LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_CLUSTER_SIZE |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES,
  };
#define LOOM_EXTRACT_U32(field, result, nonzero)                          \
  if (iree_status_is_ok(status)) {                                        \
    status = loom_kernel_launch_config_exact_u32(                         \
        fact_table, function->result_ids[result], function->name, #field, \
        nonzero, &config.field);                                          \
  }
  LOOM_EXTRACT_U32(workgroup_count.x,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_X, false);
  LOOM_EXTRACT_U32(workgroup_count.y,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Y, false);
  LOOM_EXTRACT_U32(workgroup_count.z,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Z, false);
  LOOM_EXTRACT_U32(workgroup_size.x,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_X, true);
  LOOM_EXTRACT_U32(workgroup_size.y,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Y, true);
  LOOM_EXTRACT_U32(workgroup_size.z,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Z, true);
  LOOM_EXTRACT_U32(workgroup_cluster_size.x,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X,
                   true);
  LOOM_EXTRACT_U32(workgroup_cluster_size.y,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Y,
                   true);
  LOOM_EXTRACT_U32(workgroup_cluster_size.z,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Z,
                   true);
  LOOM_EXTRACT_U32(subgroup_size,
                   LOOM_KERNEL_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE, false);
#undef LOOM_EXTRACT_U32
  if (iree_status_is_ok(status)) {
    status = loom_kernel_launch_config_exact_u64(
        fact_table,
        function->result_ids
            [LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES],
        function->name, "workgroup_storage_bytes",
        &config.workgroup_storage_bytes);
  }

  loom_pass_value_fact_owner_invalidate(fact_owner);
  if (iree_status_is_ok(status)) {
    *out_config = config;
  }
  return status;
}
