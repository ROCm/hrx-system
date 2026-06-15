// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/launch_config.h"

#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"

static bool loom_kernel_launch_config_static_u32(const loom_module_t* module,
                                                 loom_value_id_t value_id,
                                                 uint32_t* out_value) {
  *out_value = 0;
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t value_attr = loom_index_constant_value(defining_op);
  if (value_attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  int64_t value_i64 = loom_attr_as_i64(value_attr);
  if (value_i64 < 0 || value_i64 > UINT32_MAX) {
    return false;
  }
  *out_value = (uint32_t)value_i64;
  return true;
}

static bool loom_kernel_launch_config_static_u32_from_facts(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    loom_value_id_t value_id, uint32_t* out_value) {
  *out_value = 0;
  if (facts) {
    int64_t value = 0;
    if (loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(facts, value_id), &value) &&
        value >= 0 && value <= UINT32_MAX) {
      *out_value = (uint32_t)value;
      return true;
    }
  }
  return loom_kernel_launch_config_static_u32(module, value_id, out_value);
}

const loom_op_t* loom_kernel_def_launch_config_op(const loom_op_t* kernel_op) {
  if (!kernel_op || !loom_kernel_def_isa(kernel_op)) {
    return NULL;
  }
  const loom_region_t* config = loom_kernel_def_config(kernel_op);
  if (!config || config->block_count == 0) {
    return NULL;
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(config);
  if (!entry_block || entry_block->op_count == 0) {
    return NULL;
  }
  const loom_op_t* terminator = loom_block_const_last_op(entry_block);
  return terminator && loom_kernel_launch_config_isa(terminator) ? terminator
                                                                 : NULL;
}

loom_value_id_t loom_kernel_launch_config_workgroup_count_operand(
    const loom_op_t* launch_config, loom_kernel_dimension_t dimension) {
  if (!launch_config || !loom_kernel_launch_config_isa(launch_config)) {
    return LOOM_VALUE_ID_INVALID;
  }
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return loom_kernel_launch_config_workgroup_count_x(launch_config);
    case LOOM_KERNEL_DIMENSION_Y:
      return loom_kernel_launch_config_workgroup_count_y(launch_config);
    case LOOM_KERNEL_DIMENSION_Z:
      return loom_kernel_launch_config_workgroup_count_z(launch_config);
    default:
      return LOOM_VALUE_ID_INVALID;
  }
}

loom_value_id_t loom_kernel_launch_config_workgroup_size_operand(
    const loom_op_t* launch_config, loom_kernel_dimension_t dimension) {
  if (!launch_config || !loom_kernel_launch_config_isa(launch_config)) {
    return LOOM_VALUE_ID_INVALID;
  }
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return loom_kernel_launch_config_workgroup_size_x(launch_config);
    case LOOM_KERNEL_DIMENSION_Y:
      return loom_kernel_launch_config_workgroup_size_y(launch_config);
    case LOOM_KERNEL_DIMENSION_Z:
      return loom_kernel_launch_config_workgroup_size_z(launch_config);
    default:
      return LOOM_VALUE_ID_INVALID;
  }
}

bool loom_kernel_def_static_workgroup_size(
    const loom_module_t* module, const loom_op_t* kernel_op,
    loom_target_workgroup_size_t* out_size) {
  return loom_kernel_def_static_workgroup_size_from_facts(module, kernel_op,
                                                          NULL, out_size);
}

bool loom_kernel_def_static_workgroup_size_from_facts(
    const loom_module_t* module, const loom_op_t* kernel_op,
    const loom_value_fact_table_t* facts,
    loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(kernel_op);
  if (!launch_config) {
    return false;
  }
  return loom_kernel_launch_config_static_u32_from_facts(
             module, facts,
             loom_kernel_launch_config_workgroup_size_x(launch_config),
             &out_size->x) &&
         loom_kernel_launch_config_static_u32_from_facts(
             module, facts,
             loom_kernel_launch_config_workgroup_size_y(launch_config),
             &out_size->y) &&
         loom_kernel_launch_config_static_u32_from_facts(
             module, facts,
             loom_kernel_launch_config_workgroup_size_z(launch_config),
             &out_size->z);
}

bool loom_kernel_def_static_workgroup_count(
    const loom_module_t* module, const loom_op_t* kernel_op,
    loom_target_dispatch_workgroup_count_t* out_count) {
  return loom_kernel_def_static_workgroup_count_from_facts(module, kernel_op,
                                                           NULL, out_count);
}

bool loom_kernel_def_static_workgroup_count_from_facts(
    const loom_module_t* module, const loom_op_t* kernel_op,
    const loom_value_fact_table_t* facts,
    loom_target_dispatch_workgroup_count_t* out_count) {
  *out_count = (loom_target_dispatch_workgroup_count_t){0};
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(kernel_op);
  if (!launch_config) {
    return false;
  }
  return loom_kernel_launch_config_static_u32_from_facts(
             module, facts,
             loom_kernel_launch_config_workgroup_count_x(launch_config),
             &out_count->x) &&
         loom_kernel_launch_config_static_u32_from_facts(
             module, facts,
             loom_kernel_launch_config_workgroup_count_y(launch_config),
             &out_count->y) &&
         loom_kernel_launch_config_static_u32_from_facts(
             module, facts,
             loom_kernel_launch_config_workgroup_count_z(launch_config),
             &out_count->z);
}
