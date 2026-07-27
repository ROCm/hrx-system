// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/table.h"

#include <inttypes.h>

#include "loom/ir/module.h"

iree_status_t loom_low_allocation_acquire_value_scratch(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_value_scratch_t* out_scratch) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_scratch);
  *out_scratch = (loom_low_allocation_value_scratch_t){
      .module = table->module,
      .table = table,
      .value_ids = table->liveness.value_ids,
      .value_count = table->liveness.value_count,
  };
  if (table->liveness.value_count >= LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation value count exceeds local ordinal "
                            "range");
  }
  loom_module_value_ordinal_scratch_acquire(table->module);
  out_scratch->flags |= LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED;
  iree_host_size_t value_ordinal = 0;
  iree_status_t status = iree_ok_status();
  for (; value_ordinal < table->liveness.value_count; ++value_ordinal) {
    const loom_value_id_t value_id = table->liveness.value_ids[value_ordinal];
    if (value_id >= table->module->values.count) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "allocation liveness value id %u is out of range",
                           (unsigned)value_id);
      break;
    }
    loom_module_value_ordinal_scratch_set(table->module, value_id,
                                          (loom_value_ordinal_t)value_ordinal);
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < value_ordinal; ++i) {
      loom_module_value_ordinal_scratch_clear(table->module,
                                              table->liveness.value_ids[i]);
    }
    loom_module_value_ordinal_scratch_release(table->module);
    out_scratch->flags &= ~LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED;
  }
  return status;
}

void loom_low_allocation_release_value_scratch(
    loom_low_allocation_value_scratch_t* scratch) {
  IREE_ASSERT_ARGUMENT(scratch);
  if (!iree_any_bit_set(scratch->flags,
                        LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED)) {
    return;
  }
  for (iree_host_size_t i = 0; i < scratch->value_count; ++i) {
    loom_module_value_ordinal_scratch_clear(scratch->module,
                                            scratch->value_ids[i]);
  }
  loom_module_value_ordinal_scratch_release(scratch->module);
  scratch->flags &= ~LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED;
}

const loom_low_allocation_assignment_t*
loom_low_allocation_assignment_for_value_ordinal(
    const loom_low_allocation_table_t* table,
    loom_value_ordinal_t value_ordinal, uint32_t* out_assignment_index) {
  IREE_ASSERT_ARGUMENT(table);
  if (out_assignment_index) {
    *out_assignment_index = UINT32_MAX;
  }
  if (!table->assignment_indices_by_value_ordinal ||
      value_ordinal >= table->liveness.value_count) {
    return NULL;
  }
  const uint32_t assignment_index =
      table->assignment_indices_by_value_ordinal[value_ordinal];
  if (assignment_index == UINT32_MAX) {
    return NULL;
  }
  IREE_ASSERT(assignment_index < table->assignment_count);
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[assignment_index];
  IREE_ASSERT_EQ(assignment->value_id,
                 table->liveness.value_ids[value_ordinal]);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return assignment;
}

const loom_low_allocation_assignment_t*
loom_low_allocation_try_map_active_value_assignment(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  IREE_ASSERT_ARGUMENT(table);
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(table->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    if (out_assignment_index) {
      *out_assignment_index = UINT32_MAX;
    }
    return NULL;
  }
  return loom_low_allocation_assignment_for_value_ordinal(table, value_ordinal,
                                                          out_assignment_index);
}

const loom_low_allocation_assignment_t*
loom_low_allocation_map_active_value_assignment(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  IREE_ASSERT_ARGUMENT(table);
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_try_map_active_value_assignment(table, value_id,
                                                          out_assignment_index);
  IREE_ASSERT(assignment != NULL);
  return assignment;
}

const loom_low_allocation_edge_copy_group_t*
loom_low_allocation_find_edge_copy_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal) {
  IREE_ASSERT_ARGUMENT(table);
  iree_host_size_t lower = 0;
  iree_host_size_t upper = table->edge_copy_group_count;
  while (lower < upper) {
    iree_host_size_t middle = lower + (upper - lower) / 2;
    const loom_low_allocation_edge_copy_group_t* group =
        &table->edge_copy_groups[middle];
    if (source_ordinal < group->source_ordinal) {
      upper = middle;
    } else if (source_ordinal > group->source_ordinal) {
      lower = middle + 1;
    } else {
      return group;
    }
  }
  return NULL;
}

const loom_low_allocation_packet_move_temporary_group_t*
loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal) {
  IREE_ASSERT_ARGUMENT(table);
  iree_host_size_t lower = 0;
  iree_host_size_t upper = table->packet_move_temporary_group_count;
  while (lower < upper) {
    iree_host_size_t middle = lower + (upper - lower) / 2;
    const loom_low_allocation_packet_move_temporary_group_t* group =
        &table->packet_move_temporary_groups[middle];
    if (source_ordinal < group->source_ordinal) {
      upper = middle;
    } else if (source_ordinal > group->source_ordinal) {
      lower = middle + 1;
    } else {
      return group;
    }
  }
  return NULL;
}

iree_status_t loom_low_allocation_assignment_register_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t* out_register_class_name) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(assignment);
  IREE_ASSERT_ARGUMENT(out_register_class_name);
  *out_register_class_name = iree_string_view_empty();
  if (assignment->descriptor_reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      assignment->descriptor_reg_class_id >=
          table->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %" PRIu32
        " has invalid descriptor register class ID %" PRIu16,
        assignment->value_id, assignment->descriptor_reg_class_id);
  }
  const loom_low_reg_class_t* reg_class =
      &table->target.descriptor_set
           ->reg_classes[assignment->descriptor_reg_class_id];
  *out_register_class_name = loom_low_descriptor_set_string(
      table->target.descriptor_set, reg_class->name_string_offset);
  return iree_ok_status();
}
