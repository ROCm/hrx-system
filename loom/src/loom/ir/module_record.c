// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module_record.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

static int loom_module_record_compare(const void* lhs_ptr,
                                      const void* rhs_ptr) {
  const loom_module_record_t* lhs = (const loom_module_record_t*)lhs_ptr;
  const loom_module_record_t* rhs = (const loom_module_record_t*)rhs_ptr;
  int comparison = iree_string_view_compare(loom_op_vtable_name(lhs->vtable),
                                            loom_op_vtable_name(rhs->vtable));
  if (comparison != 0) return comparison;
  comparison = iree_string_view_compare(lhs->key, rhs->key);
  if (comparison != 0) return comparison;
  if (lhs->physical_ordinal < rhs->physical_ordinal) return -1;
  if (lhs->physical_ordinal > rhs->physical_ordinal) return 1;
  return 0;
}

static iree_host_size_t loom_module_record_count(const loom_module_t* module) {
  iree_host_size_t record_count = 0;
  for (uint16_t block_index = 0; block_index < module->body->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(module->body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_op_vtable_t* vtable =
          loom_context_resolve_op(module->context, op->kind);
      if (loom_op_vtable_is_keyed_module_record(vtable)) ++record_count;
    }
  }
  return record_count;
}

iree_status_t loom_module_record_plan_initialize(
    const loom_module_t* module, loom_module_record_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  iree_arena_initialize(module->arena.block_pool, &out_plan->arena);

  out_plan->record_count = loom_module_record_count(module);
  if (out_plan->record_count == 0) return iree_ok_status();

  iree_status_t status = iree_arena_allocate_array(
      &out_plan->arena, out_plan->record_count, sizeof(*out_plan->records),
      (void**)&out_plan->records);
  if (!iree_status_is_ok(status)) {
    loom_module_record_plan_deinitialize(out_plan);
    return status;
  }

  iree_host_size_t record_index = 0;
  for (uint16_t block_index = 0; block_index < module->body->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(module->body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_op_vtable_t* vtable =
          loom_context_resolve_op(module->context, op->kind);
      if (!loom_op_vtable_is_keyed_module_record(vtable)) continue;
      const loom_attribute_t key_attr =
          loom_op_const_attrs(op)[vtable->module_record_key_attr_index];
      const loom_string_id_t key_id = loom_attr_as_string_id(key_attr);
      out_plan->records[record_index] = (loom_module_record_t){
          .op = op,
          .vtable = vtable,
          .key = module->strings.entries[key_id],
          .physical_ordinal = record_index,
      };
      ++record_index;
    }
  }

  qsort(out_plan->records, out_plan->record_count, sizeof(*out_plan->records),
        loom_module_record_compare);
  return iree_ok_status();
}

void loom_module_record_plan_deinitialize(loom_module_record_plan_t* plan) {
  iree_arena_deinitialize(&plan->arena);
  memset(plan, 0, sizeof(*plan));
}
