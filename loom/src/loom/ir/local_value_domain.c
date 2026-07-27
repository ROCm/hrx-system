// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/local_value_domain.h"

#include "loom/ir/types.h"

typedef iree_status_t (*loom_local_value_domain_value_fn_t)(
    void* user_data, loom_value_id_t value_id);

typedef struct loom_local_value_domain_value_callback_t {
  // Function invoked for each visited value.
  loom_local_value_domain_value_fn_t fn;
  // Opaque callback payload passed to |fn|.
  void* user_data;
} loom_local_value_domain_value_callback_t;

static inline loom_local_value_domain_value_callback_t
loom_local_value_domain_value_callback_make(
    loom_local_value_domain_value_fn_t fn, void* user_data) {
  return (loom_local_value_domain_value_callback_t){
      .fn = fn,
      .user_data = user_data,
  };
}

static iree_status_t loom_local_value_domain_append_value_id(
    loom_local_value_domain_t* domain, iree_arena_allocator_t* arena,
    loom_value_id_t value_id, loom_value_ordinal_t* out_ordinal) {
  if (domain->value_count >= LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local value domain exceeds value ordinal range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)domain->value_count + 1;
  if (minimum_capacity > domain->value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, domain->value_count, minimum_capacity,
        sizeof(*domain->value_ids), &domain->value_capacity,
        (void**)&domain->value_ids));
  }
  const loom_value_ordinal_t value_ordinal = domain->value_count++;
  domain->value_ids[value_ordinal] = value_id;
  loom_module_value_ordinal_scratch_set(domain->module, value_id,
                                        value_ordinal);
  *out_ordinal = value_ordinal;
  return iree_ok_status();
}

iree_status_t loom_local_value_domain_register_value(
    loom_local_value_domain_t* domain, iree_arena_allocator_t* arena,
    loom_value_id_t value_id, loom_value_ordinal_t* out_ordinal) {
  IREE_ASSERT(
      iree_any_bit_set(domain->flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED));
  if (value_id >= domain->module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "local value domain saw out-of-range value id %u",
                            (unsigned)value_id);
  }
  const loom_value_ordinal_t existing_ordinal =
      loom_module_value_ordinal_scratch_lookup(domain->module, value_id);
  if (existing_ordinal != LOOM_VALUE_ORDINAL_INVALID) {
    *out_ordinal = existing_ordinal;
    return iree_ok_status();
  }
  return loom_local_value_domain_append_value_id(domain, arena, value_id,
                                                 out_ordinal);
}

typedef struct loom_local_value_domain_register_state_t {
  // Domain being populated.
  loom_local_value_domain_t* domain;
  // Arena owning domain storage.
  iree_arena_allocator_t* arena;
} loom_local_value_domain_register_state_t;

static iree_status_t loom_local_value_domain_register_value_callback(
    void* user_data, loom_value_id_t value_id) {
  loom_local_value_domain_register_state_t* state =
      (loom_local_value_domain_register_state_t*)user_data;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  return loom_local_value_domain_register_value(state->domain, state->arena,
                                                value_id, &value_ordinal);
}

static iree_status_t loom_local_value_domain_type_ref_callback(
    loom_value_id_t value_id, void* user_data) {
  loom_local_value_domain_value_callback_t* visitor =
      (loom_local_value_domain_value_callback_t*)user_data;
  return visitor->fn(visitor->user_data, value_id);
}

static iree_status_t loom_local_value_domain_for_each_type_ref(
    loom_type_t type, loom_local_value_domain_value_callback_t visitor) {
  return loom_type_walk_value_refs(
      type, loom_local_value_domain_type_ref_callback, &visitor);
}

static bool loom_local_value_domain_region_is_nested_in_op(
    const loom_op_t* owner_op, const loom_region_t* region);

static bool loom_local_value_domain_block_is_nested_in_op(
    const loom_op_t* owner_op, const loom_block_t* block) {
  return block && loom_local_value_domain_region_is_nested_in_op(
                      owner_op, block->parent_region);
}

static bool loom_local_value_domain_value_is_defined_inside_op(
    const loom_op_t* owner_op, const loom_module_t* module,
    loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_local_value_domain_block_is_nested_in_op(
        owner_op, loom_value_def_block(value));
  }
  const loom_op_t* def_op = loom_value_def_op(value);
  while (def_op) {
    if (def_op == owner_op) {
      return true;
    }
    def_op = def_op->parent_op;
  }
  return false;
}

static bool loom_local_value_domain_region_is_nested_in_op(
    const loom_op_t* owner_op, const loom_region_t* region) {
  if (!owner_op || !region) {
    return false;
  }
  loom_region_t* const* regions = loom_op_regions(owner_op);
  for (uint8_t i = 0; i < owner_op->region_count; ++i) {
    if (regions[i] == region) {
      return true;
    }
    const loom_block_t* block = NULL;
    loom_region_for_each_block(regions[i], block) {
      const loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (loom_local_value_domain_region_is_nested_in_op(op, region)) {
          return true;
        }
      }
    }
  }
  return false;
}

typedef struct loom_local_value_domain_external_use_state_t {
  // Module containing the visited regions.
  const loom_module_t* module;
  // Owner op whose nested regions are being inspected.
  const loom_op_t* owner_op;
  // Visitor invoked for each external value use.
  loom_local_value_domain_value_callback_t visitor;
} loom_local_value_domain_external_use_state_t;

static iree_status_t loom_local_value_domain_external_value_callback(
    void* user_data, loom_value_id_t value_id) {
  loom_local_value_domain_external_use_state_t* state =
      (loom_local_value_domain_external_use_state_t*)user_data;
  if (loom_local_value_domain_value_is_defined_inside_op(
          state->owner_op, state->module, value_id)) {
    return iree_ok_status();
  }
  return state->visitor.fn(state->visitor.user_data, value_id);
}

static iree_status_t loom_local_value_domain_for_each_region_external_use(
    loom_local_value_domain_external_use_state_t* state,
    const loom_region_t* region) {
  if (region == NULL) {
    return iree_ok_status();
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
          loom_block_arg_type(state->module, block, i),
          loom_local_value_domain_value_callback_make(
              loom_local_value_domain_external_value_callback, state)));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        const loom_value_id_t value_id = operands[i];
        IREE_RETURN_IF_ERROR(
            loom_local_value_domain_external_value_callback(state, value_id));
        IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
            loom_module_value_type(state->module, value_id),
            loom_local_value_domain_value_callback_make(
                loom_local_value_domain_external_value_callback, state)));
      }
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
            loom_module_value_type(state->module, results[i]),
            loom_local_value_domain_value_callback_make(
                loom_local_value_domain_external_value_callback, state)));
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_local_value_domain_for_each_region_external_use(state,
                                                                 regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_local_value_domain_for_each_nested_external_use(
    const loom_module_t* module, const loom_op_t* owner_op,
    loom_local_value_domain_value_callback_t visitor) {
  loom_local_value_domain_external_use_state_t state = {
      .module = module,
      .owner_op = owner_op,
      .visitor = visitor,
  };
  loom_region_t* const* regions = loom_op_regions(owner_op);
  for (uint8_t i = 0; i < owner_op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_region_external_use(
        &state, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_local_value_domain_for_each_op_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_local_value_domain_value_callback_t visitor) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(visitor.fn(visitor.user_data, operands[i]));
    IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
        loom_module_value_type(module, operands[i]), visitor));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
        loom_module_value_type(module, results[i]), visitor));
  }
  return loom_local_value_domain_for_each_nested_external_use(module, op,
                                                              visitor);
}

static iree_status_t loom_local_value_domain_register_region_values(
    loom_local_value_domain_t* domain, iree_arena_allocator_t* arena) {
  loom_local_value_domain_register_state_t state = {
      .domain = domain,
      .arena = arena,
  };
  loom_local_value_domain_value_callback_t visitor =
      loom_local_value_domain_value_callback_make(
          loom_local_value_domain_register_value_callback, &state);
  const loom_block_t* block = NULL;
  loom_region_for_each_block(domain->region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
          domain, arena, loom_block_arg_id(block, i), &value_ordinal));
      IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
          loom_block_arg_type(domain->module, block, i), visitor));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
            domain, arena, results[i], &value_ordinal));
      }
      IREE_RETURN_IF_ERROR(
          loom_local_value_domain_for_each_op_use(domain->module, op, visitor));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_local_value_domain_register_region_tree_values(
    loom_local_value_domain_t* domain, iree_arena_allocator_t* arena,
    const loom_region_t* region) {
  if (region == NULL) {
    return iree_ok_status();
  }
  loom_local_value_domain_register_state_t state = {
      .domain = domain,
      .arena = arena,
  };
  loom_local_value_domain_value_callback_t visitor =
      loom_local_value_domain_value_callback_make(
          loom_local_value_domain_register_value_callback, &state);
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
          domain, arena, loom_block_arg_id(block, i), &value_ordinal));
      IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
          loom_block_arg_type(domain->module, block, i), visitor));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
            domain, arena, operands[i], &value_ordinal));
        IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
            loom_module_value_type(domain->module, operands[i]), visitor));
      }
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
            domain, arena, results[i], &value_ordinal));
        IREE_RETURN_IF_ERROR(loom_local_value_domain_for_each_type_ref(
            loom_module_value_type(domain->module, results[i]), visitor));
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_local_value_domain_register_region_tree_values(domain, arena,
                                                                regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_local_value_domain_acquire(
    loom_module_t* module, const loom_region_t* region,
    loom_local_value_domain_flags_t flags, iree_arena_allocator_t* arena,
    loom_local_value_domain_t* out_domain) {
  *out_domain = (loom_local_value_domain_t){
      .module = module,
      .region = region,
  };
  loom_module_value_ordinal_scratch_acquire(module);
  out_domain->flags |= LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED | flags;
  iree_status_t status =
      iree_any_bit_set(flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)
          ? loom_local_value_domain_register_region_tree_values(out_domain,
                                                                arena, region)
          : loom_local_value_domain_register_region_values(out_domain, arena);
  if (!iree_status_is_ok(status)) {
    loom_local_value_domain_release(out_domain);
  }
  return status;
}

iree_status_t loom_local_value_domain_acquire_for_region(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_local_value_domain_t* out_domain) {
  return loom_local_value_domain_acquire(module, region, /*flags=*/0, arena,
                                         out_domain);
}

iree_status_t loom_local_value_domain_acquire_for_region_tree(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_local_value_domain_t* out_domain) {
  return loom_local_value_domain_acquire(
      module, region, LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE, arena,
      out_domain);
}

void loom_local_value_domain_release(loom_local_value_domain_t* domain) {
  if (!domain ||
      !iree_any_bit_set(domain->flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED)) {
    return;
  }
  for (loom_value_ordinal_t i = 0; i < domain->value_count; ++i) {
    loom_module_value_ordinal_scratch_clear(domain->module,
                                            domain->value_ids[i]);
  }
  loom_module_value_ordinal_scratch_release(domain->module);
  domain->flags = 0;
}
