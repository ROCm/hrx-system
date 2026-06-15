// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/value_facts.h"

#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"

static bool loom_pass_value_fact_scope_equal(loom_pass_value_fact_scope_t lhs,
                                             loom_pass_value_fact_scope_t rhs) {
  if (lhs.kind != rhs.kind || lhs.target_bundle != rhs.target_bundle) {
    return false;
  }
  switch (lhs.kind) {
    case LOOM_PASS_VALUE_FACT_SCOPE_NONE:
    case LOOM_PASS_VALUE_FACT_SCOPE_MODULE:
      return true;
    case LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION:
      return lhs.function.op == rhs.function.op &&
             lhs.function.vtable == rhs.function.vtable;
    case LOOM_PASS_VALUE_FACT_SCOPE_REGION:
      return lhs.function.op == rhs.function.op &&
             lhs.function.vtable == rhs.function.vtable &&
             lhs.region == rhs.region && lhs.parent_op == rhs.parent_op;
    default:
      return false;
  }
}

static iree_status_t loom_pass_value_fact_scope_validate(
    loom_pass_value_fact_scope_t scope) {
  switch (scope.kind) {
    case LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION:
      if (!loom_func_like_isa(scope.function)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "function fact scope requires a function");
      }
      return iree_ok_status();
    case LOOM_PASS_VALUE_FACT_SCOPE_MODULE:
      return iree_ok_status();
    case LOOM_PASS_VALUE_FACT_SCOPE_REGION:
      if (!scope.region) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "region fact scope requires a region");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported value fact scope");
  }
}

static iree_status_t loom_pass_value_fact_owner_ensure_table(
    loom_pass_value_fact_owner_t* owner, loom_module_t* module) {
  iree_host_size_t capacity = module->values.capacity;
  if (iree_any_bit_set(owner->flags,
                       LOOM_PASS_VALUE_FACT_OWNER_FLAG_TABLE_INITIALIZED) &&
      owner->module == module && owner->table.capacity >= capacity) {
    return iree_ok_status();
  }

  iree_arena_reset(&owner->storage_arena);
  iree_arena_reset(&owner->transient_arena);
  owner->module = module;
  owner->active_scope = loom_pass_value_fact_scope_none();
  owner->flags &= ~LOOM_PASS_VALUE_FACT_OWNER_FLAG_TABLE_INITIALIZED;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_with_arenas(
      &owner->table, &owner->storage_arena, &owner->transient_arena, capacity));
  loom_type_registry_configure_fact_context(&owner->table.context);
  owner->flags |= LOOM_PASS_VALUE_FACT_OWNER_FLAG_TABLE_INITIALIZED;
  return iree_ok_status();
}

static iree_status_t loom_pass_value_fact_owner_compute_module(
    loom_pass_value_fact_owner_t* owner, loom_module_t* module) {
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function) || !loom_func_like_body(function)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_compute(&owner->table, module, function));
  }
  return iree_ok_status();
}

void loom_pass_value_fact_owner_initialize(
    iree_arena_block_pool_t* block_pool,
    loom_pass_value_fact_owner_t* out_owner) {
  memset(out_owner, 0, sizeof(*out_owner));
  out_owner->block_pool = block_pool;
  out_owner->active_scope = loom_pass_value_fact_scope_none();
  iree_arena_initialize(block_pool, &out_owner->storage_arena);
  iree_arena_initialize(block_pool, &out_owner->transient_arena);
}

void loom_pass_value_fact_owner_deinitialize(
    loom_pass_value_fact_owner_t* owner) {
  loom_pass_value_fact_owner_invalidate(owner);
  iree_arena_deinitialize(&owner->transient_arena);
  iree_arena_deinitialize(&owner->storage_arena);
  memset(owner, 0, sizeof(*owner));
}

void loom_pass_value_fact_owner_invalidate(
    loom_pass_value_fact_owner_t* owner) {
  if (!iree_any_bit_set(owner->flags,
                        LOOM_PASS_VALUE_FACT_OWNER_FLAG_TABLE_INITIALIZED)) {
    owner->active_scope = loom_pass_value_fact_scope_none();
    return;
  }
  loom_value_fact_table_clear_scope(&owner->table);
  iree_arena_reset(&owner->transient_arena);
  owner->active_scope = loom_pass_value_fact_scope_none();
}

iree_status_t loom_pass_value_fact_owner_prepare(
    loom_pass_value_fact_owner_t* owner, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table) {
  *out_table = NULL;

  IREE_RETURN_IF_ERROR(loom_pass_value_fact_scope_validate(scope));
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_ensure_table(owner, module));
  loom_pass_value_fact_owner_invalidate(owner);
  owner->table.context.target_bundle = scope.target_bundle;
  *out_table = &owner->table;
  return iree_ok_status();
}

iree_status_t loom_pass_value_fact_owner_acquire(
    loom_pass_value_fact_owner_t* owner, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table) {
  *out_table = NULL;

  IREE_RETURN_IF_ERROR(loom_pass_value_fact_scope_validate(scope));
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_ensure_table(owner, module));
  if (owner->module == module &&
      loom_pass_value_fact_scope_equal(owner->active_scope, scope)) {
    *out_table = &owner->table;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_pass_value_fact_owner_prepare(owner, module, scope, out_table));
  iree_status_t status = iree_ok_status();
  switch (scope.kind) {
    case LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION:
      status =
          loom_value_fact_table_compute(&owner->table, module, scope.function);
      break;
    case LOOM_PASS_VALUE_FACT_SCOPE_REGION:
      status = loom_value_fact_table_compute_region(
          &owner->table, module, scope.function, scope.region, scope.parent_op);
      break;
    case LOOM_PASS_VALUE_FACT_SCOPE_MODULE:
      status = loom_pass_value_fact_owner_compute_module(owner, module);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported value fact scope");
      break;
  }
  if (iree_status_is_ok(status)) {
    owner->active_scope = scope;
    return iree_ok_status();
  }
  loom_pass_value_fact_owner_invalidate(owner);
  return status;
}

iree_status_t loom_pass_value_facts_prepare(
    loom_pass_t* pass, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table) {
  return loom_pass_value_fact_owner_prepare(pass->value_facts, module, scope,
                                            out_table);
}

iree_status_t loom_pass_value_facts_acquire(
    loom_pass_t* pass, loom_module_t* module,
    loom_pass_value_fact_scope_t scope, loom_value_fact_table_t** out_table) {
  return loom_pass_value_fact_owner_acquire(pass->value_facts, module, scope,
                                            out_table);
}
