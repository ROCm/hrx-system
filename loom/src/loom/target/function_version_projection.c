// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_version_projection.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/module_projection.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"

typedef struct loom_target_function_version_projection_context_table_t {
  // Context ordinals present in the function-version owner.
  iree_bitmap_t present;

  // Context ordinals with an exact authored target definition.
  iree_bitmap_t exact;

  // Projected target definition indexed by context ordinal.
  loom_symbol_ref_t* target_refs;
} loom_target_function_version_projection_context_table_t;

typedef struct loom_target_function_version_projection_plan_t {
  // Target versions reconciled against the source module symbol table.
  loom_target_function_version_snapshot_t version_snapshot;

  // Direct target-definition projection indexed by producer-owned identity.
  loom_target_function_version_projection_context_table_t contexts;

  // Number of target contexts requiring provider materialization.
  iree_host_size_t materialization_count;

  // Collision-free namespace for generated target definition names.
  iree_host_size_t generated_name_namespace;
} loom_target_function_version_projection_plan_t;

static iree_string_view_t loom_target_function_version_projection_module_name(
    const loom_module_t* module) {
  if (module->name_id == LOOM_STRING_ID_INVALID) return IREE_SV("module");
  return module->strings.entries[module->name_id];
}

static bool
loom_target_function_version_projection_parse_generated_name_namespace(
    iree_string_view_t name, uint32_t* out_namespace) {
  if (!iree_string_view_consume_prefix(&name,
                                       IREE_SV("__loom_target_context_"))) {
    return false;
  }
  const iree_host_size_t separator = iree_string_view_find_char(name, '_', 0);
  if (separator == IREE_STRING_VIEW_NPOS) return false;

  const iree_string_view_t namespace_text =
      iree_string_view_substr(name, 0, separator);
  const iree_string_view_t ordinal_text =
      iree_string_view_substr(name, separator + 1, name.size - separator - 1);
  uint32_t ignored_ordinal = 0;
  return iree_string_view_atoi_uint32_base(namespace_text, 10, out_namespace) &&
         iree_string_view_atoi_uint32_base(ordinal_text, 10, &ignored_ordinal);
}

static iree_status_t
loom_target_function_version_projection_select_generated_name_namespace(
    const loom_module_t* source_module, iree_arena_allocator_t* arena,
    iree_host_size_t* out_namespace) {
  *out_namespace = 0;
  iree_host_size_t namespace_count = 0;
  if (!iree_host_size_checked_add(source_module->symbols.count, 1,
                                  &namespace_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "projected target namespace count overflow");
  }

  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(namespace_count);
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_mul(word_count, sizeof(uint64_t),
                                  &storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "projected target namespace bitmap size overflow");
  }
  uint64_t* words = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, storage_size, (void**)&words));
  memset(words, 0, storage_size);
  const iree_bitmap_t used_namespaces = {
      .bit_count = namespace_count,
      .words = words,
  };

  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const loom_symbol_t* symbol = &source_module->symbols.entries[symbol_id];
    uint32_t namespace_ordinal = 0;
    if (loom_target_function_version_projection_parse_generated_name_namespace(
            source_module->strings.entries[symbol->name_id],
            &namespace_ordinal) &&
        namespace_ordinal < namespace_count) {
      iree_bitmap_set(used_namespaces, namespace_ordinal);
    }
  }

  *out_namespace = iree_bitmap_find_first_unset(used_namespaces, 0);
  IREE_ASSERT_LT(*out_namespace, namespace_count);
  return iree_ok_status();
}

static iree_status_t
loom_target_function_version_projection_context_table_initialize(
    iree_host_size_t context_capacity, iree_arena_allocator_t* arena,
    loom_target_function_version_projection_context_table_t* out_table) {
  *out_table = (loom_target_function_version_projection_context_table_t){
      .present = {.bit_count = context_capacity},
      .exact = {.bit_count = context_capacity},
  };
  if (context_capacity == 0) return iree_ok_status();

  const iree_host_size_t bitmap_word_count =
      iree_bitmap_calculate_words(context_capacity);
  iree_host_size_t bitmap_storage_count = 0;
  iree_host_size_t bitmap_storage_size = 0;
  if (!iree_host_size_checked_mul(bitmap_word_count, 2,
                                  &bitmap_storage_count) ||
      !iree_host_size_checked_mul(bitmap_storage_count, sizeof(uint64_t),
                                  &bitmap_storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target context bitmap size overflow");
  }
  uint64_t* bitmap_storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, bitmap_storage_size, (void**)&bitmap_storage));
  memset(bitmap_storage, 0, bitmap_storage_size);
  out_table->present.words = bitmap_storage;
  out_table->exact.words = bitmap_storage + bitmap_word_count;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, context_capacity, sizeof(*out_table->target_refs),
      (void**)&out_table->target_refs));
  for (iree_host_size_t i = 0; i < context_capacity; ++i) {
    out_table->target_refs[i] = loom_symbol_ref_null();
  }
  return iree_ok_status();
}

static iree_status_t loom_target_function_version_projection_plan_build(
    const loom_module_t* source_module,
    const loom_function_version_list_t* function_versions,
    iree_arena_allocator_t* arena,
    loom_target_function_version_projection_plan_t* out_plan) {
  *out_plan = (loom_target_function_version_projection_plan_t){0};
  if (function_versions != NULL) {
    for (iree_host_size_t i = 0; i < function_versions->count; ++i) {
      if (loom_target_function_version_const_cast(
              function_versions->values[i]) == NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "function version %zu has no target projection representation", i);
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      source_module, function_versions, arena, &out_plan->version_snapshot));

  const iree_host_size_t context_capacity =
      function_versions != NULL ? function_versions->count : 0;
  IREE_RETURN_IF_ERROR(
      loom_target_function_version_projection_context_table_initialize(
          context_capacity, arena, &out_plan->contexts));
  if (context_capacity == 0) return iree_ok_status();

  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const loom_target_function_version_t* function_version =
        loom_target_function_version_snapshot_at(&out_plan->version_snapshot,
                                                 symbol_id);
    if (function_version == NULL) continue;
    const loom_target_context_ordinal_t context_ordinal =
        function_version->target_context_ordinal;
    IREE_ASSERT_LT(context_ordinal, context_capacity);
    iree_bitmap_set(out_plan->contexts.present, context_ordinal);
    if (function_version->authored_target_is_exact) {
      iree_bitmap_set(out_plan->contexts.exact, context_ordinal);
    }
  }

  const iree_host_size_t present_count =
      iree_bitmap_count(out_plan->contexts.present);
  const iree_host_size_t exact_count =
      iree_bitmap_count(out_plan->contexts.exact);
  IREE_ASSERT_LE(exact_count, present_count);
  out_plan->materialization_count = present_count - exact_count;
  if (out_plan->materialization_count == 0) return iree_ok_status();

  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const loom_target_function_version_t* function_version =
        loom_target_function_version_snapshot_at(&out_plan->version_snapshot,
                                                 symbol_id);
    if (function_version == NULL) continue;
    const loom_target_context_ordinal_t context_ordinal =
        function_version->target_context_ordinal;
    if (iree_bitmap_test(out_plan->contexts.exact, context_ordinal)) continue;
    const loom_target_provider_t* provider =
        function_version->resolved_target.provider;
    if (provider->materialize_definition == NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target provider family '%.*s' cannot materialize target definitions",
          (int)provider->profile_type->name.size,
          provider->profile_type->name.data);
    }
  }

  return loom_target_function_version_projection_select_generated_name_namespace(
      source_module, arena, &out_plan->generated_name_namespace);
}

static iree_status_t loom_target_function_version_projection_clone_source(
    const loom_module_t* source_module, iree_host_size_t materialization_count,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    iree_allocator_t allocator, loom_ir_module_projection_t* out_projection,
    loom_module_t** out_module) {
  *out_projection = (loom_ir_module_projection_t){0};
  *out_module = NULL;

  iree_host_size_t target_string_count = 0;
  iree_host_size_t target_symbol_count = 0;
  if (!iree_host_size_checked_add(source_module->strings.count,
                                  materialization_count,
                                  &target_string_count) ||
      !iree_host_size_checked_add(source_module->symbols.count,
                                  materialization_count,
                                  &target_symbol_count) ||
      target_symbol_count > LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "projected module exceeds the target string or symbol capacity");
  }
  const loom_module_size_hints_t hints = {
      .value_count = 0,
      .string_count = target_string_count,
      .type_count = source_module->types.count,
      .encoding_count = source_module->encodings.count,
      .symbol_count = target_symbol_count,
  };
  loom_module_t* target_module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(
      source_module->context,
      loom_target_function_version_projection_module_name(source_module),
      block_pool, &hints, allocator, &target_module));
  target_module->flags = source_module->flags;

  loom_symbol_ref_t* target_symbols = NULL;
  iree_status_t status = iree_ok_status();
  if (source_module->symbols.count > 0) {
    status = iree_arena_allocate_array(
        scratch_arena, source_module->symbols.count, sizeof(*target_symbols),
        (void**)&target_symbols);
  }
  for (loom_symbol_id_t source_symbol_id = 0;
       source_symbol_id < source_module->symbols.count &&
       iree_status_is_ok(status);
       ++source_symbol_id) {
    const loom_symbol_t* source_symbol =
        &source_module->symbols.entries[source_symbol_id];
    loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
    status = loom_module_intern_string(
        target_module, source_module->strings.entries[source_symbol->name_id],
        &target_name_id);
    loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    if (iree_status_is_ok(status)) {
      status = loom_module_add_symbol(target_module, target_name_id,
                                      &target_symbol_id);
    }
    if (iree_status_is_ok(status)) {
      target_module->symbols.entries[target_symbol_id].flags =
          source_symbol->flags;
      target_symbols[source_symbol_id] = (loom_symbol_ref_t){
          .module_id = 0,
          .symbol_id = target_symbol_id,
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_ir_module_projection_initialize(
        source_module, target_module, target_symbols,
        source_module->symbols.count, out_projection);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ir_module_projection_clone(out_projection, scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    *out_module = target_module;
    target_module = NULL;
  }
  loom_module_free(target_module);
  return status;
}

static void loom_target_function_version_projection_seed_exact_contexts(
    const loom_module_t* source_module,
    const loom_ir_module_projection_t* projection,
    loom_target_function_version_projection_plan_t* plan) {
  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const loom_target_function_version_t* function_version =
        loom_target_function_version_snapshot_at(&plan->version_snapshot,
                                                 symbol_id);
    if (function_version == NULL ||
        !function_version->authored_target_is_exact) {
      continue;
    }

    const loom_symbol_ref_t source_target_ref =
        loom_func_like_target(function_version->base.function);
    IREE_ASSERT(loom_symbol_ref_is_valid(source_target_ref));
    IREE_ASSERT_EQ(source_target_ref.module_id, 0);
    IREE_ASSERT_LT(source_target_ref.symbol_id, source_module->symbols.count);
    const loom_symbol_ref_t projected_target_ref =
        loom_ir_module_projection_target_symbol(projection,
                                                source_target_ref.symbol_id);

    const loom_target_context_ordinal_t context_ordinal =
        function_version->target_context_ordinal;
    loom_symbol_ref_t* context_target_ref =
        &plan->contexts.target_refs[context_ordinal];
    if (loom_symbol_ref_is_valid(*context_target_ref)) {
      IREE_ASSERT_EQ(context_target_ref->module_id,
                     projected_target_ref.module_id);
      IREE_ASSERT_EQ(context_target_ref->symbol_id,
                     projected_target_ref.symbol_id);
    } else {
      *context_target_ref = projected_target_ref;
    }
  }
}

static iree_status_t loom_target_function_version_projection_add_target_symbol(
    loom_module_t* module, iree_host_size_t generated_name_namespace,
    iree_host_size_t materialization_ordinal,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  char name_buffer[96];
  const int name_length = snprintf(
      name_buffer, sizeof(name_buffer), "__loom_target_context_%zu_%zu",
      generated_name_namespace, materialization_ordinal);
  if (name_length < 0 || (iree_host_size_t)name_length >= sizeof(name_buffer)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "projected target symbol name is too long");
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
      &name_id));
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  *out_target_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

static iree_status_t
loom_target_function_version_projection_materialize_and_bind(
    const loom_module_t* source_module, loom_module_t* projected_module,
    const loom_ir_module_projection_t* projection,
    loom_target_function_version_projection_plan_t* plan) {
  loom_builder_t builder;
  loom_builder_initialize(projected_module, &projected_module->arena,
                          loom_module_block(projected_module), &builder);

  iree_host_size_t materialization_ordinal = 0;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < source_module->symbols.count;
       ++symbol_id) {
    const loom_target_function_version_t* function_version =
        loom_target_function_version_snapshot_at(&plan->version_snapshot,
                                                 symbol_id);
    if (function_version == NULL) continue;

    const loom_target_context_ordinal_t context_ordinal =
        function_version->target_context_ordinal;
    loom_symbol_ref_t* target_ref =
        &plan->contexts.target_refs[context_ordinal];
    if (!loom_symbol_ref_is_valid(*target_ref)) {
      loom_symbol_ref_t materialized_ref = loom_symbol_ref_null();
      IREE_RETURN_IF_ERROR(
          loom_target_function_version_projection_add_target_symbol(
              projected_module, plan->generated_name_namespace,
              materialization_ordinal, &materialized_ref));
      IREE_RETURN_IF_ERROR(
          function_version->resolved_target.provider->materialize_definition(
              &builder, &function_version->resolved_target, materialized_ref,
              LOOM_LOCATION_UNKNOWN));
      *target_ref = materialized_ref;
      ++materialization_ordinal;
    }

    const loom_symbol_ref_t projected_function_ref =
        loom_ir_module_projection_target_symbol(projection, symbol_id);
    const loom_func_like_t projected_function = loom_func_like_cast(
        projected_module,
        projected_module->symbols.entries[projected_function_ref.symbol_id]
            .defining_op);
    loom_func_like_set_target(projected_module, projected_function,
                              *target_ref);
  }
  IREE_ASSERT_EQ(materialization_ordinal, plan->materialization_count);
  return iree_ok_status();
}

iree_status_t loom_target_function_versions_project_module(
    const loom_module_t* source_module,
    const loom_function_version_list_t* function_versions,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_projected_module) {
  if (out_projected_module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_projected_module must not be NULL");
  }
  *out_projected_module = NULL;
  if (source_module == NULL || block_pool == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source_module and block_pool must not be NULL");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  loom_target_function_version_projection_plan_t plan = {0};
  loom_module_t* projected_module = NULL;
  loom_ir_module_projection_t projection = {0};

  iree_status_t status = loom_target_function_version_projection_plan_build(
      source_module, function_versions, &scratch_arena, &plan);
  if (iree_status_is_ok(status)) {
    status = loom_target_function_version_projection_clone_source(
        source_module, plan.materialization_count, block_pool, &scratch_arena,
        allocator, &projection, &projected_module);
  }
  if (iree_status_is_ok(status) && plan.contexts.present.bit_count > 0) {
    loom_target_function_version_projection_seed_exact_contexts(
        source_module, &projection, &plan);
    status = loom_target_function_version_projection_materialize_and_bind(
        source_module, projected_module, &projection, &plan);
  }
  if (iree_status_is_ok(status)) {
    *out_projected_module = projected_module;
    projected_module = NULL;
  }

  loom_module_free(projected_module);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
