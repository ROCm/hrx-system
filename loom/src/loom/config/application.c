// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/config/application.h"

#include "iree/base/api.h"
#include "loom/analysis/symbol_value_constraints.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/config/ops.h"
#include "loom/rewrite/remap.h"

iree_status_t loom_config_applied_value_sink_emit(
    loom_config_applied_value_sink_t sink, const loom_module_t* module,
    iree_string_view_t key, loom_attribute_t value) {
  if (!sink.fn) {
    return iree_ok_status();
  }
  const loom_config_applied_value_t applied_value = {
      .module = module,
      .key = key,
      .value = value,
  };
  return sink.fn(sink.user_data, &applied_value);
}

loom_value_id_t loom_config_symbol_result_value(const loom_op_t* op) {
  if (loom_config_decl_isa(op)) {
    return loom_config_decl_type(op);
  }
  if (loom_config_def_isa(op)) {
    return loom_config_def_type(op);
  }
  return LOOM_VALUE_ID_INVALID;
}

iree_string_view_t loom_config_symbol_name(const loom_module_t* module,
                                           const loom_symbol_t* symbol) {
  return loom_string_table_get(&module->strings, symbol->name_id);
}

uint16_t loom_config_find_symbol(const loom_module_t* module,
                                 iree_string_view_t key) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, key);
  return name_id == LOOM_STRING_ID_INVALID
             ? LOOM_SYMBOL_ID_INVALID
             : loom_module_find_symbol(module, name_id);
}

bool loom_config_symbol_is_config(const loom_symbol_t* symbol) {
  return symbol && symbol->defining_op &&
         loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG);
}

iree_status_t loom_config_remap_type_and_value(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_type_t source_type, loom_attribute_t source_value,
    iree_arena_block_pool_t* block_pool, loom_type_t* out_target_type,
    loom_attribute_t* out_target_value) {
  *out_target_type = (loom_type_t){0};
  *out_target_value = loom_attr_absent();

  iree_arena_allocator_t remap_arena;
  iree_arena_initialize(block_pool, &remap_arena);
  loom_ir_remap_t remap = {0};
  const loom_ir_remap_options_t remap_options = {0};
  iree_status_t status = loom_ir_remap_initialize(
      source_module, target_module, &remap_arena, &remap_options, &remap);
  if (iree_status_is_ok(status)) {
    status = loom_ir_remap_type(&remap, source_type, out_target_type);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ir_remap_attribute(&remap, source_value, out_target_value);
  }
  iree_arena_deinitialize(&remap_arena);
  return status;
}

static iree_status_t loom_config_replace_with_def(loom_module_t* module,
                                                  loom_op_t* old_op,
                                                  loom_symbol_ref_t symbol,
                                                  loom_attribute_t value,
                                                  loom_type_t type) {
  loom_block_t* block = old_op->parent_block;
  loom_op_t* before_op = old_op->next_op;
  loom_op_t* parent_op = old_op->parent_op;
  const loom_location_id_t location = old_op->location;
  const loom_value_id_t old_result = loom_config_symbol_result_value(old_op);
  const loom_attribute_t predicates = loom_config_decl_isa(old_op)
                                          ? loom_config_decl_predicates(old_op)
                                          : loom_config_def_predicates(old_op);

  IREE_RETURN_IF_ERROR(loom_op_erase(module, old_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, block, &builder);
  builder.ip.parent_op = parent_op;
  builder.ip.before_op = before_op;
  loom_op_t* new_op = NULL;
  IREE_RETURN_IF_ERROR(loom_config_def_build(
      &builder, 0, symbol, value, type, NULL, 0, NULL, 0, location, &new_op));
  if (old_result != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, old_result, loom_config_def_type(new_op)));
  }
  if (predicates.count) {
    loom_ir_remap_t remap = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_initialize(module, module, &module->arena, NULL, &remap));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&remap, old_result,
                                                 loom_config_def_type(new_op)));
    loom_predicate_t* remapped_predicates = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_predicate_list(&remap, predicates.predicate_list,
                                     predicates.count, &remapped_predicates));
    IREE_RETURN_IF_ERROR(loom_config_def_set_predicates(
        module, new_op,
        loom_attr_predicate_list(remapped_predicates, predicates.count)));
  }
  return iree_ok_status();
}

iree_status_t loom_config_apply_exact_value(loom_module_t* module,
                                            iree_string_view_t key,
                                            loom_op_t* old_op, loom_type_t type,
                                            loom_attribute_t value,
                                            iree_diagnostic_emitter_t emitter,
                                            bool* out_accepted) {
  const loom_attribute_t predicates = loom_config_decl_isa(old_op)
                                          ? loom_config_decl_predicates(old_op)
                                          : loom_config_def_predicates(old_op);
  IREE_RETURN_IF_ERROR(loom_symbol_value_constraints_verify_exact(
      module, old_op, key, type, loom_config_symbol_result_value(old_op), value,
      predicates, emitter, out_accepted));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  const loom_symbol_ref_t symbol = loom_config_decl_isa(old_op)
                                       ? loom_config_decl_symbol(old_op)
                                       : loom_config_def_symbol(old_op);
  return loom_config_replace_with_def(module, old_op, symbol, value, type);
}

iree_status_t loom_config_overlay_module(
    loom_module_t* module, const loom_module_t* config_module,
    loom_config_applied_value_sink_t applied_value_sink,
    iree_diagnostic_emitter_t emitter, iree_arena_block_pool_t* block_pool,
    loom_config_application_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(config_module);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT(module != config_module);
  IREE_ASSERT(module->context == config_module->context);

  *out_result = (loom_config_application_result_t){0};
  iree_status_t status = iree_ok_status();
  const loom_block_t* config_block =
      loom_region_const_entry_block(config_module->body);
  for (const loom_op_t* config_op = config_block->first_op;
       config_op != NULL && iree_status_is_ok(status) &&
       !out_result->error_count;
       config_op = config_op->next_op) {
    if (!loom_config_def_isa(config_op)) {
      ++out_result->error_count;
      const loom_diagnostic_emission_t emission = {
          .module = config_module,
          .op = config_op,
          .error = LOOM_ERR_SYMBOL_009,
      };
      status = iree_diagnostic_emit(emitter, &emission);
      continue;
    }
    const loom_symbol_ref_t source_ref = loom_config_def_symbol(config_op);
    const loom_symbol_t* source_symbol =
        &config_module->symbols.entries[source_ref.symbol_id];
    const iree_string_view_t key =
        loom_config_symbol_name(config_module, source_symbol);

    const uint16_t target_symbol_id = loom_config_find_symbol(module, key);
    if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      ++out_result->ignored_count;
      continue;
    }
    loom_symbol_t* target_symbol = &module->symbols.entries[target_symbol_id];
    if (!loom_config_symbol_is_config(target_symbol)) {
      ++out_result->ignored_count;
      continue;
    }
    loom_op_t* target_op = target_symbol->defining_op;
    const loom_value_id_t target_value =
        loom_config_symbol_result_value(target_op);
    const loom_type_t target_type =
        loom_module_value_type(module, target_value);
    const loom_value_id_t source_value = loom_config_def_type(config_op);

    loom_type_t remapped_type = {0};
    loom_attribute_t remapped_value = loom_attr_absent();
    status = loom_config_remap_type_and_value(
        config_module, module,
        loom_module_value_type(config_module, source_value),
        loom_config_def_value(config_op), block_pool, &remapped_type,
        &remapped_value);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    if (!loom_type_equal(remapped_type, target_type)) {
      ++out_result->error_count;
      const loom_diagnostic_param_t params[] = {
          loom_param_string(key),
          loom_param_type(target_type),
          loom_param_type(remapped_type),
      };
      const loom_diagnostic_related_op_t related = {
          .label = IREE_SV("supplied definition"),
          .module = config_module,
          .op = config_op,
      };
      const loom_diagnostic_emission_t emission = {
          .module = module,
          .op = target_op,
          .error = LOOM_ERR_SYMBOL_010,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = &related,
          .related_op_count = 1,
      };
      status = iree_diagnostic_emit(emitter, &emission);
      continue;
    }
    bool accepted = false;
    status = loom_config_apply_exact_value(module, key, target_op, target_type,
                                           remapped_value, emitter, &accepted);
    if (!accepted) {
      ++out_result->error_count;
    }
    if (!iree_status_is_ok(status) || !accepted) {
      continue;
    }
    ++out_result->materialized_count;
    status = loom_config_applied_value_sink_emit(applied_value_sink, module,
                                                 key, remapped_value);
  }
  return status;
}

iree_status_t loom_config_require_resolved_module(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    loom_config_resolution_result_t* out_result) {
  *out_result = (loom_config_resolution_result_t){0};
  const loom_symbol_t* first_unresolved = NULL;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!symbol->defining_op || !loom_config_decl_isa(symbol->defining_op)) {
      continue;
    }
    if (out_result->unresolved_count == 0) {
      first_unresolved = symbol;
    }
    ++out_result->unresolved_count;
  }
  if (!first_unresolved) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_config_symbol_name(module, first_unresolved)),
      loom_param_u64(out_result->unresolved_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = first_unresolved->defining_op,
      .error = LOOM_ERR_SYMBOL_007,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
