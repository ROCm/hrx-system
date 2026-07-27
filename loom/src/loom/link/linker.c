// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/linker.h"

#include <stdio.h>
#include <string.h>

#include "loom/analysis/symbol_dependencies.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/link/symbol_policy.h"
#include "loom/ops/config/contract.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/util/walk.h"

typedef struct loom_link_func_contract_attr_t {
  // Source declaration attribute index.
  uint8_t source_attr_index;
  // Target materialization attribute index.
  uint8_t target_attr_index;
  // Diagnostic field name for conflicts.
  iree_string_view_t field_name;
} loom_link_func_contract_attr_t;

struct loom_linker_t {
  // Context shared by all source modules and the target module.
  loom_context_t* context;
  // Block pool used for the output module and per-input temporary arenas.
  iree_arena_block_pool_t* block_pool;
  // Host allocator used for the linker object and temporary string builders.
  iree_allocator_t allocator;
  // Persistent linker scratch for target name maps.
  iree_arena_allocator_t scratch_arena;
  // Linked output module being constructed.
  loom_module_t* target_module;
  // Hash map from target-module string IDs to target symbol IDs.
  loom_symbol_map_t target_symbol_lookup;
  // Monotonic ordinal used when assigning deterministic private conflict names.
  iree_host_size_t private_name_ordinal;
  // True once finish has transferred the output module to the caller.
  bool finished;
};

typedef struct loom_linker_source_t {
  // Owning linker.
  loom_linker_t* linker;
  // Source module currently being cloned.
  const loom_module_t* module;
  // Per-input temporary arena.
  iree_arena_allocator_t* arena;
  // Source symbol index to target symbol reference table.
  loom_symbol_ref_t* target_symbols;
  // Number of entries in target_symbols.
  iree_host_size_t target_symbol_count;
  // Source symbols selected by this add operation.
  uint8_t* live_symbols;
  // Source symbols whose outgoing dependency edges have been scanned.
  uint8_t* scanned_symbols;
  // Source-module dependency graph. Built only for selective adds.
  loom_symbol_dependency_table_t dependency_table;
  // Lazily initialized source-to-target remap table.
  loom_ir_remap_t remap;
  // True when root filtering is active for this add operation.
  bool selective;
} loom_linker_source_t;

static iree_string_view_t loom_link_target_symbol_name(
    const loom_module_t* target_module, loom_symbol_ref_t target_ref) {
  const loom_symbol_t* symbol =
      &target_module->symbols.entries[target_ref.symbol_id];
  return target_module->strings.entries[symbol->name_id];
}

static iree_status_t loom_link_source_symbol_name(
    const loom_module_t* source_module, uint16_t source_symbol_id,
    iree_string_view_t* out_name) {
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_symbol_id];
  if (source_symbol->name_id >= source_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol %u name id %u is out of range",
                            (unsigned)source_symbol_id,
                            (unsigned)source_symbol->name_id);
  }
  *out_name = source_module->strings.entries[source_symbol->name_id];
  return iree_ok_status();
}

static bool loom_link_target_symbol_is_private_concrete(
    const loom_linker_t* linker, uint16_t target_symbol_id) {
  if (target_symbol_id >= linker->target_module->symbols.count) return false;
  const loom_symbol_t* symbol =
      &linker->target_module->symbols.entries[target_symbol_id];
  return loom_link_symbol_is_concrete_definition(symbol) &&
         !loom_link_symbol_has_global_identity(linker->target_module, symbol);
}

static iree_status_t loom_linker_allocate_fresh_private_name(
    loom_linker_t* linker, iree_string_view_t base_name,
    loom_string_id_t* out_name_id) {
  *out_name_id = LOOM_STRING_ID_INVALID;

  if (base_name.size > IREE_HOST_SIZE_MAX - 32) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "private symbol name is too long");
  }
  const iree_host_size_t candidate_capacity = base_name.size + 32;
  char* candidate = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      linker->allocator, candidate_capacity, (void**)&candidate));
  memcpy(candidate, base_name.data, base_name.size);

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    int suffix_length =
        snprintf(candidate + base_name.size, 32, "$link%" PRIhsz,
                 linker->private_name_ordinal++);
    if (suffix_length < 0 || suffix_length >= 32) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "private symbol suffix exceeded scratch "
                                "buffer capacity");
      break;
    }

    loom_string_id_t candidate_name_id = LOOM_STRING_ID_INVALID;
    status = loom_module_intern_string(
        linker->target_module,
        iree_make_string_view(candidate, base_name.size + suffix_length),
        &candidate_name_id);
    if (!iree_status_is_ok(status)) break;
    if (loom_symbol_map_find(&linker->target_symbol_lookup,
                             candidate_name_id) == LOOM_SYMBOL_ID_INVALID) {
      *out_name_id = candidate_name_id;
      break;
    }
  }
  iree_allocator_free(linker->allocator, candidate);
  return status;
}

static iree_status_t loom_linker_add_target_symbol(
    loom_linker_t* linker, loom_string_id_t target_name_id,
    uint16_t* out_target_symbol_id) {
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(
      linker->target_module, target_name_id, &target_symbol_id));
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
      &linker->target_symbol_lookup, &linker->scratch_arena, target_name_id,
      target_symbol_id));
  *out_target_symbol_id = target_symbol_id;
  return iree_ok_status();
}

static iree_status_t loom_linker_rename_private_target_symbol(
    loom_linker_t* linker, uint16_t target_symbol_id) {
  if (target_symbol_id >= linker->target_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target symbol %u is out of range",
                            (unsigned)target_symbol_id);
  }

  loom_symbol_t* symbol =
      &linker->target_module->symbols.entries[target_symbol_id];
  loom_string_id_t old_name_id = symbol->name_id;
  iree_string_view_t old_name =
      linker->target_module->strings.entries[old_name_id];

  loom_string_id_t new_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_linker_allocate_fresh_private_name(linker, old_name, &new_name_id));
  if (!loom_symbol_map_erase(&linker->target_symbol_lookup, old_name_id)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target symbol map did not contain '@%.*s'",
                            (int)old_name.size, old_name.data);
  }
  symbol->name_id = new_name_id;
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(&linker->target_symbol_lookup,
                                              &linker->scratch_arena,
                                              new_name_id, target_symbol_id));
  return iree_ok_status();
}

static iree_status_t loom_linker_map_source_symbol(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  if (source_symbol_id >= source->target_symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=0, symbol=%u} is out of range",
        (unsigned)source_symbol_id);
  }
  loom_symbol_ref_t cached_ref = source->target_symbols[source_symbol_id];
  if (loom_symbol_ref_is_valid(cached_ref)) {
    *out_target_ref = cached_ref;
    return iree_ok_status();
  }

  loom_linker_t* linker = source->linker;
  const loom_symbol_t* source_symbol =
      &source->module->symbols.entries[source_symbol_id];
  iree_string_view_t source_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
      source->module, source_symbol_id, &source_name));

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(linker->target_module,
                                                 source_name, &target_name_id));
  uint16_t target_symbol_id =
      loom_symbol_map_find(&linker->target_symbol_lookup, target_name_id);

  const bool source_global =
      loom_link_symbol_has_global_identity(source->module, source_symbol);
  const bool source_concrete =
      loom_link_symbol_is_concrete_definition(source_symbol);

  if (source_global && target_symbol_id != LOOM_SYMBOL_ID_INVALID &&
      loom_link_target_symbol_is_private_concrete(linker, target_symbol_id)) {
    IREE_RETURN_IF_ERROR(
        loom_linker_rename_private_target_symbol(linker, target_symbol_id));
    target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  }

  if (source_concrete && !source_global &&
      target_symbol_id != LOOM_SYMBOL_ID_INVALID) {
    const loom_symbol_t* target_symbol =
        &linker->target_module->symbols.entries[target_symbol_id];
    if (target_symbol->defining_op &&
        !loom_link_symbol_is_declaration(target_symbol)) {
      IREE_RETURN_IF_ERROR(loom_linker_allocate_fresh_private_name(
          linker, source_name, &target_name_id));
      target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    }
  }

  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_linker_add_target_symbol(linker, target_name_id,
                                                       &target_symbol_id));
  }

  loom_symbol_ref_t target_ref = {
      .module_id = 0,
      .symbol_id = target_symbol_id,
  };
  source->target_symbols[source_symbol_id] = target_ref;
  *out_target_ref = target_ref;
  return iree_ok_status();
}

static iree_status_t loom_linker_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_linker_source_t* source = (loom_linker_source_t*)user_data;
  if (source_module != source->module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link symbol remap source module mismatch");
  }
  if (target_module != source->linker->target_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link symbol remap target module mismatch");
  }
  if (source_ref.module_id != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol ref {module=%u, symbol=%u} is not "
                            "module-local",
                            (unsigned)source_ref.module_id,
                            (unsigned)source_ref.symbol_id);
  }
  if (source_ref.symbol_id >= source->target_symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=0, symbol=%u} is out of range",
        (unsigned)source_ref.symbol_id);
  }
  if (source->selective && !source->live_symbols[source_ref.symbol_id]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selective link missed reachable source symbol ref {module=0, "
        "symbol=%u}",
        (unsigned)source_ref.symbol_id);
  }
  return loom_linker_map_source_symbol(source, source_ref.symbol_id,
                                       out_target_ref);
}

static iree_status_t loom_linker_get_source_remap(loom_linker_source_t* source,
                                                  loom_ir_remap_t** out_remap) {
  if (!source->remap.source_module) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
        source->module, source->linker->target_module, source->arena,
        &(loom_ir_remap_options_t){
            .remap_symbol = loom_ir_remap_symbol_callback_make(
                loom_linker_remap_symbol, source),
        },
        &source->remap));
  }
  *out_remap = &source->remap;
  return iree_ok_status();
}

static bool loom_link_op_symbol_ref(const loom_module_t* module,
                                    const loom_op_t* op,
                                    loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->symbol_def || !vtable->attr_descriptors) {
    return false;
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return false;
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[symbol_attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) {
    return false;
  }
  *out_ref = loom_attr_as_symbol(loom_op_const_attrs(op)[symbol_attr_index]);
  return loom_symbol_ref_is_valid(*out_ref) && out_ref->module_id == 0 &&
         out_ref->symbol_id < module->symbols.count;
}

static iree_status_t loom_link_duplicate_config_definition_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                          "duplicate config definition '@%.*s' has "
                          "incompatible %.*s",
                          (int)name.size, name.data, (int)field_name.size,
                          field_name.data);
}

static iree_status_t loom_link_remap_config_def_type_and_value(
    loom_linker_t* linker, const loom_module_t* source_module,
    const loom_op_t* source_op, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_attribute_t* out_value) {
  loom_value_id_t source_value_id = loom_config_def_type(source_op);
  if (source_value_id == LOOM_VALUE_ID_INVALID ||
      source_value_id >= source_module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config definition has no result value");
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      source_module, linker->target_module, arena, /*options=*/NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_type(
      &remap, loom_module_value_type(source_module, source_value_id),
      out_type));
  return loom_ir_remap_attribute(&remap, loom_config_def_value(source_op),
                                 out_value);
}

static iree_status_t loom_link_check_duplicate_config_definition(
    loom_linker_t* linker, loom_op_t* existing_op,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    bool* out_merge) {
  *out_merge = false;
  if (!loom_config_def_isa(existing_op) || !loom_config_def_isa(source_op)) {
    return iree_ok_status();
  }

  loom_type_t existing_type = {0};
  loom_attribute_t existing_value = {0};
  IREE_RETURN_IF_ERROR(loom_link_remap_config_def_type_and_value(
      linker, linker->target_module, existing_op, arena, &existing_type,
      &existing_value));

  loom_type_t new_type = {0};
  loom_attribute_t new_value = {0};
  IREE_RETURN_IF_ERROR(loom_link_remap_config_def_type_and_value(
      linker, source_module, source_op, arena, &new_type, &new_value));

  if (!loom_type_equal(existing_type, new_type)) {
    return loom_link_duplicate_config_definition_status(linker, target_ref,
                                                        IREE_SV("type"));
  }
  if (!loom_attribute_equal(&existing_value, &new_value)) {
    return loom_link_duplicate_config_definition_status(linker, target_ref,
                                                        IREE_SV("value"));
  }

  *out_merge = true;
  return iree_ok_status();
}

static iree_status_t loom_link_incompatible_contract_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has incompatible "
                          "function contract field '%.*s'",
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_link_signature_count_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name, iree_host_size_t source_count,
    iree_host_size_t target_count) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has %" PRIhsz
                          " %.*s but selected "
                          "symbol has %" PRIhsz,
                          (int)symbol_name.size, symbol_name.data, source_count,
                          (int)field_name.size, field_name.data, target_count);
}

static iree_status_t loom_link_signature_type_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name, iree_host_size_t ordinal) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has incompatible "
                          "%.*s type at index %" PRIhsz,
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data, ordinal);
}

static iree_status_t loom_link_map_func_signature_values(
    loom_ir_remap_t* remap, loom_func_like_t source_func,
    loom_func_like_t target_func, const loom_linker_t* linker,
    loom_symbol_ref_t target_ref) {
  uint16_t source_arg_count = 0;
  const loom_value_id_t* source_args =
      loom_func_like_arg_ids(source_func, &source_arg_count);
  uint16_t target_arg_count = 0;
  const loom_value_id_t* target_args =
      loom_func_like_arg_ids(target_func, &target_arg_count);
  if (source_arg_count != target_arg_count) {
    return loom_link_signature_count_status(linker, target_ref, IREE_SV("args"),
                                            source_arg_count, target_arg_count);
  }
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(remap, source_args, target_args,
                                                source_arg_count));

  const uint16_t source_result_count = source_func.op->result_count;
  const uint16_t target_result_count = target_func.op->result_count;
  if (source_result_count != target_result_count) {
    return loom_link_signature_count_status(
        linker, target_ref, IREE_SV("results"), source_result_count,
        target_result_count);
  }
  return loom_ir_remap_map_values(remap, loom_op_const_results(source_func.op),
                                  loom_op_const_results(target_func.op),
                                  source_result_count);
}

static iree_status_t loom_link_check_func_signature_types(
    loom_ir_remap_t* remap, const loom_module_t* source_module,
    loom_func_like_t source_func, loom_func_like_t target_func,
    const loom_linker_t* linker, loom_symbol_ref_t target_ref) {
  uint16_t source_arg_count = 0;
  const loom_value_id_t* source_args =
      loom_func_like_arg_ids(source_func, &source_arg_count);
  uint16_t target_arg_count = 0;
  const loom_value_id_t* target_args =
      loom_func_like_arg_ids(target_func, &target_arg_count);
  if (source_arg_count != target_arg_count) {
    return loom_link_signature_count_status(linker, target_ref, IREE_SV("args"),
                                            source_arg_count, target_arg_count);
  }
  for (uint16_t i = 0; i < source_arg_count; ++i) {
    loom_type_t source_type =
        loom_module_value_type(source_module, source_args[i]);
    loom_type_t remapped_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(remap, source_type, &remapped_type));
    loom_type_t target_type =
        loom_module_value_type(linker->target_module, target_args[i]);
    if (!loom_type_equal(remapped_type, target_type)) {
      return loom_link_signature_type_status(linker, target_ref, IREE_SV("arg"),
                                             i);
    }
  }

  const loom_value_id_t* source_results = loom_op_const_results(source_func.op);
  const loom_value_id_t* target_results = loom_op_const_results(target_func.op);
  if (source_func.op->result_count != target_func.op->result_count) {
    return loom_link_signature_count_status(
        linker, target_ref, IREE_SV("results"), source_func.op->result_count,
        target_func.op->result_count);
  }
  for (uint16_t i = 0; i < source_func.op->result_count; ++i) {
    loom_type_t source_type =
        loom_module_value_type(source_module, source_results[i]);
    loom_type_t remapped_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(remap, source_type, &remapped_type));
    loom_type_t target_type =
        loom_module_value_type(linker->target_module, target_results[i]);
    if (!loom_type_equal(remapped_type, target_type)) {
      return loom_link_signature_type_status(linker, target_ref,
                                             IREE_SV("result"), i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_check_func_tied_results(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    loom_func_like_t source_func, loom_func_like_t target_func) {
  if (source_func.op->tied_result_count != target_func.op->tied_result_count) {
    return loom_link_signature_count_status(
        linker, target_ref, IREE_SV("tied results"),
        source_func.op->tied_result_count, target_func.op->tied_result_count);
  }
  if (source_func.op->tied_result_count == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t byte_count =
      (iree_host_size_t)source_func.op->tied_result_count *
      sizeof(loom_tied_result_t);
  if (memcmp(loom_op_tied_results(source_func.op),
             loom_op_tied_results(target_func.op), byte_count) != 0) {
    return loom_link_incompatible_contract_status(linker, target_ref,
                                                  IREE_SV("tied_results"));
  }
  return iree_ok_status();
}

static bool loom_link_func_attr_present(loom_func_like_t func,
                                        uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(func.op)[attr_index]);
}

static iree_status_t loom_link_merge_func_attr(
    loom_ir_remap_t* remap, const loom_linker_t* linker,
    loom_symbol_ref_t target_ref, loom_func_like_t source_func,
    uint8_t source_attr_index, loom_func_like_t target_func,
    uint8_t target_attr_index, iree_string_view_t field_name) {
  if (!loom_link_func_attr_present(source_func, source_attr_index)) {
    return iree_ok_status();
  }
  if (target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_link_incompatible_contract_status(linker, target_ref,
                                                  field_name);
  }

  loom_attribute_t source_attr =
      loom_op_attrs(source_func.op)[source_attr_index];
  loom_attribute_t remapped_attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_attribute(remap, source_attr, &remapped_attr));

  loom_attribute_t* target_attr =
      &loom_op_attrs(target_func.op)[target_attr_index];
  if (loom_attr_is_absent(*target_attr)) {
    *target_attr = remapped_attr;
    return iree_ok_status();
  }
  if (!loom_attribute_equal(target_attr, &remapped_attr)) {
    return loom_link_incompatible_contract_status(linker, target_ref,
                                                  field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_merge_func_contract(
    loom_linker_t* linker, loom_linker_source_t* link_source,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    loom_op_t* target_op) {
  if (source_op == target_op) return iree_ok_status();

  loom_func_like_t source_func = loom_func_like_cast(source_module, source_op);
  loom_func_like_t target_func =
      loom_func_like_cast(linker->target_module, target_op);
  if (!loom_func_like_isa(source_func) || !loom_func_like_isa(target_func)) {
    return loom_link_incompatible_contract_status(linker, target_ref,
                                                  IREE_SV("func_like"));
  }

  loom_ir_remap_symbol_callback_t symbol_callback =
      loom_ir_remap_symbol_callback_empty();
  if (source_module != linker->target_module) {
    if (!link_source) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cross-module contract merge requires source "
                              "remap state");
    }
    symbol_callback = loom_ir_remap_symbol_callback_make(
        loom_linker_remap_symbol, link_source);
  }

  loom_ir_remap_t contract_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(source_module, linker->target_module, arena,
                               &(loom_ir_remap_options_t){
                                   .remap_symbol = symbol_callback,
                               },
                               &contract_remap));
  IREE_RETURN_IF_ERROR(loom_link_map_func_signature_values(
      &contract_remap, source_func, target_func, linker, target_ref));
  IREE_RETURN_IF_ERROR(loom_link_check_func_signature_types(
      &contract_remap, source_module, source_func, target_func, linker,
      target_ref));
  IREE_RETURN_IF_ERROR(loom_link_check_func_tied_results(
      linker, target_ref, source_func, target_func));

  const loom_link_func_contract_attr_t attrs[] = {
      {
          .source_attr_index = source_func.vtable->visibility_attr_index,
          .target_attr_index = target_func.vtable->visibility_attr_index,
          .field_name = IREE_SV("visibility"),
      },
      {
          .source_attr_index = source_func.vtable->import_module_attr_index,
          .target_attr_index = target_func.vtable->import_module_attr_index,
          .field_name = IREE_SV("import_module"),
      },
      {
          .source_attr_index = source_func.vtable->import_symbol_attr_index,
          .target_attr_index = target_func.vtable->import_symbol_attr_index,
          .field_name = IREE_SV("import_symbol"),
      },
      {
          .source_attr_index = source_func.vtable->cc_attr_index,
          .target_attr_index = target_func.vtable->cc_attr_index,
          .field_name = IREE_SV("cc"),
      },
      {
          .source_attr_index = source_func.vtable->purity_attr_index,
          .target_attr_index = target_func.vtable->purity_attr_index,
          .field_name = IREE_SV("purity"),
      },
      {
          .source_attr_index = source_func.vtable->temperature_attr_index,
          .target_attr_index = target_func.vtable->temperature_attr_index,
          .field_name = IREE_SV("temperature"),
      },
      {
          .source_attr_index = source_func.vtable->inline_policy_attr_index,
          .target_attr_index = target_func.vtable->inline_policy_attr_index,
          .field_name = IREE_SV("inline_policy"),
      },
      {
          .source_attr_index = source_func.vtable->target_attr_index,
          .target_attr_index = target_func.vtable->target_attr_index,
          .field_name = IREE_SV("target"),
      },
      {
          .source_attr_index = source_func.vtable->abi_attr_index,
          .target_attr_index = target_func.vtable->abi_attr_index,
          .field_name = IREE_SV("abi"),
      },
      {
          .source_attr_index = source_func.vtable->abi_attrs_attr_index,
          .target_attr_index = target_func.vtable->abi_attrs_attr_index,
          .field_name = IREE_SV("abi_attrs"),
      },
      {
          .source_attr_index = source_func.vtable->export_symbol_attr_index,
          .target_attr_index = target_func.vtable->export_symbol_attr_index,
          .field_name = IREE_SV("export_symbol"),
      },
      {
          .source_attr_index = source_func.vtable->export_attrs_attr_index,
          .target_attr_index = target_func.vtable->export_attrs_attr_index,
          .field_name = IREE_SV("export_attrs"),
      },
      {
          .source_attr_index = source_func.vtable->export_linkage_attr_index,
          .target_attr_index = target_func.vtable->export_linkage_attr_index,
          .field_name = IREE_SV("export_linkage"),
      },
      {
          .source_attr_index = source_func.vtable->predicates_attr_index,
          .target_attr_index = target_func.vtable->predicates_attr_index,
          .field_name = IREE_SV("predicates"),
      },
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(attrs); ++i) {
    IREE_RETURN_IF_ERROR(loom_link_merge_func_attr(
        &contract_remap, linker, target_ref, source_func,
        attrs[i].source_attr_index, target_func, attrs[i].target_attr_index,
        attrs[i].field_name));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_incompatible_config_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked config declaration for '@%.*s' has "
                          "incompatible field '%.*s'",
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_link_append_config_decl_predicates(
    loom_linker_t* linker, loom_symbol_ref_t target_ref, loom_op_t* target_op,
    loom_attribute_t predicates) {
  if (predicates.kind == LOOM_ATTR_ABSENT || predicates.count == 0) {
    return iree_ok_status();
  }
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST ||
      !predicates.predicate_list || target_op->attribute_count <= 1) {
    return loom_link_incompatible_config_status(linker, target_ref,
                                                IREE_SV("predicates"));
  }

  loom_attribute_t old_predicates = loom_op_attrs(target_op)[1];
  if (old_predicates.kind != LOOM_ATTR_ABSENT &&
      old_predicates.kind != LOOM_ATTR_PREDICATE_LIST) {
    return loom_link_incompatible_config_status(linker, target_ref,
                                                IREE_SV("predicates"));
  }
  if (old_predicates.count > 0 && !old_predicates.predicate_list) {
    return loom_link_incompatible_config_status(linker, target_ref,
                                                IREE_SV("predicates"));
  }

  iree_host_size_t total_count =
      (iree_host_size_t)old_predicates.count + predicates.count;
  if (total_count > UINT16_MAX) {
    iree_string_view_t symbol_name =
        loom_link_target_symbol_name(linker->target_module, target_ref);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "linked config declaration for '@%.*s' has %" PRIhsz
                            " merged predicates, max %u",
                            (int)symbol_name.size, symbol_name.data,
                            total_count, (unsigned)UINT16_MAX);
  }

  loom_predicate_t* merged_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &linker->target_module->arena, total_count, sizeof(*merged_predicates),
      (void**)&merged_predicates));
  if (old_predicates.count > 0) {
    memcpy(merged_predicates, old_predicates.predicate_list,
           (iree_host_size_t)old_predicates.count * sizeof(*merged_predicates));
  }
  memcpy(merged_predicates + old_predicates.count, predicates.predicate_list,
         (iree_host_size_t)predicates.count * sizeof(*merged_predicates));
  loom_op_attrs(target_op)[1] =
      loom_attr_predicate_list(merged_predicates, (uint16_t)total_count);
  return iree_ok_status();
}

static iree_status_t loom_link_merge_config_contract(
    loom_linker_t* linker, loom_linker_source_t* link_source,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    loom_op_t* target_op) {
  if (source_op == target_op) return iree_ok_status();
  if (!loom_config_decl_isa(source_op)) {
    return loom_link_incompatible_config_status(linker, target_ref,
                                                IREE_SV("declaration"));
  }

  loom_value_id_t target_value_id = loom_config_symbol_result_value(target_op);
  if (target_value_id == LOOM_VALUE_ID_INVALID) {
    return loom_link_incompatible_config_status(linker, target_ref,
                                                IREE_SV("definition"));
  }
  loom_type_t target_type =
      loom_module_value_type(linker->target_module, target_value_id);

  loom_value_id_t source_value_id = loom_config_decl_type(source_op);
  loom_ir_remap_symbol_callback_t symbol_callback =
      loom_ir_remap_symbol_callback_empty();
  if (source_module != linker->target_module) {
    if (!link_source) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cross-module contract merge requires source "
                              "remap state");
    }
    symbol_callback = loom_ir_remap_symbol_callback_make(
        loom_linker_remap_symbol, link_source);
  }

  loom_ir_remap_t contract_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(source_module, linker->target_module, arena,
                               &(loom_ir_remap_options_t){
                                   .remap_symbol = symbol_callback,
                               },
                               &contract_remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&contract_remap, source_value_id,
                                               target_value_id));
  loom_type_t remapped_source_type = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_type(
      &contract_remap, loom_module_value_type(source_module, source_value_id),
      &remapped_source_type));
  if (!loom_type_equal(remapped_source_type, target_type)) {
    return loom_link_incompatible_config_status(linker, target_ref,
                                                IREE_SV("type"));
  }

  loom_attribute_t source_predicates = loom_config_decl_predicates(source_op);
  loom_predicate_t* remapped_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      &contract_remap, source_predicates.predicate_list,
      source_predicates.count, &remapped_predicates));
  loom_attribute_t predicate_attr =
      loom_attr_predicate_list(remapped_predicates, source_predicates.count);

  if (loom_config_def_isa(target_op)) {
    return loom_config_check_value_constraints(
        loom_link_target_symbol_name(linker->target_module, target_ref),
        target_type, target_value_id, loom_config_def_value(target_op),
        predicate_attr);
  }
  if (loom_config_decl_isa(target_op)) {
    return loom_link_append_config_decl_predicates(linker, target_ref,
                                                   target_op, predicate_attr);
  }
  return loom_link_incompatible_config_status(linker, target_ref,
                                              IREE_SV("definition"));
}

static iree_status_t loom_link_merge_symbol_contract(
    loom_linker_t* linker, loom_linker_source_t* link_source,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    loom_op_t* target_op) {
  if (source_op == target_op) return iree_ok_status();
  if (loom_func_like_isa(
          loom_func_like_cast(linker->target_module, target_op))) {
    return loom_link_merge_func_contract(linker, link_source, source_module,
                                         source_op, arena, target_ref,
                                         target_op);
  }
  if (loom_config_decl_isa(target_op) || loom_config_def_isa(target_op)) {
    return loom_link_merge_config_contract(linker, link_source, source_module,
                                           source_op, arena, target_ref,
                                           target_op);
  }
  return loom_link_incompatible_contract_status(linker, target_ref,
                                                IREE_SV("declaration"));
}

static iree_status_t loom_linker_clone_source_op(loom_linker_source_t* source,
                                                 const loom_op_t* source_op,
                                                 loom_op_t* before_op,
                                                 loom_op_t** out_cloned_op) {
  loom_builder_t builder;
  loom_builder_initialize(
      source->linker->target_module, &source->linker->target_module->arena,
      loom_module_block(source->linker->target_module), &builder);
  if (before_op) {
    loom_builder_set_before(&builder, before_op);
  }

  loom_ir_remap_t* remap = NULL;
  IREE_RETURN_IF_ERROR(loom_linker_get_source_remap(source, &remap));
  return loom_ir_clone_op(&builder, source_op, remap, out_cloned_op);
}

static iree_status_t loom_linker_duplicate_concrete_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref) {
  iree_string_view_t name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                          "duplicate concrete symbol definition '@%.*s'",
                          (int)name.size, name.data);
}

static iree_status_t loom_linker_clone_or_merge_symbol_op(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t target_ref) {
  loom_linker_t* linker = source->linker;
  const loom_symbol_t* source_symbol =
      &source->module->symbols.entries[source_symbol_id];
  loom_op_t* source_op = source_symbol->defining_op;
  if (!source_op) return iree_ok_status();

  loom_symbol_t* target_symbol =
      &linker->target_module->symbols.entries[target_ref.symbol_id];
  loom_op_t* target_op = target_symbol->defining_op;
  const bool source_declaration =
      loom_link_symbol_is_declaration(source_symbol);

  if (!target_op) {
    loom_op_t* cloned_op = NULL;
    return loom_linker_clone_source_op(source, source_op, /*before_op=*/NULL,
                                       &cloned_op);
  }

  const bool target_declaration =
      loom_link_symbol_is_declaration(target_symbol);
  if (target_declaration && !source_declaration) {
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_linker_clone_source_op(source, source_op, target_op, &cloned_op));
    IREE_RETURN_IF_ERROR(loom_link_merge_symbol_contract(
        linker, source, linker->target_module, target_op, source->arena,
        target_ref, cloned_op));
    IREE_RETURN_IF_ERROR(loom_op_erase(linker->target_module, target_op));
    loom_module_link_symbol_defining_op(
        linker->target_module, cloned_op,
        loom_op_vtable(linker->target_module, cloned_op));
    return iree_ok_status();
  }

  if (source_declaration) {
    return loom_link_merge_symbol_contract(linker, source, source->module,
                                           source_op, source->arena, target_ref,
                                           target_op);
  }

  bool merge_duplicate_config_definition = false;
  IREE_RETURN_IF_ERROR(loom_link_check_duplicate_config_definition(
      linker, target_op, source->module, source_op, source->arena, target_ref,
      &merge_duplicate_config_definition));
  if (merge_duplicate_config_definition) {
    return iree_ok_status();
  }
  return loom_linker_duplicate_concrete_status(linker, target_ref);
}

static iree_status_t loom_linker_mark_source_symbol_live(
    loom_linker_source_t* source, uint16_t source_symbol_id) {
  if (source_symbol_id >= source->target_symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol %u is out of range",
                            (unsigned)source_symbol_id);
  }
  if (source->live_symbols[source_symbol_id]) {
    return iree_ok_status();
  }
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_linker_map_source_symbol(source, source_symbol_id, &target_ref));
  source->live_symbols[source_symbol_id] = 1;
  return iree_ok_status();
}

static bool loom_linker_source_symbol_is_func_provider(
    const loom_symbol_t* symbol) {
  return symbol && (symbol->kind == LOOM_SYMBOL_FUNC_TEMPLATE ||
                    symbol->kind == LOOM_SYMBOL_FUNC_UKERNEL);
}

static iree_status_t loom_linker_mark_contract_providers_live(
    loom_linker_source_t* source, iree_string_view_t contract) {
  if (iree_string_view_is_empty(contract)) return iree_ok_status();
  for (uint16_t symbol_id = 0; symbol_id < source->target_symbol_count;
       ++symbol_id) {
    const loom_symbol_t* symbol = &source->module->symbols.entries[symbol_id];
    if (!loom_linker_source_symbol_is_func_provider(symbol)) continue;
    loom_func_like_t provider =
        loom_func_like_cast(source->module, symbol->defining_op);
    if (!loom_func_like_isa(provider)) continue;

    const loom_string_id_t contract_id = loom_func_like_implements(provider);
    if (contract_id == LOOM_STRING_ID_INVALID ||
        contract_id >= source->module->strings.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func provider symbol has an invalid "
                              "implementation contract key");
    }
    if (!iree_string_view_equal(source->module->strings.entries[contract_id],
                                contract)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_linker_mark_source_symbol_live(source, symbol_id));
  }
  return iree_ok_status();
}

typedef struct loom_linker_apply_dependency_walk_t {
  // Source module currently being selectively linked.
  loom_linker_source_t* source;

  // Module containing the func.apply operations being scanned.
  const loom_module_t* apply_module;
} loom_linker_apply_dependency_walk_t;

static iree_status_t loom_linker_visit_apply_dependency(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_func_apply_isa(op)) return iree_ok_status();

  loom_linker_apply_dependency_walk_t* walk =
      (loom_linker_apply_dependency_walk_t*)user_data;
  loom_linker_source_t* source = walk->source;
  const loom_string_id_t contract_id = loom_func_apply_contract(op);
  if (contract_id == LOOM_STRING_ID_INVALID ||
      contract_id >= walk->apply_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func.apply has an invalid contract string id");
  }
  return loom_linker_mark_contract_providers_live(
      source, walk->apply_module->strings.entries[contract_id]);
}

static iree_status_t loom_linker_mark_function_apply_dependencies_live(
    loom_linker_source_t* source, const loom_module_t* apply_module,
    loom_func_like_t function) {
  if (!loom_func_like_isa(function)) return iree_ok_status();
  loom_linker_apply_dependency_walk_t walk = {
      .source = source,
      .apply_module = apply_module,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(apply_module, function, LOOM_WALK_PRE_ORDER,
                            (loom_walk_callback_t){
                                .fn = loom_linker_visit_apply_dependency,
                                .user_data = &walk,
                            },
                            source->arena, &walk_result);
}

static iree_status_t loom_linker_mark_apply_dependencies_live(
    loom_linker_source_t* source, const loom_symbol_t* symbol) {
  if (!symbol || !symbol->defining_op) return iree_ok_status();
  loom_func_like_t function =
      loom_func_like_cast(source->module, symbol->defining_op);
  return loom_linker_mark_function_apply_dependencies_live(
      source, source->module, function);
}

static iree_string_view_t loom_link_normalize_root_name(
    iree_string_view_t root_name) {
  if (iree_string_view_starts_with_char(root_name, '@')) {
    root_name = iree_string_view_remove_prefix(root_name, 1);
  }
  return root_name;
}

static iree_status_t loom_linker_mark_root_symbols_live(
    loom_linker_source_t* source, const loom_linker_add_options_t* options) {
  for (iree_host_size_t root_index = 0;
       root_index < options->root_symbols.count; ++root_index) {
    iree_string_view_t root_name =
        loom_link_normalize_root_name(options->root_symbols.values[root_index]);
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "root symbol name must not be empty");
    }
    for (iree_host_size_t symbol_index = 0;
         symbol_index < source->target_symbol_count; ++symbol_index) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          source->module, (uint16_t)symbol_index, &source_name));
      if (!iree_string_view_equal(source_name, root_name)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_linker_mark_source_symbol_live(source, (uint16_t)symbol_index));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_mark_existing_target_anchors_live(
    loom_linker_source_t* source) {
  loom_linker_t* linker = source->linker;
  for (iree_host_size_t symbol_index = 0;
       symbol_index < source->target_symbol_count; ++symbol_index) {
    iree_string_view_t source_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
        source->module, (uint16_t)symbol_index, &source_name));
    loom_string_id_t target_name_id =
        loom_module_lookup_string(linker->target_module, source_name);
    if (target_name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    uint16_t target_symbol_id =
        loom_symbol_map_find(&linker->target_symbol_lookup, target_name_id);
    if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    const loom_symbol_t* source_symbol =
        &source->module->symbols.entries[symbol_index];
    const loom_symbol_t* target_symbol =
        &linker->target_module->symbols.entries[target_symbol_id];
    const bool target_needs_materialization =
        !target_symbol->defining_op ||
        loom_link_symbol_is_declaration(target_symbol);
    if (target_needs_materialization ||
        loom_link_symbol_has_global_identity(source->module, source_symbol)) {
      IREE_RETURN_IF_ERROR(
          loom_linker_mark_source_symbol_live(source, (uint16_t)symbol_index));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_mark_existing_target_apply_dependencies_live(
    loom_linker_source_t* source) {
  loom_linker_t* linker = source->linker;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(linker->target_module, symbol) {
    if (!symbol->defining_op) continue;
    loom_func_like_t function =
        loom_func_like_cast(linker->target_module, symbol->defining_op);
    IREE_RETURN_IF_ERROR(loom_linker_mark_function_apply_dependencies_live(
        source, linker->target_module, function));
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_resolve_live_symbols(
    loom_linker_source_t* source) {
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(
      source->module, source->arena, &source->dependency_table));

  bool changed = true;
  while (changed) {
    changed = false;
    for (uint16_t symbol_id = 0; symbol_id < source->target_symbol_count;
         ++symbol_id) {
      if (!source->live_symbols[symbol_id] ||
          source->scanned_symbols[symbol_id]) {
        continue;
      }
      source->scanned_symbols[symbol_id] = 1;
      changed = true;
      if (symbol_id >= source->dependency_table.symbol_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source symbol %u is out of range for "
                                "dependency table with %" PRIhsz " symbols",
                                (unsigned)symbol_id,
                                source->dependency_table.symbol_count);
      }

      loom_symbol_dependency_edge_id_t edge_id =
          source->dependency_table.symbols[symbol_id].first_outgoing_edge_id;
      while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
        const loom_symbol_dependency_edge_t* edge =
            &source->dependency_table.edges[edge_id];
        IREE_RETURN_IF_ERROR(loom_linker_mark_source_symbol_live(
            source, edge->target_symbol_id));
        edge_id = edge->next_outgoing_edge_id;
      }

      const loom_symbol_t* symbol = &source->module->symbols.entries[symbol_id];
      IREE_RETURN_IF_ERROR(
          loom_linker_mark_apply_dependencies_live(source, symbol));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_clone_module_body(
    loom_linker_source_t* source) {
  const loom_block_t* source_block =
      loom_region_const_entry_block(source->module->body);
  for (const loom_op_t* source_op = source_block->first_op; source_op;
       source_op = source_op->next_op) {
    loom_symbol_ref_t source_ref = loom_symbol_ref_null();
    const bool has_symbol_ref =
        loom_link_op_symbol_ref(source->module, source_op, &source_ref);
    if (!has_symbol_ref) {
      if (source->selective) continue;
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(loom_linker_clone_source_op(
          source, source_op, /*before_op=*/NULL, &cloned_op));
      continue;
    }

    if (source->selective && !source->live_symbols[source_ref.symbol_id]) {
      continue;
    }
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_linker_map_source_symbol(
        source, source_ref.symbol_id, &target_ref));
    IREE_RETURN_IF_ERROR(loom_linker_clone_or_merge_symbol_op(
        source, source_ref.symbol_id, target_ref));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_validate_add_options(
    const loom_linker_add_options_t* options) {
  if (!options || options->root_symbols.count == 0) {
    return iree_ok_status();
  }
  if (!options->root_symbols.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  return iree_ok_status();
}

static iree_host_size_t loom_link_add_root_symbol_count(
    const loom_linker_add_options_t* options) {
  return options ? options->root_symbols.count : 0;
}

iree_status_t loom_linker_create(loom_context_t* context,
                                 const loom_linker_options_t* options,
                                 iree_arena_block_pool_t* block_pool,
                                 iree_allocator_t allocator,
                                 loom_linker_t** out_linker) {
  *out_linker = NULL;
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "context must not be NULL");
  }
  if (!block_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block pool must not be NULL");
  }

  loom_linker_t* linker = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*linker), (void**)&linker));
  memset(linker, 0, sizeof(*linker));
  linker->context = context;
  linker->block_pool = block_pool;
  linker->allocator = allocator;
  iree_arena_initialize(block_pool, &linker->scratch_arena);

  iree_string_view_t module_name =
      options && !iree_string_view_is_empty(options->module_name)
          ? options->module_name
          : IREE_SV("linked");
  iree_status_t status =
      loom_module_allocate(context, module_name, block_pool, /*hints=*/NULL,
                           allocator, &linker->target_module);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&linker->scratch_arena);
    iree_allocator_free(allocator, linker);
    return status;
  }

  *out_linker = linker;
  return iree_ok_status();
}

void loom_linker_free(loom_linker_t* linker) {
  if (!linker) return;
  if (linker->target_module) {
    loom_module_free(linker->target_module);
  }
  iree_arena_deinitialize(&linker->scratch_arena);
  iree_allocator_free(linker->allocator, linker);
}

iree_status_t loom_linker_add_module(loom_linker_t* linker,
                                     const loom_module_t* source_module,
                                     const loom_linker_add_options_t* options) {
  if (!linker || !source_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linker and source module must not be NULL");
  }
  if (linker->finished || !linker->target_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot add a module after linker finish");
  }
  if (source_module->context != linker->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module context does not match linker");
  }
  IREE_RETURN_IF_ERROR(loom_link_validate_add_options(options));

  iree_arena_allocator_t source_arena;
  iree_arena_initialize(linker->block_pool, &source_arena);

  loom_linker_source_t source = {
      .linker = linker,
      .module = source_module,
      .arena = &source_arena,
      .target_symbol_count = source_module->symbols.count,
      .selective = loom_link_add_root_symbol_count(options) > 0,
  };

  iree_status_t status = iree_ok_status();
  if (source.target_symbol_count > 0) {
    status = iree_arena_allocate_array(
        &source_arena, source.target_symbol_count,
        sizeof(*source.target_symbols), (void**)&source.target_symbols);
  }
  if (iree_status_is_ok(status) && source.target_symbol_count > 0) {
    for (iree_host_size_t i = 0; i < source.target_symbol_count; ++i) {
      source.target_symbols[i] = loom_symbol_ref_null();
    }
  }
  if (iree_status_is_ok(status) && source.selective) {
    status = iree_arena_allocate_array(
        &source_arena, source.target_symbol_count, sizeof(*source.live_symbols),
        (void**)&source.live_symbols);
  }
  if (iree_status_is_ok(status) && source.selective) {
    status = iree_arena_allocate_array(
        &source_arena, source.target_symbol_count,
        sizeof(*source.scanned_symbols), (void**)&source.scanned_symbols);
  }
  if (iree_status_is_ok(status) && source.selective) {
    memset(source.live_symbols, 0,
           source.target_symbol_count * sizeof(*source.live_symbols));
    memset(source.scanned_symbols, 0,
           source.target_symbol_count * sizeof(*source.scanned_symbols));
    status = loom_linker_mark_root_symbols_live(&source, options);
  }
  if (iree_status_is_ok(status) && source.selective) {
    status = loom_linker_mark_existing_target_anchors_live(&source);
  }
  if (iree_status_is_ok(status) && source.selective) {
    status = loom_linker_mark_existing_target_apply_dependencies_live(&source);
  }
  if (iree_status_is_ok(status) && source.selective) {
    status = loom_linker_resolve_live_symbols(&source);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_clone_module_body(&source);
  }

  iree_arena_deinitialize(&source_arena);
  return status;
}

iree_status_t loom_linker_finish(loom_linker_t* linker,
                                 loom_module_t** out_module) {
  *out_module = NULL;
  if (!linker) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linker must not be NULL");
  }
  if (linker->finished || !linker->target_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "linker has already been finished");
  }

  IREE_RETURN_IF_ERROR(loom_module_compute_uses(linker->target_module));
  *out_module = linker->target_module;
  linker->target_module = NULL;
  linker->finished = true;
  return iree_ok_status();
}

static iree_status_t loom_link_validate_inputs(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count) {
  if (!source_modules || source_module_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one source module is required");
  }
  if (!source_modules[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module 0 is NULL");
  }
  loom_context_t* context = source_modules[0]->context;
  for (iree_host_size_t i = 0; i < source_module_count; ++i) {
    if (!source_modules[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source module %" PRIhsz " is NULL", i);
    }
    if (source_modules[i]->context != context) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source modules must share one context");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_validate_options(
    const loom_link_options_t* options) {
  if (!options || options->root_symbols.count == 0) {
    return iree_ok_status();
  }
  if (!options->root_symbols.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  return iree_ok_status();
}

static iree_status_t loom_link_module_contains_root(
    const loom_module_t* module, const loom_link_options_t* options,
    bool* out_contains_root) {
  *out_contains_root = false;
  for (iree_host_size_t root_index = 0;
       root_index < options->root_symbols.count; ++root_index) {
    iree_string_view_t root_name =
        loom_link_normalize_root_name(options->root_symbols.values[root_index]);
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "root symbol name must not be empty");
    }
    for (iree_host_size_t symbol_index = 0;
         symbol_index < module->symbols.count; ++symbol_index) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          module, (uint16_t)symbol_index, &source_name));
      if (iree_string_view_equal(source_name, root_name)) {
        *out_contains_root = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_check_finished_roots(
    const loom_linker_t* linker, const loom_link_options_t* options) {
  if (!options) return iree_ok_status();
  for (iree_host_size_t i = 0; i < options->root_symbols.count; ++i) {
    iree_string_view_t root_name =
        loom_link_normalize_root_name(options->root_symbols.values[i]);
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "root symbol name must not be empty");
    }
    loom_string_id_t target_name_id =
        loom_module_lookup_string(linker->target_module, root_name);
    if (target_name_id == LOOM_STRING_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "root symbol '@%.*s' was not found",
                              (int)root_name.size, root_name.data);
    }
    uint16_t target_symbol_id =
        loom_symbol_map_find(&linker->target_symbol_lookup, target_name_id);
    if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "root symbol '@%.*s' was not found",
                              (int)root_name.size, root_name.data);
    }
    const loom_symbol_t* target_symbol =
        &linker->target_module->symbols.entries[target_symbol_id];
    if (!target_symbol->defining_op) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "root symbol '@%.*s' has no materialized definition or declaration",
          (int)root_name.size, root_name.data);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_link_materialized_modules(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count, const loom_link_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module) {
  *out_module = NULL;
  IREE_RETURN_IF_ERROR(
      loom_link_validate_inputs(source_modules, source_module_count));
  IREE_RETURN_IF_ERROR(loom_link_validate_options(options));

  loom_linker_t* linker = NULL;
  iree_status_t status =
      loom_linker_create(source_modules[0]->context,
                         &(loom_linker_options_t){
                             .module_name = options ? options->module_name
                                                    : iree_string_view_empty(),
                         },
                         block_pool, allocator, &linker);

  const iree_host_size_t root_symbol_count =
      options ? options->root_symbols.count : 0;
  uint8_t* root_module_flags = NULL;
  if (iree_status_is_ok(status) && root_symbol_count > 0) {
    status = iree_allocator_malloc(
        allocator, source_module_count * sizeof(*root_module_flags),
        (void**)&root_module_flags);
  }
  if (iree_status_is_ok(status) && root_module_flags) {
    memset(root_module_flags, 0,
           source_module_count * sizeof(*root_module_flags));
    for (iree_host_size_t i = 0;
         i < source_module_count && iree_status_is_ok(status); ++i) {
      bool contains_root = false;
      status = loom_link_module_contains_root(source_modules[i], options,
                                              &contains_root);
      root_module_flags[i] = contains_root ? 1 : 0;
    }
  }

  loom_linker_add_options_t add_options = {
      .root_symbols =
          {
              .count = root_symbol_count,
              .values = root_symbol_count ? options->root_symbols.values : NULL,
          },
  };
  if (iree_status_is_ok(status) && root_module_flags) {
    for (iree_host_size_t i = 0;
         i < source_module_count && iree_status_is_ok(status); ++i) {
      if (!root_module_flags[i]) continue;
      status = loom_linker_add_module(linker, source_modules[i], &add_options);
    }
  }
  for (iree_host_size_t i = 0;
       i < source_module_count && iree_status_is_ok(status); ++i) {
    if (root_module_flags && root_module_flags[i]) continue;
    status = loom_linker_add_module(linker, source_modules[i], &add_options);
  }
  if (iree_status_is_ok(status) && root_symbol_count > 0) {
    status = loom_link_check_finished_roots(linker, options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_finish(linker, out_module);
  }
  iree_allocator_free(allocator, root_module_flags);
  loom_linker_free(linker);
  return status;
}
