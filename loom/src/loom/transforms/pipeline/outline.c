// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/outline.h"

#include "loom/ir/local_value_domain.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

static const loom_pass_info_t kPipelineOutlinePassInfo = {
    .name = IREE_SVL("outline-pipeline-strands"),
    .description =
        IREE_SVL("Outline strand bodies with typed capture arguments."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_pipeline_outline_pass_info(void) {
  return &kPipelineOutlinePassInfo;
}

typedef struct loom_pipeline_outline_entry_t {
  // Source declaration whose body becomes an ordinary callable.
  loom_op_t* strand;
  // Stable worker environment retained before the construction is changed.
  const loom_target_facts_t* target_facts;
} loom_pipeline_outline_entry_t;

typedef struct loom_pipeline_outline_collection_t {
  // Module containing the construction being visited.
  loom_module_t* module;
  // Arena owning the source operation list.
  iree_arena_allocator_t* arena;
  // Arena retaining worker environments across source-scope invalidation.
  iree_arena_allocator_t* target_arena;
  // Canonical target scopes seeded during the construction traversal.
  loom_value_fact_table_t* facts;
  // Strands in preorder; reverse consumption factors nested declarations first.
  loom_pipeline_outline_entry_t* strands;
  // Number of initialized strand entries.
  iree_host_size_t count;
  // Allocated strand entries.
  iree_host_size_t capacity;
} loom_pipeline_outline_collection_t;

static iree_status_t loom_pipeline_outline_collect(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_pipeline_outline_collection_t* collection = user_data;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_nested_target_scopes(
      collection->facts, collection->module, op));
  if (!loom_pipeline_strand_isa(op)) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_pipeline_strand_body(op);
  const loom_op_t* first = loom_region_entry_block(body)->first_op;
  if (body->block_count == 1 &&
      (loom_pipeline_end_isa(first) ||
       (loom_func_call_isa(first) && loom_pipeline_end_isa(first->next_op)))) {
    return iree_ok_status();
  }
  if (collection->count == collection->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        collection->arena, collection->count, collection->count + 1,
        sizeof(*collection->strands), &collection->capacity,
        (void**)&collection->strands));
  }
  loom_target_facts_t* target_facts = NULL;
  const loom_target_facts_t* source_target =
      loom_value_fact_table_block_target_facts(collection->facts,
                                               loom_region_entry_block(body));
  if (source_target != NULL) {
    IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
        source_target, collection->target_arena, &target_facts));
  }
  collection->strands[collection->count++] = (loom_pipeline_outline_entry_t){
      .strand = op,
      .target_facts = target_facts,
  };
  return iree_ok_status();
}

static iree_status_t loom_pipeline_outline_symbol(
    loom_module_t* module, loom_symbol_map_t* names,
    iree_arena_allocator_t* arena, uint32_t* next_name,
    loom_symbol_ref_t* out_symbol) {
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  do {
    char text[48];
    iree_snprintf(text, sizeof(text), "__pipeline_strand_%u", (*next_name)++);
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, iree_make_cstring_view(text), &name));
  } while (loom_symbol_map_find(names, name) != LOOM_SYMBOL_ID_INVALID);
  loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &symbol));
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(names, arena, name, symbol));
  *out_symbol = (loom_symbol_ref_t){0, symbol};
  return iree_ok_status();
}

static iree_status_t loom_pipeline_outline_strand(loom_rewriter_t* rewriter,
                                                  loom_op_t* strand,
                                                  loom_symbol_ref_t symbol,
                                                  iree_arena_allocator_t* arena,
                                                  loom_op_t** out_function) {
  loom_module_t* module = rewriter->module;
  loom_region_t* source_body = loom_pipeline_strand_body(strand);
  loom_local_value_domain_t domain;
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      module, source_body, arena, &domain));
  const iree_host_size_t capture_count =
      domain.value_count - domain.definition_count;
  const loom_value_id_t* captures = domain.value_ids + domain.definition_count;
  loom_local_value_domain_release(&domain);

  loom_type_t* capture_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capture_count, sizeof(*capture_types), (void**)&capture_types));
  for (iree_host_size_t i = 0; i < capture_count; ++i) {
    capture_types[i] = loom_module_value_type(module, captures[i]);
  }
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder, LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY, 0, 0, 0, 0, 0,
      LOOM_INLINE_POLICY_INLINE, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), symbol, capture_types, capture_count, NULL,
      0, NULL, 0, NULL, 0, strand->location, &function));
  loom_region_t* target_body = loom_func_def_body(function);
  loom_block_t* entry = loom_region_entry_block(target_body);
  IREE_RETURN_IF_ERROR(loom_ir_remap_assign_value_types(
      module, captures, entry->arg_ids, capture_count));
  for (iree_host_size_t i = 0; i < capture_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, captures[i], entry->arg_ids[i]));
  }
  const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
  loom_ir_remap_t remap;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(module, module, arena, &options, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&remap, captures,
                                                entry->arg_ids, capture_count));

  loom_block_t* moved_entry = NULL;
  IREE_RETURN_IF_ERROR(loom_rewriter_move_region_blocks(
      rewriter, source_body, strand, target_body, 1, function, &moved_entry));
  loom_builder_enter_region(&builder, function, target_body);
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_br_build(&builder, moved_entry, NULL, 0,
                                         strand->location, &branch));
  for (uint16_t i = 1; i < target_body->block_count; ++i) {
    loom_block_t* block = target_body->blocks[i];
    for (uint16_t a = 0; a < block->arg_count; ++a) {
      const loom_value_id_t value = loom_block_arg_id(block, a);
      loom_type_t type = loom_module_value_type(module, value);
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(&remap, type, &type));
      IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(rewriter, value, type));
    }
    loom_op_t* op = block->first_op;
    while (op) {
      loom_op_t* next = op->next_op;
      if (loom_pipeline_end_isa(op)) {
        loom_builder_set_before(&rewriter->builder, op);
        loom_op_t* return_op = NULL;
        IREE_RETURN_IF_ERROR(loom_func_return_build(&rewriter->builder, NULL, 0,
                                                    op->location, &return_op));
        IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      } else {
        IREE_RETURN_IF_ERROR(loom_ir_remap_op_references(rewriter, op, &remap));
      }
      op = next;
    }
  }
  loom_builder_enter_region(&rewriter->builder, strand, source_body);
  loom_op_t* call = NULL;
  IREE_RETURN_IF_ERROR(loom_func_call_build(
      &rewriter->builder, 0, 0, 0, 0, symbol, captures, capture_count, NULL, 0,
      NULL, 0, strand->location, &call));
  loom_op_t* end = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_end_build(&rewriter->builder, strand->location, &end));
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_outline_register_worker(
    loom_pass_t* pass, loom_module_t* module, loom_op_t* function,
    const loom_target_facts_t* target_facts,
    const loom_target_environment_t* environment,
    loom_function_version_owner_t* versions,
    iree_host_size_t* next_target_context) {
  if (target_facts == NULL || versions == NULL || environment == NULL) {
    return iree_ok_status();
  }
  const loom_target_provider_t* provider =
      loom_target_environment_lookup_fact_provider(environment,
                                                   target_facts->fact_type);
  if (provider == NULL) {
    return iree_ok_status();
  }
  if (*next_target_context >= LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "outlined workers exceed %u target contexts",
                            (unsigned)LOOM_TARGET_CONTEXT_ORDINAL_INVALID);
  }
  IREE_RETURN_IF_ERROR(loom_target_facts_builder_select_execution(
      target_facts, versions->arena, &target_facts));
  const loom_symbol_ref_t symbol = loom_func_def_callee(function);
  const loom_func_symbol_facts_t function_facts = {
      .func_op = function,
      .name = loom_string_table_get(
          &module->strings, module->symbols.entries[symbol.symbol_id].name_id),
  };
  bool contract_valid = false;
  const loom_target_facts_t* function_target_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_internal_facts(
      module, &function_facts, loom_target_facts_identity_name(target_facts),
      target_facts, pass->diagnostic_emitter, versions->arena, &contract_valid,
      &function_target_facts));
  loom_target_function_version_t* version = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(versions->arena, sizeof(*version), (void**)&version));
  *version = (loom_target_function_version_t){
      .base =
          {
              .type = &loom_target_function_version_type,
              .function = loom_func_like_cast(module, function),
          },
      .resolved_target = {.provider = provider, .facts = target_facts},
      .target_context_ordinal =
          (loom_target_context_ordinal_t)(*next_target_context)++,
      .function_target_facts = function_target_facts,
  };
  return loom_function_version_owner_append(versions, &version->base);
}

iree_status_t loom_pipeline_outline_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  loom_function_version_owner_t* versions =
      loom_target_pass_capability_function_version_owner(capability);
  const loom_target_environment_t* environment =
      loom_target_pass_capability_target_environment(capability);
  loom_pipeline_outline_collection_t collection = {
      .module = module,
      .arena = pass->arena,
      .target_arena = versions ? versions->arena : pass->arena,
  };
  loom_target_function_version_snapshot_t snapshot = {0};
  bool has_snapshot = false;
  // Construction helpers have been expanded before normal compilation reaches
  // this pass. Ordinary functions and kernels need no body discovery or facts.
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    loom_op_t* definition = module->symbols.entries[i].defining_op;
    if (!loom_pipeline_def_isa(definition)) {
      continue;
    }
    if (!has_snapshot) {
      IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
          module, loom_target_pass_capability_function_versions(capability),
          pass->arena, &snapshot));
      has_snapshot = true;
    }
    const loom_func_like_t function = loom_func_like_cast(module, definition);
    const loom_target_function_version_t* version =
        loom_target_function_version_snapshot_at(&snapshot,
                                                 (loom_symbol_id_t)i);
    const loom_target_facts_t* target_facts =
        version ? version->function_target_facts : NULL;
    if (version == NULL) {
      bool resolved = false;
      IREE_RETURN_IF_ERROR(loom_target_pass_resolve_function_facts(
          pass, module, function, &resolved, &target_facts));
    }
    IREE_RETURN_IF_ERROR(loom_pass_value_facts_prepare(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(function, target_facts),
        &collection.facts));
    loom_region_t* body = loom_pipeline_def_body(definition);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_root_target_scope(
        collection.facts, module, body, definition));
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_RETURN_IF_ERROR(loom_walk_region(
        module, body, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_pipeline_outline_collect, &collection},
        &walk_result));
  }
  if (!collection.count) {
    return iree_ok_status();
  }
  loom_symbol_map_t names = {0};
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_string_id_t name = module->symbols.entries[i].name_id;
    if (name != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_symbol_map_insert(&names, pass->arena, name,
                                                  (loom_symbol_id_t)i));
    }
  }
  uint32_t next_name = 0;
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_host_size_t next_target_context = snapshot.target_context_capacity;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = collection.count;
       i > 0 && iree_status_is_ok(status); --i) {
    const loom_pipeline_outline_entry_t* entry = &collection.strands[i - 1];
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    status = loom_pipeline_outline_symbol(module, &names, pass->arena,
                                          &next_name, &symbol);
    if (iree_status_is_ok(status)) {
      loom_op_t* function = NULL;
      status = loom_pipeline_outline_strand(&rewriter, entry->strand, symbol,
                                            pass->arena, &function);
      if (iree_status_is_ok(status)) {
        status = loom_pipeline_outline_register_worker(
            pass, module, function, entry->target_facts, environment, versions,
            &next_target_context);
      }
    }
  }
  if (iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
