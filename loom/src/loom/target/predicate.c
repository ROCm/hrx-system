// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/predicate.h"

#include <stddef.h>
#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"
#include "loom/target/selection.h"

static bool loom_target_pass_predicate_is_supported_attr(
    iree_string_view_t name) {
  return iree_string_view_equal(name, IREE_SV("target")) ||
         iree_string_view_equal(name, IREE_SV("target_op")) ||
         iree_string_view_equal(name, IREE_SV("bundle")) ||
         iree_string_view_equal(name, IREE_SV("snapshot")) ||
         iree_string_view_equal(name, IREE_SV("codegen")) ||
         iree_string_view_equal(name, IREE_SV("artifact_format")) ||
         iree_string_view_equal(name, IREE_SV("abi")) ||
         iree_string_view_equal(name, IREE_SV("config")) ||
         iree_string_view_equal(name, IREE_SV("contract"));
}

static bool loom_target_pass_predicate_attrs_contain(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_host_size_t end_index, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < end_index; ++i) {
    if (attrs.entries[i].name_id >= module->strings.count) {
      continue;
    }
    if (iree_string_view_equal(
            module->strings.entries[attrs.entries[i].name_id], name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_target_pass_predicate_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t* out_string) {
  if (string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid pass.where target predicate string id");
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_target_pass_predicate_attr_strings(
    const loom_module_t* module, const loom_named_attr_t* attr,
    iree_string_view_t* out_name, iree_string_view_t* out_value) {
  IREE_RETURN_IF_ERROR(loom_target_pass_predicate_string_from_id(
      module, attr->name_id, out_name));
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.where target predicate attr '%.*s' must be string",
        (int)out_name->size, out_name->data);
  }
  return loom_target_pass_predicate_string_from_id(
      module, loom_attr_as_string_id(attr->value), out_value);
}

static iree_status_t loom_target_pass_predicate_verify(
    void* user_data, const loom_pass_predicate_verify_context_t* context) {
  if (!iree_string_view_equal(context->predicate, IREE_SV("target"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported target pass.where predicate '%.*s'",
                            (int)context->predicate.size,
                            context->predicate.data);
  }
  if (context->anchor_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.where target predicate requires func anchor");
  }
  const loom_named_attr_slice_t attrs =
      loom_pass_where_attrs(context->where_op);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_target_pass_predicate_attr_strings(
        context->pipeline_module, &attrs.entries[i], &name, &value));
    (void)value;
    if (!loom_target_pass_predicate_is_supported_attr(name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass.where target predicate does not accept attr '%.*s'",
          (int)name.size, name.data);
    }
    if (loom_target_pass_predicate_attrs_contain(context->pipeline_module,
                                                 attrs, i, name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass.where target predicate has duplicate attr '%.*s'",
          (int)name.size, name.data);
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_target_pass_predicate_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_target_pass_predicate_trim_symbol(
    iree_string_view_t name) {
  name = iree_string_view_trim(name);
  while (name.size > 0 && name.data[0] == '@') {
    name = iree_string_view_substr(name, 1, IREE_STRING_VIEW_NPOS);
  }
  return name;
}

static bool loom_target_pass_predicate_symbol_matches(
    iree_string_view_t actual, iree_string_view_t expected) {
  return iree_string_view_equal(
      actual, loom_target_pass_predicate_trim_symbol(expected));
}

static bool loom_target_pass_predicate_symbol_id(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_symbol_id_t* out_symbol_id) {
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!module || !symbol || module->symbols.count == 0) {
    return false;
  }
  const loom_symbol_t* first = module->symbols.entries;
  const loom_symbol_t* last = first + module->symbols.count;
  if (symbol < first || symbol >= last) {
    return false;
  }
  *out_symbol_id = (loom_symbol_id_t)(symbol - first);
  return true;
}

typedef struct loom_target_pass_predicate_target_facts_t {
  // Resolved function facts for the current function.
  const loom_func_symbol_facts_t* func;
  // Effective target record symbol selected for |func|.
  loom_symbol_ref_t target_ref;
  // Target record op named by |target_ref|.
  const loom_op_t* target_op;
  // Target record facts named by |target_ref|.
  const loom_target_symbol_facts_t* target;
  // True after function contract resolution has run.
  bool contract_resolved;
  // True when |contract_storage| contains a valid function contract.
  bool contract_valid;
  // Function contract storage resolved from |func| and |target|.
  loom_target_bundle_storage_t contract_storage;
} loom_target_pass_predicate_target_facts_t;

static iree_status_t loom_target_pass_predicate_resolve_facts(
    const loom_pass_predicate_evaluate_context_t* context,
    iree_arena_allocator_t* arena,
    loom_target_pass_predicate_target_facts_t* out_facts, bool* out_valid) {
  *out_facts = (loom_target_pass_predicate_target_facts_t){0};
  *out_valid = false;

  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_target_pass_predicate_symbol_id(context->target_module,
                                            context->symbol, &symbol_id)) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
      &fact_table, context->target_module, symbol_id, &base_facts));
  out_facts->func = loom_func_symbol_facts_cast(base_facts);
  if (!out_facts->func) {
    return iree_ok_status();
  }
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_environment(context->environment);
  out_facts->target_ref = loom_target_effective_target_ref(
      out_facts->func->target_symbol, target_capability);
  if (!loom_symbol_ref_is_valid(out_facts->target_ref)) {
    return iree_ok_status();
  }
  if (out_facts->target_ref.module_id != 0 ||
      out_facts->target_ref.symbol_id >=
          context->target_module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &fact_table, context->target_module, out_facts->target_ref,
      &target_base_facts));
  out_facts->target = loom_target_symbol_facts_cast(target_base_facts);
  if (!out_facts->target) {
    return iree_ok_status();
  }
  out_facts->target_op =
      context->target_module->symbols.entries[out_facts->target_ref.symbol_id]
          .defining_op;

  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_target_pass_predicate_resolve_contract(
    const loom_pass_predicate_evaluate_context_t* context,
    loom_target_pass_predicate_target_facts_t* facts) {
  if (facts->contract_resolved) {
    return iree_ok_status();
  }
  facts->contract_resolved = true;
  return loom_target_function_contract_resolve_from_bundle(
      context->target_module, facts->func, facts->target->name,
      &facts->target->storage.bundle, (iree_diagnostic_emitter_t){0},
      &facts->contract_valid, &facts->contract_storage);
}

static iree_status_t loom_target_pass_predicate_match_attr(
    const loom_pass_predicate_evaluate_context_t* context,
    loom_target_pass_predicate_target_facts_t* facts, iree_string_view_t name,
    iree_string_view_t expected, bool* out_match) {
  *out_match = false;
  const loom_target_bundle_storage_t* storage = &facts->target->storage;
  if (iree_string_view_equal(name, IREE_SV("target"))) {
    *out_match = loom_target_pass_predicate_symbol_matches(
        loom_target_pass_predicate_symbol_name(context->target_module,
                                               facts->target_ref),
        expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("target_op"))) {
    if (facts->target_op == NULL) {
      return iree_ok_status();
    }
    *out_match = iree_string_view_equal(
        loom_op_name(context->target_module, facts->target_op), expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("bundle"))) {
    *out_match = iree_string_view_equal(storage->bundle.name, expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("snapshot"))) {
    *out_match = iree_string_view_equal(storage->snapshot.name, expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("codegen"))) {
    *out_match = iree_string_view_equal(
        loom_target_codegen_format_name(storage->snapshot.codegen_format),
        expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("artifact_format"))) {
    *out_match = iree_string_view_equal(
        loom_target_artifact_format_name(storage->snapshot.artifact_format),
        expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("abi"))) {
    IREE_RETURN_IF_ERROR(
        loom_target_pass_predicate_resolve_contract(context, facts));
    if (!facts->contract_valid) {
      return iree_ok_status();
    }
    *out_match = iree_string_view_equal(
        loom_target_abi_kind_name(facts->contract_storage.export_plan.abi_kind),
        expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("config"))) {
    *out_match = iree_string_view_equal(storage->config.name, expected);
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("contract"))) {
    *out_match =
        iree_string_view_equal(storage->config.contract_set_key, expected);
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_target_pass_predicate_evaluate(
    void* user_data, const loom_pass_predicate_evaluate_context_t* context,
    bool* out_match) {
  *out_match = false;
  if (!iree_string_view_equal(context->predicate, IREE_SV("target"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported target pass.where predicate '%.*s'",
                            (int)context->predicate.size,
                            context->predicate.data);
  }
  if (context->anchor_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.where target predicate requires func anchor");
  }

  loom_target_pass_predicate_provider_storage_t* storage =
      (loom_target_pass_predicate_provider_storage_t*)user_data;
  if (!storage || !storage->block_pool) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target pass predicate provider requires a block pool");
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(storage->block_pool, &arena);
  loom_target_pass_predicate_target_facts_t facts = {0};
  bool facts_valid = false;
  iree_status_t status = loom_target_pass_predicate_resolve_facts(
      context, &arena, &facts, &facts_valid);
  if (iree_status_is_ok(status) && facts_valid) {
    *out_match = true;
    const loom_named_attr_slice_t attrs =
        loom_pass_where_attrs(context->where_op);
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      iree_string_view_t name = iree_string_view_empty();
      iree_string_view_t expected = iree_string_view_empty();
      status = loom_target_pass_predicate_attr_strings(
          context->pipeline_module, &attrs.entries[i], &name, &expected);
      if (!iree_status_is_ok(status)) {
        break;
      }
      bool attr_match = false;
      status = loom_target_pass_predicate_match_attr(context, &facts, name,
                                                     expected, &attr_match);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (!attr_match) {
        *out_match = false;
        break;
      }
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}

void loom_target_pass_predicate_provider_storage_initialize(
    iree_arena_block_pool_t* block_pool,
    loom_target_pass_predicate_provider_storage_t* out_storage) {
  *out_storage = (loom_target_pass_predicate_provider_storage_t){
      .block_pool = block_pool,
  };
}

loom_pass_predicate_provider_t loom_target_pass_predicate_provider(
    loom_target_pass_predicate_provider_storage_t* storage) {
  return (loom_pass_predicate_provider_t){
      .verify = loom_target_pass_predicate_verify,
      .evaluate = loom_target_pass_predicate_evaluate,
      .user_data = storage,
  };
}
