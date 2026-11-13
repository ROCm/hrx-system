// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/compile/request.h"

#include <stdlib.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/link/linker.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/target/entry_selection.h"
#include "loom/target/module_specialization.h"
#include "loom/target/projection.h"

static iree_string_view_t loom_compile_entry_kind_name(
    loom_compile_entry_kind_t kind) {
  switch (kind) {
    case LOOM_COMPILE_ENTRY_KIND_KERNEL:
      return IREE_SV("kernel");
    case LOOM_COMPILE_ENTRY_KIND_COMMAND:
      return IREE_SV("command");
    case LOOM_COMPILE_ENTRY_KIND_MODULE:
      return IREE_SV("module");
    case LOOM_COMPILE_ENTRY_KIND_INVALID:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

static iree_status_t loom_compile_entry_selection_lookup_root(
    const loom_module_t* module, iree_string_view_t root_name,
    const loom_symbol_t** out_symbol) {
  *out_symbol = NULL;
  root_name = loom_target_entry_normalize_symbol_name(root_name);
  if (iree_string_view_is_empty(root_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root symbol name must not be empty");
  }
  const loom_string_id_t name_id = loom_module_lookup_string(module, root_name);
  const loom_symbol_id_t symbol_id =
      name_id != LOOM_STRING_ID_INVALID
          ? loom_module_find_symbol(module, name_id)
          : LOOM_SYMBOL_ID_INVALID;
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "root symbol '@%.*s' was not found",
                            (int)root_name.size, root_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "root symbol '@%.*s' has no materialized definition or declaration",
        (int)root_name.size, root_name.data);
  }
  *out_symbol = symbol;
  return iree_ok_status();
}

static iree_status_t loom_compile_entry_selection_emit_unresolved_root(
    const loom_compile_request_options_t* options, const loom_module_t* module,
    const loom_symbol_t* symbol) {
  const iree_string_view_t symbol_name =
      loom_string_table_get(&module->strings, symbol->name_id);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
  };
  const loom_diagnostic_emission_t emission = {
      .op = symbol->defining_op,
      .error = LOOM_ERR_SYMBOL_002,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(options->diagnostic_emitter, &emission);
}

static bool loom_compile_entry_selection_is_array_program(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  const loom_func_like_t function =
      loom_func_like_const_cast(module, symbol->defining_op);
  return loom_func_like_isa(function) &&
         loom_func_like_abi(function) == LOOM_TARGET_ABI_ARRAY_PROGRAM;
}

static iree_status_t loom_compile_entry_selection_classify_symbol(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_entry_kind_t* out_kind) {
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    const loom_symbol_product_carrier_t carrier =
        loom_symbol_definition_product_carrier(symbol->definition,
                                               symbol->defining_op);
    switch (carrier) {
      case LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED:
      case 0:
        *out_kind = LOOM_COMPILE_ENTRY_KIND_MODULE;
        return iree_ok_status();
      case LOOM_PIPELINE_SCOPE_KERNEL:
        *out_kind = LOOM_COMPILE_ENTRY_KIND_KERNEL;
        return iree_ok_status();
      default: {
        const iree_string_view_t symbol_name =
            loom_string_table_get(&module->strings, symbol->name_id);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pipeline root '@%.*s' has unsupported entry scope %u",
            (int)symbol_name.size, symbol_name.data, (unsigned)carrier);
      }
    }
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM)) {
    *out_kind = LOOM_COMPILE_ENTRY_KIND_COMMAND;
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL) ||
      loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY) ||
      loom_compile_entry_selection_is_array_program(module, symbol)) {
    *out_kind = LOOM_COMPILE_ENTRY_KIND_KERNEL;
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    *out_kind = LOOM_COMPILE_ENTRY_KIND_MODULE;
    return iree_ok_status();
  }
  const iree_string_view_t symbol_name =
      loom_string_table_get(&module->strings, symbol->name_id);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "root symbol '@%.*s' does not define a compilable entry",
      (int)symbol_name.size, symbol_name.data);
}

static iree_status_t loom_compile_entry_selection_kernel_target_type(
    const loom_module_t* module, const loom_symbol_t* symbol,
    const loom_target_fact_type_t** out_fact_type) {
  *out_fact_type = NULL;
  const loom_func_like_t function =
      loom_func_like_const_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "kernel root does not implement FuncLike");
  }
  const loom_symbol_ref_t target_ref = loom_func_like_target(function);
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_ok_status();
  }
  if (target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel root has an invalid target reference");
  }
  const loom_symbol_t* target_symbol =
      &module->symbols.entries[target_ref.symbol_id];
  const loom_target_like_descriptor_t* descriptor = loom_target_like_descriptor(
      loom_target_like_cast(module, target_symbol->defining_op));
  if (descriptor == NULL || descriptor->fact_type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel root target is not a target definition");
  }
  *out_fact_type = descriptor->fact_type;
  return iree_ok_status();
}

static iree_status_t loom_compile_entry_selection_merge_kernel_target(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_entry_selection_t* selection) {
  const loom_target_fact_type_t* fact_type = NULL;
  IREE_RETURN_IF_ERROR(loom_compile_entry_selection_kernel_target_type(
      module, symbol, &fact_type));
  if (fact_type == NULL) {
    ++selection->untargeted_kernel_count;
    return iree_ok_status();
  }
  if (selection->target_fact_type != NULL &&
      selection->target_fact_type != fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected kernel roots use multiple target families ('%.*s' and "
        "'%.*s')",
        (int)selection->target_fact_type->name.size,
        selection->target_fact_type->name.data, (int)fact_type->name.size,
        fact_type->name.data);
  }
  selection->target_fact_type = fact_type;
  return iree_ok_status();
}

static iree_status_t loom_compile_entry_selection_merge_root(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_entry_selection_t* selection) {
  loom_compile_entry_kind_t kind = LOOM_COMPILE_ENTRY_KIND_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_compile_entry_selection_classify_symbol(module, symbol, &kind));
  if (selection->roots.count != 0 && selection->kind != kind) {
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected roots mix entry categories '%.*s' and '%.*s' at '@%.*s'",
        (int)loom_compile_entry_kind_name(selection->kind).size,
        loom_compile_entry_kind_name(selection->kind).data,
        (int)loom_compile_entry_kind_name(kind).size,
        loom_compile_entry_kind_name(kind).data, (int)symbol_name.size,
        symbol_name.data);
  }
  selection->kind = kind;
  ++selection->roots.count;
  return kind == LOOM_COMPILE_ENTRY_KIND_KERNEL
             ? loom_compile_entry_selection_merge_kernel_target(module, symbol,
                                                                selection)
             : iree_ok_status();
}

static loom_compile_entry_kind_t loom_compile_entry_selection_default_kind(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol->defining_op == NULL ||
      loom_symbol_definition_is_declaration(symbol->definition)) {
    return LOOM_COMPILE_ENTRY_KIND_INVALID;
  }
  const bool is_public = iree_any_bit_set(
      symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN);
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM)) {
    return is_public ? LOOM_COMPILE_ENTRY_KIND_COMMAND
                     : LOOM_COMPILE_ENTRY_KIND_INVALID;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    const loom_symbol_product_carrier_t carrier =
        loom_symbol_definition_product_carrier(symbol->definition,
                                               symbol->defining_op);
    if (carrier == LOOM_PIPELINE_SCOPE_KERNEL) {
      return is_public ? LOOM_COMPILE_ENTRY_KIND_KERNEL
                       : LOOM_COMPILE_ENTRY_KIND_INVALID;
    }
    return is_public &&
                   (carrier == LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED ||
                    carrier == 0) &&
                   loom_symbol_implements(symbol,
                                          LOOM_SYMBOL_INTERFACE_FUNC_LIKE)
               ? LOOM_COMPILE_ENTRY_KIND_MODULE
               : LOOM_COMPILE_ENTRY_KIND_INVALID;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY)) {
    return LOOM_COMPILE_ENTRY_KIND_KERNEL;
  }
  if (loom_compile_entry_selection_is_array_program(module, symbol)) {
    return is_public ? LOOM_COMPILE_ENTRY_KIND_KERNEL
                     : LOOM_COMPILE_ENTRY_KIND_INVALID;
  }
  if (is_public &&
      loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) &&
      !loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL)) {
    return LOOM_COMPILE_ENTRY_KIND_MODULE;
  }
  return LOOM_COMPILE_ENTRY_KIND_INVALID;
}

static bool loom_compile_entry_selection_name_equal(iree_string_view_t lhs,
                                                    iree_string_view_t rhs) {
  return iree_string_view_equal(loom_target_entry_normalize_symbol_name(lhs),
                                loom_target_entry_normalize_symbol_name(rhs));
}

static bool loom_compile_entry_selection_list_contains(
    iree_string_view_list_t roots, iree_string_view_t root_name) {
  for (iree_host_size_t i = 0; i < roots.count; ++i) {
    if (loom_compile_entry_selection_name_equal(roots.values[i], root_name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_compile_entry_selection_reject_command(void) {
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "command-program roots require loomc_cmd_program_product_build");
}

typedef struct loom_compile_default_root_summary_t {
  iree_host_size_t count;
  iree_host_size_t name_bytes;
} loom_compile_default_root_summary_t;

static iree_status_t loom_compile_entry_selection_summarize_defaults(
    const loom_module_t* module,
    loom_compile_default_root_summary_t summaries[4]) {
  memset(summaries, 0, sizeof(*summaries) * 4);
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const loom_compile_entry_kind_t kind =
        loom_compile_entry_selection_default_kind(module, symbol);
    if (kind == LOOM_COMPILE_ENTRY_KIND_INVALID) {
      continue;
    }
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    loom_compile_default_root_summary_t* summary = &summaries[kind];
    if (!iree_host_size_checked_add(summary->name_bytes, symbol_name.size,
                                    &summary->name_bytes)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "compile root names are too large");
    }
    ++summary->count;
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_entry_selection_apply_exclusions(
    const loom_module_t* module, iree_string_view_list_t excluded_roots,
    loom_compile_entry_kind_t selected_kind,
    loom_compile_default_root_summary_t* selected_summary) {
  for (iree_host_size_t i = 0; i < excluded_roots.count; ++i) {
    const iree_string_view_t excluded_name = excluded_roots.values[i];
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (loom_compile_entry_selection_name_equal(excluded_name,
                                                  excluded_roots.values[j])) {
        const iree_string_view_t normalized_name =
            loom_target_entry_normalize_symbol_name(excluded_name);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "excluded root '@%.*s' is repeated",
            (int)normalized_name.size, normalized_name.data);
      }
    }

    const loom_symbol_t* symbol = NULL;
    IREE_RETURN_IF_ERROR(loom_compile_entry_selection_lookup_root(
        module, excluded_name, &symbol));
    loom_compile_entry_kind_t kind = LOOM_COMPILE_ENTRY_KIND_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_compile_entry_selection_classify_symbol(module, symbol, &kind));
    const iree_string_view_t normalized_name =
        loom_target_entry_normalize_symbol_name(excluded_name);
    if (kind != selected_kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "excluded root '@%.*s' has entry category '%.*s', expected '%.*s'",
          (int)normalized_name.size, normalized_name.data,
          (int)loom_compile_entry_kind_name(kind).size,
          loom_compile_entry_kind_name(kind).data,
          (int)loom_compile_entry_kind_name(selected_kind).size,
          loom_compile_entry_kind_name(selected_kind).data);
    }
    if (loom_compile_entry_selection_default_kind(module, symbol) != kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "excluded root '@%.*s' is not selected by entry category '%.*s'",
          (int)normalized_name.size, normalized_name.data,
          (int)loom_compile_entry_kind_name(kind).size,
          loom_compile_entry_kind_name(kind).data);
    }
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    IREE_ASSERT_GT(selected_summary->count, 0u);
    IREE_ASSERT_GE(selected_summary->name_bytes, symbol_name.size);
    --selected_summary->count;
    selected_summary->name_bytes -= symbol_name.size;
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_entry_selection_require_default_roots(
    loom_compile_entry_kind_t kind, iree_host_size_t root_count,
    bool has_exclusions) {
  if (root_count != 0) {
    return iree_ok_status();
  }
  if (has_exclusions) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "excluded roots empty the default %.*s root set",
                            (int)loom_compile_entry_kind_name(kind).size,
                            loom_compile_entry_kind_name(kind).data);
  }
  switch (kind) {
    case LOOM_COMPILE_ENTRY_KIND_KERNEL:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "entry category 'kernel' requires a nonempty root set, "
          "public or retained kernel-scoped pipelines, or array programs");
    case LOOM_COMPILE_ENTRY_KIND_MODULE:
    case LOOM_COMPILE_ENTRY_KIND_COMMAND:
    case LOOM_COMPILE_ENTRY_KIND_INVALID:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "compile entry category is invalid");
}

static iree_status_t loom_compile_entry_selection_copy_default_roots(
    const loom_module_t* module, loom_compile_entry_kind_t kind,
    iree_string_view_list_t excluded_roots,
    loom_compile_default_root_summary_t root_summary,
    iree_arena_allocator_t* arena,
    loom_compile_entry_selection_t* out_selection) {
  iree_string_view_t* root_names = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, root_summary.count, sizeof(*root_names), (void**)&root_names));
  char* root_name_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, root_summary.name_bytes,
                                           (void**)&root_name_storage));

  loom_compile_entry_selection_t selection = {
      .kind = kind,
      .roots = {.values = root_names},
  };
  char* next_root_name = root_name_storage;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (loom_compile_entry_selection_default_kind(module, symbol) != kind) {
      continue;
    }
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    if (loom_compile_entry_selection_list_contains(excluded_roots,
                                                   symbol_name)) {
      continue;
    }
    memcpy(next_root_name, symbol_name.data, symbol_name.size);
    root_names[selection.roots.count] =
        iree_make_string_view(next_root_name, symbol_name.size);
    next_root_name += symbol_name.size;
    ++selection.roots.count;
    if (kind == LOOM_COMPILE_ENTRY_KIND_KERNEL) {
      IREE_RETURN_IF_ERROR(loom_compile_entry_selection_merge_kernel_target(
          module, symbol, &selection));
    }
  }
  IREE_ASSERT_EQ(selection.roots.count, root_summary.count);
  IREE_ASSERT_EQ(next_root_name, root_name_storage + root_summary.name_bytes);
  *out_selection = selection;
  return iree_ok_status();
}

static iree_status_t loom_compile_entry_selection_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    iree_arena_allocator_t* arena,
    loom_compile_entry_selection_t* out_selection, uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_selection);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_selection = (loom_compile_entry_selection_t){0};
  *out_error_count = 0;
  const iree_string_view_list_t explicit_roots = options->roots;
  const iree_string_view_list_t excluded_roots = options->excluded_roots;
  if (explicit_roots.count != 0 && explicit_roots.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root count is nonzero but roots are NULL");
  }
  if (explicit_roots.count != 0 && excluded_roots.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "explicit roots and excluded roots cannot be combined");
  }
  if (excluded_roots.count != 0 && excluded_roots.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "excluded root count is nonzero but excluded roots are NULL");
  }

  if (explicit_roots.count != 0) {
    loom_compile_entry_selection_t selection = {
        .roots = {.values = explicit_roots.values},
    };
    for (iree_host_size_t i = 0; i < explicit_roots.count; ++i) {
      const loom_symbol_t* symbol = NULL;
      IREE_RETURN_IF_ERROR(loom_compile_entry_selection_lookup_root(
          module, explicit_roots.values[i], &symbol));
      if (loom_symbol_definition_is_declaration(symbol->definition)) {
        IREE_RETURN_IF_ERROR(loom_compile_entry_selection_emit_unresolved_root(
            options, module, symbol));
        *out_error_count = 1;
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_compile_entry_selection_merge_root(module, symbol, &selection));
    }
    if (selection.kind == LOOM_COMPILE_ENTRY_KIND_COMMAND) {
      return loom_compile_entry_selection_reject_command();
    }
    *out_selection = selection;
    return iree_ok_status();
  }

  loom_compile_default_root_summary_t summaries[4];
  IREE_RETURN_IF_ERROR(
      loom_compile_entry_selection_summarize_defaults(module, summaries));
  const iree_host_size_t category_count =
      (summaries[LOOM_COMPILE_ENTRY_KIND_COMMAND].count != 0) +
      (summaries[LOOM_COMPILE_ENTRY_KIND_KERNEL].count != 0) +
      (summaries[LOOM_COMPILE_ENTRY_KIND_MODULE].count != 0);
  if (category_count > 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unrooted module has mixed default entry categories; select roots "
        "explicitly");
  }
  const loom_compile_entry_kind_t selected_kind =
      summaries[LOOM_COMPILE_ENTRY_KIND_COMMAND].count != 0
          ? LOOM_COMPILE_ENTRY_KIND_COMMAND
      : summaries[LOOM_COMPILE_ENTRY_KIND_KERNEL].count != 0
          ? LOOM_COMPILE_ENTRY_KIND_KERNEL
          : LOOM_COMPILE_ENTRY_KIND_MODULE;
  if (selected_kind == LOOM_COMPILE_ENTRY_KIND_COMMAND) {
    return loom_compile_entry_selection_reject_command();
  }
  loom_compile_default_root_summary_t selected_summary =
      summaries[selected_kind];
  IREE_RETURN_IF_ERROR(loom_compile_entry_selection_apply_exclusions(
      module, excluded_roots, selected_kind, &selected_summary));

  if (selected_kind == LOOM_COMPILE_ENTRY_KIND_MODULE &&
      excluded_roots.count == 0) {
    out_selection->kind = LOOM_COMPILE_ENTRY_KIND_MODULE;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_compile_entry_selection_require_default_roots(
      selected_kind, selected_summary.count, excluded_roots.count != 0));
  return loom_compile_entry_selection_copy_default_roots(
      module, selected_kind, excluded_roots, selected_summary, arena,
      out_selection);
}

static iree_status_t loom_compile_request_select_named_format(
    iree_string_view_t format,
    const loom_target_environment_t* target_environment,
    const loom_target_emitter_t** out_target_emitter) {
  *out_target_emitter = NULL;
  const loom_target_emitter_t* target_emitter =
      loom_target_environment_lookup_emitter(target_environment, format);
  if (target_emitter == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format '%.*s' is not available in this binary",
                            (int)format.size, format.data);
  }
  *out_target_emitter = target_emitter;
  return iree_ok_status();
}

static iree_status_t loom_compile_request_select_canonical_kernel_emitter(
    const loom_target_fact_type_t* target_fact_type,
    const loom_target_environment_t* target_environment,
    const loom_target_emitter_t** out_target_emitter) {
  *out_target_emitter = NULL;
  const loom_target_emitter_t* canonical_emitter =
      loom_target_environment_lookup_canonical_kernel_emitter(
          target_environment, target_fact_type);
  if (canonical_emitter == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "no canonical kernel format is configured for target family '%.*s'",
        (int)target_fact_type->name.size, target_fact_type->name.data);
  }
  *out_target_emitter = canonical_emitter;
  return iree_ok_status();
}

static iree_status_t loom_compile_request_select_emitter(
    loom_compile_entry_kind_t kind, iree_string_view_t explicit_format,
    const loom_target_fact_type_t* target_fact_type,
    const loom_target_environment_t* target_environment,
    const loom_target_emitter_t** out_target_emitter) {
  explicit_format = iree_string_view_trim(explicit_format);
  if (!iree_string_view_is_empty(explicit_format)) {
    return loom_compile_request_select_named_format(
        explicit_format, target_environment, out_target_emitter);
  }
  switch (kind) {
    case LOOM_COMPILE_ENTRY_KIND_KERNEL:
      return loom_compile_request_select_canonical_kernel_emitter(
          target_fact_type, target_environment, out_target_emitter);
    case LOOM_COMPILE_ENTRY_KIND_MODULE: {
      const loom_target_emitter_t* canonical_emitter =
          target_fact_type != NULL
              ? loom_target_environment_lookup_canonical_module_emitter(
                    target_environment, target_fact_type)
              : NULL;
      if (canonical_emitter != NULL) {
        *out_target_emitter = canonical_emitter;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "module entries require an explicit format or a target profile "
          "with a canonical module format");
    }
    case LOOM_COMPILE_ENTRY_KIND_COMMAND:
    case LOOM_COMPILE_ENTRY_KIND_INVALID:
      break;
  }
  IREE_ASSERT_UNREACHABLE("resolved compile entry category");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_compile_request_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    const loom_target_environment_t* target_environment,
    iree_arena_allocator_t* arena, loom_compile_request_t* out_request,
    uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_request);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_request = (loom_compile_request_t){0};
  *out_error_count = 0;

  loom_compile_entry_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_compile_entry_selection_resolve(
      module, options, arena, &selection, out_error_count));
  if (*out_error_count != 0) {
    return iree_ok_status();
  }

  loom_compile_request_t request = {
      .selection = selection,
      .target_profile = options->target_profile,
  };
  if (request.target_profile != NULL && selection.target_fact_type != NULL &&
      request.target_profile->type->fact_type != selection.target_fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target family '%.*s' cannot specialize roots authored for target "
        "family '%.*s'",
        (int)request.target_profile->type->name.size,
        request.target_profile->type->name.data,
        (int)selection.target_fact_type->name.size,
        selection.target_fact_type->name.data);
  }
  const loom_target_fact_type_t* target_fact_type =
      request.target_profile != NULL ? request.target_profile->type->fact_type
                                     : selection.target_fact_type;
  if (selection.kind == LOOM_COMPILE_ENTRY_KIND_KERNEL &&
      request.target_profile == NULL &&
      selection.untargeted_kernel_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel entries require an explicit target when %u selected root%s "
        "omit target(...) attrs",
        (unsigned)selection.untargeted_kernel_count,
        selection.untargeted_kernel_count == 1 ? "" : "s");
  }
  if (selection.kind == LOOM_COMPILE_ENTRY_KIND_KERNEL &&
      target_fact_type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel entries require a target");
  }
  if (target_fact_type != NULL &&
      loom_target_environment_lookup_fact_provider(target_environment,
                                                   target_fact_type) == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target family '%.*s' is not available in this binary",
        (int)target_fact_type->name.size, target_fact_type->name.data);
  }
  IREE_RETURN_IF_ERROR(loom_compile_request_select_emitter(
      request.selection.kind, options->format, target_fact_type,
      target_environment, &request.target_emitter));
  *out_request = request;
  return iree_ok_status();
}

static int loom_compile_request_compare_symbol_refs(const void* lhs_ptr,
                                                    const void* rhs_ptr) {
  const loom_symbol_ref_t lhs = *(const loom_symbol_ref_t*)lhs_ptr;
  const loom_symbol_ref_t rhs = *(const loom_symbol_ref_t*)rhs_ptr;
  return (lhs.symbol_id > rhs.symbol_id) - (lhs.symbol_id < rhs.symbol_id);
}

static iree_status_t loom_compile_request_build_specializations(
    const loom_module_t* module, loom_linker_target_symbol_list_t root_symbols,
    const loom_target_profile_t* target_profile, iree_arena_allocator_t* arena,
    loom_target_specialization_request_list_t* out_specializations) {
  *out_specializations = (loom_target_specialization_request_list_t){0};
  if (root_symbols.count > 1) {
    qsort(root_symbols.values, root_symbols.count, sizeof(*root_symbols.values),
          loom_compile_request_compare_symbol_refs);
  }

  const iree_host_size_t capacity =
      root_symbols.count != 0 ? root_symbols.count : module->symbols.count;
  loom_target_specialization_request_t* specializations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*specializations), (void**)&specializations));
  iree_host_size_t specialization_count = 0;
  if (root_symbols.count != 0) {
    for (iree_host_size_t i = 0; i < root_symbols.count; ++i) {
      const loom_symbol_ref_t root = root_symbols.values[i];
      if (i != 0 && root.symbol_id == root_symbols.values[i - 1].symbol_id) {
        continue;
      }
      const loom_symbol_t* symbol = &module->symbols.entries[root.symbol_id];
      if (loom_func_like_body(
              loom_func_like_const_cast(module, symbol->defining_op)) == NULL) {
        continue;
      }
      specializations[specialization_count++] =
          (loom_target_specialization_request_t){
              .function_ref = {.module_id = 0, .symbol_id = root.symbol_id},
              .target_profile = target_profile,
          };
    }
  } else {
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &module->symbols.entries[i];
      const loom_func_like_t function =
          loom_func_like_const_cast(module, symbol->defining_op);
      if (loom_func_like_body(function) == NULL ||
          (loom_func_like_is_module_internal(function) &&
           !iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_RETAIN))) {
        continue;
      }
      specializations[specialization_count++] =
          (loom_target_specialization_request_t){
              .function_ref = {.module_id = 0, .symbol_id = i},
              .target_profile = target_profile,
          };
    }
  }
  *out_specializations = (loom_target_specialization_request_list_t){
      .values = specializations,
      .count = specialization_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_compile_request_materialize_roots(
    const loom_compile_request_t* request,
    const loom_target_entry_options_t* entry_options,
    loom_linker_target_symbol_list_t root_targets,
    const loom_module_t* source_module, loom_source_table_projection_t* sources,
    iree_arena_block_pool_t* block_pool, loom_module_t** out_module,
    uint32_t* out_error_count) {
  *out_module = NULL;
  const loom_module_t* const source_modules[] = {source_module};
  iree_string_view_t module_name = iree_string_view_empty();
  if (source_module->name_id < source_module->strings.count) {
    module_name =
        loom_string_table_get(&source_module->strings, source_module->name_id);
  }
  loom_target_entry_diagnostic_emitter_t emitter;
  loom_target_entry_diagnostic_emitter_initialize(source_module, entry_options,
                                                  LOOM_EMITTER_PASS, &emitter);
  const loom_source_table_resolver_t input_sources = sources->table;
  iree_status_t status = loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = module_name,
          .root_symbols = request->selection.roots,
          .root_target_symbols = root_targets,
          .diagnostic_emitter = loom_target_entry_emitter(&emitter),
          .source_callback = {.fn = loom_source_table_project,
                              .user_data = sources},
      },
      block_pool, source_module->host_allocator, out_module);
  *out_error_count = emitter.error_count;
  if (!iree_status_is_ok(status) || *out_module == NULL) {
    sources->table = input_sources;
  }
  return status;
}

static void loom_compile_request_publish_module_roots(
    loom_module_t* module, loom_linker_target_symbol_list_t root_targets) {
  for (iree_host_size_t i = 0; i < root_targets.count; ++i) {
    const loom_symbol_ref_t root = root_targets.values[i];
    IREE_ASSERT_EQ(root.module_id, 0u);
    IREE_ASSERT_LT(root.symbol_id, module->symbols.count);
    loom_symbol_t* symbol = &module->symbols.entries[root.symbol_id];
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    IREE_ASSERT(loom_func_like_isa(function));
    loom_op_attrs(function.op)[function.vtable->visibility_attr_index] =
        loom_attr_enum(/*public=*/1);
    symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
  }
}

iree_status_t loom_compile_request_materialize(
    const loom_compile_request_t* request,
    const loom_target_environment_t* target_environment,
    const loom_target_entry_options_t* entry_options,
    const loom_module_t* source_module,
    loom_compile_request_source_ownership_t source_ownership,
    loom_source_table_projection_t* sources, iree_arena_allocator_t* arena,
    iree_arena_block_pool_t* block_pool, loom_module_t** out_module,
    loom_target_specialization_request_list_t* out_target_specializations,
    uint32_t* out_error_count) {
  *out_module = source_ownership == LOOM_COMPILE_REQUEST_SOURCE_TRANSFERRED
                    ? (loom_module_t*)source_module
                    : NULL;
  *out_target_specializations = (loom_target_specialization_request_list_t){0};
  *out_error_count = 0;

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  const bool requires_root_targets =
      request->target_profile != NULL ||
      request->selection.kind == LOOM_COMPILE_ENTRY_KIND_MODULE;
  const iree_host_size_t root_target_count =
      requires_root_targets ? request->selection.roots.count : 0;
  loom_symbol_ref_t* root_target_values = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, root_target_count, sizeof(*root_target_values),
      (void**)&root_target_values);
  loom_linker_target_symbol_list_t root_targets = {
      .count = root_target_count,
      .values = root_target_values,
  };
  const bool must_link =
      source_ownership == LOOM_COMPILE_REQUEST_SOURCE_BORROWED ||
      request->selection.roots.count != 0;
  if (iree_status_is_ok(status) && must_link) {
    loom_module_t* materialized_module = NULL;
    status = loom_compile_request_materialize_roots(
        request, entry_options, root_targets, source_module, sources,
        block_pool, &materialized_module, out_error_count);
    if (iree_status_is_ok(status)) {
      if (source_ownership == LOOM_COMPILE_REQUEST_SOURCE_TRANSFERRED) {
        loom_module_free(*out_module);
      }
      *out_module = materialized_module;
    }
  }
  if (iree_status_is_ok(status) && *out_error_count == 0 &&
      root_targets.count != 0 &&
      request->selection.kind == LOOM_COMPILE_ENTRY_KIND_MODULE) {
    loom_compile_request_publish_module_roots(*out_module, root_targets);
  }

  loom_target_specialization_request_list_t specializations = {0};
  if (iree_status_is_ok(status) && *out_error_count == 0 &&
      request->target_profile != NULL) {
    iree_arena_allocator_t* specialization_arena =
        request->selection.kind == LOOM_COMPILE_ENTRY_KIND_KERNEL
            ? &scratch_arena
            : arena;
    status = loom_compile_request_build_specializations(
        *out_module, root_targets, request->target_profile,
        specialization_arena, &specializations);
  }
  if (iree_status_is_ok(status) && *out_error_count == 0 &&
      request->selection.kind == LOOM_COMPILE_ENTRY_KIND_KERNEL &&
      request->target_profile != NULL) {
    loom_target_entry_diagnostic_emitter_t diagnostic_emitter;
    loom_target_entry_diagnostic_emitter_initialize(
        *out_module, entry_options, LOOM_EMITTER_PASS, &diagnostic_emitter);
    status = loom_target_specialize_module(
        target_environment, specializations,
        (loom_target_declaration_binding_list_t){0},
        loom_target_entry_emitter(&diagnostic_emitter), block_pool,
        (*out_module)->host_allocator, out_module, out_error_count);
    sources->table.module = *out_module;
  } else if (iree_status_is_ok(status) && *out_error_count == 0 &&
             request->selection.kind == LOOM_COMPILE_ENTRY_KIND_MODULE) {
    *out_target_specializations = specializations;
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
