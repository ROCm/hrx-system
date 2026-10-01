// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/compile/request.h"

#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/target/entry_selection.h"
#include "loom/target/projection.h"

iree_string_view_t loom_compile_product_name(loom_compile_product_t product) {
  switch (product) {
    case LOOM_COMPILE_PRODUCT_KERNEL:
      return IREE_SV("kernel");
    case LOOM_COMPILE_PRODUCT_COMMAND:
      return IREE_SV("command");
    case LOOM_COMPILE_PRODUCT_MODULE:
      return IREE_SV("module");
    case LOOM_COMPILE_PRODUCT_INVALID:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

static iree_status_t loom_compile_product_selection_lookup_root(
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

static bool loom_compile_product_selection_is_array_program(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  const loom_func_like_t function =
      loom_func_like_const_cast(module, symbol->defining_op);
  return loom_func_like_isa(function) &&
         loom_func_like_abi(function) == LOOM_TARGET_ABI_ARRAY_PROGRAM;
}

static iree_status_t loom_compile_product_selection_classify_symbol(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_product_t* out_product) {
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    const loom_symbol_product_carrier_t carrier =
        loom_symbol_definition_product_carrier(symbol->definition,
                                               symbol->defining_op);
    switch (carrier) {
      case LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED:
      case 0:
        *out_product = LOOM_COMPILE_PRODUCT_MODULE;
        return iree_ok_status();
      case LOOM_PIPELINE_DEF_SCOPE_KERNEL:
        *out_product = LOOM_COMPILE_PRODUCT_KERNEL;
        return iree_ok_status();
      default: {
        const iree_string_view_t symbol_name =
            loom_string_table_get(&module->strings, symbol->name_id);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pipeline root '@%.*s' has unsupported product scope %u",
            (int)symbol_name.size, symbol_name.data, (unsigned)carrier);
      }
    }
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM)) {
    *out_product = LOOM_COMPILE_PRODUCT_COMMAND;
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL) ||
      loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY) ||
      loom_compile_product_selection_is_array_program(module, symbol)) {
    *out_product = LOOM_COMPILE_PRODUCT_KERNEL;
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    *out_product = LOOM_COMPILE_PRODUCT_MODULE;
    return iree_ok_status();
  }
  const iree_string_view_t symbol_name =
      loom_string_table_get(&module->strings, symbol->name_id);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "root symbol '@%.*s' does not define a compilable product",
      (int)symbol_name.size, symbol_name.data);
}

static iree_status_t loom_compile_product_selection_kernel_target_type(
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

static iree_status_t loom_compile_product_selection_merge_kernel_target(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_product_selection_t* selection) {
  const loom_target_fact_type_t* fact_type = NULL;
  IREE_RETURN_IF_ERROR(loom_compile_product_selection_kernel_target_type(
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

static iree_status_t loom_compile_product_selection_merge_root(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_product_selection_t* selection) {
  loom_compile_product_t product = LOOM_COMPILE_PRODUCT_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_compile_product_selection_classify_symbol(module, symbol, &product));
  if (selection->roots.count != 0 && selection->product != product) {
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected roots mix '%.*s' and '%.*s' products at '@%.*s'",
        (int)loom_compile_product_name(selection->product).size,
        loom_compile_product_name(selection->product).data,
        (int)loom_compile_product_name(product).size,
        loom_compile_product_name(product).data, (int)symbol_name.size,
        symbol_name.data);
  }
  selection->product = product;
  ++selection->roots.count;
  return product == LOOM_COMPILE_PRODUCT_KERNEL
             ? loom_compile_product_selection_merge_kernel_target(
                   module, symbol, selection)
             : iree_ok_status();
}

static loom_compile_product_t loom_compile_product_selection_default_product(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol->defining_op == NULL ||
      loom_symbol_definition_is_declaration(symbol->definition)) {
    return LOOM_COMPILE_PRODUCT_INVALID;
  }
  const bool is_public = iree_any_bit_set(
      symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN);
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM)) {
    return is_public ? LOOM_COMPILE_PRODUCT_COMMAND
                     : LOOM_COMPILE_PRODUCT_INVALID;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    const loom_symbol_product_carrier_t carrier =
        loom_symbol_definition_product_carrier(symbol->definition,
                                               symbol->defining_op);
    if (carrier == LOOM_PIPELINE_DEF_SCOPE_KERNEL) {
      return is_public ? LOOM_COMPILE_PRODUCT_KERNEL
                       : LOOM_COMPILE_PRODUCT_INVALID;
    }
    return is_public &&
                   (carrier == LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED ||
                    carrier == 0) &&
                   loom_symbol_implements(symbol,
                                          LOOM_SYMBOL_INTERFACE_FUNC_LIKE)
               ? LOOM_COMPILE_PRODUCT_MODULE
               : LOOM_COMPILE_PRODUCT_INVALID;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY)) {
    return LOOM_COMPILE_PRODUCT_KERNEL;
  }
  if (loom_compile_product_selection_is_array_program(module, symbol)) {
    return is_public ? LOOM_COMPILE_PRODUCT_KERNEL
                     : LOOM_COMPILE_PRODUCT_INVALID;
  }
  if (is_public &&
      loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) &&
      !loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL)) {
    return LOOM_COMPILE_PRODUCT_MODULE;
  }
  return LOOM_COMPILE_PRODUCT_INVALID;
}

static bool loom_compile_product_selection_name_equal(iree_string_view_t lhs,
                                                      iree_string_view_t rhs) {
  return iree_string_view_equal(loom_target_entry_normalize_symbol_name(lhs),
                                loom_target_entry_normalize_symbol_name(rhs));
}

static bool loom_compile_product_selection_list_contains(
    iree_string_view_list_t roots, iree_string_view_t root_name) {
  for (iree_host_size_t i = 0; i < roots.count; ++i) {
    if (loom_compile_product_selection_name_equal(roots.values[i], root_name)) {
      return true;
    }
  }
  return false;
}

typedef struct loom_compile_default_root_summary_t {
  iree_host_size_t count;
  iree_host_size_t name_bytes;
} loom_compile_default_root_summary_t;

static iree_status_t loom_compile_product_selection_summarize_defaults(
    const loom_module_t* module,
    loom_compile_default_root_summary_t summaries[4]) {
  memset(summaries, 0, sizeof(*summaries) * 4);
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const loom_compile_product_t product =
        loom_compile_product_selection_default_product(module, symbol);
    if (product == LOOM_COMPILE_PRODUCT_INVALID) {
      continue;
    }
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    loom_compile_default_root_summary_t* summary = &summaries[product];
    if (!iree_host_size_checked_add(summary->name_bytes, symbol_name.size,
                                    &summary->name_bytes)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "compile root names are too large");
    }
    ++summary->count;
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_product_selection_apply_exclusions(
    const loom_module_t* module, iree_string_view_list_t excluded_roots,
    loom_compile_product_t selected_product,
    loom_compile_default_root_summary_t* selected_summary) {
  for (iree_host_size_t i = 0; i < excluded_roots.count; ++i) {
    const iree_string_view_t excluded_name = excluded_roots.values[i];
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (loom_compile_product_selection_name_equal(excluded_name,
                                                    excluded_roots.values[j])) {
        const iree_string_view_t normalized_name =
            loom_target_entry_normalize_symbol_name(excluded_name);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "excluded root '@%.*s' is repeated",
            (int)normalized_name.size, normalized_name.data);
      }
    }

    const loom_symbol_t* symbol = NULL;
    IREE_RETURN_IF_ERROR(loom_compile_product_selection_lookup_root(
        module, excluded_name, &symbol));
    loom_compile_product_t product = LOOM_COMPILE_PRODUCT_INVALID;
    IREE_RETURN_IF_ERROR(loom_compile_product_selection_classify_symbol(
        module, symbol, &product));
    const iree_string_view_t normalized_name =
        loom_target_entry_normalize_symbol_name(excluded_name);
    if (product != selected_product) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "excluded root '@%.*s' has product '%.*s', expected '%.*s'",
          (int)normalized_name.size, normalized_name.data,
          (int)loom_compile_product_name(product).size,
          loom_compile_product_name(product).data,
          (int)loom_compile_product_name(selected_product).size,
          loom_compile_product_name(selected_product).data);
    }
    if (loom_compile_product_selection_default_product(module, symbol) !=
        product) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "excluded root '@%.*s' is not selected by product '%.*s'",
          (int)normalized_name.size, normalized_name.data,
          (int)loom_compile_product_name(product).size,
          loom_compile_product_name(product).data);
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

static iree_status_t loom_compile_product_selection_require_default_roots(
    loom_compile_product_t product, iree_host_size_t root_count,
    bool has_exclusions) {
  if (root_count != 0) {
    return iree_ok_status();
  }
  if (has_exclusions) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "excluded roots empty the default %.*s root set",
                            (int)loom_compile_product_name(product).size,
                            loom_compile_product_name(product).data);
  }
  switch (product) {
    case LOOM_COMPILE_PRODUCT_COMMAND:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product 'command' requires a nonempty set of public or retained "
          "command-program roots");
    case LOOM_COMPILE_PRODUCT_KERNEL:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product 'kernel' requires a nonempty root set of kernel entries, "
          "public or retained kernel-scoped pipelines, or array programs");
    case LOOM_COMPILE_PRODUCT_MODULE:
    case LOOM_COMPILE_PRODUCT_INVALID:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "compile product selection is invalid");
}

static iree_status_t loom_compile_product_selection_copy_default_roots(
    const loom_module_t* module, loom_compile_product_t product,
    iree_string_view_list_t excluded_roots,
    loom_compile_default_root_summary_t root_summary,
    iree_arena_allocator_t* arena,
    loom_compile_product_selection_t* out_selection) {
  iree_string_view_t* root_names = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, root_summary.count, sizeof(*root_names), (void**)&root_names));
  char* root_name_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, root_summary.name_bytes,
                                           (void**)&root_name_storage));

  loom_compile_product_selection_t selection = {
      .product = product,
      .roots = {.values = root_names},
  };
  char* next_root_name = root_name_storage;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (loom_compile_product_selection_default_product(module, symbol) !=
        product) {
      continue;
    }
    const iree_string_view_t symbol_name =
        loom_string_table_get(&module->strings, symbol->name_id);
    if (loom_compile_product_selection_list_contains(excluded_roots,
                                                     symbol_name)) {
      continue;
    }
    memcpy(next_root_name, symbol_name.data, symbol_name.size);
    root_names[selection.roots.count] =
        iree_make_string_view(next_root_name, symbol_name.size);
    next_root_name += symbol_name.size;
    ++selection.roots.count;
    if (product == LOOM_COMPILE_PRODUCT_KERNEL) {
      IREE_RETURN_IF_ERROR(loom_compile_product_selection_merge_kernel_target(
          module, symbol, &selection));
    }
  }
  IREE_ASSERT_EQ(selection.roots.count, root_summary.count);
  IREE_ASSERT_EQ(next_root_name, root_name_storage + root_summary.name_bytes);
  *out_selection = selection;
  return iree_ok_status();
}

static iree_status_t loom_compile_product_selection_resolve(
    const loom_module_t* module, iree_string_view_list_t explicit_roots,
    iree_string_view_list_t excluded_roots,
    loom_compile_product_t product_constraint, iree_arena_allocator_t* arena,
    loom_compile_product_selection_t* out_selection) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_selection);
  IREE_ASSERT(product_constraint >= LOOM_COMPILE_PRODUCT_INVALID &&
              product_constraint <= LOOM_COMPILE_PRODUCT_MODULE);
  *out_selection = (loom_compile_product_selection_t){0};
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
    loom_compile_product_selection_t selection = {
        .roots = {.values = explicit_roots.values},
    };
    for (iree_host_size_t i = 0; i < explicit_roots.count; ++i) {
      const loom_symbol_t* symbol = NULL;
      IREE_RETURN_IF_ERROR(loom_compile_product_selection_lookup_root(
          module, explicit_roots.values[i], &symbol));
      IREE_RETURN_IF_ERROR(loom_compile_product_selection_merge_root(
          module, symbol, &selection));
    }
    if (product_constraint != LOOM_COMPILE_PRODUCT_INVALID &&
        product_constraint != selection.product) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected roots infer product '%.*s'; product constraint '%.*s' "
          "cannot reinterpret them",
          (int)loom_compile_product_name(selection.product).size,
          loom_compile_product_name(selection.product).data,
          (int)loom_compile_product_name(product_constraint).size,
          loom_compile_product_name(product_constraint).data);
    }
    *out_selection = selection;
    return iree_ok_status();
  }

  loom_compile_default_root_summary_t summaries[4];
  IREE_RETURN_IF_ERROR(
      loom_compile_product_selection_summarize_defaults(module, summaries));
  loom_compile_product_t selected_product = product_constraint;
  if (selected_product == LOOM_COMPILE_PRODUCT_INVALID) {
    selected_product = summaries[LOOM_COMPILE_PRODUCT_COMMAND].count != 0
                           ? LOOM_COMPILE_PRODUCT_COMMAND
                       : summaries[LOOM_COMPILE_PRODUCT_KERNEL].count != 0
                           ? LOOM_COMPILE_PRODUCT_KERNEL
                           : LOOM_COMPILE_PRODUCT_MODULE;
  }
  loom_compile_default_root_summary_t selected_summary =
      summaries[selected_product];
  IREE_RETURN_IF_ERROR(loom_compile_product_selection_apply_exclusions(
      module, excluded_roots, selected_product, &selected_summary));

  if (selected_product == LOOM_COMPILE_PRODUCT_MODULE &&
      excluded_roots.count == 0) {
    out_selection->product = LOOM_COMPILE_PRODUCT_MODULE;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_compile_product_selection_require_default_roots(
      selected_product, selected_summary.count, excluded_roots.count != 0));
  return loom_compile_product_selection_copy_default_roots(
      module, selected_product, excluded_roots, selected_summary, arena,
      out_selection);
}

static iree_status_t loom_compile_request_select_explicit_target(
    iree_string_view_t target_value,
    const loom_target_environment_t* target_environment,
    loom_compile_target_selection_t* out_target) {
  *out_target = (loom_compile_target_selection_t){0};
  target_value = iree_string_view_trim(target_value);
  if (iree_string_view_is_empty(target_value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_specification_parse(
      target_value, &out_target->specification));
  IREE_RETURN_IF_ERROR(loom_target_environment_select_profile(
      target_environment, &out_target->specification, &out_target->profile));
  return iree_ok_status();
}

static iree_status_t loom_compile_request_parse_product(
    iree_string_view_t value, loom_compile_product_t* out_product) {
  *out_product = LOOM_COMPILE_PRODUCT_INVALID;
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  for (loom_compile_product_t product = LOOM_COMPILE_PRODUCT_KERNEL;
       product <= LOOM_COMPILE_PRODUCT_MODULE; ++product) {
    if (iree_string_view_equal(value, loom_compile_product_name(product))) {
      *out_product = product;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown --product='%.*s'; expected 'kernel', "
                          "'command', or 'module'",
                          (int)value.size, value.data);
}

static iree_status_t loom_compile_request_select_named_format(
    loom_compile_product_t product, iree_string_view_t format,
    const loom_target_environment_t* target_environment,
    const loom_target_emitter_t** out_target_emitter) {
  *out_target_emitter = NULL;
  const bool is_command_format =
      iree_string_view_equal(format, IREE_SV("loom-command"));
  const loom_target_emitter_t* target_emitter =
      loom_target_environment_lookup_emitter(target_environment, format);
  if (!is_command_format && target_emitter == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format '%.*s' is not available in this binary",
                            (int)format.size, format.data);
  }
  if (is_command_format && target_emitter != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "format '%.*s' has multiple configured producers",
                            (int)format.size, format.data);
  }

  if (is_command_format) {
    if (product != LOOM_COMPILE_PRODUCT_COMMAND) {
      const iree_string_view_t product_name =
          loom_compile_product_name(product);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "format 'loom-command' cannot emit product '%.*s'",
          (int)product_name.size, product_name.data);
    }
    return iree_ok_status();
  }
  if (product == LOOM_COMPILE_PRODUCT_COMMAND) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format '%.*s' cannot emit product 'command'",
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
    loom_compile_product_t product, iree_string_view_t explicit_format,
    const loom_target_fact_type_t* target_fact_type,
    const loom_target_environment_t* target_environment,
    const loom_target_emitter_t** out_target_emitter) {
  explicit_format = iree_string_view_trim(explicit_format);
  if (!iree_string_view_is_empty(explicit_format)) {
    return loom_compile_request_select_named_format(
        product, explicit_format, target_environment, out_target_emitter);
  }
  switch (product) {
    case LOOM_COMPILE_PRODUCT_KERNEL:
      return loom_compile_request_select_canonical_kernel_emitter(
          target_fact_type, target_environment, out_target_emitter);
    case LOOM_COMPILE_PRODUCT_COMMAND:
      return iree_ok_status();
    case LOOM_COMPILE_PRODUCT_MODULE: {
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
          "module product requires --format or a --target with a canonical "
          "module format");
    }
    case LOOM_COMPILE_PRODUCT_INVALID:
      break;
  }
  IREE_ASSERT_UNREACHABLE("resolved compile product");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_compile_request_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    const loom_target_environment_t* target_environment,
    iree_arena_allocator_t* arena, loom_compile_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = (loom_compile_request_t){0};

  loom_compile_product_t product_constraint = LOOM_COMPILE_PRODUCT_INVALID;
  IREE_RETURN_IF_ERROR(loom_compile_request_parse_product(options->product,
                                                          &product_constraint));
  loom_compile_product_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_compile_product_selection_resolve(
      module, options->roots, options->excluded_roots, product_constraint,
      arena, &selection));

  loom_compile_request_t request = {
      .selection = selection,
  };
  IREE_RETURN_IF_ERROR(loom_compile_request_select_explicit_target(
      options->target, target_environment, &request.explicit_target));
  if (request.explicit_target.profile != NULL &&
      selection.product == LOOM_COMPILE_PRODUCT_COMMAND) {
    const iree_string_view_t product_name =
        loom_compile_product_name(selection.product);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--target is not valid for product '%.*s'",
                            (int)product_name.size, product_name.data);
  }
  if (request.explicit_target.profile != NULL &&
      selection.target_fact_type != NULL &&
      request.explicit_target.profile->type->fact_type !=
          selection.target_fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--target family '%.*s' cannot specialize roots authored for target "
        "family '%.*s'",
        (int)request.explicit_target.profile->type->name.size,
        request.explicit_target.profile->type->name.data,
        (int)selection.target_fact_type->name.size,
        selection.target_fact_type->name.data);
  }
  const loom_target_fact_type_t* target_fact_type =
      request.explicit_target.profile != NULL
          ? request.explicit_target.profile->type->fact_type
          : selection.target_fact_type;
  if (selection.product == LOOM_COMPILE_PRODUCT_KERNEL &&
      request.explicit_target.profile == NULL &&
      selection.untargeted_kernel_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel product requires --target when %u selected kernel root%s "
        "omit target(...) attrs",
        (unsigned)selection.untargeted_kernel_count,
        selection.untargeted_kernel_count == 1 ? "" : "s");
  }
  if (selection.product == LOOM_COMPILE_PRODUCT_KERNEL &&
      target_fact_type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product requires a target");
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
      request.selection.product, options->format, target_fact_type,
      target_environment, &request.target_emitter));
  *out_request = request;
  return iree_ok_status();
}
