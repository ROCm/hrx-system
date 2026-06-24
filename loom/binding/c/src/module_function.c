// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "diagnostic.h"
#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"

typedef struct loomc_module_function_resolved_query_options_t {
  // Optional symbol selector after stripping a leading '@'.
  loomc_string_view_t function_symbol;

  // Requested function kind.
  loomc_module_function_kind_t kind;
} loomc_module_function_resolved_query_options_t;

static bool loomc_module_function_string_view_is_valid(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_status_t loomc_module_function_validate_string_view(
    loomc_string_view_t value) {
  if (!loomc_module_function_string_view_is_valid(value)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_string_view_t loomc_module_function_normalize_symbol(
    loomc_string_view_t symbol) {
  if (symbol.size != 0 && symbol.data[0] == '@') {
    return loomc_make_string_view(symbol.data + 1, symbol.size - 1);
  }
  return symbol;
}

static bool loomc_module_function_kind_is_valid(
    loomc_module_function_kind_t kind) {
  switch (kind) {
    case LOOMC_MODULE_FUNCTION_KIND_UNKNOWN:
    case LOOMC_MODULE_FUNCTION_KIND_FUNCTION:
    case LOOMC_MODULE_FUNCTION_KIND_KERNEL:
    case LOOMC_MODULE_FUNCTION_KIND_TARGET_FUNCTION:
    case LOOMC_MODULE_FUNCTION_KIND_TARGET_KERNEL:
      return true;
  }
  return false;
}

static loomc_status_t loomc_module_function_resolve_query_options(
    const loomc_module_function_query_options_t* options,
    loomc_module_function_resolved_query_options_t* out_options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module function query options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module function query options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "module function query option extensions are not supported");
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_module_function_validate_string_view(options->function_symbol));
    if (!loomc_module_function_kind_is_valid(options->kind)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "module function query kind is not supported");
    }
  }

  out_options->function_symbol =
      options ? loomc_module_function_normalize_symbol(options->function_symbol)
              : loomc_string_view_empty();
  out_options->kind =
      options ? options->kind : LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  return loomc_ok_status();
}

static bool loomc_module_function_kind_matches(
    const loomc_module_function_resolved_query_options_t* options,
    loomc_module_function_kind_t kind) {
  return options->kind == LOOMC_MODULE_FUNCTION_KIND_UNKNOWN ||
         options->kind == kind;
}

static bool loomc_module_function_symbol_kind(
    const loom_symbol_t* symbol, loomc_module_function_kind_t* out_kind) {
  *out_kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  const loom_op_t* op = symbol ? symbol->defining_op : NULL;
  if (loom_kernel_def_isa(op)) {
    *out_kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL;
    return true;
  }
  if (loom_func_def_isa(op)) {
    *out_kind = LOOMC_MODULE_FUNCTION_KIND_FUNCTION;
    return true;
  }
  if (loom_low_kernel_def_isa(op)) {
    *out_kind = LOOMC_MODULE_FUNCTION_KIND_TARGET_KERNEL;
    return true;
  }
  if (loom_low_func_def_isa(op)) {
    *out_kind = LOOMC_MODULE_FUNCTION_KIND_TARGET_FUNCTION;
    return true;
  }
  return false;
}

static iree_string_view_t loomc_module_function_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (module == NULL || symbol == NULL ||
      symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[symbol->name_id];
}

static bool loomc_module_function_symbol_has_export_info(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return false;
  }
  return loom_func_like_export_symbol(func) != LOOM_STRING_ID_INVALID;
}

static bool loomc_module_function_symbol_is_public(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return true;
  }
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  return loom_func_like_isa(func) && loom_func_like_visibility(func) != 0;
}

static bool loomc_module_function_populate_from_symbol(
    const loom_module_t* module, loomc_host_size_t function_ordinal,
    const loom_symbol_t* symbol, loomc_module_function_t* out_function) {
  *out_function = (loomc_module_function_t){0};
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_symbol_kind(symbol, &kind)) {
    return false;
  }
  iree_string_view_t symbol_name =
      loomc_module_function_symbol_name(module, symbol);
  if (iree_string_view_is_empty(symbol_name)) {
    return false;
  }

  out_function->function_ordinal = function_ordinal;
  out_function->symbol_name = loomc_string_view_from_iree(symbol_name);
  out_function->kind = kind;
  if (loomc_module_function_symbol_is_public(module, symbol)) {
    out_function->flags |= LOOMC_MODULE_FUNCTION_FLAG_PUBLIC;
  }
  if (loomc_module_function_symbol_has_export_info(module, symbol)) {
    out_function->flags |= LOOMC_MODULE_FUNCTION_FLAG_HAS_EXPORT_INFO;
  }
  return true;
}

static bool loomc_module_function_try_resolve_at(
    const loom_module_t* module, loomc_host_size_t function_ordinal,
    const loom_symbol_t** out_symbol, loomc_module_function_kind_t* out_kind) {
  *out_symbol = NULL;
  *out_kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (module == NULL) {
    return false;
  }

  loomc_host_size_t current_function_ordinal = 0;
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loomc_module_function_symbol_kind(symbol, &kind)) {
      continue;
    }
    if (current_function_ordinal == function_ordinal) {
      *out_symbol = symbol;
      *out_kind = kind;
      return true;
    }
    ++current_function_ordinal;
  }
  return false;
}

static bool loomc_module_function_symbol_function_ordinal(
    const loom_module_t* module, const loom_symbol_t* expected_symbol,
    loomc_host_size_t* out_function_ordinal) {
  *out_function_ordinal = 0;
  loomc_host_size_t current_function_ordinal = 0;
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loomc_module_function_symbol_kind(symbol, &kind)) {
      continue;
    }
    if (symbol == expected_symbol) {
      *out_function_ordinal = current_function_ordinal;
      return true;
    }
    ++current_function_ordinal;
  }
  return false;
}

static bool loomc_module_function_symbol_name_equal(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loomc_string_view_t symbol_name) {
  return loomc_string_view_equal(
      loomc_string_view_from_iree(
          loomc_module_function_symbol_name(module, symbol)),
      symbol_name);
}

static const loom_symbol_t* loomc_module_function_find_named_symbol(
    const loom_module_t* module, loomc_string_view_t symbol_name) {
  loom_string_id_t name_id = loom_module_lookup_string(
      module, iree_string_view_from_loomc(symbol_name));
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return NULL;
  }
  return &module->symbols.entries[symbol_id];
}

static loomc_status_t loomc_module_function_fail_result(
    loomc_result_t* result, loomc_string_view_t code, loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status);
}

static loomc_status_t loomc_module_function_fail_not_found_result(
    loomc_result_t* result) {
  return loomc_module_function_fail_result(
      result, loomc_make_cstring_view("MODULE_FUNCTION/NOT_FOUND"),
      loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                        "module function symbol was not found"));
}

static loomc_host_size_t loomc_module_function_count_matching_symbols(
    const loom_module_t* module,
    const loomc_module_function_resolved_query_options_t* options) {
  loomc_host_size_t count = 0;
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
    if (loomc_module_function_symbol_kind(&module->symbols.entries[i], &kind) &&
        loomc_module_function_kind_matches(options, kind)) {
      ++count;
    }
  }
  return count;
}

static loomc_host_size_t loomc_module_function_fill_matching_symbols(
    const loom_module_t* module,
    const loomc_module_function_resolved_query_options_t* options,
    loomc_host_size_t function_capacity,
    loomc_module_function_t* out_functions) {
  loomc_host_size_t written_count = 0;
  loomc_host_size_t function_ordinal = 0;
  for (loomc_host_size_t i = 0;
       i < module->symbols.count && written_count < function_capacity; ++i) {
    loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loomc_module_function_symbol_kind(symbol, &kind)) {
      continue;
    }
    if (!loomc_module_function_kind_matches(options, kind)) {
      ++function_ordinal;
      continue;
    }
    if (loomc_module_function_populate_from_symbol(
            module, function_ordinal, symbol, &out_functions[written_count])) {
      ++written_count;
    }
    ++function_ordinal;
  }
  return written_count;
}

static loomc_status_t loomc_module_function_query_named(
    const loom_module_t* module,
    const loomc_module_function_resolved_query_options_t* options,
    loomc_host_size_t function_capacity, loomc_module_function_t* out_functions,
    loomc_host_size_t* out_function_count, loomc_result_t* result) {
  const loom_symbol_t* symbol =
      loomc_module_function_find_named_symbol(module, options->function_symbol);
  if (symbol == NULL) {
    *out_function_count = 0;
    return loomc_module_function_fail_not_found_result(result);
  }

  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_symbol_kind(symbol, &kind)) {
    *out_function_count = 0;
    return loomc_module_function_fail_not_found_result(result);
  }
  if (!loomc_module_function_kind_matches(options, kind)) {
    *out_function_count = 0;
    return loomc_module_function_fail_not_found_result(result);
  }

  *out_function_count = 1;
  if (function_capacity != 0) {
    loomc_host_size_t function_ordinal = 0;
    if (!loomc_module_function_symbol_function_ordinal(module, symbol,
                                                       &function_ordinal)) {
      *out_function_count = 0;
      return loomc_module_function_fail_not_found_result(result);
    }
    (void)loomc_module_function_populate_from_symbol(module, function_ordinal,
                                                     symbol, &out_functions[0]);
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_module_function_query_all(
    const loom_module_t* module,
    const loomc_module_function_resolved_query_options_t* options,
    loomc_host_size_t function_capacity, loomc_module_function_t* out_functions,
    loomc_host_size_t* out_function_count) {
  *out_function_count =
      loomc_module_function_count_matching_symbols(module, options);
  if (function_capacity != 0) {
    (void)loomc_module_function_fill_matching_symbols(
        module, options, function_capacity, out_functions);
  }
  return loomc_ok_status();
}

static bool loomc_module_function_try_resolve(
    const loomc_module_t* module, const loomc_module_function_t* function,
    const loom_module_t** out_internal_module, const loom_symbol_t** out_symbol,
    loomc_module_function_kind_t* out_kind) {
  *out_internal_module = NULL;
  *out_symbol = NULL;
  *out_kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL || function == NULL) {
    return false;
  }

  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_try_resolve_at(
          internal_module, function->function_ordinal, &symbol, &kind) ||
      kind != function->kind ||
      !loomc_module_function_string_view_is_valid(function->symbol_name) ||
      !loomc_module_function_symbol_name_equal(internal_module, symbol,
                                               function->symbol_name)) {
    return false;
  }
  *out_internal_module = internal_module;
  *out_symbol = symbol;
  *out_kind = kind;
  return true;
}

static loomc_status_t loomc_module_function_resolve(
    const loomc_module_t* module, const loomc_module_function_t* function,
    const loom_module_t** out_internal_module, const loom_symbol_t** out_symbol,
    loomc_module_function_kind_t* out_kind) {
  if (module == NULL || function == NULL || out_internal_module == NULL ||
      out_symbol == NULL || out_kind == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module, function, and output parameters must not be NULL");
  }
  if (!loomc_module_function_try_resolve(module, function, out_internal_module,
                                         out_symbol, out_kind)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module function metadata is stale or does not belong to module");
  }
  return loomc_ok_status();
}

static bool loomc_module_function_populate_export_info(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loomc_module_function_export_info_t* out_info) {
  *out_info = (loomc_module_function_export_info_t){0};
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return false;
  }

  loom_string_id_t export_symbol_id = loom_func_like_export_symbol(func);
  if (export_symbol_id != LOOM_STRING_ID_INVALID) {
    if (export_symbol_id >= module->strings.count) {
      return false;
    }
    out_info->export_symbol =
        loomc_string_view_from_iree(module->strings.entries[export_symbol_id]);
    out_info->flags |= LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL;
  }
  return out_info->flags != 0;
}

loomc_status_t loomc_module_query_functions(
    const loomc_module_t* module,
    const loomc_module_function_query_options_t* options,
    loomc_allocator_t allocator, loomc_host_size_t function_capacity,
    loomc_module_function_t* out_functions,
    loomc_host_size_t* out_function_count, loomc_result_t** out_result) {
  if (module == NULL || out_function_count == NULL || out_result == NULL ||
      (function_capacity != 0 && out_functions == NULL)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module, output parameters, and function storage must be valid");
  }
  *out_function_count = 0;
  *out_result = NULL;
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }

  loomc_module_function_resolved_query_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_module_function_resolve_query_options(options, &resolved_options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_status_t status = loomc_ok_status();
  if (loomc_string_view_is_empty(resolved_options.function_symbol)) {
    status = loomc_module_function_query_all(internal_module, &resolved_options,
                                             function_capacity, out_functions,
                                             out_function_count);
  } else {
    status = loomc_module_function_query_named(
        internal_module, &resolved_options, function_capacity, out_functions,
        out_function_count, result);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

bool loomc_module_try_get_function_at(const loomc_module_t* module,
                                      loomc_host_size_t function_ordinal,
                                      loomc_module_function_t* out_function) {
  if (out_function == NULL) {
    return false;
  }
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_try_resolve_at(internal_module, function_ordinal,
                                            &symbol, &kind)) {
    *out_function = (loomc_module_function_t){0};
    return false;
  }
  return loomc_module_function_populate_from_symbol(
      internal_module, function_ordinal, symbol, out_function);
}

loomc_status_t loomc_module_get_function_at(
    const loomc_module_t* module, loomc_host_size_t function_ordinal,
    loomc_module_function_t* out_function) {
  if (module == NULL || out_function == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_function must not be NULL");
  }
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_try_resolve_at(internal_module, function_ordinal,
                                            &symbol, &kind) ||
      !loomc_module_function_populate_from_symbol(
          internal_module, function_ordinal, symbol, out_function)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module function ordinal was not found");
  }
  return loomc_ok_status();
}

bool loomc_module_try_lookup_function(const loomc_module_t* module,
                                      loomc_string_view_t symbol_name,
                                      loomc_module_function_t* out_function) {
  if (out_function == NULL ||
      !loomc_module_function_string_view_is_valid(symbol_name)) {
    return false;
  }
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return false;
  }
  symbol_name = loomc_module_function_normalize_symbol(symbol_name);
  const loom_symbol_t* symbol =
      loomc_module_function_find_named_symbol(internal_module, symbol_name);
  if (symbol == NULL) {
    return false;
  }
  loomc_host_size_t function_ordinal = 0;
  if (!loomc_module_function_symbol_function_ordinal(internal_module, symbol,
                                                     &function_ordinal)) {
    return false;
  }
  return loomc_module_function_populate_from_symbol(
      internal_module, function_ordinal, symbol, out_function);
}

loomc_status_t loomc_module_lookup_function(
    const loomc_module_t* module, loomc_string_view_t symbol_name,
    loomc_module_function_t* out_function) {
  if (module == NULL || out_function == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_function must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_module_function_validate_string_view(symbol_name));
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  symbol_name = loomc_module_function_normalize_symbol(symbol_name);
  const loom_symbol_t* symbol =
      loomc_module_function_find_named_symbol(internal_module, symbol_name);
  if (symbol == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module function symbol was not found");
  }
  loomc_host_size_t function_ordinal = 0;
  if (!loomc_module_function_symbol_function_ordinal(internal_module, symbol,
                                                     &function_ordinal) ||
      !loomc_module_function_populate_from_symbol(
          internal_module, function_ordinal, symbol, out_function)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module function symbol was not found");
  }
  return loomc_ok_status();
}

bool loomc_module_function_try_get_export_info_at(
    const loomc_module_t* module, loomc_host_size_t function_ordinal,
    loomc_module_function_export_info_t* out_info) {
  if (out_info == NULL) {
    return false;
  }
  *out_info = (loomc_module_function_export_info_t){0};
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_try_resolve_at(internal_module, function_ordinal,
                                            &symbol, &kind)) {
    return false;
  }
  return loomc_module_function_populate_export_info(internal_module, symbol,
                                                    out_info);
}

loomc_status_t loomc_module_function_get_export_info_at(
    const loomc_module_t* module, loomc_host_size_t function_ordinal,
    loomc_module_function_export_info_t* out_info) {
  if (module == NULL || out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_info must not be NULL");
  }
  *out_info = (loomc_module_function_export_info_t){0};
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_try_resolve_at(internal_module, function_ordinal,
                                            &symbol, &kind)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module function ordinal was not found");
  }
  if (!loomc_module_function_populate_export_info(internal_module, symbol,
                                                  out_info)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module function has no export metadata");
  }
  return loomc_ok_status();
}

bool loomc_module_function_try_get_export_info(
    const loomc_module_t* module, const loomc_module_function_t* function,
    loomc_module_function_export_info_t* out_info) {
  if (out_info == NULL) {
    return false;
  }
  *out_info = (loomc_module_function_export_info_t){0};
  const loom_module_t* internal_module = NULL;
  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  if (!loomc_module_function_try_resolve(module, function, &internal_module,
                                         &symbol, &kind)) {
    return false;
  }
  return loomc_module_function_populate_export_info(internal_module, symbol,
                                                    out_info);
}

loomc_status_t loomc_module_function_get_export_info(
    const loomc_module_t* module, const loomc_module_function_t* function,
    loomc_module_function_export_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  const loom_module_t* internal_module = NULL;
  const loom_symbol_t* symbol = NULL;
  loomc_module_function_kind_t kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_module_function_resolve(
      module, function, &internal_module, &symbol, &kind));
  if (!loomc_module_function_populate_export_info(internal_module, symbol,
                                                  out_info)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module function has no export metadata");
  }
  return loomc_ok_status();
}
