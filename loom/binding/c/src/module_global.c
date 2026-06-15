// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "diagnostic.h"
#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/op_defs.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"

typedef struct loomc_module_global_resolved_query_options_t {
  // Optional symbol selector after stripping a leading '@'.
  loomc_string_view_t global_symbol;

  // Requested global kind.
  loomc_module_global_kind_t kind;
} loomc_module_global_resolved_query_options_t;

static bool loomc_module_global_string_view_is_valid(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_status_t loomc_module_global_validate_string_view(
    loomc_string_view_t value) {
  if (!loomc_module_global_string_view_is_valid(value)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_string_view_t loomc_module_global_normalize_symbol(
    loomc_string_view_t symbol) {
  if (symbol.size != 0 && symbol.data[0] == '@') {
    return loomc_make_string_view(symbol.data + 1, symbol.size - 1);
  }
  return symbol;
}

static bool loomc_module_global_kind_is_valid(loomc_module_global_kind_t kind) {
  switch (kind) {
    case LOOMC_MODULE_GLOBAL_KIND_UNKNOWN:
    case LOOMC_MODULE_GLOBAL_KIND_CONSTANT:
    case LOOMC_MODULE_GLOBAL_KIND_VARIABLE:
      return true;
  }
  return false;
}

static loomc_status_t loomc_module_global_resolve_query_options(
    const loomc_module_global_query_options_t* options,
    loomc_module_global_resolved_query_options_t* out_options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_GLOBAL_QUERY_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module global query options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module global query options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "module global query option extensions are not supported");
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_module_global_validate_string_view(options->global_symbol));
    if (!loomc_module_global_kind_is_valid(options->kind)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "module global query kind is not supported");
    }
  }

  out_options->global_symbol =
      options ? loomc_module_global_normalize_symbol(options->global_symbol)
              : loomc_string_view_empty();
  out_options->kind =
      options ? options->kind : LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  return loomc_ok_status();
}

static bool loomc_module_global_kind_matches(
    const loomc_module_global_resolved_query_options_t* options,
    loomc_module_global_kind_t kind) {
  return options->kind == LOOMC_MODULE_GLOBAL_KIND_UNKNOWN ||
         options->kind == kind;
}

static bool loomc_module_global_symbol_kind(
    const loom_symbol_t* symbol, loomc_module_global_kind_t* out_kind) {
  *out_kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  const loom_op_t* op = symbol ? symbol->defining_op : NULL;
  if (loom_global_constant_isa(op)) {
    *out_kind = LOOMC_MODULE_GLOBAL_KIND_CONSTANT;
    return true;
  }
  if (loom_global_variable_isa(op)) {
    *out_kind = LOOMC_MODULE_GLOBAL_KIND_VARIABLE;
    return true;
  }
  return false;
}

static iree_string_view_t loomc_module_global_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (module == NULL || symbol == NULL ||
      symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[symbol->name_id];
}

static bool loomc_module_global_populate_from_symbol(
    const loom_module_t* module, loomc_host_size_t global_ordinal,
    const loom_symbol_t* symbol, loomc_module_global_t* out_global) {
  *out_global = (loomc_module_global_t){0};
  loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  if (!loomc_module_global_symbol_kind(symbol, &kind)) {
    return false;
  }
  iree_string_view_t symbol_name =
      loomc_module_global_symbol_name(module, symbol);
  if (iree_string_view_is_empty(symbol_name)) {
    return false;
  }

  out_global->global_ordinal = global_ordinal;
  out_global->symbol_name = loomc_string_view_from_iree(symbol_name);
  out_global->kind = kind;
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    out_global->flags |= LOOMC_MODULE_GLOBAL_FLAG_PUBLIC;
  }
  return true;
}

static bool loomc_module_global_try_resolve_at(
    const loom_module_t* module, loomc_host_size_t global_ordinal,
    const loom_symbol_t** out_symbol, loomc_module_global_kind_t* out_kind) {
  *out_symbol = NULL;
  *out_kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  if (module == NULL) {
    return false;
  }

  loomc_host_size_t current_global_ordinal = 0;
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loomc_module_global_symbol_kind(symbol, &kind)) {
      continue;
    }
    if (current_global_ordinal == global_ordinal) {
      *out_symbol = symbol;
      *out_kind = kind;
      return true;
    }
    ++current_global_ordinal;
  }
  return false;
}

static bool loomc_module_global_symbol_global_ordinal(
    const loom_module_t* module, const loom_symbol_t* expected_symbol,
    loomc_host_size_t* out_global_ordinal) {
  *out_global_ordinal = 0;
  loomc_host_size_t current_global_ordinal = 0;
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loomc_module_global_symbol_kind(symbol, &kind)) {
      continue;
    }
    if (symbol == expected_symbol) {
      *out_global_ordinal = current_global_ordinal;
      return true;
    }
    ++current_global_ordinal;
  }
  return false;
}

static const loom_symbol_t* loomc_module_global_find_named_symbol(
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

static loomc_status_t loomc_module_global_fail_result(loomc_result_t* result,
                                                      loomc_string_view_t code,
                                                      loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status);
}

static loomc_status_t loomc_module_global_fail_not_found_result(
    loomc_result_t* result) {
  return loomc_module_global_fail_result(
      result, loomc_make_cstring_view("MODULE_GLOBAL/NOT_FOUND"),
      loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                        "module global symbol was not found"));
}

static loomc_host_size_t loomc_module_global_count_matching_symbols(
    const loom_module_t* module,
    const loomc_module_global_resolved_query_options_t* options) {
  loomc_host_size_t count = 0;
  for (loomc_host_size_t i = 0; i < module->symbols.count; ++i) {
    loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
    if (loomc_module_global_symbol_kind(&module->symbols.entries[i], &kind) &&
        loomc_module_global_kind_matches(options, kind)) {
      ++count;
    }
  }
  return count;
}

static loomc_host_size_t loomc_module_global_fill_matching_symbols(
    const loom_module_t* module,
    const loomc_module_global_resolved_query_options_t* options,
    loomc_host_size_t global_capacity, loomc_module_global_t* out_globals) {
  loomc_host_size_t written_count = 0;
  loomc_host_size_t global_ordinal = 0;
  for (loomc_host_size_t i = 0;
       i < module->symbols.count && written_count < global_capacity; ++i) {
    loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loomc_module_global_symbol_kind(symbol, &kind)) {
      continue;
    }
    if (!loomc_module_global_kind_matches(options, kind)) {
      ++global_ordinal;
      continue;
    }
    if (loomc_module_global_populate_from_symbol(module, global_ordinal, symbol,
                                                 &out_globals[written_count])) {
      ++written_count;
    }
    ++global_ordinal;
  }
  return written_count;
}

static loomc_status_t loomc_module_global_query_named(
    const loom_module_t* module,
    const loomc_module_global_resolved_query_options_t* options,
    loomc_host_size_t global_capacity, loomc_module_global_t* out_globals,
    loomc_host_size_t* out_global_count, loomc_result_t* result) {
  const loom_symbol_t* symbol =
      loomc_module_global_find_named_symbol(module, options->global_symbol);
  if (symbol == NULL) {
    *out_global_count = 0;
    return loomc_module_global_fail_not_found_result(result);
  }

  loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  if (!loomc_module_global_symbol_kind(symbol, &kind)) {
    *out_global_count = 0;
    return loomc_module_global_fail_not_found_result(result);
  }
  if (!loomc_module_global_kind_matches(options, kind)) {
    *out_global_count = 0;
    return loomc_module_global_fail_not_found_result(result);
  }

  *out_global_count = 1;
  if (global_capacity != 0) {
    loomc_host_size_t global_ordinal = 0;
    if (!loomc_module_global_symbol_global_ordinal(module, symbol,
                                                   &global_ordinal)) {
      *out_global_count = 0;
      return loomc_module_global_fail_not_found_result(result);
    }
    (void)loomc_module_global_populate_from_symbol(module, global_ordinal,
                                                   symbol, &out_globals[0]);
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_module_global_query_all(
    const loom_module_t* module,
    const loomc_module_global_resolved_query_options_t* options,
    loomc_host_size_t global_capacity, loomc_module_global_t* out_globals,
    loomc_host_size_t* out_global_count) {
  *out_global_count =
      loomc_module_global_count_matching_symbols(module, options);
  if (global_capacity != 0) {
    (void)loomc_module_global_fill_matching_symbols(
        module, options, global_capacity, out_globals);
  }
  return loomc_ok_status();
}

loomc_status_t loomc_module_query_globals(
    const loomc_module_t* module,
    const loomc_module_global_query_options_t* options,
    loomc_allocator_t allocator, loomc_host_size_t global_capacity,
    loomc_module_global_t* out_globals, loomc_host_size_t* out_global_count,
    loomc_result_t** out_result) {
  if (module == NULL || out_global_count == NULL || out_result == NULL ||
      (global_capacity != 0 && out_globals == NULL)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module, output parameters, and global storage must be valid");
  }
  *out_global_count = 0;
  *out_result = NULL;
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }

  loomc_module_global_resolved_query_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_module_global_resolve_query_options(options, &resolved_options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_status_t status = loomc_ok_status();
  if (loomc_string_view_is_empty(resolved_options.global_symbol)) {
    status = loomc_module_global_query_all(internal_module, &resolved_options,
                                           global_capacity, out_globals,
                                           out_global_count);
  } else {
    status = loomc_module_global_query_named(internal_module, &resolved_options,
                                             global_capacity, out_globals,
                                             out_global_count, result);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

bool loomc_module_try_get_global_at(const loomc_module_t* module,
                                    loomc_host_size_t global_ordinal,
                                    loomc_module_global_t* out_global) {
  if (out_global == NULL) {
    return false;
  }
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  const loom_symbol_t* symbol = NULL;
  loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  if (!loomc_module_global_try_resolve_at(internal_module, global_ordinal,
                                          &symbol, &kind)) {
    *out_global = (loomc_module_global_t){0};
    return false;
  }
  return loomc_module_global_populate_from_symbol(
      internal_module, global_ordinal, symbol, out_global);
}

loomc_status_t loomc_module_get_global_at(const loomc_module_t* module,
                                          loomc_host_size_t global_ordinal,
                                          loomc_module_global_t* out_global) {
  if (module == NULL || out_global == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_global must not be NULL");
  }
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  const loom_symbol_t* symbol = NULL;
  loomc_module_global_kind_t kind = LOOMC_MODULE_GLOBAL_KIND_UNKNOWN;
  if (!loomc_module_global_try_resolve_at(internal_module, global_ordinal,
                                          &symbol, &kind) ||
      !loomc_module_global_populate_from_symbol(internal_module, global_ordinal,
                                                symbol, out_global)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module global ordinal was not found");
  }
  return loomc_ok_status();
}

bool loomc_module_try_lookup_global(const loomc_module_t* module,
                                    loomc_string_view_t symbol_name,
                                    loomc_module_global_t* out_global) {
  if (out_global == NULL) {
    return false;
  }
  *out_global = (loomc_module_global_t){0};
  if (!loomc_module_global_string_view_is_valid(symbol_name)) {
    return false;
  }
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return false;
  }
  symbol_name = loomc_module_global_normalize_symbol(symbol_name);
  const loom_symbol_t* symbol =
      loomc_module_global_find_named_symbol(internal_module, symbol_name);
  if (symbol == NULL) {
    return false;
  }
  loomc_host_size_t global_ordinal = 0;
  if (!loomc_module_global_symbol_global_ordinal(internal_module, symbol,
                                                 &global_ordinal)) {
    return false;
  }
  return loomc_module_global_populate_from_symbol(
      internal_module, global_ordinal, symbol, out_global);
}

loomc_status_t loomc_module_lookup_global(const loomc_module_t* module,
                                          loomc_string_view_t symbol_name,
                                          loomc_module_global_t* out_global) {
  if (module == NULL || out_global == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_global must not be NULL");
  }
  *out_global = (loomc_module_global_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_module_global_validate_string_view(symbol_name));
  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  if (internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  symbol_name = loomc_module_global_normalize_symbol(symbol_name);
  const loom_symbol_t* symbol =
      loomc_module_global_find_named_symbol(internal_module, symbol_name);
  if (symbol == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module global symbol was not found");
  }
  loomc_host_size_t global_ordinal = 0;
  if (!loomc_module_global_symbol_global_ordinal(internal_module, symbol,
                                                 &global_ordinal) ||
      !loomc_module_global_populate_from_symbol(internal_module, global_ordinal,
                                                symbol, out_global)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "module global symbol was not found");
  }
  return loomc_ok_status();
}
