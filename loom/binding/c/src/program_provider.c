// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "program_provider.h"

#include "loom/ir/module.h"
#include "loomc/iree.h"

static loomc_string_view_t loomc_program_provider_normalize_root_name(
    loomc_string_view_t name) {
  if (name.size != 0 && name.data[0] == '@') {
    return loomc_make_string_view(name.data + 1, name.size - 1);
  }
  return name;
}

static loomc_status_t loomc_program_provider_lookup_root(
    const loom_module_t* module, loomc_string_view_t name,
    const loom_op_t** out_root_op) {
  *out_root_op = NULL;
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, iree_string_view_from_loomc(name));
  if (name_id == LOOM_STRING_ID_INVALID) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "program root symbol was not found");
  }
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "program root symbol was not found");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (symbol->defining_op == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "program root must be a definition");
  }
  *out_root_op = symbol->defining_op;
  return loomc_ok_status();
}

loomc_status_t loomc_program_provider_select_roots(
    const loomc_program_provider_set_t* provider_set,
    const loom_module_t* module, const loomc_string_view_t* root_names,
    loomc_host_size_t root_count, iree_arena_allocator_t* arena,
    loomc_program_provider_selection_t* out_selection) {
  *out_selection = (loomc_program_provider_selection_t){0};

  const loom_op_t** root_ops = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_arena_allocate_array(
      arena, root_count, sizeof(*root_ops), (void**)&root_ops)));
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    const loomc_string_view_t name =
        loomc_program_provider_normalize_root_name(root_names[i]);
    for (loomc_host_size_t j = 0; j < i; ++j) {
      if (loomc_string_view_equal(
              name,
              loomc_program_provider_normalize_root_name(root_names[j]))) {
        return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                                 "program root names must be unique");
      }
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_program_provider_lookup_root(module, name, &root_ops[i]));

    const loomc_program_provider_t* root_provider = NULL;
    for (loomc_host_size_t provider_index = 0;
         provider_index < provider_set->count; ++provider_index) {
      const loomc_program_provider_t* provider =
          provider_set->values[provider_index];
      if (!provider->owns_root(module, root_ops[i])) continue;
      if (root_provider != NULL) {
        return loomc_make_status(
            LOOMC_STATUS_INTERNAL,
            "multiple program providers own the same selected root");
      }
      root_provider = provider;
    }
    if (root_provider == NULL) {
      return loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "no linked program provider owns a selected root");
    }
    if (out_selection->provider != NULL &&
        out_selection->provider != root_provider) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "selected roots span multiple program provider families");
    }
    out_selection->provider = root_provider;
  }

  out_selection->root_ops = root_ops;
  out_selection->root_count = root_count;
  return loomc_ok_status();
}
