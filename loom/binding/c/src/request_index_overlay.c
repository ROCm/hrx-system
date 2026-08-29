// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "request_index_overlay.h"

#include "context.h"
#include "link_index.h"
#include "loomc/iree.h"
#include "product.h"

static loomc_status_t loomc_request_index_overlay_map_roots(
    const loom_link_module_index_t* module_index,
    iree_host_size_t provider_ordinal, const loomc_request_t* request,
    iree_host_size_t* out_symbol_ordinals) {
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(module_index, provider_ordinal);
  if (provider == NULL) {
    return loomc_make_status(LOOMC_STATUS_INTERNAL,
                             "request provider is absent from its index");
  }

  const loomc_request_root_t* roots = loomc_request_roots(request);
  const iree_host_size_t root_count = loomc_request_root_count(request);
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    const loomc_request_root_t root = roots[i];
    if (root.module_ordinal >= provider->module_count) {
      return loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "request root module ordinal exceeds its bytecode source");
    }
    const loom_link_module_index_module_t* module =
        loom_link_module_index_module_at(
            module_index, provider->module_start_ordinal + root.module_ordinal);
    if (root.symbol_ordinal >= module->symbol_count) {
      return loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "request root symbol ordinal exceeds its bytecode module");
    }
    out_symbol_ordinals[i] = module->symbol_start_ordinal + root.symbol_ordinal;
  }
  return loomc_ok_status();
}

loomc_status_t loomc_request_index_overlay_initialize(
    loomc_context_t* context, iree_arena_block_pool_t* block_pool,
    const loomc_link_index_t* library_index, const loomc_request_t* request,
    loomc_result_t* result, iree_arena_allocator_t* scratch_arena,
    loomc_allocator_t allocator, loomc_request_index_overlay_t* out_overlay) {
  *out_overlay = (loomc_request_index_overlay_t){0};

  loom_link_module_index_t* module_index = NULL;
  const iree_allocator_t iree_allocator = iree_allocator_from_loomc(allocator);
  loomc_status_t status =
      library_index != NULL
          ? loomc_status_from_iree(loom_link_module_index_allocate_overlay(
                loomc_link_index_module_index(library_index), block_pool,
                iree_allocator, &module_index))
          : loomc_status_from_iree(loom_link_module_index_allocate(
                loomc_context_loom_context(context), block_pool, iree_allocator,
                &module_index));

  iree_host_size_t request_provider_ordinal = IREE_HOST_SIZE_MAX;
  if (loomc_status_is_ok(status)) {
    const loomc_link_index_source_options_t source_options = {
        .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    status = loomc_link_index_add_source_to_module_index(
        context, module_index, loomc_request_source(request), &source_options,
        result, &request_provider_ordinal);
  }

  const iree_host_size_t root_count = loomc_request_root_count(request);
  iree_host_size_t* root_symbol_ordinals = NULL;
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      root_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        scratch_arena, root_count, sizeof(*root_symbol_ordinals),
        (void**)&root_symbol_ordinals));
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_request_index_overlay_map_roots(
        module_index, request_provider_ordinal, request, root_symbol_ordinals);
  }

  if (loomc_status_is_ok(status)) {
    *out_overlay = (loomc_request_index_overlay_t){
        .module_index = module_index,
        .request_provider_ordinal = request_provider_ordinal,
        .root_symbol_ordinals = root_symbol_ordinals,
        .root_symbol_count = root_count,
    };
    module_index = NULL;
  }
  loom_link_module_index_free(module_index);
  return status;
}

void loomc_request_index_overlay_deinitialize(
    loomc_request_index_overlay_t* overlay) {
  if (overlay == NULL) return;
  loom_link_module_index_free(overlay->module_index);
  *overlay = (loomc_request_index_overlay_t){0};
}
