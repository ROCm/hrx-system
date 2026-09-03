// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/cmd/product_provider.h"

#include "loom/codegen/low/repr.h"
#include "loom/link/materialization_environment.h"
#include "loom/link/module_index.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_builder.h"
#include "loom/target/arch/cmd/product.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/compile/product.h"

void loom_cmd_product_build_options_initialize(
    loom_cmd_product_build_options_t* out_options) {
  *out_options = (loom_cmd_product_build_options_t){0};
}

// Selects command roots from source roles retained by the module index. The
// compact indexed-symbol scan replaces source-IR walks and allocates the result
// once at its maximum possible size.
static iree_status_t loom_cmd_product_provider_collect_roots(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* indexed_module,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t** out_root_symbol_ordinals,
    iree_host_size_t* out_root_count) {
  *out_root_symbol_ordinals = NULL;
  *out_root_count = 0;
  if (indexed_module->symbol_count == 0) return iree_ok_status();

  iree_host_size_t* root_symbol_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, indexed_module->symbol_count,
      sizeof(*root_symbol_ordinals), (void**)&root_symbol_ordinals));

  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < indexed_module->symbol_count; ++i) {
    const iree_host_size_t symbol_ordinal =
        indexed_module->symbol_start_ordinal + i;
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, symbol_ordinal);
    IREE_ASSERT(symbol != NULL);
    if (!iree_any_bit_set(symbol->facets.schema.interfaces,
                          LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM) ||
        !iree_any_bit_set(symbol->flags,
                          LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION) ||
        !iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC |
                                             LOOM_LINK_SYMBOL_FLAG_RETAIN)) {
      continue;
    }
    root_symbol_ordinals[root_count++] = symbol_ordinal;
  }
  *out_root_symbol_ordinals = root_symbol_ordinals;
  *out_root_count = root_count;
  return iree_ok_status();
}

static iree_status_t loom_cmd_product_provider_build(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  *out_product = NULL;
  if (request->low_descriptor_registry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command product compilation requires a low descriptor registry");
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(request->block_pool, &scratch_arena);
  loom_link_module_index_t* index = NULL;
  iree_status_t status = loom_link_module_index_allocate(
      request->module->context, request->block_pool, request->allocator,
      &index);
  iree_host_size_t provider_ordinal = 0;
  if (iree_status_is_ok(status)) {
    status = loom_link_module_index_add_materialized(
        index, request->module,
        &(loom_link_module_index_add_options_t){
            .provider_name = provider->name,
        },
        &provider_ordinal);
  }

  const loom_link_module_index_module_t* indexed_module = NULL;
  if (iree_status_is_ok(status)) {
    const loom_link_module_index_provider_t* indexed_provider =
        loom_link_module_index_provider_at(index, provider_ordinal);
    IREE_ASSERT(indexed_provider != NULL);
    IREE_ASSERT_EQ(indexed_provider->module_count, 1u);
    indexed_module = loom_link_module_index_module_at(
        index, indexed_provider->module_start_ordinal);
    IREE_ASSERT(indexed_module != NULL);
  }

  iree_host_size_t* root_symbol_ordinals = NULL;
  iree_host_size_t root_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_cmd_product_provider_collect_roots(
        index, indexed_module, &scratch_arena, &root_symbol_ordinals,
        &root_count);
  }
  if (iree_status_is_ok(status) && root_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command product requires at least one retained command program root");
  }

  const loom_compile_options_t* compile_options = request->compile_options;
  const loom_target_entry_options_t diagnostic_options = {
      .diagnostic_sink = compile_options->diagnostic_sink,
      .source_resolver = compile_options->source_resolver,
      .max_errors = compile_options->max_errors,
  };
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      request->module, &diagnostic_options, LOOM_EMITTER_PASS,
      &diagnostic_emitter);

  loom_low_repr_environment_t low_repr_environment = {0};
  loom_low_repr_environment_initialize(
      &request->low_descriptor_registry->registry, &low_repr_environment);
  const loom_link_plan_materialization_environment_t
      materialization_environment = {
          .context = request->module->context,
          .block_pool = request->block_pool,
          .low_repr_environment = low_repr_environment,
          .allocator = request->allocator,
      };
  const loom_cmd_product_build_options_t* product_options =
      (const loom_cmd_product_build_options_t*)request->option_chain;
  loom_cmd_program_plan_index_options_t plan_options;
  loom_cmd_program_plan_index_options_initialize(&plan_options);
  if (product_options != NULL) {
    plan_options.kernel_request_sink = product_options->kernel_request_sink;
  }

  bool plan_valid = false;
  loom_cmd_program_artifact_set_t artifact_set = {0};
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_artifact_set_build_from_index(
        index, root_symbol_ordinals, root_count,
        &(loom_cmd_program_artifact_builder_options_t){
            .plan_options = plan_options.kernel_request_sink.publish != NULL
                                ? &plan_options
                                : NULL,
            .pass_registry = loom_pass_builtin_registry(),
            .diagnostic_emitter =
                loom_target_entry_emitter(&diagnostic_emitter),
            .materialization_environment = &materialization_environment,
        },
        &scratch_arena, &plan_valid, &artifact_set, request->allocator);
  }
  if (iree_status_is_ok(status) && !plan_valid &&
      diagnostic_emitter.error_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "command program preparation failed without a diagnostic");
  }
  if (iree_status_is_ok(status) && plan_valid) {
    status =
        loom_cmd_product_create(&artifact_set, request->allocator, out_product);
  }

  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  loom_link_module_index_free(index);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

const loom_product_format_provider_t loom_cmd_product_provider = {
    .name = IREE_SVL("command"),
    .operation = &loom_cmd_product_operation,
    .format = &loom_cmd_product_format,
    .flags = LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    .build = loom_cmd_product_provider_build,
};
