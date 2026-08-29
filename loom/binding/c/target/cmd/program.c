// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cmd/program.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "loom/binding/c/src/config.h"
#include "loom/binding/c/src/context.h"
#include "loom/binding/c/src/diagnostic.h"
#include "loom/binding/c/src/link_index.h"
#include "loom/binding/c/src/link_materialization.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/result.h"
#include "loom/binding/c/src/target.h"
#include "loom/binding/c/src/workspace.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_builder.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/transforms/kernel/kernel_class_materializer.h"
#include "loomc/iree.h"

enum {
  LOOMC_CMD_PROGRAM_PRODUCT_KNOWN_FLAGS =
      LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
};

struct loomc_cmd_program_product_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for product-owned storage.
  loomc_allocator_t allocator;

  // Owned serialized programs and copied requirement metadata.
  loom_cmd_program_artifact_set_t artifact_set;
};

typedef struct loomc_cmd_program_product_invocation_t {
  // Public composition resources borrowed by this invocation.
  struct {
    // Context shared by every source and returned request module.
    loomc_context_t* context;

    // Workspace retained independently by each returned request module.
    loomc_workspace_t* workspace;

    // Allocator used for public request module handles.
    loomc_allocator_t allocator;
  } composition;

  // Operation result receiving command preparation diagnostics.
  loomc_result_t* result;

  // Optional embedding sink accepting source-backed kernel requests.
  loomc_cmd_kernel_request_sink_t request_sink;

  // True after the embedding request callback returns a non-OK status.
  bool request_callback_failed;
} loomc_cmd_program_product_invocation_t;

static bool loomc_cmd_program_product_any_flag_set(
    loomc_cmd_program_product_flags_t flags,
    loomc_cmd_program_product_flags_t bits) {
  return (flags & bits) != 0;
}

static loomc_status_t loomc_cmd_program_product_validate_options(
    loomc_workspace_t* workspace,
    const loomc_cmd_program_product_options_t* options,
    loomc_allocator_t allocator,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
  if (workspace == NULL || options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "workspace and product options must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command product options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command product options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, out_target_specialization));
  if (options->link_index == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link_index must not be NULL");
  }
  if (options->root_symbol_count != 0 &&
      options->root_symbol_ordinals == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "root_symbol_count is non-zero but root_symbol_ordinals is NULL");
  }
  if ((options->flags & ~LOOMC_CMD_PROGRAM_PRODUCT_KNOWN_FLAGS) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command product options contain unknown flags");
  }
  if (options->root_symbol_count == 0 &&
      !loomc_cmd_program_product_any_flag_set(
          options->flags,
          LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command product construction requires an explicit or INPUT root");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_target_specialization_options_validate_environment(
          *out_target_specialization,
          loomc_context_target_environment(
              loomc_link_index_context(options->link_index))));
  return loomc_config_validate_options(&options->config);
}

static bool loomc_cmd_program_product_symbol_is_command_root(
    const loom_link_module_index_symbol_t* symbol) {
  return symbol != NULL &&
         loom_link_module_index_symbol_facet_ordinal(
             symbol, LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION) !=
             LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
}

static bool loomc_cmd_program_product_root_is_marked(
    const uint64_t* root_bits, iree_host_size_t symbol_ordinal) {
  return (root_bits[symbol_ordinal / 64] &
          (UINT64_C(1) << (symbol_ordinal % 64))) != 0;
}

static void loomc_cmd_program_product_mark_root(
    uint64_t* root_bits, iree_host_size_t symbol_ordinal) {
  root_bits[symbol_ordinal / 64] |= UINT64_C(1) << (symbol_ordinal % 64);
}

// Adds automatic INPUT roots without adding work to the exact-root JIT path.
static loomc_status_t loomc_cmd_program_product_resolve_roots(
    const loomc_cmd_program_product_options_t* options,
    iree_arena_allocator_t* scratch_arena,
    const iree_host_size_t** out_root_symbol_ordinals,
    iree_host_size_t* out_root_symbol_count) {
  *out_root_symbol_ordinals = options->root_symbol_ordinals;
  *out_root_symbol_count = options->root_symbol_count;
  if (!loomc_cmd_program_product_any_flag_set(
          options->flags,
          LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS)) {
    return loomc_ok_status();
  }

  const loom_link_module_index_t* index =
      loomc_link_index_module_index(options->link_index);
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(index);
  const loom_link_module_index_symbol_ordinal_list_t input_exports =
      loom_link_module_index_input_exports(index);
  iree_host_size_t root_capacity = 0;
  if (!iree_host_size_checked_add(options->root_symbol_count,
                                  input_exports.count, &root_capacity)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "command root table is too large");
  }

  iree_host_size_t* roots = NULL;
  if (root_capacity != 0) {
    LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_arena_allocate_array(
        scratch_arena, root_capacity, sizeof(*roots), (void**)&roots)));
  }
  const iree_host_size_t root_bit_count =
      symbol_count / 64 + (symbol_count % 64 != 0);
  uint64_t* root_bits = NULL;
  if (root_bit_count != 0) {
    LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
        iree_arena_allocate_array(scratch_arena, root_bit_count,
                                  sizeof(*root_bits), (void**)&root_bits)));
    memset(root_bits, 0, root_bit_count * sizeof(*root_bits));
  }

  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < options->root_symbol_count; ++i) {
    const iree_host_size_t symbol_ordinal = options->root_symbol_ordinals[i];
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, symbol_ordinal);
    if (!loomc_cmd_program_product_symbol_is_command_root(symbol)) {
      return loomc_status_from_iree(
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "root symbol ordinal %" PRIhsz
                           " does not expose a command implementation",
                           symbol_ordinal));
    }
    if (!loomc_cmd_program_product_root_is_marked(root_bits, symbol_ordinal)) {
      loomc_cmd_program_product_mark_root(root_bits, symbol_ordinal);
    }
    roots[root_count++] = symbol_ordinal;
  }
  for (iree_host_size_t i = 0; i < input_exports.count; ++i) {
    const iree_host_size_t symbol_ordinal = input_exports.values[i];
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, symbol_ordinal);
    if (!loomc_cmd_program_product_symbol_is_command_root(symbol) ||
        loomc_cmd_program_product_root_is_marked(root_bits, symbol_ordinal)) {
      continue;
    }
    loomc_cmd_program_product_mark_root(root_bits, symbol_ordinal);
    roots[root_count++] = symbol_ordinal;
  }
  if (root_count == 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "the selected INPUT providers export no command definitions");
  }
  *out_root_symbol_ordinals = roots;
  *out_root_symbol_count = root_count;
  return loomc_ok_status();
}

static iree_string_view_t loomc_cmd_program_product_kernel_name(
    const loom_kernel_class_product_t* product) {
  IREE_ASSERT_ARGUMENT(product);
  IREE_ASSERT_ARGUMENT(product->module);
  IREE_ASSERT(loom_symbol_ref_is_valid(product->kernel));
  IREE_ASSERT_EQ(product->kernel.module_id, 0u);
  IREE_ASSERT_LT(product->kernel.symbol_id, product->module->symbols.count);
  const loom_symbol_t* symbol =
      &product->module->symbols.entries[product->kernel.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, product->module->strings.count);
  return product->module->strings.entries[symbol->name_id];
}

static iree_status_t loomc_cmd_program_product_publish_kernel_request(
    void* user_data, loom_cmd_program_kernel_request_t request) {
  loomc_cmd_program_product_invocation_t* invocation =
      (loomc_cmd_program_product_invocation_t*)user_data;
  const iree_string_view_t root_symbol =
      loomc_cmd_program_product_kernel_name(&request.source.product);

  loomc_module_t* module = NULL;
  loomc_status_t status = loomc_module_create_empty(
      invocation->composition.context, invocation->composition.workspace,
      invocation->composition.allocator, &module);
  if (loomc_status_is_ok(status)) {
    status =
        loomc_module_set_loom_module(module, request.source.product.module);
    if (loomc_status_is_ok(status)) {
      request.source.product.module = NULL;
    }
  }
  if (loomc_status_is_ok(status)) {
    loomc_cmd_kernel_request_t public_request = {
        .entry_requirement_ordinal = request.entry_requirement_index,
        .root_symbol = loomc_string_view_from_iree(root_symbol),
        .member_count = request.source.member_count,
        .module = module,
    };
    module = NULL;
    status = invocation->request_sink.publish(
        invocation->request_sink.user_data, public_request);
    invocation->request_callback_failed = !loomc_status_is_ok(status);
  }

  loomc_module_release(module);
  loom_kernel_class_product_deinitialize(&request.source.product);
  return iree_status_from_loomc(status);
}

static iree_status_t loomc_cmd_program_product_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_cmd_program_product_invocation_t* invocation =
      (loomc_cmd_program_product_invocation_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      invocation->result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
}

static loomc_status_t loomc_cmd_program_product_translate_plan_status(
    loomc_cmd_program_product_invocation_t* invocation,
    loomc_host_size_t diagnostic_count_before, iree_status_t plan_status) {
  if (iree_status_is_ok(plan_status)) {
    return loomc_ok_status();
  }
  if (invocation->request_callback_failed) {
    return loomc_status_from_iree(plan_status);
  }
  if (!loomc_result_succeeded(invocation->result) ||
      loomc_result_diagnostic_count(invocation->result) !=
          diagnostic_count_before) {
    iree_status_free(plan_status);
    return loomc_result_set_state(invocation->result,
                                  LOOMC_RESULT_STATE_FAILED);
  }
  return loomc_status_from_iree(plan_status);
}

static loomc_status_t loomc_cmd_program_product_allocate(
    loom_cmd_program_artifact_set_t* artifact_set, loomc_allocator_t allocator,
    loomc_cmd_program_product_t** out_product) {
  *out_product = NULL;
  loomc_cmd_program_product_t* product = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*product), (void**)&product));
  memset(product, 0, sizeof(*product));
  iree_atomic_ref_count_init(&product->ref_count);
  product->allocator = allocator;
  product->artifact_set = *artifact_set;
  *artifact_set = (loom_cmd_program_artifact_set_t){0};
  *out_product = product;
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_product_build(
    loomc_workspace_t* workspace,
    const loomc_cmd_program_product_options_t* options,
    loomc_allocator_t allocator, loomc_cmd_program_product_t** out_product,
    loomc_result_t** out_result) {
  if (out_product == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_product and out_result must not be NULL");
  }
  *out_product = NULL;
  *out_result = NULL;
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_product_validate_options(
      workspace, options, allocator, &target_specialization));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loom_cmd_program_artifact_set_t artifact_set = {0};
  loomc_cmd_program_product_t* product = NULL;
  loomc_cmd_program_product_invocation_t invocation = {
      .composition =
          {
              .context = loomc_link_index_context(options->link_index),
              .workspace = workspace,
              .allocator = allocator,
          },
      .result = result,
      .request_sink = options->kernel_request_sink,
  };
  loomc_status_t status = loomc_ok_status();

  const iree_host_size_t* root_symbol_ordinals = NULL;
  iree_host_size_t root_symbol_count = 0;
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_product_resolve_roots(
        options, &scratch_arena, &root_symbol_ordinals, &root_symbol_count);
  }

  loomc_link_materialization_state_t materialization_state = {0};
  loomc_link_materialization_state_initialize(
      invocation.composition.context, workspace, options->link_index,
      &options->config, target_specialization, result, allocator,
      &materialization_state);
  const loom_link_plan_materialization_environment_t
      materialization_environment =
          loomc_link_materialization_state_environment(&materialization_state);
  loom_cmd_program_plan_index_options_t plan_options;
  loom_cmd_program_plan_index_options_initialize(&plan_options);
  if (options->kernel_request_sink.publish != NULL) {
    plan_options.kernel_request_sink = (loom_cmd_program_kernel_request_sink_t){
        .publish = loomc_cmd_program_product_publish_kernel_request,
        .user_data = &invocation,
    };
  }

  bool plan_valid = false;
  const loomc_host_size_t diagnostic_count_before =
      loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status)) {
    const iree_status_t plan_status =
        loom_cmd_program_artifact_set_build_from_index(
            loomc_link_index_module_index(options->link_index),
            root_symbol_ordinals, root_symbol_count,
            &(loom_cmd_program_artifact_builder_options_t){
                .plan_options = options->kernel_request_sink.publish != NULL
                                    ? &plan_options
                                    : NULL,
                .pass_registry = loom_pass_builtin_registry(),
                .diagnostic_emitter =
                    {
                        .fn = loomc_cmd_program_product_capture_diagnostic,
                        .user_data = &invocation,
                    },
                .materialization_environment = &materialization_environment,
            },
            &scratch_arena, &plan_valid, &artifact_set,
            iree_allocator_from_loomc(allocator));
    status = loomc_cmd_program_product_translate_plan_status(
        &invocation, diagnostic_count_before, plan_status);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      !plan_valid) {
    if (loomc_result_diagnostic_count(result) == diagnostic_count_before) {
      status = loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "command program preparation failed without a diagnostic");
    } else {
      status = loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status =
        loomc_cmd_program_product_allocate(&artifact_set, allocator, &product);
  }
  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_product = product;
      product = NULL;
    }
    *out_result = result;
    result = NULL;
  }

  loomc_cmd_program_product_release(product);
  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  iree_arena_deinitialize(&scratch_arena);
  loomc_result_release(result);
  return status;
}

void loomc_cmd_program_product_retain(loomc_cmd_program_product_t* product) {
  if (product == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&product->ref_count);
}

void loomc_cmd_program_product_release(loomc_cmd_program_product_t* product) {
  if (product == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&product->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = product->allocator;
  loom_cmd_program_artifact_set_deinitialize(&product->artifact_set);
  loomc_allocator_free(allocator, product);
}

loomc_host_size_t loomc_cmd_program_product_program_count(
    const loomc_cmd_program_product_t* product) {
  return product ? product->artifact_set.programs.count : 0;
}

bool loomc_cmd_program_product_program_at(
    const loomc_cmd_program_product_t* product, loomc_host_size_t ordinal,
    loomc_cmd_program_t* out_program) {
  if (product == NULL || out_program == NULL ||
      ordinal >= product->artifact_set.programs.count) {
    return false;
  }
  const loom_cmd_program_artifact_t* artifact =
      &product->artifact_set.programs.values[ordinal];
  const loomc_string_view_t symbol =
      loomc_string_view_from_iree(artifact->symbol);
  *out_program = (loomc_cmd_program_t){
      .symbol = symbol,
      .artifact =
          {
              .kind = LOOMC_ARTIFACT_KIND_EXECUTABLE,
              .format =
                  loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM),
              .identifier = symbol,
              .contents = loomc_byte_sequence_from_iree(artifact->data),
          },
      .entry_requirement_ordinals = artifact->entry_requirement_indices,
      .entry_requirement_count = artifact->entry_requirement_count,
  };
  return true;
}

loomc_host_size_t loomc_cmd_program_product_entry_requirement_count(
    const loomc_cmd_program_product_t* product) {
  return product ? product->artifact_set.entries.count : 0;
}

bool loomc_cmd_program_product_entry_requirement_at(
    const loomc_cmd_program_product_t* product, loomc_host_size_t ordinal,
    loomc_cmd_entry_requirement_t* out_requirement) {
  if (product == NULL || out_requirement == NULL ||
      ordinal >= product->artifact_set.entries.count) {
    return false;
  }
  const loom_cmd_program_artifact_entry_t* entry =
      &product->artifact_set.entries.values[ordinal];
  *out_requirement = (loomc_cmd_entry_requirement_t){
      .symbol = loomc_string_view_from_iree(entry->symbol),
      .has_source_request = entry->has_source_request,
  };
  return true;
}
