// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cmd/program.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/binding/c/src/config.h"
#include "loom/binding/c/src/context.h"
#include "loom/binding/c/src/diagnostic.h"
#include "loom/binding/c/src/link_index.h"
#include "loom/binding/c/src/link_materialization.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/module_bytecode.h"
#include "loom/binding/c/src/product.h"
#include "loom/binding/c/src/request_index_overlay.h"
#include "loom/binding/c/src/result.h"
#include "loom/binding/c/src/target.h"
#include "loom/binding/c/src/workspace.h"
#include "loom/binding/c/target/cmd/compiler.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/target/arch/cmd/lower/program_plan_index.h"
#include "loom/transforms/cleanup/configured.h"
#include "loom/transforms/kernel/kernel_class_materializer.h"
#include "loomc/compile.h"
#include "loomc/iree.h"

enum {
  LOOMC_CMD_PROGRAM_PRODUCT_KNOWN_FLAGS =
      LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
};

typedef struct loomc_cmd_program_product_impl_t {
  // Generic immutable product interface exposed to callers.
  loomc_product_t base;

  // Allocator used for product-owned storage.
  loomc_allocator_t allocator;

  // Owned serialized programs and copied requirement metadata.
  loom_cmd_program_artifact_set_t artifact_set;
} loomc_cmd_program_product_impl_t;

static void loomc_cmd_program_product_destroy(loomc_product_t* base_product) {
  loomc_cmd_program_product_impl_t* product =
      (loomc_cmd_program_product_impl_t*)base_product;
  loomc_allocator_t allocator = product->allocator;
  loom_cmd_program_artifact_set_deinitialize(&product->artifact_set);
  loomc_allocator_free(allocator, product);
}

static const loomc_product_descriptor_t loomc_cmd_program_product_descriptor_ =
    {
        .destroy = loomc_cmd_program_product_destroy,
};

static const loomc_cmd_program_product_impl_t*
loomc_cmd_program_product_const_cast(const loomc_product_t* product) {
  if (!loomc_product_isa(product, &loomc_cmd_program_product_descriptor_)) {
    return NULL;
  }
  return (const loomc_cmd_program_product_impl_t*)product;
}

typedef struct loomc_cmd_program_product_invocation_t {
  // Resources used to serialize immutable requests.
  struct {
    // Context providing the bytecode representation environment.
    loomc_context_t* context;

    // Workspace block pool used while materializing request sources.
    iree_arena_block_pool_t* block_pool;

    // Allocator used for request sources and handles.
    loomc_allocator_t allocator;
  } request;

  // Operation result receiving command planning diagnostics.
  loomc_result_t* result;

  // Optional embedding sink accepting source-backed kernel requests.
  loomc_request_sink_t request_sink;

  // True after the embedding request callback returns a non-OK status.
  bool request_callback_failed;
} loomc_cmd_program_product_invocation_t;

static bool loomc_cmd_program_product_any_flag_set(
    loomc_cmd_program_product_flags_t flags,
    loomc_cmd_program_product_flags_t bits) {
  return (flags & bits) != 0;
}

static loomc_status_t loomc_cmd_program_validate_artifact_flags(
    loomc_emit_artifact_flags_t flags) {
  const loomc_emit_artifact_flags_t known_flags =
      LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY |
      LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING;
  if ((flags & ~known_flags) != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command product options contain unknown artifact flags");
  }
  return loomc_ok_status();
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
  LOOMC_RETURN_IF_ERROR(
      loomc_cmd_program_validate_artifact_flags(options->artifact_flags));
  if (options->compiler == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compiler must not be NULL");
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
  LOOMC_RETURN_IF_ERROR(loomc_link_materialization_validate_targets(
      options->target_profile, *out_target_specialization,
      loomc_context_target_environment(loomc_link_index_context(
          loomc_cmd_compiler_index(options->compiler)))));
  return loomc_config_validate_text_options(&options->config);
}

static loomc_status_t loomc_cmd_program_product_validate_request_options(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_request_t* request,
    const loomc_cmd_program_request_options_t* options,
    loomc_allocator_t allocator,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
  if (context == NULL || workspace == NULL || request == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, and request must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }
  if (loomc_request_product_descriptor(request) !=
      loomc_cmd_program_product_descriptor()) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "request does not require a command-program product");
  }
  if (loomc_source_format(loomc_request_source(request)) !=
      LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "request source is not Loom bytecode");
  }
  if (loomc_request_root_count(request) == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command request must contain at least one root");
  }
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_REQUEST_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command request options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command request options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, out_target_specialization));
  LOOMC_RETURN_IF_ERROR(
      loomc_cmd_program_validate_artifact_flags(options->artifact_flags));
  if (options->library_index != NULL) {
    if (loomc_link_index_context(options->library_index) != context) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command request library index was created with another context");
    }
    if (loom_link_module_index_input_provider_count(
            loomc_link_index_module_index(options->library_index)) != 0) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command request library index contains INPUT providers");
    }
  }
  LOOMC_RETURN_IF_ERROR(loomc_link_materialization_validate_targets(
      options->target_profile, *out_target_specialization,
      loomc_context_target_environment(context)));
  return loomc_config_validate_text_options(&options->config);
}

static loomc_status_t loomc_cmd_program_product_validate_module_options(
    loomc_workspace_t* workspace, const loomc_module_t* module,
    const loomc_cmd_program_module_options_t* options,
    loomc_allocator_t allocator,
    const loomc_target_specialization_options_t** out_target_specialization) {
  *out_target_specialization = NULL;
  if (workspace == NULL || module == NULL || options == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workspace, module, and module options must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command module options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command module options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(loomc_target_specialization_options_resolve(
      options->next, out_target_specialization));
  LOOMC_RETURN_IF_ERROR(
      loomc_cmd_program_validate_artifact_flags(options->artifact_flags));
  loomc_context_t* context = loomc_module_context(module);
  if (context == NULL || loomc_module_const_loom_module(module) == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  if (options->library_index != NULL) {
    if (loomc_link_index_context(options->library_index) != context) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command module library index was created with another context");
    }
    if (loom_link_module_index_input_provider_count(
            loomc_link_index_module_index(options->library_index)) != 0) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command module library index contains INPUT providers");
    }
  }
  if (options->provider_name.data == NULL && options->provider_name.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "provider name has length but no data");
  }
  if (options->root_symbol_count != 0 && options->root_symbols == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "root_symbol_count is non-zero but root_symbols is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->root_symbol_count; ++i) {
    if (options->root_symbols[i].data == NULL &&
        options->root_symbols[i].size != 0) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "command root has length but no data");
    }
  }
  if ((options->flags & ~LOOMC_CMD_PROGRAM_PRODUCT_KNOWN_FLAGS) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command module options contain unknown flags");
  }
  if (options->root_symbol_count == 0 &&
      !loomc_cmd_program_product_any_flag_set(
          options->flags,
          LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command product construction requires an explicit or INPUT root");
  }
  LOOMC_RETURN_IF_ERROR(loomc_link_materialization_validate_targets(
      options->target_profile, *out_target_specialization,
      loomc_context_target_environment(context)));
  return loomc_config_validate_text_options(&options->config);
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
    const loom_link_module_index_t* index,
    const iree_host_size_t* explicit_root_symbol_ordinals,
    iree_host_size_t explicit_root_symbol_count,
    loomc_cmd_program_product_flags_t flags,
    iree_arena_allocator_t* scratch_arena,
    const iree_host_size_t** out_root_symbol_ordinals,
    iree_host_size_t* out_root_symbol_count) {
  *out_root_symbol_ordinals = explicit_root_symbol_ordinals;
  *out_root_symbol_count = explicit_root_symbol_count;
  if (!loomc_cmd_program_product_any_flag_set(
          flags, LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS)) {
    return loomc_ok_status();
  }

  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(index);
  const loom_link_module_index_symbol_ordinal_list_t input_exports =
      loom_link_module_index_input_exports(index);
  iree_host_size_t root_capacity = 0;
  if (!iree_host_size_checked_add(explicit_root_symbol_count,
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
  for (iree_host_size_t i = 0; i < explicit_root_symbol_count; ++i) {
    const iree_host_size_t symbol_ordinal = explicit_root_symbol_ordinals[i];
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

static loomc_status_t loomc_cmd_program_product_resolve_module_roots(
    const loom_link_module_index_t* index,
    iree_host_size_t module_provider_ordinal,
    const loomc_string_view_t* root_symbols, iree_host_size_t root_symbol_count,
    iree_arena_allocator_t* scratch_arena,
    const iree_host_size_t** out_root_symbol_ordinals) {
  *out_root_symbol_ordinals = NULL;
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index, module_provider_ordinal);
  if (provider == NULL || provider->module_count != 1) {
    return loomc_make_status(
        LOOMC_STATUS_INTERNAL,
        "materialized command provider did not index exactly one module");
  }
  const loom_link_module_index_module_t* module =
      loom_link_module_index_module_at(index, provider->module_start_ordinal);
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INTERNAL,
                             "materialized command module was not indexed");
  }
  iree_host_size_t* root_symbol_ordinals = NULL;
  if (root_symbol_count != 0) {
    LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_arena_allocate_array(
        scratch_arena, root_symbol_count, sizeof(*root_symbol_ordinals),
        (void**)&root_symbol_ordinals)));
  }
  for (iree_host_size_t i = 0; i < root_symbol_count; ++i) {
    const iree_string_view_t root_symbol =
        iree_string_view_from_loomc(root_symbols[i]);
    const loom_link_module_index_symbol_t* root =
        loom_link_module_index_lookup_private(index, module, root_symbol);
    if (root == NULL) {
      for (root = loom_link_module_index_lookup_global(index, root_symbol);
           root != NULL && root->module_ordinal != module->ordinal;
           root = loom_link_module_index_next_global_duplicate(index, root)) {
      }
    }
    if (root == NULL) {
      return loomc_status_from_iree(
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "command root '@%.*s' was not found in module",
                           (int)root_symbol.size, root_symbol.data));
    }
    if (!loomc_cmd_program_product_symbol_is_command_root(root)) {
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "module symbol '@%.*s' does not expose a command implementation",
          (int)root_symbol.size, root_symbol.data));
    }
    root_symbol_ordinals[i] = root->ordinal;
  }
  *out_root_symbol_ordinals = root_symbol_ordinals;
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
  return loom_string_table_get(&product->module->strings, symbol->name_id);
}

static iree_status_t loomc_cmd_program_product_publish_kernel_request(
    void* user_data, const loom_cmd_program_kernel_request_t* request) {
  loomc_cmd_program_product_invocation_t* invocation =
      (loomc_cmd_program_product_invocation_t*)user_data;
  loom_kernel_class_product_t kernel_product = {0};
  loomc_status_t status =
      loomc_status_from_iree(loom_cmd_program_kernel_request_materialize(
          request, invocation->request.block_pool,
          iree_allocator_from_loomc(invocation->request.allocator),
          &kernel_product));
  loomc_source_t* source = NULL;
  loom_symbol_id_t bytecode_symbol_ordinal = LOOM_SYMBOL_ID_INVALID;
  if (loomc_status_is_ok(status)) {
    const iree_string_view_t root_symbol =
        loomc_cmd_program_product_kernel_name(&kernel_product);
    const loom_symbol_id_t module_symbol_id = kernel_product.kernel.symbol_id;
    const loomc_module_symbol_projection_t projection = {
        .module_symbol_ids = &module_symbol_id,
        .bytecode_symbol_ordinals = &bytecode_symbol_ordinal,
        .count = 1,
    };
    status = loomc_module_serialize_internal_bytecode_to_source(
        invocation->request.context, kernel_product.module,
        loomc_string_view_from_iree(root_symbol), &projection,
        invocation->request.allocator, &source);
  }

  loomc_request_t* public_request = NULL;
  if (loomc_status_is_ok(status)) {
    const loomc_request_root_t root = {
        .module_ordinal = 0,
        .symbol_ordinal = bytecode_symbol_ordinal,
    };
    const loomc_request_binding_t binding = {
        .requirement_ordinal = request->entry_requirement_index,
        .root_ordinal = 0,
    };
    status = loomc_request_create_take_source(
        loomc_compiled_module_product_descriptor(), &source, &root, 1, &binding,
        1, invocation->request.allocator, &public_request);
  }
  if (loomc_status_is_ok(status)) {
    loomc_request_t* transferred_request = public_request;
    public_request = NULL;
    status = invocation->request_sink.publish(
        invocation->request_sink.user_data, transferred_request);
    invocation->request_callback_failed = !loomc_status_is_ok(status);
  }

  loomc_request_release(public_request);
  loomc_source_release(source);
  loom_kernel_class_product_deinitialize(&kernel_product);
  return iree_status_from_loomc(status);
}

static iree_status_t loomc_cmd_program_product_capture_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_cmd_program_product_invocation_t* invocation =
      (loomc_cmd_program_product_invocation_t*)user_data;
  const loomc_target_pass_environment_t* pass_environment =
      loomc_context_target_pass_environment(invocation->request.context);
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      invocation->result, /*module=*/NULL, (loom_source_resolver_t){0},
      LOOM_EMITTER_PASS, emission,
      pass_environment ? &pass_environment->diagnostic_type_print_options
                       : NULL));
}

static iree_status_t loomc_cmd_program_product_verify_module(
    void* user_data, const loom_module_t* module, bool* out_valid) {
  loomc_cmd_program_product_invocation_t* invocation =
      (loomc_cmd_program_product_invocation_t*)user_data;
  *out_valid = false;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_result_verify_loom_module(
      invocation->request.context, module, (loom_source_resolver_t){0},
      /*max_errors=*/0, invocation->result)));
  *out_valid = loomc_result_succeeded(invocation->result);
  return iree_ok_status();
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
    loomc_result_set_status_code(
        invocation->result, (loomc_status_code_t)iree_status_code(plan_status));
    iree_status_free(plan_status);
    return loomc_ok_status();
  }
  return loomc_status_from_iree(plan_status);
}

static loomc_status_t loomc_cmd_program_product_create(
    loom_cmd_program_artifact_set_t* artifact_set, loomc_allocator_t allocator,
    loomc_product_t** out_product) {
  *out_product = NULL;
  iree_host_size_t artifact_storage_size = 0;
  iree_host_size_t allocation_size = sizeof(loomc_cmd_program_product_impl_t);
  if (!iree_host_size_checked_mul(artifact_set->programs.count,
                                  sizeof(loomc_artifact_t),
                                  &artifact_storage_size) ||
      !iree_host_size_checked_add(allocation_size, artifact_storage_size,
                                  &allocation_size)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "command product metadata is too large");
  }

  loomc_cmd_program_product_impl_t* product = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&product));
  memset(product, 0, sizeof(*product));
  product->allocator = allocator;
  product->artifact_set = *artifact_set;
  *artifact_set = (loom_cmd_program_artifact_set_t){0};

  loomc_artifact_t* artifacts = (loomc_artifact_t*)(product + 1);
  for (iree_host_size_t i = 0; i < product->artifact_set.programs.count; ++i) {
    const loom_cmd_program_artifact_t* program =
        &product->artifact_set.programs.values[i];
    const loomc_string_view_t symbol =
        loomc_string_view_from_iree(program->symbol);
    artifacts[i] = (loomc_artifact_t){
        .kind = LOOMC_ARTIFACT_KIND_EXECUTABLE,
        .format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM),
        .identifier = symbol,
        .contents = loomc_byte_sequence_from_iree(program->data),
    };
  }
  loomc_product_initialize(&loomc_cmd_program_product_descriptor_, artifacts,
                           product->artifact_set.programs.count,
                           product->artifact_set.programs.count,
                           product->artifact_set.entries.count, &product->base);
  *out_product = &product->base;
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_program_capture_listing(
    const loom_cmd_program_plan_t* plan, loomc_result_t* result) {
  const loomc_allocator_t allocator = loomc_result_allocator(result);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loomc_status_t status = loomc_status_from_iree(
      loom_cmd_program_plan_format_listing(plan, &builder));
  char* storage = NULL;
  if (loomc_status_is_ok(status)) {
    const iree_host_size_t length = iree_string_builder_size(&builder);
    storage = iree_string_builder_take_storage(&builder);
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_TEXT,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM_LISTING),
        loomc_make_cstring_view("command-program.listing"),
        loomc_make_byte_span(storage, length));
  }
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, storage);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loomc_cmd_program_publish_native_source(
    void* user_data,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  return loomc_cmd_native_source_cache_publish(
      user_data, environment->user_data, environment, source_symbol_ordinal,
      site_module, sites, site_count, collection_options, sink, scratch_arena,
      out_site_classes);
}

static loomc_status_t loomc_cmd_program_product_build_indexed(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loom_link_module_index_t* module_index,
    loomc_cmd_native_source_cache_t* native_sources,
    const iree_host_size_t* root_symbol_ordinals,
    iree_host_size_t root_symbol_count,
    const loom_link_plan_materialization_environment_t*
        materialization_environment,
    loomc_request_sink_t request_sink,
    loomc_emit_artifact_flags_t artifact_flags,
    iree_arena_allocator_t* scratch_arena, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_product_t** out_product) {
  *out_product = NULL;
  loom_cmd_program_plan_t plan = {0};
  loom_cmd_program_artifact_set_t artifact_set = {0};
  loomc_product_t* product = NULL;
  loomc_cmd_program_product_invocation_t invocation = {
      .request =
          {
              .context = context,
              .block_pool = loomc_workspace_block_pool(workspace),
              .allocator = allocator,
          },
      .result = result,
      .request_sink = request_sink,
  };
  loom_cmd_program_plan_index_options_t plan_options;
  loom_cmd_program_plan_index_options_initialize(&plan_options);
  plan_options.verification.fn = loomc_cmd_program_product_verify_module;
  plan_options.verification.user_data = &invocation;
  loomc_link_materialization_state_t* materialization_state =
      (loomc_link_materialization_state_t*)
          materialization_environment->user_data;
  if (materialization_state->specialization.deployment_profile != NULL) {
    plan_options.live_kernel_entry_validator.fn =
        loomc_link_materialization_validate_command_kernel_entries;
    plan_options.live_kernel_entry_validator.user_data = materialization_state;
  }
  if (request_sink.publish != NULL) {
    if (native_sources != NULL) {
      plan_options.kernel_request_provider = (loom_kernel_request_provider_t){
          .publish = loomc_cmd_program_publish_native_source,
          .user_data = native_sources,
      };
    }
    plan_options.kernel_request_sink = (loom_cmd_program_kernel_request_sink_t){
        .publish = loomc_cmd_program_product_publish_kernel_request,
        .user_data = &invocation,
    };
  }

  bool plan_valid = false;
  const loomc_host_size_t diagnostic_count_before =
      loomc_result_diagnostic_count(result);
  iree_status_t plan_status = loom_cmd_program_plan_build_from_index(
      module_index, root_symbol_ordinals, root_symbol_count, &plan_options,
      loom_pass_builtin_registry(),
      loom_cleanup_configured_pattern_provider_set(),
      (iree_diagnostic_emitter_t){
          .fn = loomc_cmd_program_product_capture_diagnostic,
          .user_data = &invocation,
      },
      materialization_environment, scratch_arena, &plan_valid, &plan);
  if (iree_status_is_ok(plan_status) && plan_valid) {
    plan_status = loom_cmd_program_artifact_set_build(
        &plan, iree_allocator_from_loomc(allocator), &artifact_set);
  }
  loomc_status_t status = loomc_cmd_program_product_translate_plan_status(
      &invocation, diagnostic_count_before, plan_status);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      !plan_valid) {
    if (loomc_result_diagnostic_count(result) == diagnostic_count_before) {
      status = loomc_make_status(
          LOOMC_STATUS_INTERNAL,
          "command program planning failed without a diagnostic");
    } else {
      loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      (artifact_flags & LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING)) {
    status = loomc_cmd_program_capture_listing(&plan, result);
  }
  loom_cmd_program_plan_deinitialize(&plan);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status =
        loomc_cmd_program_product_create(&artifact_set, allocator, &product);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    *out_product = product;
    product = NULL;
  }

  loomc_product_release(product);
  loom_cmd_program_artifact_set_deinitialize(&artifact_set);
  return status;
}

loomc_status_t loomc_cmd_program_product_build(
    loomc_workspace_t* workspace,
    const loomc_cmd_program_product_options_t* options,
    loomc_allocator_t allocator, loomc_product_t** out_product,
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
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK,
      loomc_context_source_retention(loomc_link_index_context(
          loomc_cmd_compiler_index(options->compiler))),
      allocator, &result));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loomc_product_t* product = NULL;
  loomc_status_t status = loomc_ok_status();

  const iree_host_size_t* root_symbol_ordinals = NULL;
  iree_host_size_t root_symbol_count = 0;
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_product_resolve_roots(
        loomc_link_index_module_index(
            loomc_cmd_compiler_index(options->compiler)),
        options->root_symbol_ordinals, options->root_symbol_count,
        options->flags, &scratch_arena, &root_symbol_ordinals,
        &root_symbol_count);
  }

  if (loomc_status_is_ok(status)) {
    loomc_link_materialization_state_t materialization_state = {0};
    status = loomc_link_materialization_state_initialize(
        loomc_link_index_context(loomc_cmd_compiler_index(options->compiler)),
        workspace, loomc_cmd_compiler_index(options->compiler),
        &options->config, target_specialization, result, &scratch_arena,
        allocator, &materialization_state);
    materialization_state.specialization.deployment_profile =
        options->target_profile;
    materialization_state.specialization.deployment_phase =
        LOOMC_LINK_DEPLOYMENT_SPECIALIZATION_PHASE_COMMAND_PROJECTION;
    const loom_link_plan_materialization_environment_t
        materialization_environment =
            loomc_link_materialization_state_environment(
                &materialization_state);
    if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
      status = loomc_cmd_program_product_build_indexed(
          loomc_link_index_context(loomc_cmd_compiler_index(options->compiler)),
          workspace,
          loomc_link_index_module_index(
              loomc_cmd_compiler_index(options->compiler)),
          loomc_cmd_compiler_sources(options->compiler), root_symbol_ordinals,
          root_symbol_count, &materialization_environment,
          options->request_sink, options->artifact_flags, &scratch_arena,
          result, allocator, &product);
    }
    loomc_link_materialization_state_deinitialize(&materialization_state);
  }
  if (loomc_status_is_ok(status)) {
    *out_product = product;
    product = NULL;
    *out_result = result;
    result = NULL;
  }

  loomc_product_release(product);
  iree_arena_deinitialize(&scratch_arena);
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_cmd_program_product_build_module(
    loomc_workspace_t* workspace, const loomc_module_t* module,
    const loomc_cmd_program_module_options_t* options,
    loomc_allocator_t allocator, loomc_product_t** out_product,
    loomc_result_t** out_result) {
  if (out_product == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_product and out_result must not be NULL");
  }
  *out_product = NULL;
  *out_result = NULL;
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_product_validate_module_options(
      workspace, module, options, allocator, &target_specialization));

  loomc_context_t* context = loomc_module_context(module);
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(context), allocator,
      &result));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loom_link_module_index_t* module_index = NULL;
  loomc_product_t* product = NULL;
  const loom_link_module_index_t* library_index =
      loomc_link_index_module_index(options->library_index);
  loomc_status_t status =
      library_index != NULL
          ? loomc_status_from_iree(loom_link_module_index_allocate_overlay(
                library_index, loomc_workspace_block_pool(workspace),
                iree_allocator_from_loomc(allocator), &module_index))
          : loomc_status_from_iree(loom_link_module_index_allocate(
                loomc_context_loom_context(context),
                loomc_workspace_block_pool(workspace),
                iree_allocator_from_loomc(allocator), &module_index));
  iree_host_size_t module_provider_ordinal = IREE_HOST_SIZE_MAX;
  if (loomc_status_is_ok(status)) {
    const loom_link_module_index_add_options_t add_options = {
        .provider_name = iree_string_view_from_loomc(options->provider_name),
        .role = LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    status = loomc_status_from_iree(loom_link_module_index_add_materialized(
        module_index, loomc_module_const_loom_module(module), &add_options,
        &module_provider_ordinal));
  }

  const iree_host_size_t* explicit_root_symbol_ordinals = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_product_resolve_module_roots(
        module_index, module_provider_ordinal, options->root_symbols,
        options->root_symbol_count, &scratch_arena,
        &explicit_root_symbol_ordinals);
  }
  const iree_host_size_t* root_symbol_ordinals = NULL;
  iree_host_size_t root_symbol_count = 0;
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_product_resolve_roots(
        module_index, explicit_root_symbol_ordinals, options->root_symbol_count,
        options->flags, &scratch_arena, &root_symbol_ordinals,
        &root_symbol_count);
  }
  if (loomc_status_is_ok(status)) {
    loomc_link_materialization_state_t materialization_state = {0};
    status = loomc_link_materialization_state_initialize_overlay(
        context, workspace, options->library_index, module_provider_ordinal,
        /*appended_source=*/NULL, &options->config, target_specialization,
        result, &scratch_arena, allocator, &materialization_state);
    materialization_state.specialization.deployment_profile =
        options->target_profile;
    materialization_state.specialization.deployment_phase =
        LOOMC_LINK_DEPLOYMENT_SPECIALIZATION_PHASE_COMMAND_PROJECTION;
    const loom_link_plan_materialization_environment_t
        materialization_environment =
            loomc_link_materialization_state_environment(
                &materialization_state);
    if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
      status = loomc_cmd_program_product_build_indexed(
          context, workspace, module_index, /*native_sources=*/NULL,
          root_symbol_ordinals, root_symbol_count, &materialization_environment,
          options->request_sink, options->artifact_flags, &scratch_arena,
          result, allocator, &product);
    }
    loomc_link_materialization_state_deinitialize(&materialization_state);
  }
  if (loomc_status_is_ok(status)) {
    *out_product = product;
    product = NULL;
    *out_result = result;
    result = NULL;
  }

  loomc_product_release(product);
  loom_link_module_index_free(module_index);
  iree_arena_deinitialize(&scratch_arena);
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_cmd_program_product_build_request(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_request_t* request,
    const loomc_cmd_program_request_options_t* options,
    loomc_allocator_t allocator, loomc_product_t** out_product,
    loomc_result_t** out_result) {
  if (out_product == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_product and out_result must not be NULL");
  }
  *out_product = NULL;
  *out_result = NULL;
  const loomc_target_specialization_options_t* target_specialization = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_product_validate_request_options(
      context, workspace, request, options, allocator, &target_specialization));

  const loomc_cmd_program_request_options_t default_options = {0};
  if (options == NULL) {
    options = &default_options;
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(context), allocator,
      &result));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loomc_request_index_overlay_t request_overlay = {0};
  loomc_product_t* product = NULL;
  loomc_status_t status = loomc_request_index_overlay_initialize(
      context, loomc_workspace_block_pool(workspace), options->library_index,
      request, result, &scratch_arena, allocator, &request_overlay);

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    loomc_link_materialization_state_t materialization_state = {0};
    status = loomc_link_materialization_state_initialize_overlay(
        context, workspace, options->library_index,
        request_overlay.request_provider_ordinal, loomc_request_source(request),
        &options->config, target_specialization, result, &scratch_arena,
        allocator, &materialization_state);
    materialization_state.specialization.deployment_profile =
        options->target_profile;
    materialization_state.specialization.deployment_phase =
        LOOMC_LINK_DEPLOYMENT_SPECIALIZATION_PHASE_COMMAND_PROJECTION;
    const loom_link_plan_materialization_environment_t
        materialization_environment =
            loomc_link_materialization_state_environment(
                &materialization_state);
    if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
      status = loomc_cmd_program_product_build_indexed(
          context, workspace, request_overlay.module_index,
          /*native_sources=*/NULL, request_overlay.root_symbol_ordinals,
          request_overlay.root_symbol_count, &materialization_environment,
          options->request_sink, options->artifact_flags, &scratch_arena,
          result, allocator, &product);
    }
    loomc_link_materialization_state_deinitialize(&materialization_state);
  }
  if (loomc_status_is_ok(status)) {
    *out_product = product;
    product = NULL;
    *out_result = result;
    result = NULL;
  }

  loomc_product_release(product);
  loomc_request_index_overlay_deinitialize(&request_overlay);
  iree_arena_deinitialize(&scratch_arena);
  loomc_result_release(result);
  return status;
}

const loomc_product_descriptor_t* loomc_cmd_program_product_descriptor(void) {
  return &loomc_cmd_program_product_descriptor_;
}

loomc_host_size_t loomc_cmd_program_product_program_count(
    const loomc_product_t* base_product) {
  const loomc_cmd_program_product_impl_t* product =
      loomc_cmd_program_product_const_cast(base_product);
  return product ? product->artifact_set.programs.count : 0;
}

bool loomc_cmd_program_product_program_at(const loomc_product_t* base_product,
                                          loomc_host_size_t ordinal,
                                          loomc_cmd_program_t* out_program) {
  const loomc_cmd_program_product_impl_t* product =
      loomc_cmd_program_product_const_cast(base_product);
  if (product == NULL || out_program == NULL ||
      ordinal >= product->artifact_set.programs.count) {
    return false;
  }
  const loom_cmd_program_artifact_t* artifact =
      &product->artifact_set.programs.values[ordinal];
  const loomc_artifact_t* generic_artifact =
      loomc_product_artifact_at(base_product, ordinal);
  *out_program = (loomc_cmd_program_t){
      .symbol = loomc_string_view_from_iree(artifact->symbol),
      .artifact = *generic_artifact,
      .entry_requirement_ordinals = artifact->entry_requirement_indices,
      .entry_requirement_count = artifact->entry_requirement_count,
  };
  return true;
}

loomc_host_size_t loomc_cmd_program_product_entry_requirement_count(
    const loomc_product_t* base_product) {
  const loomc_cmd_program_product_impl_t* product =
      loomc_cmd_program_product_const_cast(base_product);
  return product ? product->artifact_set.entries.count : 0;
}

bool loomc_cmd_program_product_entry_requirement_at(
    const loomc_product_t* base_product, loomc_host_size_t ordinal,
    loomc_cmd_entry_requirement_t* out_requirement) {
  const loomc_cmd_program_product_impl_t* product =
      loomc_cmd_program_product_const_cast(base_product);
  if (product == NULL || out_requirement == NULL ||
      ordinal >= product->artifact_set.entries.count) {
    return false;
  }
  const loom_cmd_program_artifact_entry_t* entry =
      &product->artifact_set.entries.values[ordinal];
  *out_requirement = (loomc_cmd_entry_requirement_t){
      .symbol = loomc_string_view_from_iree(entry->symbol),
  };
  return true;
}
