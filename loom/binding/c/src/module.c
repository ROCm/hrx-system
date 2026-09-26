// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "module.h"

#include <stddef.h>
#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/function_version.h"
#include "loom/target/function_version_projection.h"
#include "loomc/iree.h"
#include "result.h"
#include "source.h"
#include "target.h"
#include "workspace.h"

enum {
  LOOMC_MODULE_PROJECTION_BLOCK_SIZE = 32 * 1024,
};

struct loomc_module_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release module handle storage.
  loomc_allocator_t allocator;

  // Context retained by the module handle.
  loomc_context_t* context;

  // Workspace retained for the module arena block pool.
  loomc_workspace_t* workspace;

  // Internal linked, parsed, or optimized module.
  loom_module_t* module;

  // Products retained across result release and separate artifact emission.
  struct {
    // Arena owning applied configuration and live concrete function versions.
    iree_arena_allocator_t arena;
    // Concrete versions retained while subsequent invocations transform the IR.
    loom_function_version_owner_t function_versions;
    // Applied invocation bindings with copied key/value strings.
    loomc_config_binding_list_t config_bindings;
  } compilation;

  // Physical layouts of completed Low, invalidated independently of versions.
  struct {
    // Lazily backed arena reset before subsequent compiler mutation.
    iree_arena_allocator_t arena;
    // Workgroup-only snapshots indexed by the current module's symbols.
    loom_low_workgroup_layouts_t layouts;
  } workgroup_storage;

  // Input invariants retained across successful compiler transforms. A failed
  // mutation invalidates both independently owned admission facts.
  struct {
    // Structural verification established by source admission or the binding.
    bool structural;
    // Target-Low verification against the context's immutable target tables.
    bool context_target;
  } verification;
};

typedef struct loomc_module_ir_projection_t {
  // Module to serialize, borrowing the source or owned by this projection.
  const loom_module_t* module;

  // Derived projected module, or NULL when the source is self-contained.
  loom_module_t* owned_module;

  // Block pool backing |owned_module| while it is live.
  iree_arena_block_pool_t block_pool;
} loomc_module_ir_projection_t;

typedef struct loomc_module_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;

  // Source associated with emitted diagnostics.
  const loomc_source_t* source;
} loomc_module_diagnostic_capture_t;

typedef struct loomc_module_deserialize_state_t {
  // Result receiving deserialization diagnostics.
  loomc_result_t* result;

  // Public module handle owning the IR under construction.
  loomc_module_t* module;

  // Internal module before ownership transfers to |module|.
  loom_module_t* internal_module;

  // Diagnostic count before the format decoder runs.
  loomc_host_size_t before_diagnostic_count;
} loomc_module_deserialize_state_t;

static void loomc_module_destroy(loomc_module_t* module) {
  loomc_allocator_t allocator = module->allocator;
  iree_arena_deinitialize(&module->workgroup_storage.arena);
  iree_arena_deinitialize(&module->compilation.arena);
  loom_module_free(module->module);
  loomc_workspace_release(module->workspace);
  loomc_context_release(module->context);
  loomc_allocator_free(allocator, module);
}

static loomc_string_view_t loomc_module_default_identifier(
    loomc_source_format_t format) {
  switch (format) {
    case LOOMC_SOURCE_FORMAT_TEXT:
      return loomc_make_cstring_view("module.loom");
    case LOOMC_SOURCE_FORMAT_BYTECODE:
      return loomc_make_cstring_view("module.loombc");
    default:
      return loomc_string_view_empty();
  }
}

static loomc_status_t loomc_module_ir_projection_initialize(
    const loomc_module_t* source_module,
    const loom_module_t* source_internal_module, loomc_allocator_t allocator,
    loomc_module_ir_projection_t* out_projection) {
  *out_projection = (loomc_module_ir_projection_t){
      .module = source_internal_module,
  };
  const loom_function_version_list_t* function_versions =
      loomc_module_function_versions(source_module);
  if (function_versions == NULL) {
    return loomc_ok_status();
  }

  iree_arena_block_pool_initialize(LOOMC_MODULE_PROJECTION_BLOCK_SIZE,
                                   iree_allocator_from_loomc(allocator),
                                   &out_projection->block_pool);
  loomc_status_t status =
      loomc_status_from_iree(loom_target_function_versions_project_module(
          source_internal_module, function_versions,
          &out_projection->block_pool, iree_allocator_from_loomc(allocator),
          NULL, &out_projection->owned_module));
  if (loomc_status_is_ok(status)) {
    out_projection->module = out_projection->owned_module;
  } else {
    iree_arena_block_pool_deinitialize(&out_projection->block_pool);
    *out_projection = (loomc_module_ir_projection_t){0};
  }
  return status;
}

static void loomc_module_ir_projection_deinitialize(
    loomc_module_ir_projection_t* projection) {
  if (projection->owned_module != NULL) {
    loom_module_free(projection->owned_module);
    iree_arena_block_pool_deinitialize(&projection->block_pool);
  }
  *projection = (loomc_module_ir_projection_t){0};
}

static bool loomc_module_serialize_options_has_field(
    const loomc_module_serialize_options_t* options, iree_host_size_t offset,
    iree_host_size_t length) {
  return options != NULL && options->structure_size >= offset + length;
}

#define LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options, field)  \
  loomc_module_serialize_options_has_field(                       \
      options, offsetof(loomc_module_serialize_options_t, field), \
      sizeof(((loomc_module_serialize_options_t*)0)->field))

static loomc_status_t loomc_module_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_module_validate_serialize_options(
    const loomc_module_serialize_options_t* options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "module serialize option extensions are not supported");
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_module_validate_string_view(options->identifier));
  }
  loomc_module_text_presentation_t text_presentation =
      LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT;
  if (LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options, text_presentation)) {
    text_presentation = options->text_presentation;
  }
  switch (text_presentation) {
    case LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT:
    case LOOMC_MODULE_TEXT_PRESENTATION_GENERIC:
    case LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM:
      break;
    default:
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize text_presentation is not supported");
  }

  return loomc_ok_status();
}

static loomc_status_t loomc_module_resolve_serialize_options(
    const loomc_module_serialize_options_t* options,
    loomc_source_format_t required_format,
    loomc_module_resolved_serialize_options_t* out_options) {
  LOOMC_RETURN_IF_ERROR(loomc_module_validate_serialize_options(options));

  const loomc_source_format_t option_format =
      options ? options->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (option_format != LOOMC_SOURCE_FORMAT_UNKNOWN &&
      option_format != required_format) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module serialize format contradicts the selected entry point");
  }

  loomc_string_view_t identifier =
      options ? options->identifier : loomc_string_view_empty();
  if (loomc_string_view_is_empty(identifier)) {
    identifier = loomc_module_default_identifier(required_format);
  }

  loomc_module_text_presentation_t text_presentation =
      LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT;
  if (LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options, text_presentation)) {
    text_presentation = options->text_presentation;
  }

  out_options->format = required_format;
  out_options->identifier = identifier;
  out_options->text_presentation = text_presentation;
  return loomc_ok_status();
}

loomc_status_t loomc_module_validate_deserialize_options(
    const loomc_module_deserialize_options_t* options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module deserialize options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module deserialize options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "module deserialize option extensions are not supported");
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_module_validate_string_view(options->identifier));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_module_resolve_deserialize_options(
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_source_format_t required_format,
    loomc_module_resolved_deserialize_options_t* out_options) {
  LOOMC_RETURN_IF_ERROR(loomc_module_validate_deserialize_options(options));

  const loomc_source_format_t option_format =
      options ? options->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (option_format != LOOMC_SOURCE_FORMAT_UNKNOWN &&
      option_format != required_format) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module deserialize format contradicts the selected entry point");
  }

  loomc_string_view_t identifier =
      options ? options->identifier : loomc_string_view_empty();
  if (loomc_string_view_is_empty(identifier)) {
    identifier = loomc_source_identifier(source);
  }
  if (loomc_string_view_is_empty(identifier)) {
    identifier = loomc_module_default_identifier(required_format);
  }

  out_options->format = required_format;
  out_options->identifier = identifier;
  return loomc_ok_status();
}

static iree_status_t loomc_module_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_module_diagnostic_capture_t* capture =
      (loomc_module_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      capture->result, capture->source, diagnostic));
}

static loomc_status_t loomc_module_mark_deserialize_failed(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_host_size_t before_diagnostic_count) {
  if (loomc_result_diagnostic_count(result) != before_diagnostic_count) {
    return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  loomc_status_t status = loomc_make_status(
      LOOMC_STATUS_INVALID_ARGUMENT, "module source did not deserialize");
  loomc_status_t add_status = loomc_result_add_status_diagnostic(
      result, source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("MODULE/DESERIALIZE"), status);
  loomc_status_free(status);
  LOOMC_RETURN_IF_ERROR(add_status);
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}

static loomc_status_t loomc_module_deserialize_initialize(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_allocator_t allocator, loomc_module_deserialize_state_t* out_state) {
  *out_state = (loomc_module_deserialize_state_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                            allocator, &out_state->result));
  loomc_status_t status = loomc_module_create_empty(
      context, workspace, allocator, &out_state->module);
  out_state->before_diagnostic_count =
      loomc_result_diagnostic_count(out_state->result);
  return status;
}

static loomc_status_t loomc_module_deserialize_complete(
    const loomc_source_t* source, loomc_status_t status,
    loomc_module_deserialize_state_t* state, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (loomc_status_is_ok(status) && state->internal_module == NULL) {
    status = loomc_module_mark_deserialize_failed(
        state->result, source, state->before_diagnostic_count);
  }
  if (loomc_status_is_ok(status) && state->internal_module != NULL) {
    loomc_module_set_loom_module(state->module, state->internal_module,
                                 LOOMC_MODULE_INPUT_UNVERIFIED);
    state->internal_module = NULL;
  }
  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(state->result)) {
      *out_module = state->module;
      state->module = NULL;
    }
    *out_result = state->result;
    state->result = NULL;
  }
  loom_module_free(state->internal_module);
  loomc_module_release(state->module);
  loomc_result_release(state->result);
  *state = (loomc_module_deserialize_state_t){0};
  return status;
}

loomc_status_t loomc_module_validate_source_arguments(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (out_module != NULL) {
    *out_module = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (context == NULL || workspace == NULL || source == NULL ||
      out_module == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, source, out_module, and out_result must not be "
        "NULL");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_module_deserialize_explicit_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_source_format_t required_format,
    loomc_module_source_decoder_fn_t decoder, loomc_allocator_t allocator,
    loomc_module_t** out_module, loomc_result_t** out_result) {
  LOOMC_RETURN_IF_ERROR(loomc_module_validate_source_arguments(
      context, workspace, source, out_module, out_result));

  loomc_module_resolved_deserialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_module_resolve_deserialize_options(
      source, options, required_format, &resolved_options));

  loomc_module_deserialize_state_t state;
  loomc_status_t status = loomc_module_deserialize_initialize(
      context, workspace, allocator, &state);
  if (loomc_status_is_ok(status)) {
    loomc_module_diagnostic_capture_t capture = {
        .result = state.result,
        .source = source,
    };
    const loom_diagnostic_sink_t diagnostic_sink = {
        .fn = loomc_module_capture_diagnostic,
        .user_data = &capture,
    };
    status = decoder(context, source, &resolved_options, diagnostic_sink,
                     allocator, state.module, &state.internal_module);
  }
  return loomc_module_deserialize_complete(source, status, &state, out_module,
                                           out_result);
}

static loomc_status_t loomc_module_require_internal(
    const loomc_module_t* module, const loom_module_t** out_internal_module) {
  if (module == NULL || out_internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_internal_module must not be NULL");
  }
  if (module->module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  *out_internal_module = module->module;
  return loomc_ok_status();
}

loomc_status_t loomc_module_serialize_explicit_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_source_format_t required_format,
    loomc_module_source_encoder_fn_t encoder, loomc_allocator_t allocator,
    loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;

  loomc_module_resolved_serialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_module_resolve_serialize_options(
      options, required_format, &resolved_options));
  const loom_module_t* internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(module, &internal_module));

  loomc_module_ir_projection_t projection = {0};
  LOOMC_RETURN_IF_ERROR(loomc_module_ir_projection_initialize(
      module, internal_module, allocator, &projection));
  loomc_status_t status = encoder(module, projection.module, &resolved_options,
                                  allocator, out_source);
  loomc_module_ir_projection_deinitialize(&projection);
  return status;
}

loomc_status_t loomc_module_create_empty(loomc_context_t* context,
                                         loomc_workspace_t* workspace,
                                         loomc_allocator_t allocator,
                                         loomc_module_t** out_module) {
  if (context == NULL || workspace == NULL || out_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context, workspace, and out_module must not be "
                             "NULL");
  }
  *out_module = NULL;

  loomc_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->allocator = allocator;
  module->context = context;
  loomc_context_retain(context);
  module->workspace = workspace;
  loomc_workspace_retain(workspace);
  iree_arena_initialize(loomc_workspace_block_pool(workspace),
                        &module->compilation.arena);
  iree_arena_initialize(loomc_workspace_block_pool(workspace),
                        &module->workgroup_storage.arena);
  loom_function_version_owner_initialize(
      &module->compilation.arena, &module->compilation.function_versions);

  *out_module = module;
  return loomc_ok_status();
}

loomc_allocator_t loomc_module_allocator(const loomc_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  return module->allocator;
}

loomc_context_t* loomc_module_context(const loomc_module_t* module) {
  return module ? module->context : NULL;
}

iree_arena_block_pool_t* loomc_module_block_pool(loomc_module_t* module) {
  return module ? loomc_workspace_block_pool(module->workspace) : NULL;
}

void loomc_module_set_loom_module(loomc_module_t* module,
                                  loom_module_t* internal_module,
                                  loomc_module_input_state_t input_state) {
  IREE_ASSERT(!module->module, "module storage can only be transferred once");
  module->module = internal_module;
  module->verification.structural =
      input_state == LOOMC_MODULE_INPUT_STRUCTURALLY_VERIFIED;
}

loom_module_t* loomc_module_loom_module(loomc_module_t* module) {
  return module ? module->module : NULL;
}

const loom_module_t* loomc_module_const_loom_module(
    const loomc_module_t* module) {
  return module ? module->module : NULL;
}

typedef struct loomc_module_verify_capture_t {
  // Result receiving verification diagnostics.
  loomc_result_t* result;
  // Borrowed module owning operation locations during verification.
  const loom_module_t* module;
} loomc_module_verify_capture_t;

static iree_status_t loomc_module_capture_verify_emission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  const loomc_module_verify_capture_t* capture = user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      capture->result, capture->module, LOOM_EMITTER_VERIFIER, emission));
}

loomc_status_t loomc_module_verify(
    loomc_module_t* module,
    const loomc_target_environment_t* target_environment,
    loomc_result_t* result) {
  const bool context_environment =
      target_environment == loomc_context_target_environment(module->context);
  if (module->verification.context_target && context_environment) {
    return loomc_ok_status();
  }

  if (!module->verification.structural) {
    LOOMC_RETURN_IF_ERROR(
        loomc_result_verify_loom_module(module->module, result));
    if (!loomc_result_succeeded(result)) {
      return loomc_ok_status();
    }
    module->verification.structural = true;
  }

  const loomc_target_pass_environment_t* pass_environment =
      loomc_target_environment_pass_environment(target_environment);
  loomc_module_verify_capture_t capture = {.result = result,
                                           .module = module->module};
  const loom_low_verify_options_t options = {
      .descriptor_registry =
          pass_environment ? &pass_environment->low_descriptor_registry.registry
                           : NULL,
      .function_versions = loomc_module_function_versions(module),
      .emitter = {.fn = loomc_module_capture_verify_emission,
                  .user_data = &capture},
      .provider_list = pass_environment
                           ? loom_target_environment_low_verify_provider_list(
                                 pass_environment->target_environment)
                           : loom_low_verify_provider_list_empty(),
      .max_errors = 20,
  };
  loom_low_verify_scratch_t scratch =
      loom_low_verify_scratch_for_module(module->module);
  loom_low_verify_result_t verify_result = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_low_verify_module(
      module->module, &options, &scratch, &verify_result)));
  if (verify_result.error_count != 0) {
    return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  if (context_environment) {
    module->verification.context_target = true;
  }
  return loomc_ok_status();
}

void loomc_module_invalidate_verification(loomc_module_t* module) {
  module->verification.structural = false;
  module->verification.context_target = false;
}

loom_function_version_owner_t* loomc_module_function_version_owner(
    loomc_module_t* module) {
  return &module->compilation.function_versions;
}

void loomc_module_invalidate_compilation(loomc_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  loomc_module_invalidate_workgroup_layouts(module);
  iree_arena_reset(&module->compilation.arena);
  module->compilation.config_bindings = (loomc_config_binding_list_t){0};
  loom_function_version_owner_initialize(
      &module->compilation.arena, &module->compilation.function_versions);
}

iree_arena_allocator_t* loomc_module_workgroup_layout_arena(
    loomc_module_t* module) {
  return &module->workgroup_storage.arena;
}

loom_low_workgroup_layouts_t* loomc_module_mutable_workgroup_layouts(
    loomc_module_t* module) {
  return &module->workgroup_storage.layouts;
}

const loom_low_workgroup_layouts_t* loomc_module_workgroup_layouts(
    const loomc_module_t* module) {
  return module->workgroup_storage.layouts.count != 0
             ? &module->workgroup_storage.layouts
             : NULL;
}

void loomc_module_invalidate_workgroup_layouts(loomc_module_t* module) {
  iree_arena_reset(&module->workgroup_storage.arena);
  module->workgroup_storage.layouts = (loom_low_workgroup_layouts_t){0};
}

static iree_status_t loomc_module_record_config_binding(
    void* user_data, const loom_tooling_config_binding_t* binding) {
  loomc_module_t* module = (loomc_module_t*)user_data;
  return loomc_config_binding_list_append(&module->compilation.config_bindings,
                                          binding, &module->compilation.arena);
}

loom_tooling_config_binding_sink_t loomc_module_config_binding_sink(
    loomc_module_t* module) {
  return (loom_tooling_config_binding_sink_t){
      .fn = loomc_module_record_config_binding,
      .user_data = module,
  };
}

const loomc_config_binding_list_t* loomc_module_config_bindings(
    const loomc_module_t* module) {
  return &module->compilation.config_bindings;
}

const loom_function_version_list_t* loomc_module_function_versions(
    const loomc_module_t* module) {
  return module && module->compilation.function_versions.list.count != 0
             ? &module->compilation.function_versions.list
             : NULL;
}

void loomc_module_retain(loomc_module_t* module) {
  if (module == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&module->ref_count);
}

void loomc_module_release(loomc_module_t* module) {
  if (module == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    loomc_module_destroy(module);
  }
}

loomc_status_t loomc_module_clone(const loomc_module_t* source_module,
                                  loomc_workspace_t* workspace,
                                  loomc_allocator_t allocator,
                                  loomc_module_t** out_module) {
  if (out_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_module must not be NULL");
  }
  *out_module = NULL;
  if (workspace == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "workspace must not be NULL");
  }

  const loom_module_t* source_internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(source_module, &source_internal_module));

  loomc_module_t* module = NULL;
  loom_module_t* cloned_internal_module = NULL;
  loomc_status_t status = loomc_module_create_empty(
      source_module->context, workspace, allocator, &module);
  if (loomc_status_is_ok(status)) {
    status =
        loomc_status_from_iree(loom_target_function_versions_project_module(
            source_internal_module,
            loomc_module_function_versions(source_module),
            loomc_module_block_pool(module),
            iree_allocator_from_loomc(allocator),
            &module->compilation.function_versions, &cloned_internal_module));
  }
  if (loomc_status_is_ok(status)) {
    loomc_module_set_loom_module(module, cloned_internal_module,
                                 LOOMC_MODULE_INPUT_UNVERIFIED);
    cloned_internal_module = NULL;
  }
  for (const loomc_config_binding_record_t* binding =
           source_module->compilation.config_bindings.head;
       binding != NULL && loomc_status_is_ok(status); binding = binding->next) {
    status = loomc_status_from_iree(loomc_config_binding_list_append(
        &module->compilation.config_bindings, &binding->binding,
        &module->compilation.arena));
  }
  if (loomc_status_is_ok(status)) {
    *out_module = module;
    module = NULL;
  }

  loom_module_free(cloned_internal_module);
  loomc_module_release(module);
  return status;
}
