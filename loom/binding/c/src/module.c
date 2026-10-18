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
#include "loom/format/text/printer.h"
#include "loom/ir/function_version.h"
#include "loom/target/function_version_projection.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "option_chain.h"
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

  // Exact source snapshots indexed by the active module's source IDs.
  loom_source_storage_t sources;

  // Products retained across result release and separate artifact emission.
  struct {
    // Arena owning applied configuration and live concrete function versions.
    iree_arena_allocator_t arena;
    // Concrete versions retained while subsequent invocations transform the IR.
    loom_function_version_owner_t function_versions;
    // Applied invocation bindings with copied key/value strings.
    loomc_config_binding_list_t config_bindings;
  } compilation;

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

static loomc_status_t loomc_module_capture_source(
    loomc_module_t* module, loomc_string_view_t identifier,
    loomc_byte_span_t contents) {
  if (loomc_string_view_is_empty(identifier)) {
    return loomc_ok_status();
  }
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_module_register_source(
      module->module, iree_string_view_from_loomc(identifier), &source_id)));
  return loomc_module_insert_source_snapshot(
      module, source_id, iree_string_view_from_loomc(identifier),
      iree_make_string_view((const char*)contents.data, contents.data_length));
}

static void loomc_module_destroy(loomc_module_t* module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loomc_allocator_t allocator = module->allocator;
  loom_source_storage_deinitialize(&module->sources);
  iree_arena_deinitialize(&module->compilation.arena);
  loom_module_free(module->module);
  loomc_workspace_release(module->workspace);
  loomc_context_release(module->context);
  loomc_allocator_free(allocator, module);
  IREE_TRACE_ZONE_END(z0);
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

static loomc_status_t loomc_module_mark_deserialize_failed(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_host_size_t before_diagnostic_count) {
  if (loomc_result_diagnostic_count(result) != before_diagnostic_count) {
    loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
    return loomc_ok_status();
  }
  const loomc_diagnostic_t diagnostic = {
      .severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      .code = loomc_make_cstring_view("MODULE/DESERIALIZE"),
      .message = loomc_make_cstring_view("module source did not deserialize"),
      .range = {.source = source},
  };
  LOOMC_RETURN_IF_ERROR(loomc_result_add_diagnostic(result, &diagnostic));
  loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
  return loomc_ok_status();
}

static loomc_status_t loomc_module_deserialize_initialize(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_allocator_t allocator, loomc_module_deserialize_state_t* out_state) {
  *out_state = (loomc_module_deserialize_state_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(context), allocator,
      &out_state->result));
  loomc_status_t status = loomc_module_create_empty(
      context, workspace, allocator, &out_state->module);
  out_state->before_diagnostic_count =
      loomc_result_diagnostic_count(out_state->result);
  return status;
}

static loomc_status_t loomc_module_deserialize_complete(
    const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loomc_status_t status, loomc_module_deserialize_state_t* state,
    loomc_module_t** out_module, loomc_result_t** out_result) {
  if (loomc_status_is_ok(status) && state->internal_module == NULL) {
    status = loomc_module_mark_deserialize_failed(
        state->result, source, state->before_diagnostic_count);
  }
  if (loomc_status_is_ok(status) && state->internal_module != NULL) {
    loomc_module_set_loom_module(state->module, state->internal_module,
                                 LOOMC_MODULE_INPUT_UNVERIFIED);
    state->internal_module = NULL;
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(state->result) &&
      options->format == LOOMC_SOURCE_FORMAT_TEXT) {
    status = loomc_module_capture_source(state->module, options->identifier,
                                         loomc_source_contents(source));
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
    loomc_diagnostic_capture_t capture;
    loomc_diagnostic_capture_initialize(state.result, source, /*module=*/NULL,
                                        (loom_source_resolver_t){0},
                                        LOOM_EMITTER_PARSER,
                                        /*text_print_options=*/NULL, &capture);
    const loom_diagnostic_sink_t diagnostic_sink = {
        .fn = loomc_diagnostic_capture,
        .user_data = &capture,
    };
    status = decoder(context, source, &resolved_options, diagnostic_sink,
                     allocator, state.module, &state.internal_module);
  }
  return loomc_module_deserialize_complete(source, &resolved_options, status,
                                           &state, out_module, out_result);
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
  IREE_TRACE_ZONE_BEGIN(z0);
  if (context == NULL || workspace == NULL || out_module == NULL) {
    loomc_status_t return_status =
        loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                          "context, workspace, and out_module must not be "
                          "NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  *out_module = NULL;

  loomc_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->allocator = allocator;
  module->context = context;
  loomc_context_retain(context);
  module->workspace = workspace;
  loomc_workspace_retain(workspace);
  loom_source_storage_initialize(iree_allocator_from_loomc(allocator),
                                 &module->sources);
  iree_arena_initialize(loomc_workspace_block_pool(workspace),
                        &module->compilation.arena);
  loom_function_version_owner_initialize(
      &module->compilation.arena, &module->compilation.function_versions);

  *out_module = module;
  IREE_TRACE_ZONE_END(z0);
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

loomc_status_t loomc_module_insert_source_snapshot(loomc_module_t* module,
                                                   loom_source_id_t source_id,
                                                   iree_string_view_t filename,
                                                   iree_string_view_t source) {
  IREE_ASSERT_ARGUMENT(module);
  if (loomc_context_source_retention(module->context) ==
      LOOMC_SOURCE_RETENTION_METADATA_ONLY) {
    return loomc_ok_status();
  }
  return loomc_status_from_iree(loom_source_storage_insert(
      &module->sources, source_id, filename, source));
}

void loomc_module_set_loom_module(loomc_module_t* module,
                                  loom_module_t* internal_module,
                                  loomc_module_input_state_t input_state) {
  IREE_ASSERT(!module->module, "module storage can only be transferred once");
  loomc_module_adopt_loom_module_replacement(module, internal_module,
                                             input_state);
}

void loomc_module_adopt_loom_module_replacement(
    loomc_module_t* module, loom_module_t* internal_module,
    loomc_module_input_state_t input_state) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(internal_module);
  module->module = internal_module;
  module->sources.table.module = internal_module;
  module->verification.structural =
      input_state == LOOMC_MODULE_INPUT_STRUCTURALLY_VERIFIED;
  module->verification.context_target = false;
}

loom_module_t* loomc_module_loom_module(loomc_module_t* module) {
  return module ? module->module : NULL;
}

const loom_module_t* loomc_module_const_loom_module(
    const loomc_module_t* module) {
  return module ? module->module : NULL;
}

loom_source_resolver_t loomc_module_source_resolver(
    const loomc_module_t* module) {
  return module ? loom_source_storage_resolver(&module->sources)
                : (loom_source_resolver_t){0};
}

const loom_source_table_resolver_t* loomc_module_source_table(
    const loomc_module_t* module) {
  return module ? &module->sources.table : NULL;
}

loomc_status_t loomc_module_replace_source_table(
    loomc_module_t* module, const loom_module_t* internal_module,
    const loom_source_table_resolver_t* source_table) {
  if (loomc_context_source_retention(module->context) ==
      LOOMC_SOURCE_RETENTION_METADATA_ONLY) {
    loomc_module_clear_sources(module, internal_module);
    return loomc_ok_status();
  }
  if (source_table->entries == module->sources.table.entries) {
    module->sources.table.module = internal_module;
    module->sources.table.count = source_table->count;
    return loomc_ok_status();
  }
  loom_source_storage_t replacement;
  loom_source_storage_initialize(module->sources.allocator, &replacement);
  replacement.table.module = internal_module;
  loomc_status_t status = loomc_ok_status();
  for (iree_host_size_t i = 0;
       i < source_table->count && loomc_status_is_ok(status); ++i) {
    const loom_source_entry_t* entry = &source_table->entries[i];
    if (entry->source_id == LOOM_SOURCE_ID_INVALID) {
      continue;
    }
    status = loomc_status_from_iree(loom_source_storage_insert(
        &replacement, entry->source_id, entry->filename, entry->source));
  }
  if (loomc_status_is_ok(status)) {
    loom_source_storage_deinitialize(&module->sources);
    module->sources = replacement;
    replacement = (loom_source_storage_t){0};
  }
  loom_source_storage_deinitialize(&replacement);
  return status;
}

void loomc_module_clear_sources(loomc_module_t* module,
                                const loom_module_t* internal_module) {
  loom_source_storage_deinitialize(&module->sources);
  loom_source_storage_initialize(iree_allocator_from_loomc(module->allocator),
                                 &module->sources);
  module->sources.table.module = internal_module;
}

loomc_status_t loomc_module_admit_structural(loomc_module_t* module,
                                             loomc_result_t* result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (module->verification.structural) {
    IREE_TRACE_ZONE_END(z0);
    return loomc_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_result_verify_loom_module(module->module,
                                          loomc_module_source_resolver(module),
                                          /*max_errors=*/0, result));
  if (loomc_result_succeeded(result)) {
    module->verification.structural = true;
  }
  IREE_TRACE_ZONE_END(z0);
  return loomc_ok_status();
}

static loomc_status_t loomc_module_verify_target_into_result(
    loomc_module_t* module,
    const loomc_target_environment_t* target_environment, uint32_t max_errors,
    loomc_result_t* result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const loomc_target_pass_environment_t* pass_environment =
      loomc_target_environment_pass_environment(target_environment);
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(
      result, /*source=*/NULL, module->module,
      loomc_module_source_resolver(module), LOOM_EMITTER_VERIFIER,
      pass_environment ? &pass_environment->diagnostic_type_print_options
                       : NULL,
      &capture);
  const loom_low_verify_options_t options = {
      .descriptor_registry =
          pass_environment ? &pass_environment->low_descriptor_registry.registry
                           : NULL,
      .function_versions = loomc_module_function_versions(module),
      .emitter = {.fn = loomc_diagnostic_capture_emission,
                  .user_data = &capture},
      .provider_list = pass_environment
                           ? loom_target_environment_low_verify_provider_list(
                                 pass_environment->target_environment)
                           : loom_low_verify_provider_list_empty(),
      .max_errors = max_errors ? max_errors : 20,
  };
  loom_low_verify_scratch_t scratch =
      loom_low_verify_scratch_for_module(module->module);
  loom_low_verify_result_t verify_result = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loomc_status_from_iree(loom_low_verify_module(
              module->module, &options, &scratch, &verify_result)));
  if (verify_result.error_count != 0) {
    loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
    IREE_TRACE_ZONE_END(z0);
    return loomc_ok_status();
  }
  if (target_environment == loomc_context_target_environment(module->context)) {
    module->verification.context_target = true;
  }
  IREE_TRACE_ZONE_END(z0);
  return loomc_ok_status();
}

loomc_status_t loomc_module_admit(
    loomc_module_t* module,
    const loomc_target_environment_t* target_environment,
    loomc_result_t* result) {
  if (module->verification.context_target &&
      target_environment == loomc_context_target_environment(module->context)) {
    return loomc_ok_status();
  }
  LOOMC_RETURN_IF_ERROR(loomc_module_admit_structural(module, result));
  return loomc_result_succeeded(result)
             ? loomc_module_verify_target_into_result(
                   module, target_environment, /*max_errors=*/0, result)
             : loomc_ok_status();
}

loomc_status_t loomc_module_verify(loomc_module_t* module,
                                   const loomc_module_verify_options_t* options,
                                   loomc_allocator_t allocator,
                                   loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module verify options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module verify options structure_size is too small");
    }
    loomc_option_chain_t option_chain = {0};
    LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
        options->next, /*allowed_options=*/0, &option_chain));
  }
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(module->context),
      allocator, &result));
  const uint32_t max_errors = options ? options->max_errors : 0;
  loomc_module_invalidate_verification(module);
  loomc_status_t status = loomc_result_verify_loom_module(
      module->module, loomc_module_source_resolver(module), max_errors, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    module->verification.structural = true;
    status = loomc_module_verify_target_into_result(
        module, loomc_context_target_environment(module->context), max_errors,
        result);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

void loomc_module_invalidate_verification(loomc_module_t* module) {
  module->verification.structural = false;
  module->verification.context_target = false;
}

void loomc_module_invalidate_target_verification(loomc_module_t* module) {
  module->verification.context_target = false;
}

loom_function_version_owner_t* loomc_module_function_version_owner(
    loomc_module_t* module) {
  return &module->compilation.function_versions;
}

void loomc_module_invalidate_compilation(loomc_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  iree_arena_reset(&module->compilation.arena);
  module->compilation.config_bindings = (loomc_config_binding_list_t){0};
  loom_function_version_owner_initialize(
      &module->compilation.arena, &module->compilation.function_versions);
}

static iree_status_t loomc_module_record_config_value(
    void* user_data, const loom_config_applied_value_t* applied_value) {
  loomc_module_t* module = (loomc_module_t*)user_data;
  iree_string_builder_t builder;
  iree_string_builder_initialize(applied_value->module->allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_text_print_attribute(
      &applied_value->value, applied_value->module, &stream);
  if (iree_status_is_ok(status)) {
    status = loomc_config_binding_list_append(
        &module->compilation.config_bindings, applied_value->key,
        iree_string_builder_view(&builder), &module->compilation.arena);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

loom_config_applied_value_sink_t loomc_module_config_applied_value_sink(
    loomc_module_t* module) {
  return (loom_config_applied_value_sink_t){
      .fn = loomc_module_record_config_value,
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
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_module == NULL) {
    loomc_status_t return_status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT, "out_module must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }
  *out_module = NULL;
  if (workspace == NULL) {
    loomc_status_t return_status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT, "workspace must not be NULL");
    IREE_TRACE_ZONE_END(z0);
    return return_status;
  }

  const loom_module_t* source_internal_module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
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
    status = loomc_module_replace_source_table(
        module, cloned_internal_module,
        loomc_module_source_table(source_module));
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
        &module->compilation.config_bindings, binding->key, binding->value,
        &module->compilation.arena));
  }
  if (loomc_status_is_ok(status)) {
    *out_module = module;
    module = NULL;
  }

  loom_module_free(cloned_internal_module);
  loomc_module_release(module);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
