// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/link_dependency.h"

#include "iree/base/internal/arena.h"
#include "link_index.h"
#include "loom/link/dependency_analysis.h"
#include "loom/link/dependency_report.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "result.h"
#include "workspace.h"

enum {
  LOOMC_LINK_DEPENDENCY_KNOWN_ARTIFACT_FLAGS =
      LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON,
};

static loomc_status_t loomc_link_dependency_validate_options(
    const loomc_link_dependency_analysis_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "link dependency analysis options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "link dependency analysis options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "link dependency analysis option extensions are not supported");
  }
  if (options->direct_provider_count != 0 &&
      options->direct_provider_ordinals == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "direct_provider_count is non-zero but direct_provider_ordinals is "
        "NULL");
  }
  if (options->component_name.data == NULL &&
      options->component_name.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "component_name has length but no data");
  }
  if ((options->artifact_flags & ~LOOMC_LINK_DEPENDENCY_KNOWN_ARTIFACT_FLAGS) !=
      0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "link dependency analysis options contain unknown artifact flag bits");
  }
  if (options->report_identifier.data == NULL &&
      options->report_identifier.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "report_identifier has length but no data");
  }
  if (!loomc_string_view_is_empty(options->report_identifier) &&
      !iree_any_bit_set(options->artifact_flags,
                        LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "report_identifier requires the report JSON artifact flag");
  }
  return loomc_ok_status();
}

static const loomc_source_t* loomc_link_dependency_requirement_source(
    const loomc_link_index_t* link_index,
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement) {
  const loom_link_module_index_symbol_t* source_symbol = NULL;
  if (requirement->first_source_symbol_ordinal !=
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    source_symbol = loom_link_module_index_symbol_at(
        analysis->index, requirement->first_source_symbol_ordinal);
  } else if (requirement->kind ==
             LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
    source_symbol = loom_link_module_index_symbol_at(
        analysis->index, requirement->target.symbol_ordinal);
  }
  if (source_symbol == NULL) {
    return NULL;
  }
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(analysis->index, source_symbol);
  return loomc_link_index_source_for_provider(link_index, provider->ordinal);
}

static loomc_status_t loomc_link_dependency_add_diagnostics(
    const loomc_link_index_t* link_index,
    const loom_link_dependency_analysis_t* analysis,
    loomc_string_view_t component_name, loomc_result_t* result) {
  iree_string_builder_t message_builder;
  iree_string_builder_initialize(
      iree_allocator_from_loomc(loomc_result_allocator(result)),
      &message_builder);
  loom_output_stream_t message_stream;
  loom_output_stream_for_builder(&message_builder, &message_stream);

  loomc_status_t status = loomc_ok_status();
  for (iree_host_size_t i = 0;
       loomc_status_is_ok(status) && i < analysis->requirements.count; ++i) {
    const loom_link_dependency_requirement_t* requirement =
        &analysis->requirements.values[i];
    if (loom_link_dependency_requirement_satisfied(requirement)) {
      continue;
    }
    iree_string_builder_reset(&message_builder);
    status = loomc_status_from_iree(loom_link_dependency_format_diagnostic(
        analysis, requirement, iree_string_view_from_loomc(component_name),
        &message_stream));
    if (!loomc_status_is_ok(status)) {
      break;
    }
    const loomc_diagnostic_t diagnostic = {
        .severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        .code = loomc_string_view_from_iree(
            loom_link_dependency_diagnostic_code(requirement)),
        .message = loomc_string_view_from_iree(
            iree_string_builder_view(&message_builder)),
        .range =
            {
                .source = loomc_link_dependency_requirement_source(
                    link_index, analysis, requirement),
            },
    };
    status = loomc_result_add_diagnostic(result, &diagnostic);
  }

  iree_string_builder_deinitialize(&message_builder);
  return status;
}

static loomc_status_t loomc_link_dependency_add_report(
    const loom_link_dependency_analysis_t* analysis,
    loomc_string_view_t component_name, loomc_string_view_t identifier,
    loomc_result_t* result) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  iree_string_builder_t report_builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &report_builder);
  loom_output_stream_t report_stream;
  loom_output_stream_for_builder(&report_builder, &report_stream);
  loomc_status_t status =
      loomc_status_from_iree(loom_link_dependency_format_json(
          analysis, iree_string_view_from_loomc(component_name),
          &report_stream));

  char* report_storage = NULL;
  iree_host_size_t report_length = 0;
  if (loomc_status_is_ok(status)) {
    report_length = iree_string_builder_size(&report_builder);
    report_storage = iree_string_builder_take_storage(&report_builder);
  }
  if (loomc_string_view_is_empty(identifier)) {
    identifier = loomc_make_cstring_view("dependency-report.json");
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_REPORT,
        loomc_make_cstring_view(
            LOOMC_ARTIFACT_FORMAT_LINK_DEPENDENCY_REPORT_JSON),
        identifier, loomc_make_byte_span(report_storage, report_length));
  }
  if (loomc_status_is_ok(status)) {
    report_storage = NULL;
  }

  loomc_allocator_free(allocator, report_storage);
  iree_string_builder_deinitialize(&report_builder);
  return status;
}

loomc_status_t loomc_link_analyze_dependencies(
    const loomc_link_index_t* link_index, loomc_workspace_t* workspace,
    const loomc_link_dependency_analysis_options_t* options,
    loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (link_index == NULL || workspace == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link_index and workspace must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_link_dependency_validate_options(options));

  const loom_link_dependency_analysis_options_t internal_options = {
      .direct_provider_ordinals =
          options ? options->direct_provider_ordinals : NULL,
      .direct_provider_count = options ? options->direct_provider_count : 0,
  };
  loomc_allocator_t allocator = loomc_link_index_allocator(link_index);
  iree_arena_allocator_t arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &arena);
  loom_link_dependency_analysis_t analysis = {0};
  loomc_status_t status = loomc_status_from_iree(loom_link_dependency_analyze(
      loomc_link_index_module_index(link_index), &internal_options,
      loomc_workspace_block_pool(workspace), &arena,
      iree_allocator_from_loomc(allocator), &analysis));

  loomc_result_t* result = NULL;
  if (loomc_status_is_ok(status)) {
    const loomc_result_state_t state =
        loom_link_dependency_analysis_succeeded(&analysis)
            ? LOOMC_RESULT_STATE_SUCCEEDED
            : LOOMC_RESULT_STATE_FAILED;
    status = loomc_result_create(state, allocator, &result);
  }
  const loomc_string_view_t component_name =
      options ? options->component_name : loomc_string_view_empty();
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
    status = loomc_link_dependency_add_diagnostics(link_index, &analysis,
                                                   component_name, result);
  }
  if (loomc_status_is_ok(status) && options != NULL &&
      iree_any_bit_set(options->artifact_flags,
                       LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON)) {
    status = loomc_link_dependency_add_report(
        &analysis, component_name, options->report_identifier, result);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }

  loomc_result_release(result);
  iree_arena_deinitialize(&arena);
  return status;
}
