// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-analyze/session.h"

#include <string.h>

#include "loom/tooling/cli/loomc_result.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/target/configured_ir.h"

struct loom_analyze_session_t {
  iree_allocator_t allocator;
  loomc_context_t* context;
  loomc_workspace_t* workspace;
};

iree_status_t loom_analyze_session_create(
    iree_allocator_t allocator, loom_analyze_session_t** out_session) {
  *out_session = NULL;
  loom_analyze_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*session), (void**)&session));
  session->allocator = allocator;
  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(allocator);
  loomc_target_environment_t* environment = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_configured_ir(
          compiler_allocator, &environment));
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(loomc_context_create(
        &context_options, compiler_allocator, &session->context));
  }
  loomc_target_environment_release(environment);
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_workspace_create(NULL, compiler_allocator, &session->workspace));
  }
  if (!iree_status_is_ok(status)) {
    loom_analyze_session_free(session);
    return status;
  }
  *out_session = session;
  return iree_ok_status();
}

void loom_analyze_session_free(loom_analyze_session_t* session) {
  if (!session) {
    return;
  }
  loomc_workspace_release(session->workspace);
  loomc_context_release(session->context);
  iree_allocator_free(session->allocator, session);
}

// Prints |result| to |diagnostics|, or only reports acceptance when NULL.
static iree_status_t loom_analyze_session_consume_result(loomc_result_t* result,
                                                         FILE* diagnostics,
                                                         bool* out_accepted) {
  if (diagnostics) {
    return loom_tooling_cli_print_loomc_result(diagnostics, result,
                                               out_accepted);
  }
  *out_accepted = loomc_result_succeeded(result);
  return iree_ok_status();
}

iree_status_t loom_analyze_session_load(loom_analyze_session_t* session,
                                        iree_string_view_t filename,
                                        iree_const_byte_span_t contents,
                                        FILE* diagnostics,
                                        loomc_module_t** out_module,
                                        const loom_module_t** out_native_module,
                                        bool* out_accepted) {
  *out_module = NULL;
  *out_native_module = NULL;
  *out_accepted = false;
  const loomc_allocator_t compiler_allocator =
      loomc_allocator_from_iree(session->allocator);
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_string_view_from_iree(filename),
      .contents = loomc_byte_span_from_iree(contents),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_source_create(&source_options, compiler_allocator, &source)));
  loomc_result_t* result = NULL;
  bool accepted = false;
  iree_status_t status =
      iree_status_from_loomc(loomc_module_deserialize_from_source(
          session->context, session->workspace, source, NULL,
          compiler_allocator, out_module, &result));
  loomc_source_release(source);
  if (iree_status_is_ok(status)) {
    status =
        loom_analyze_session_consume_result(result, diagnostics, &accepted);
  }
  loomc_result_release(result);
  result = NULL;

  // The interop view verifies the module and exposes native IR read-only.
  loomc_module_interop_view_t view;
  memset(&view, 0, sizeof(view));
  if (iree_status_is_ok(status) && accepted) {
    status = iree_status_from_loomc(loomc_module_get_interop_view(
        *out_module, compiler_allocator, &view, &result));
    if (iree_status_is_ok(status)) {
      status =
          loom_analyze_session_consume_result(result, diagnostics, &accepted);
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status) && accepted) {
    *out_native_module = view.module;
    *out_accepted = true;
  }
  return status;
}
