// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Public module ownership and compiler invocation for checker modes.

#ifndef LOOM_TOOLS_LOOM_CHECK_COMPILE_H_
#define LOOM_TOOLS_LOOM_CHECK_COMPILE_H_

#include "loom/sanitizer/options.h"
#include "loom/tooling/input/loomc.h"
#include "loom/tools/loom-check/execute.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/sanitizer.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler integrations supplied by the final loom-check binary. The generic
// runner does not choose a configured target set or optional source frontend.
typedef struct loom_check_compile_provider_t {
  // Optional foreign-source importer dispatch.
  loom_tooling_input_import_loomc_fn_t import;
  // Opaque state forwarded to |import|.
  void* import_user_data;
} loom_check_compile_provider_t;

// Lazily prepared public compiler state shared by checker cases. The runner
// supplies the retained target environment; the first use prepares the context,
// workspace, and compiler together.
typedef struct loom_check_compile_session_t {
  // Binary-owned compiler integrations used when the session is prepared.
  const loom_check_compile_provider_t* provider;
  // Host allocator used for every session-owned public handle.
  iree_allocator_t host_allocator;
  // Retained public target environment composed by the checker runner.
  loomc_target_environment_t* target_environment;
  // Public context, or NULL before the first use.
  loomc_context_t* context;
  // Reusable invocation workspace, or NULL before the first use.
  loomc_workspace_t* workspace;
  // Prepared compiler, or NULL before the first use.
  loomc_compiler_t* compiler;
  // Cached empty, source-to-Low CFG, and source-to-Low structured programs.
  loomc_pass_program_t* artifact_pass_programs[3];
} loom_check_compile_session_t;

// Prepares the session's context, workspace, and compiler on first use.
iree_status_t loom_check_compile_session_prepare(
    loom_check_compile_session_t* session);

typedef struct loom_check_compile_options_t {
  // Lazily prepared public compiler state owned by the runner.
  loom_check_compile_session_t* session;
  // Selected compiler profile.
  loomc_target_profile_t* target_profile;
  // Compile-time bindings applied while preparing the artifact request.
  const loomc_config_options_t* config;
  // Instrumentation attached to the default target pass program, or NULL.
  const loomc_sanitizer_options_t* sanitizer;
} loom_check_compile_options_t;

// Public artifact compile request used by emit providers.
typedef struct loom_check_compile_artifact_options_t {
  // Public artifact format selecting the target emitter.
  iree_string_view_t artifact_format;
  // Optional explicit root function symbol.
  iree_string_view_t root;
  // Optional public target profile borrowed until artifact compilation returns.
  loomc_target_profile_t* target_profile;
  // Whether to lower source IR to Low before target emission.
  bool lower_source_to_low;
  // Source-to-Low control-flow shape when lowering source IR.
  loomc_target_control_flow_lowering_t control_flow_lowering;
  // Optional public target emission report requested with the artifact.
  const loomc_compile_report_options_t* report;
} loom_check_compile_artifact_options_t;

// One source-to-Low checker compilation request.
typedef struct loom_check_compile_source_low_options_t {
  // Pass-program boundary to prepare and execute.
  loomc_target_pipeline_kind_t pipeline_kind;
  // Source-to-Low legality diagnostics requested by the checker case.
  loomc_source_to_low_diagnostic_flags_t diagnostic_flags;
  // Control-flow representation selected for lowering.
  loomc_target_control_flow_lowering_t control_flow_lowering;
  // Sanitizer instrumentation and reporting policy.
  loom_sanitizer_options_t sanitizer;
  // Optional function symbol to specialize.
  iree_string_view_t function_name;
  // Optional public target profile borrowed until source lowering returns.
  loomc_target_profile_t* target_profile;
  // Optional public compile report artifact requested from the pipeline.
  const loomc_compile_report_options_t* report;
} loom_check_compile_source_low_options_t;

// Releases all public compiler state prepared by |session|.
void loom_check_compile_session_deinitialize(
    loom_check_compile_session_t* session);

// Selects a complete target profile from the prepared session environment.
iree_status_t loom_check_compile_session_select_target_profile(
    loom_check_compile_session_t* session, iree_string_view_t specification,
    loomc_target_profile_t** out_target_profile);

// Admits one checker case through the shared public compiler session. Public
// diagnostics are appended to |diagnostic_collector|. |max_errors| limits text
// parser recovery; zero selects the parser default. A successful module is
// owned by the caller and must be released with loomc_module_release.
// |filename| is the reader's logical source identity, already path-mapped.
// Foreign importers receive |input_request->path| for include resolution and
// apply the request's path policy to sources they discover.
iree_status_t loom_check_compile_admit_module(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    loom_check_compile_session_t* session,
    const loom_check_environment_t* environment, uint32_t max_errors,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    loomc_module_t** out_module);

// Selects the explicit function name, sole definition, or unique public/kernel
// entry among private helpers. The returned name borrows module storage.
iree_status_t loom_check_compile_select_source_function(
    const loomc_module_t* module, iree_string_view_t function_name,
    loom_check_diagnostic_collector_t* collector,
    iree_string_view_t* out_function_name);

// Compiles the request's admitted public module to target artifacts. Public
// diagnostics are appended to |request->diagnostic_collector|. On success,
// |out_result| owns the artifacts until loomc_result_release. Compiler
// rejection with error diagnostics retains the result when it carries artifacts
// such as a failure report; otherwise it leaves NULL for annotation matching.
// Rejection without an error diagnostic fails. Consumers select and map only
// the artifacts they inspect, independently of diagnostic matching.
iree_status_t loom_check_compile_artifact(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_artifact_options_t* options,
    loomc_result_t** out_result);

// Emits artifacts from the request's prepared public module without running
// passes or materializing roots. Result ownership and diagnostic handling are
// the same as loom_check_compile_artifact.
iree_status_t loom_check_emit_artifact(
    const loom_check_emit_provider_request_t* request,
    const loomc_emit_options_t* options, loomc_result_t** out_result);

// Selects a required output from a compiler result by kind and
// format. The returned artifact borrows |result|. Missing requested output is
// a check failure, including optional outputs unsupported by an emitter.
iree_status_t loom_check_compile_select_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    iree_string_view_t format, const loomc_artifact_t** out_artifact);

// Lowers the request's admitted public module through LoomC. User diagnostics
// leave |out_compiled| false and return OK for normal annotation matching.
// When non-NULL, |out_result| receives a successful result or a diagnosed
// rejection carrying artifacts and owns it until released; otherwise the result
// is discarded after diagnostics are copied. A retained report does not imply
// successful lowering: |out_compiled| remains false for rejected programs.
iree_status_t loom_check_compile_source_low(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_source_low_options_t* options, bool* out_compiled,
    loomc_result_t** out_result);

// Admits and compiles one source case through LoomC's final artifact producer.
// Success requires a nonempty artifact or exactly
// matched diagnostic annotations. RUN goldens, XFAIL, and external execution
// requirements do not describe this independent compile outcome. No device is
// opened and no artifact is loaded or executed.
iree_status_t loom_check_execute_compile(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_compile_options_t* options,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_COMPILE_H_
