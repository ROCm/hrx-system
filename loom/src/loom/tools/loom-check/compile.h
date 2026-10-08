// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Offline compiler qualification, independent of execution resources.

#ifndef LOOM_TOOLS_LOOM_CHECK_COMPILE_H_
#define LOOM_TOOLS_LOOM_CHECK_COMPILE_H_

#include "loom/sanitizer/options.h"
#include "loom/target/pipeline_options.h"
#include "loom/target/reporting/report.h"
#include "loom/tools/loom-check/execute.h"
#include "loomc/compile.h"
#include "loomc/sanitizer.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Imports one optional foreign source format through LoomC. Builtin Loom text
// and bytecode admission do not use this callback.
typedef iree_status_t (*loom_check_compile_import_fn_t)(
    void* user_data, iree_string_view_t format,
    iree_string_view_t input_options,
    const loom_tooling_source_path_options_t* source_path_options,
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

// Compiler integrations supplied by the final loom-check binary. The generic
// runner does not choose a configured target set or optional source frontend.
typedef struct loom_check_compile_provider_t {
  // Optional foreign-source importer dispatch.
  loom_check_compile_import_fn_t import;
  // Opaque state forwarded to |import|.
  void* import_user_data;
} loom_check_compile_provider_t;

// Lazily prepared public compiler state shared by checker cases. Initializing a
// session does not construct a target environment, context, workspace, or
// compiler; the first public compile operation prepares all four together.
typedef struct loom_check_compile_session_t {
  // Binary-owned compiler integrations used when the session is prepared.
  const loom_check_compile_provider_t* provider;
  // Host allocator used for every session-owned public handle.
  iree_allocator_t host_allocator;
  // Retained public target environment composed by the checker runner.
  loomc_target_environment_t* target_environment;
  // Public context, or NULL before the first compile operation.
  loomc_context_t* context;
  // Reusable invocation workspace, or NULL before the first compile operation.
  loomc_workspace_t* workspace;
  // Prepared compiler, or NULL before the first compile operation.
  loomc_compiler_t* compiler;
  // Cached empty, source-to-Low CFG, and source-to-Low structured programs.
  loomc_pass_program_t* artifact_pass_programs[3];
} loom_check_compile_session_t;

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
  // Optional complete family:selector target profile specification.
  iree_string_view_t target;
  // Whether to lower source IR to Low before target emission.
  bool lower_source_to_low;
  // Source-to-Low control-flow shape when lowering source IR.
  loomc_target_control_flow_lowering_t control_flow_lowering;
} loom_check_compile_artifact_options_t;

// Source-Low pass-program boundary selected by a checker provider.
typedef enum loom_check_compile_source_low_pipeline_e {
  // Full source-to-Low target pipeline.
  LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DEFAULT = 0,
  // Full target pipeline prepared for required low-asm output.
  LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_ARTIFACT = 1,
  // Raw diagnostic boundary prepared for required low-asm output.
  LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DIAGNOSTIC_ARTIFACT = 2,
  // Raw source-to-Low diagnostic pass without target pipeline preparation.
  LOOM_CHECK_COMPILE_SOURCE_LOW_PIPELINE_DIAGNOSTIC = 3,
} loom_check_compile_source_low_pipeline_t;

// One source-to-Low checker compilation request.
typedef struct loom_check_compile_source_low_options_t {
  // Pass-program boundary to prepare and execute.
  loom_check_compile_source_low_pipeline_t pipeline;
  // Source-to-Low legality diagnostics requested by the checker case.
  loom_target_low_legality_diagnostic_flags_t diagnostic_flags;
  // Control-flow representation selected for lowering.
  loom_target_control_flow_lowering_t control_flow_lowering;
  // Sanitizer instrumentation and reporting policy.
  loom_sanitizer_options_t sanitizer;
  // Optional function symbol to specialize.
  iree_string_view_t function_name;
  // Optional complete target profile specification.
  iree_string_view_t target;
  // Optional caller-owned native compile report populated by the pipeline.
  loom_target_compile_report_t* report;
} loom_check_compile_source_low_options_t;

// Borrowed public and exact-version views of one compiled source module.
typedef struct loom_check_compile_source_low_view_t {
  // Public module retained by the compile helper during the callback.
  const loomc_module_t* public_module;
  // Context-target-verified native module owned by |public_module|.
  const loom_module_t* module;
  // Exact source snapshots for native diagnostics during the callback.
  loom_source_resolver_t source_resolver;
} loom_check_compile_source_low_view_t;

// Consumes one verified mutable native projection of an admitted public
// module. The projection remains valid only for the callback invocation.
typedef iree_status_t (*loom_check_compile_native_module_consumer_fn_t)(
    void* user_data, const loom_check_emit_native_module_t* native_module);

// Consumes one successfully compiled source-Low module before it is released.
typedef iree_status_t (*loom_check_compile_source_low_consumer_fn_t)(
    void* user_data, const loom_check_compile_source_low_view_t* view);

// Releases all public compiler state prepared by |session|.
void loom_check_compile_session_deinitialize(
    loom_check_compile_session_t* session);

// Selects a complete target profile from the prepared session environment.
iree_status_t loom_check_compile_session_select_target_profile(
    loom_check_compile_session_t* session, iree_string_view_t specification,
    loomc_target_profile_t** out_target_profile);

// Admits one checker case through the shared public compiler session. Public
// diagnostics are appended to |diagnostic_collector|. A successful module is
// owned by the caller and must be released with loomc_module_release.
iree_status_t loom_check_compile_admit_module(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_input_request_t* input_request,
    loom_check_compile_session_t* session,
    const loom_check_environment_t* environment,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    loomc_module_t** out_module);

// Projects, invokes, and re-verifies the admitted public module in |request|.
// Public verification diagnostics suppress the callback and return OK for
// normal loom-check annotation matching.
iree_status_t loom_check_compile_with_native_module(
    const loom_check_emit_provider_request_t* request,
    loom_check_compile_native_module_consumer_fn_t consumer, void* user_data);

// Compiles the request's admitted public module to one target artifact. Public
// diagnostics are appended to |request->diagnostic_collector|. A successful
// primary artifact is returned as an immutable source owned by the caller.
iree_status_t loom_check_compile_artifact(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_artifact_options_t* options,
    loomc_source_t** out_artifact_source);

// Lowers the request's admitted public module through LoomC, then invokes
// |consumer| on the successfully compiled module. User diagnostics suppress
// the callback and return OK for normal loom-check annotation matching.
iree_status_t loom_check_compile_source_low(
    const loom_check_emit_provider_request_t* request,
    const loom_check_compile_source_low_options_t* options,
    loom_check_compile_source_low_consumer_fn_t consumer, void* user_data);

// Admits and compiles one source case through LoomC's final artifact producer.
// Success requires a nonempty artifact or exactly
// matched diagnostic annotations. RUN goldens, XFAIL, and external execution
// requirements do not describe this independent compile outcome. No device is
// opened and no artifact is loaded or executed.
iree_status_t loom_check_execute_compile(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_input_request_t* input_request,
    const loom_check_compile_options_t* options,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_COMPILE_H_
