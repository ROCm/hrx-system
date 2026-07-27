// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only pass infrastructure harness.
//
// This keeps pass verifier/compiler/interpreter/tooling tests on the synthetic
// pass registry and test dialect without repeatedly hand-assembling contexts,
// traces, parser options, and run options.

#ifndef LOOM_PASS_TEST_HARNESS_H_
#define LOOM_PASS_TEST_HARNESS_H_

#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/emitter.h"
#include "loom/ir/context.h"
#include "loom/ir/ir.h"
#include "loom/pass/environment.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/program.h"
#include "loom/pass/report.h"
#include "loom/pass/test/registry.h"
#include "loom/pass/tooling.h"
#include "loom/pass/verify.h"

namespace loom {

class PassProgramStorage {
 public:
  ~PassProgramStorage();

  // Program owned by this storage object.
  loom_pass_program_t program = {};
};

class PassReportStorage {
 public:
  PassReportStorage();
  ~PassReportStorage();

  // Report owned by this storage object.
  loom_pass_report_t report = {};
};

struct PassTestPredicateCapture {
  // Number of provider verify callbacks observed.
  int verify_count = 0;
  // Number of provider evaluate callbacks observed.
  int evaluate_count = 0;
  // Function symbol selected by the synthetic target predicate.
  iree_string_view_t selected_symbol = iree_string_view_empty();
};

// Returns a provider for `where target(...)` predicates that selects functions
// by |capture->selected_symbol|. The callback counts let tests prove verify and
// evaluation ran at the intended phase and granularity.
loom_pass_predicate_provider_t PassTestTargetPredicateProvider(
    PassTestPredicateCapture* capture);

// Returns a pass environment satisfying the synthetic `target.record`
// descriptor requirement.
loom_pass_environment_t PassTestTargetRecordEnvironment();

// Asserts one chronological synthetic pass trace event.
void ExpectTraceEvent(const loom_test_pass_trace_t& trace,
                      iree_host_size_t event_index,
                      iree_string_view_t pass_name,
                      iree_string_view_t symbol_name);

class PassTestHarness : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  loom_module_t* Parse(iree_string_view_t source);
  loom_module_t* Parse(iree_string_view_t source,
                       iree_string_view_t source_name);
  loom_module_t* AllocateModule();
  loom_module_t* AllocateModule(iree_string_view_t name);

  const loom_op_t* ModuleOp(loom_module_t* module, iree_host_size_t index);
  const loom_op_t* Pipeline(loom_module_t* module, iree_host_size_t index);
  const loom_op_t* PipelineBodyOp(loom_module_t* module,
                                  iree_host_size_t pipeline_index,
                                  iree_host_size_t op_index);
  loom_func_like_t Function(loom_module_t* module, iree_host_size_t index);

  iree_status_t Verify(iree_string_view_t source,
                       loom_pass_environment_t environment = {},
                       loom_pass_predicate_provider_t predicate_provider = {});
  iree_status_t VerifyModule(
      const loom_module_t* module, loom_pass_environment_t environment = {},
      loom_pass_predicate_provider_t predicate_provider = {});
  void ExpectVerifyStatus(
      iree_status_code_t expected_code, iree_string_view_t source,
      loom_pass_environment_t environment = {},
      loom_pass_predicate_provider_t predicate_provider = {});

  iree_status_t Compile(loom_module_t* module, const loom_op_t* pipeline_op,
                        loom_pass_program_t* out_program,
                        loom_pass_predicate_provider_t predicate_provider = {});
  iree_status_t Compile(loom_module_t* module, const loom_op_t* pipeline_op,
                        loom_pass_program_t* out_program,
                        loom_pass_environment_t environment,
                        loom_pass_predicate_provider_t predicate_provider);

  loom_pass_interpreter_options_t InterpreterOptions(
      loom_test_pass_trace_t* trace,
      iree_diagnostic_emitter_t diagnostic_emitter = {},
      loom_pass_report_t* report = nullptr,
      loom_pass_predicate_provider_t predicate_provider = {});

  loom_pass_tool_run_options_t ToolOptions(
      loom_test_pass_trace_t* trace,
      loom_pass_predicate_provider_t predicate_provider = {},
      loom_pass_environment_t environment = {});

  iree_status_t RunModulePipeline(
      loom_module_t* module, iree_host_size_t pipeline_index,
      loom_test_pass_trace_t* trace,
      loom_pass_predicate_provider_t predicate_provider = {});
  iree_status_t RunFunctionPipeline(
      loom_module_t* module, iree_host_size_t pipeline_index,
      iree_host_size_t function_index, loom_test_pass_trace_t* trace,
      loom_pass_predicate_provider_t predicate_provider = {});
  iree_status_t RunPipelineSymbol(loom_module_t* module,
                                  iree_string_view_t pipeline_symbol,
                                  loom_test_pass_trace_t* trace);
  iree_status_t RunFlatPipeline(loom_module_t* module,
                                iree_string_view_t pass_list,
                                loom_test_pass_trace_t* trace);

  iree_arena_block_pool_t* block_pool() { return &block_pool_; }
  iree_arena_allocator_t* scratch_arena() { return &scratch_arena_; }
  loom_context_t* context() { return &context_; }

 private:
  loom_pass_environment_t EnvironmentWithTrace(
      loom_test_pass_trace_t* trace,
      loom_pass_environment_t base_environment = {});

  // Block pool shared by parsing, compilation, and interpreter scratch.
  iree_arena_block_pool_t block_pool_ = {};
  // Scratch arena used by verifier/compiler helpers in tests.
  iree_arena_allocator_t scratch_arena_ = {};
  // Context containing only the pass and test dialects.
  loom_context_t context_ = {};
  // Trace capability storage used while one interpreter/tool options object
  // created by this harness is live.
  loom_test_pass_trace_capability_t trace_capability_ = {};
  // Capability pointer table used while one interpreter/tool options object
  // created by this harness is live.
  const loom_pass_environment_capability_t* environment_capabilities_[8] = {};
  // Parsed or allocated modules freed during TearDown.
  std::vector<loom_module_t*> modules_;
};

}  // namespace loom

#endif  // LOOM_PASS_TEST_HARNESS_H_
