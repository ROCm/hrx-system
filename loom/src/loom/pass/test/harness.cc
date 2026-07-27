// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/test/harness.h"

#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

static iree_string_view_t PredicateContextSymbolName(
    const loom_pass_predicate_evaluate_context_t* context) {
  if (!context->symbol ||
      context->symbol->name_id >= context->target_module->strings.count) {
    return IREE_SV("<none>");
  }
  return context->target_module->strings.entries[context->symbol->name_id];
}

static iree_status_t VerifyTargetPredicate(
    void* user_data, const loom_pass_predicate_verify_context_t* context) {
  PassTestPredicateCapture* capture =
      static_cast<PassTestPredicateCapture*>(user_data);
  ++capture->verify_count;
  if (!iree_string_view_equal(context->predicate, IREE_SV("target"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected test predicate");
  }
  if (context->anchor_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target predicate requires func anchor");
  }
  return iree_ok_status();
}

static iree_status_t EvaluateTargetPredicate(
    void* user_data, const loom_pass_predicate_evaluate_context_t* context,
    bool* out_match) {
  PassTestPredicateCapture* capture =
      static_cast<PassTestPredicateCapture*>(user_data);
  ++capture->evaluate_count;
  *out_match = iree_string_view_equal(PredicateContextSymbolName(context),
                                      capture->selected_symbol);
  return iree_ok_status();
}

}  // namespace

PassProgramStorage::~PassProgramStorage() {
  loom_pass_program_deinitialize(&program);
}

PassReportStorage::PassReportStorage() {
  loom_pass_report_initialize(iree_allocator_system(), &report);
}

PassReportStorage::~PassReportStorage() {
  loom_pass_report_deinitialize(&report);
}

loom_pass_predicate_provider_t PassTestTargetPredicateProvider(
    PassTestPredicateCapture* capture) {
  return (loom_pass_predicate_provider_t){
      /*.verify=*/VerifyTargetPredicate,
      /*.evaluate=*/EvaluateTargetPredicate,
      /*.user_data=*/capture,
  };
}

loom_pass_environment_t PassTestTargetRecordEnvironment() {
  static const loom_pass_environment_capability_t* const capabilities[] = {
      &loom_test_pass_target_record_capability,
  };
  return loom_pass_environment_make(capabilities, IREE_ARRAYSIZE(capabilities));
}

void ExpectTraceEvent(const loom_test_pass_trace_t& trace,
                      iree_host_size_t event_index,
                      iree_string_view_t pass_name,
                      iree_string_view_t symbol_name) {
  ASSERT_LT(event_index, trace.event_count);
  EXPECT_TRUE(
      iree_string_view_equal(trace.events[event_index].pass_name, pass_name));
  EXPECT_TRUE(iree_string_view_equal(trace.events[event_index].symbol_name,
                                     symbol_name));
}

void PassTestHarness::SetUp() {
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool_);
  iree_arena_initialize(&block_pool_, &scratch_arena_);
  loom_context_initialize(iree_allocator_system(), &context_);

  iree_host_size_t pass_count = 0;
  const loom_op_vtable_t* const* pass_vtables =
      loom_pass_dialect_vtables(&pass_count);
  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_PASS, pass_vtables, (uint16_t)pass_count));

  iree_host_size_t test_count = 0;
  const loom_op_vtable_t* const* test_vtables =
      loom_test_dialect_vtables(&test_count);
  IREE_ASSERT_OK(loom_context_register_dialect(
      &context_, LOOM_DIALECT_TEST, test_vtables, (uint16_t)test_count));

  IREE_ASSERT_OK(loom_context_finalize(&context_));
}

void PassTestHarness::TearDown() {
  for (loom_module_t* module : modules_) {
    loom_module_free(module);
  }
  loom_context_deinitialize(&context_);
  iree_arena_deinitialize(&scratch_arena_);
  iree_arena_block_pool_deinitialize(&block_pool_);
}

loom_module_t* PassTestHarness::Parse(iree_string_view_t source) {
  return Parse(source, IREE_SV("pass_test.loom"));
}

loom_module_t* PassTestHarness::Parse(iree_string_view_t source,
                                      iree_string_view_t source_name) {
  loom_text_parse_options_t options = {
      /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
      /*.max_errors=*/20,
  };
  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(loom_text_parse(source, source_name, &context_, &block_pool_,
                                 &options, &module));
  EXPECT_NE(module, nullptr);
  if (module) {
    modules_.push_back(module);
  }
  return module;
}

loom_module_t* PassTestHarness::AllocateModule() {
  return AllocateModule(IREE_SV("test"));
}

loom_module_t* PassTestHarness::AllocateModule(iree_string_view_t name) {
  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(loom_module_allocate(&context_, name, &block_pool_, nullptr,
                                      iree_allocator_system(), &module));
  EXPECT_NE(module, nullptr);
  if (module) {
    modules_.push_back(module);
  }
  return module;
}

const loom_op_t* PassTestHarness::ModuleOp(loom_module_t* module,
                                           iree_host_size_t index) {
  const loom_op_t* op = loom_block_const_op(loom_module_block(module), index);
  EXPECT_NE(op, nullptr);
  return op;
}

const loom_op_t* PassTestHarness::Pipeline(loom_module_t* module,
                                           iree_host_size_t index) {
  const loom_op_t* op = ModuleOp(module, index);
  if (!op) {
    return nullptr;
  }
  EXPECT_TRUE(loom_pass_pipeline_isa(op));
  return op;
}

const loom_op_t* PassTestHarness::PipelineBodyOp(
    loom_module_t* module, iree_host_size_t pipeline_index,
    iree_host_size_t op_index) {
  const loom_op_t* pipeline = Pipeline(module, pipeline_index);
  if (!pipeline) {
    return nullptr;
  }
  loom_region_t* body = loom_pass_pipeline_body(pipeline);
  const loom_op_t* op =
      loom_block_const_op(loom_region_entry_block(body), op_index);
  EXPECT_NE(op, nullptr);
  return op;
}

loom_func_like_t PassTestHarness::Function(loom_module_t* module,
                                           iree_host_size_t index) {
  const loom_op_t* op = ModuleOp(module, index);
  if (!op) {
    return {};
  }
  loom_func_like_t function =
      loom_func_like_cast(module, const_cast<loom_op_t*>(op));
  EXPECT_TRUE(loom_func_like_isa(function));
  return function;
}

iree_status_t PassTestHarness::Verify(
    iree_string_view_t source, loom_pass_environment_t environment,
    loom_pass_predicate_provider_t predicate_provider) {
  loom_module_t* module = Parse(source);
  if (!module) {
    return iree_make_status(IREE_STATUS_INTERNAL, "parse failed");
  }
  return VerifyModule(module, environment, predicate_provider);
}

iree_status_t PassTestHarness::VerifyModule(
    const loom_module_t* module, loom_pass_environment_t environment,
    loom_pass_predicate_provider_t predicate_provider) {
  loom_pass_verify_options_t options = {
      /*.registry=*/loom_test_pass_registry(),
      /*.environment=*/environment,
      /*.predicate_provider=*/predicate_provider,
  };
  return loom_pass_verify_module(module, &options, &scratch_arena_);
}

void PassTestHarness::ExpectVerifyStatus(
    iree_status_code_t expected_code, iree_string_view_t source,
    loom_pass_environment_t environment,
    loom_pass_predicate_provider_t predicate_provider) {
  IREE_EXPECT_STATUS_IS(expected_code,
                        Verify(source, environment, predicate_provider));
}

iree_status_t PassTestHarness::Compile(
    loom_module_t* module, const loom_op_t* pipeline_op,
    loom_pass_program_t* out_program,
    loom_pass_predicate_provider_t predicate_provider) {
  return Compile(module, pipeline_op, out_program, {}, predicate_provider);
}

iree_status_t PassTestHarness::Compile(
    loom_module_t* module, const loom_op_t* pipeline_op,
    loom_pass_program_t* out_program, loom_pass_environment_t environment,
    loom_pass_predicate_provider_t predicate_provider) {
  loom_pass_program_compile_options_t options = {
      /*.registry=*/loom_test_pass_registry(),
      /*.environment=*/environment,
      /*.predicate_provider=*/predicate_provider,
  };
  return loom_pass_program_compile_pipeline(module, pipeline_op, &options,
                                            &block_pool_, out_program);
}

loom_pass_environment_t PassTestHarness::EnvironmentWithTrace(
    loom_test_pass_trace_t* trace, loom_pass_environment_t base_environment) {
  iree_host_size_t capability_count = 0;
  for (iree_host_size_t i = 0; i < base_environment.capability_count; ++i) {
    IREE_ASSERT(capability_count < IREE_ARRAYSIZE(environment_capabilities_));
    environment_capabilities_[capability_count++] =
        base_environment.capabilities[i];
  }
  if (trace) {
    IREE_ASSERT(capability_count < IREE_ARRAYSIZE(environment_capabilities_));
    trace_capability_ = loom_test_pass_trace_capability_make(trace);
    environment_capabilities_[capability_count++] = &trace_capability_.base;
  }
  return loom_pass_environment_make(environment_capabilities_,
                                    capability_count);
}

loom_pass_interpreter_options_t PassTestHarness::InterpreterOptions(
    loom_test_pass_trace_t* trace, iree_diagnostic_emitter_t diagnostic_emitter,
    loom_pass_report_t* report,
    loom_pass_predicate_provider_t predicate_provider) {
  return (loom_pass_interpreter_options_t){
      /*.block_pool=*/&block_pool_,
      /*.predicate_provider=*/predicate_provider,
      /*.diagnostic_emitter=*/diagnostic_emitter,
      /*.environment=*/EnvironmentWithTrace(trace),
      /*.report=*/report,
  };
}

loom_pass_tool_run_options_t PassTestHarness::ToolOptions(
    loom_test_pass_trace_t* trace,
    loom_pass_predicate_provider_t predicate_provider,
    loom_pass_environment_t environment) {
  return (loom_pass_tool_run_options_t){
      /*.registry=*/loom_test_pass_registry(),
      /*.environment=*/EnvironmentWithTrace(trace, environment),
      /*.predicate_provider=*/predicate_provider,
      /*.block_pool=*/&block_pool_,
  };
}

iree_status_t PassTestHarness::RunModulePipeline(
    loom_module_t* module, iree_host_size_t pipeline_index,
    loom_test_pass_trace_t* trace,
    loom_pass_predicate_provider_t predicate_provider) {
  PassProgramStorage storage;
  IREE_RETURN_IF_ERROR(Compile(module, Pipeline(module, pipeline_index),
                               &storage.program, predicate_provider));
  loom_pass_interpreter_options_t options =
      InterpreterOptions(trace, {}, nullptr, predicate_provider);
  loom_pass_run_result_t result = {0};
  return loom_pass_interpreter_run_module(&storage.program, module, &options,
                                          &result);
}

iree_status_t PassTestHarness::RunFunctionPipeline(
    loom_module_t* module, iree_host_size_t pipeline_index,
    iree_host_size_t function_index, loom_test_pass_trace_t* trace,
    loom_pass_predicate_provider_t predicate_provider) {
  PassProgramStorage storage;
  IREE_RETURN_IF_ERROR(Compile(module, Pipeline(module, pipeline_index),
                               &storage.program, predicate_provider));
  loom_pass_interpreter_options_t options =
      InterpreterOptions(trace, {}, nullptr, predicate_provider);
  loom_pass_run_result_t result = {0};
  return loom_pass_interpreter_run_function(&storage.program, module,
                                            Function(module, function_index),
                                            &options, &result);
}

iree_status_t PassTestHarness::RunPipelineSymbol(
    loom_module_t* module, iree_string_view_t pipeline_symbol,
    loom_test_pass_trace_t* trace) {
  loom_pass_tool_run_options_t options = ToolOptions(trace);
  loom_pass_run_result_t result = {0};
  return loom_pass_tool_run_pipeline_symbol(module, pipeline_symbol, &options,
                                            &result);
}

iree_status_t PassTestHarness::RunFlatPipeline(loom_module_t* module,
                                               iree_string_view_t pass_list,
                                               loom_test_pass_trace_t* trace) {
  loom_pass_tool_run_options_t options = ToolOptions(trace);
  loom_pass_run_result_t result = {0};
  return loom_pass_tool_run_flat_pipeline(module, pass_list, &options, &result);
}

}  // namespace loom
