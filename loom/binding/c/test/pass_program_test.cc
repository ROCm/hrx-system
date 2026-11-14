// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cfenv>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/interop.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/pass_trace.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string SourceContents(const loomc_source_t* source) {
  const auto contents = loomc_source_contents(source);
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
  return WorkspacePtr(workspace);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(contents, strlen(contents)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_status_code(result), LOOMC_STATUS_OK);
  EXPECT_EQ(loomc_result_diagnostic_count(result), 0u);
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

ModulePtr CreatePipelineModule(loomc_context_t* context,
                               loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("pipelines.loom", R"(
pass.pipeline<module> @cleanup pipeline {
  for func {
    canonicalize
  }
  call @finish
}

pass.pipeline<module> @finish pipeline {
  for func {
    dce
  }
}
)");
  return DeserializeModule(context, workspace, source.get());
}

void ExpectInvalidPassProgramResult(const loomc_result_t* result,
                                    const char* code = "PASS_PROGRAM/INVALID") {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), code);
}

TEST(PassProgramTest, RetainRelease) {
  ContextPtr context = CreateContext();
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), nullptr, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);

  loomc_pass_program_retain(pass_program_ptr.get());
  loomc_pass_program_release(pass_program_ptr.get());
}

TEST(PassProgramTest, PrintsEmptyProgram) {
  auto context = CreateContext();
  loomc_pass_program_t* raw_program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context.get(), nullptr, loomc_allocator_system(), &raw_program));
  PassProgramPtr program(raw_program);
  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_print_text_to_source(
      program.get(), nullptr, loomc_allocator_system(), &raw_source));
  SourcePtr source(raw_source);
  EXPECT_EQ(loomc_source_format(source.get()), LOOMC_SOURCE_FORMAT_TEXT);

  auto workspace = CreateWorkspace();
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_source_t* raw_roundtrip = nullptr;
  LOOMC_ASSERT_OK(loomc_module_print_text_to_source(
      module.get(), nullptr, loomc_allocator_system(), &raw_roundtrip));
  SourcePtr roundtrip(raw_roundtrip);
  EXPECT_EQ(SourceContents(source.get()), SourceContents(roundtrip.get()));
}

TEST(PassProgramTest, CreatesFromPipelineText) {
  ContextPtr context = CreateContext();

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context.get(), loomc_make_cstring_view("canonicalize,dce"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());

  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_print_text_to_source(
      pass_program_ptr.get(), nullptr, loomc_allocator_system(), &raw_source));
  SourcePtr source(raw_source);
  auto workspace = CreateWorkspace();
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_source_t* raw_roundtrip = nullptr;
  LOOMC_ASSERT_OK(loomc_module_print_text_to_source(
      module.get(), nullptr, loomc_allocator_system(), &raw_roundtrip));
  SourcePtr roundtrip(raw_roundtrip);
  EXPECT_EQ(SourceContents(source.get()), SourceContents(roundtrip.get()));
}

TEST(PassProgramTest, CreatesFromModuleSymbol) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreatePipelineModule(context.get(), workspace.get());
  loomc_source_t* raw_expected = nullptr;
  LOOMC_ASSERT_OK(loomc_module_print_text_to_source(
      module.get(), nullptr, loomc_allocator_system(), &raw_expected));
  SourcePtr expected(raw_expected);

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      module.get(), loomc_make_cstring_view("@cleanup"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());

  module.reset();
  workspace.reset();
  context.reset();
  const loomc_module_serialize_options_t print_options = {
      .identifier = loomc_make_cstring_view("prepared.loom"),
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_PRESERVE,
  };
  loomc_source_t* raw_printed = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_print_text_to_source(
      pass_program_ptr.get(), &print_options, loomc_allocator_system(),
      &raw_printed));
  SourcePtr printed(raw_printed);
  pass_program_ptr.reset();
  EXPECT_EQ(ToString(loomc_source_identifier(printed.get())), "prepared.loom");
  EXPECT_EQ(SourceContents(printed.get()), SourceContents(expected.get()));
}

TEST(PassProgramTest, RejectsInvalidPrintArguments) {
  auto context = CreateContext();
  loomc_pass_program_t* raw_program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context.get(), nullptr, loomc_allocator_system(), &raw_program));
  PassProgramPtr program(raw_program);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_print_text_to_source(
          program.get(), nullptr, loomc_allocator_system(), nullptr));
  loomc_source_t* source = reinterpret_cast<loomc_source_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_print_text_to_source(
          nullptr, nullptr, loomc_allocator_system(), &source));
  EXPECT_EQ(source, nullptr);
  const loomc_module_serialize_options_t invalid_options[] = {
      {.format = LOOMC_SOURCE_FORMAT_BYTECODE},
      {.identifier = loomc_make_string_view(nullptr, 1)},
      {.type = LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS},
  };
  for (const auto& options : invalid_options) {
    source = reinterpret_cast<loomc_source_t*>(0x1);
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_INVALID_ARGUMENT,
        loomc_pass_program_print_text_to_source(
            program.get(), &options, loomc_allocator_system(), &source));
    EXPECT_EQ(source, nullptr);
  }
}

TEST(PassProgramTest, ReportsInvalidPipelineTextInResult) {
  ContextPtr context = CreateContext();

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context.get(), loomc_make_cstring_view("definitely-not-a-pass"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(pass_program, nullptr);
  ResultPtr result_ptr(result);
  ASSERT_NE(result_ptr.get(), nullptr);
  ExpectInvalidPassProgramResult(result_ptr.get());
}

TEST(PassProgramTest, ReportsMissingModuleSymbolInResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreatePipelineModule(context.get(), workspace.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      module.get(), loomc_make_cstring_view("@missing"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(pass_program, nullptr);
  ResultPtr result_ptr(result);
  ExpectInvalidPassProgramResult(result_ptr.get(), "SYMBOL/017");
  EXPECT_EQ(loomc_result_status_code(result_ptr.get()),
            LOOMC_STATUS_INVALID_ARGUMENT);
}

TEST(PassProgramTest, ReportsNonPipelineModuleSymbolInResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("not_pipeline.loom", R"(
func.def @not_pipeline() {
  func.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      module.get(), loomc_make_cstring_view("@not_pipeline"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(pass_program, nullptr);
  ResultPtr result_ptr(result);
  ExpectInvalidPassProgramResult(result_ptr.get(), "SYMBOL/003");
  const auto* diagnostic = loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic->range.source, nullptr);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "not_pipeline.loom");
  EXPECT_EQ(diagnostic->range.start_line, 2u);
  EXPECT_EQ(diagnostic->range.end_line, 4u);
  EXPECT_GT(loomc_source_contents(diagnostic->range.source).data_length, 0u);
}

TEST(PassProgramTest, RejectsUnknownOptionStructure) {
  ContextPtr context = CreateContext();

  loomc_pass_program_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), &options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(pass_program, nullptr);
}

constexpr const char* kRunInput = R"(
func.def public @entry(%x: i32) -> (i32) {
  %dead = scalar.addi %x, %x : i32
  func.return %x : i32
}
)";

constexpr const char* kRunOutput =
    "func.def public @entry(%x: i32) -> (i32) {\n"
    "  func.return %x : i32\n"
    "}\n";

SourcePtr SerializeModule(loomc_module_t* module,
                          loomc_source_format_t format) {
  const loomc_module_serialize_options_t options = {.format = format};
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

PassProgramPtr PrepareText(loomc_context_t* context, const char* text) {
  loomc_pass_program_t* program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view(text), nullptr, loomc_allocator_system(),
      &program, &result));
  ResultPtr result_owner(result);
  ExpectSucceededResult(result);
  return PassProgramPtr(program);
}

TEST(PassProgramTest, RunPreservesIndependentInputAndOutputFormats) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("input.loom", kRunInput);
  auto original =
      DeserializeModule(context.get(), workspace.get(), source.get());
  auto program = PrepareText(context.get(), "dce");
  std::vector<SourcePtr> outputs;
  for (auto input_format :
       {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    auto input = SerializeModule(original.get(), input_format);
    auto module =
        DeserializeModule(context.get(), workspace.get(), input.get());
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_run(program.get(), workspace.get(),
                                           module.get(), nullptr,
                                           loomc_allocator_system(), &result));
    ResultPtr result_owner(result);
    ExpectSucceededResult(result);
    EXPECT_EQ(loomc_result_artifact_count(result), 0u);
    for (auto output_format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      outputs.push_back(SerializeModule(module.get(), output_format));
    }
  }
  original.reset();
  source.reset();
  program.reset();
  context.reset();
  loomc_workspace_trim(workspace.get());
  workspace.reset();

  auto replay_context = CreateContext();
  auto replay_workspace = CreateWorkspace();
  for (const auto& output : outputs) {
    auto replay = DeserializeModule(replay_context.get(),
                                    replay_workspace.get(), output.get());
    EXPECT_EQ(
        SourceContents(
            SerializeModule(replay.get(), LOOMC_SOURCE_FORMAT_TEXT).get()),
        kRunOutput);
  }
}

TEST(PassProgramTest, RunSharesPreparedSnapshotAcrossIndependentWorkers) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto pipeline = CreatePipelineModule(context.get(), workspace.get());
  loomc_pass_program_t* raw_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_module_symbol(
      pipeline.get(), loomc_make_cstring_view("@cleanup"), nullptr,
      loomc_allocator_system(), &raw_program, &result));
  PassProgramPtr program(raw_program);
  ResultPtr result_owner(result);
  ExpectSucceededResult(result);
  pipeline.reset();
  workspace.reset();

  std::array<std::thread, 4> workers;
  for (auto& worker : workers) {
    worker = std::thread([&] {
      auto worker_workspace = CreateWorkspace();
      auto source = CreateTextSource("worker.loom", kRunInput);
      for (int iteration = 0; iteration < 3; ++iteration) {
        auto module = DeserializeModule(context.get(), worker_workspace.get(),
                                        source.get());
        loomc_result_t* run_result = nullptr;
        LOOMC_ASSERT_OK(loomc_pass_program_run(
            program.get(), worker_workspace.get(), module.get(), nullptr,
            loomc_allocator_system(), &run_result));
        ResultPtr run_owner(run_result);
        ExpectSucceededResult(run_result);
        EXPECT_EQ(
            SourceContents(
                SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get()),
            kRunOutput);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

TEST(PassProgramTest, RunRejectsPublicPreconditionsBeforeMutation) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("input.loom", kRunInput);
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  const auto before = SourceContents(
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get());
  auto other_context = CreateContext();
  auto other_program = PrepareText(other_context.get(), "dce");
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_run(other_program.get(), workspace.get(), module.get(),
                             nullptr, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
  auto program = PrepareText(context.get(), "dce");
  const loomc_pass_run_options_t invalid_options[] = {
      {.type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS},
      {.structure_size = sizeof(loomc_pass_run_options_t) - 1},
  };
  for (const auto& options : invalid_options) {
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_INVALID_ARGUMENT,
        loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                               &options, loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
  }
  const loomc_pass_program_options_t unsupported = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      .structure_size = sizeof(unsupported),
  };
  const loomc_pass_run_options_t options = {.next = &unsupported};
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNIMPLEMENTED,
      loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                             &options, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(SourceContents(
                SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get()),
            before);
}

TEST(PassProgramTest, RunAdmitsInputBeforeExecutingPasses) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource(
      "invalid.loom", "func.def @entry() -> (i32) { func.return }\n");
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  auto program = PrepareText(context.get(), "dce");
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_run(program.get(), workspace.get(),
                                         module.get(), nullptr,
                                         loomc_allocator_system(), &result));
  ResultPtr result_owner(result);
  EXPECT_FALSE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_status_code(result), LOOMC_STATUS_FAILED_PRECONDITION);
  ASSERT_GT(loomc_result_diagnostic_count(result), 0u);
  module.reset();
  source.reset();
  program.reset();
  context.reset();
  workspace.reset();
  const auto* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic->range.source, nullptr);
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "invalid.loom");
}

TEST(PassProgramTest, RunRestoresFloatingPointStateAfterSuccessAndFailure) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("input.loom", kRunInput);
  auto pipeline_source = CreateTextSource("failure.loom", R"(
pass.pipeline<module> @failure pipeline {
  for func { dce }
  fail "execution sentinel"
}
)");
  auto pipeline =
      DeserializeModule(context.get(), workspace.get(), pipeline_source.get());
  loomc_pass_program_t* raw_program = nullptr;
  loomc_result_t* prepared_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_module_symbol(
      pipeline.get(), loomc_make_cstring_view("@failure"), nullptr,
      loomc_allocator_system(), &raw_program, &prepared_result));
  PassProgramPtr failed_program(raw_program);
  ResultPtr preparation(prepared_result);
  ExpectSucceededResult(prepared_result);
  auto succeeded_program = PrepareText(context.get(), "dce");
  for (bool succeeds : {true, false}) {
    auto module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    const int original_rounding = std::fegetround();
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_pass_program_run(
        succeeds ? succeeded_program.get() : failed_program.get(),
        workspace.get(), module.get(), nullptr, loomc_allocator_system(),
        &result);
    const int actual_rounding = std::fegetround();
    EXPECT_EQ(std::fesetround(original_rounding), 0);
    ResultPtr result_owner(result);
    LOOMC_ASSERT_OK(status);
    EXPECT_EQ(actual_rounding, FE_DOWNWARD);
    EXPECT_EQ(loomc_result_succeeded(result), succeeds);
    EXPECT_EQ(loomc_result_status_code(result),
              succeeds ? LOOMC_STATUS_OK : LOOMC_STATUS_FAILED_PRECONDITION);
    EXPECT_EQ(
        SourceContents(
            SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get()),
        kRunOutput);
    if (!succeeds) {
      ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
      EXPECT_EQ(ToString(loomc_result_diagnostic_at(result, 0)->code),
                "PASS_PROGRAM/EXECUTION");
    }
  }
}

TEST(PassProgramTest, RunPropagatesTraceFailureAndClosesOpenedArtifacts) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("input.loom", kRunInput);
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  auto program = PrepareText(context.get(), "dce");
  int closes = 0;
  const loomc_pass_trace_options_t trace = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(trace),
      .flags = LOOMC_PASS_TRACE_FLAG_BEFORE_ALL,
      .sink =
          {
              .write = [](void*,
                          loomc_string_view_t) { return loomc_ok_status(); },
          },
      .artifact_sink = {
          .open =
              [](void*, const loomc_pass_trace_event_t*,
                 loomc_pass_trace_artifact_t* artifact) {
                *artifact = {
                    .reference = loomc_make_cstring_view("trace.loom"),
                    .sink =
                        {
                            .write =
                                [](void*, loomc_string_view_t) {
                                  return loomc_status_from_code(
                                      LOOMC_STATUS_INVALID_ARGUMENT);
                                },
                        },
                };
                return loomc_ok_status();
              },
          .close =
              [](void* user_data, loomc_pass_trace_artifact_t*) {
                ++*static_cast<int*>(user_data);
                return loomc_ok_status();
              },
          .user_data = &closes,
      },
  };
  const loomc_pass_run_options_t options = {.next = &trace};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                             &options, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(closes, 1);
}

TEST(PassProgramTest, RunStopsReportingAfterTraceInfrastructureFailure) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("input.loom", kRunInput);
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  auto program = PrepareText(context.get(), "dce");
  const loomc_pass_trace_options_t trace = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(trace),
      .flags = LOOMC_PASS_TRACE_FLAG_AFTER_ALL,
      .sink =
          {
              .write =
                  [](void*, loomc_string_view_t) {
                    return loomc_status_from_code(
                        LOOMC_STATUS_RESOURCE_EXHAUSTED);
                  },
          },
  };
  iree_arena_block_pool_t report_block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                   &report_block_pool);
  loom_pass_report_t report;
  loom_pass_report_initialize(&report_block_pool, &report);
  const loomc_native_pass_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .next = &trace,
      .report = &report,
  };
  const loomc_pass_run_options_t options = {.next = &report_options};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_RESOURCE_EXHAUSTED,
      loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                             &options, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(report.invocation_count, 0u);
  loom_pass_report_deinitialize(&report);
  iree_arena_block_pool_deinitialize(&report_block_pool);
}

TEST(PassProgramTest, RunAllocationFailureLeavesNoResultOrMutation) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("input.loom", kRunInput);
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  const auto before = SourceContents(
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get());
  auto program = PrepareText(context.get(), "dce");
  const loomc_allocator_t allocator = {
      .ctl =
          [](void*, loomc_allocator_command_t, const void*, void**) {
            return loomc_status_from_code(LOOMC_STATUS_RESOURCE_EXHAUSTED);
          },
  };
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_RESOURCE_EXHAUSTED,
      loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                             nullptr, allocator, &result));
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(SourceContents(
                SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get()),
            before);
}

TEST(PassProgramTest, FunctionRootVisitsBodiesAndLeavesConfigurationUnapplied) {
  auto context = CreateContext();
  auto workspace = CreateWorkspace();
  auto pipeline_source = CreateTextSource("function_pipeline.loom", R"(
pass.pipeline<func> @cleanup pipeline { dce }
pass.pipeline<module> @empty pipeline {}
)");
  auto pipeline_module =
      DeserializeModule(context.get(), workspace.get(), pipeline_source.get());
  auto source = CreateTextSource("functions.loom", R"(
config.def @value = 7 : i32
func.decl @external(%x: i32) -> (i32)

func.def public @first() -> (i32) {
  %dead = scalar.constant 1 : i32
  %value = config.get @value : i32
  func.return %value : i32
}

func.def public @second() -> (i32) {
  %dead = scalar.constant 2 : i32
  %value = config.get @value : i32
  func.return %value : i32
}
)");
  for (const char* symbol : {"empty", "cleanup"}) {
    loomc_pass_program_t* raw_program = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_from_module_symbol(
        pipeline_module.get(), loomc_make_cstring_view(symbol), nullptr,
        loomc_allocator_system(), &raw_program, &result));
    PassProgramPtr program(raw_program);
    ResultPtr result_owner(result);
    ExpectSucceededResult(result);
    result_owner.reset();
    auto module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    const auto before = SourceContents(
        SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get());
    result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_run(program.get(), workspace.get(),
                                           module.get(), nullptr,
                                           loomc_allocator_system(), &result));
    result_owner.reset(result);
    ExpectSucceededResult(result);
    EXPECT_EQ(loomc_result_artifact_count(result), 0u);
    const auto after = SourceContents(
        SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT).get());
    if (std::strcmp(symbol, "empty") == 0) {
      EXPECT_EQ(after, before);
    } else {
      EXPECT_EQ(after, R"(config.def @value = 7 : i32

func.decl @external(%x: i32) -> (i32)

func.def public @first() -> (i32) {
  %value = config.get @value : i32
  func.return %value : i32
}

func.def public @second() -> (i32) {
  %value = config.get @value : i32
  func.return %value : i32
}
)");
    }
  }
}

}  // namespace
