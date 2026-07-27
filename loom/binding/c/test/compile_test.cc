// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/compile.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;

using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;

using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;

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

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
  };
  loomc_compiler_t* compiler = nullptr;
  loomc_status_t status = loomc_compiler_create(
      context, &options, loomc_allocator_system(), &compiler);
  LOOMC_EXPECT_OK(status);
  return CompilerPtr(compiler);
}

PassProgramPtr CreateEmptyPassProgram(loomc_context_t* context) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context, nullptr, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  return PassProgramPtr(pass_program);
}

PassProgramPtr CreatePassProgramFromPipelineText(loomc_context_t* context,
                                                 const char* pipeline_text) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view(pipeline_text), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return PassProgramPtr(pass_program);
}

PassProgramPtr CreatePassProgramFromModuleSymbol(
    const loomc_module_t* pipeline_module, const char* pipeline_symbol) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      pipeline_module, loomc_make_cstring_view(pipeline_symbol), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return PassProgramPtr(pass_program);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
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
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
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
  EXPECT_EQ(loomc_result_diagnostic_count(result), 0u);
}

void ExpectFailedResultCode(const loomc_result_t* result, const char* code) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  const loomc_host_size_t diagnostic_count =
      loomc_result_diagnostic_count(result);
  ASSERT_NE(diagnostic_count, 0u);
  bool found = false;
  for (loomc_host_size_t i = 0; i < diagnostic_count; ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    ASSERT_NE(diagnostic, nullptr);
    found |= ToString(diagnostic->code) == code;
  }
  EXPECT_TRUE(found);
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact == nullptr) {
      continue;
    }
    if (artifact->kind == kind &&
        loomc_string_view_equal(artifact->format,
                                loomc_make_cstring_view(format))) {
      return artifact;
    }
  }
  return nullptr;
}

std::string SerializeModuleToText(const loomc_module_t* module) {
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("compiled.loom"),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  if (!loomc_status_is_ok(status)) {
    return std::string();
  }
  SourcePtr source_ptr(source);
  return ToString(loomc_source_contents(source_ptr.get()));
}

ModulePtr CreateValidModule(loomc_context_t* context,
                            loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("compile.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  return DeserializeModule(context, workspace, source.get());
}

ModulePtr CreateConfigModule(loomc_context_t* context,
                             loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("config.loom", R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]

func.def public @entry() -> (index) {
  %hidden = config.get @model36.model.hidden_size : index
  func.return %hidden : index
}
)");
  return DeserializeModule(context, workspace, source.get());
}

TEST(CompileTest, CompilerRetainRelease) {
  ContextPtr context = CreateContext();
  CompilerPtr compiler = CreateCompiler(context.get());
  loomc_compiler_retain(compiler.get());
  loomc_compiler_release(compiler.get());
}

TEST(CompileTest, CompileModuleRunsPreparedPassProgram) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program =
      CreatePassProgramFromPipelineText(context.get(), "canonicalize,dce");
  ModulePtr module = CreateValidModule(context.get(), workspace.get());

  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("tile_m"),
          /*.value=*/loomc_make_cstring_view("128"),
      },
  };
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.artifact_flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_make_cstring_view("{\"tile_n\":\"64\"}"),
      },
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 0u);
  EXPECT_EQ(loomc_result_artifact_count(result_ptr.get()), 0u);
}

TEST(CompileTest, CompileModuleRunsPassProgramFromReleasedModuleSymbol) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  SourcePtr pipeline_source = CreateTextSource("pipelines.loom", R"(
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
  ModulePtr pipeline_module =
      DeserializeModule(context.get(), workspace.get(), pipeline_source.get());
  PassProgramPtr pass_program =
      CreatePassProgramFromModuleSymbol(pipeline_module.get(), "@cleanup");
  ASSERT_NE(pass_program.get(), nullptr);
  pipeline_module.reset();
  pipeline_source.reset();
  ModulePtr module = CreateValidModule(context.get(), workspace.get());

  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      nullptr, loomc_allocator_system(), &result);
  LOOMC_ASSERT_OK(status);
  ResultPtr result_ptr(result);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 0u);
  EXPECT_EQ(loomc_result_artifact_count(result_ptr.get()), 0u);
}

TEST(CompileTest, CompileModuleMaterializesInvocationConfig) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigModule(context.get(), workspace.get());

  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("@model36.model.hidden_size"),
          /*.value=*/loomc_make_cstring_view("4096"),
      },
  };
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_make_cstring_view(R"({
                "model36": {
                  "model": {"hidden_size": 2048},
                  "unused": 1
                }
              })"),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());

  std::string text = SerializeModuleToText(module.get());
  EXPECT_NE(text.find("config.def @model36.model.hidden_size = 4096 : index"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
  EXPECT_EQ(text.find("2048"), std::string::npos);
}

TEST(CompileTest, CompileModuleEmitsRequestedArtifacts) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigModule(context.get(), workspace.get());

  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("@model36.model.hidden_size"),
          /*.value=*/loomc_make_cstring_view("4096"),
      },
  };
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT |
          LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
          LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_EQ(loomc_result_artifact_count(result_ptr.get()), 3u);

  const loomc_artifact_t* text_artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                   LOOMC_ARTIFACT_FORMAT_LOOM_TEXT);
  ASSERT_NE(text_artifact, nullptr);
  EXPECT_EQ(ToString(text_artifact->identifier), "jit_kernel.loom");
  std::string text = ToString(text_artifact->contents);
  EXPECT_NE(text.find("config.def @model36.model.hidden_size = 4096 : index"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);

  const loomc_artifact_t* bytecode_artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                   LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  ASSERT_NE(bytecode_artifact, nullptr);
  EXPECT_EQ(ToString(bytecode_artifact->identifier), "jit_kernel.loombc");
  EXPECT_NE(bytecode_artifact->contents.data_length, 0u);
  loomc_source_t* bytecode_source = nullptr;
  status = loomc_artifact_create_source(
      bytecode_artifact, LOOMC_SOURCE_FORMAT_UNKNOWN, loomc_allocator_system(),
      &bytecode_source);
  LOOMC_EXPECT_OK(status);
  SourcePtr bytecode_source_ptr(bytecode_source);
  EXPECT_EQ(loomc_source_format(bytecode_source_ptr.get()),
            LOOMC_SOURCE_FORMAT_BYTECODE);
  ModulePtr bytecode_module = DeserializeModule(context.get(), workspace.get(),
                                                bytecode_source_ptr.get());
  EXPECT_NE(bytecode_module.get(), nullptr);

  const loomc_artifact_t* report_artifact = FindArtifact(
      result_ptr.get(), LOOMC_ARTIFACT_KIND_REPORT, LOOMC_ARTIFACT_FORMAT_JSON);
  ASSERT_NE(report_artifact, nullptr);
  EXPECT_EQ(ToString(report_artifact->identifier),
            "jit_kernel.compile-report.json");
  std::string report = ToString(report_artifact->contents);
  EXPECT_NE(report.find(R"("kind":"loomc.compile")"), std::string::npos);
  EXPECT_NE(report.find(R"("state":"succeeded")"), std::string::npos);
  EXPECT_NE(report.find(R"("artifact_count":2)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_binding_count":1)"), std::string::npos);
}

TEST(CompileTest, CompileModuleReportsUnknownConfigAsResultDiagnostic) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());

  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("tile_m"),
          /*.value=*/loomc_make_cstring_view("128"),
      },
  };
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/1,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN,
      },
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectFailedResultCode(result_ptr.get(), "CONFIG/INVALID");
  ASSERT_EQ(loomc_result_artifact_count(result_ptr.get()), 1u);
  const loomc_artifact_t* report_artifact = FindArtifact(
      result_ptr.get(), LOOMC_ARTIFACT_KIND_REPORT, LOOMC_ARTIFACT_FORMAT_JSON);
  ASSERT_NE(report_artifact, nullptr);
  std::string report = ToString(report_artifact->contents);
  EXPECT_NE(report.find(R"("state":"failed")"), std::string::npos);
  EXPECT_NE(report.find(R"("diagnostic_count":1)"), std::string::npos);
}

TEST(CompileTest, CompileModuleReportsUnresolvedConfigAsResultDiagnostic) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigModule(context.get(), workspace.get());

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/nullptr,
          /*.binding_count=*/0,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectFailedResultCode(result_ptr.get(), "CONFIG/INVALID");
}

TEST(CompileTest, CompileModuleRejectsInvalidInvocationConfig) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/nullptr,
          /*.binding_count=*/1,
      },
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

TEST(CompileTest, CompileModuleRejectsUnknownOptionStructure) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

}  // namespace
