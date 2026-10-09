// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/import/cxx.h"

#include <string>

#include "iree/testing/gtest.h"
#include "loomc/compile.h"
#include "loomc/target/amdgpu.h"
#include "test/util.h"

namespace {
using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using EnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ProgramPtr = HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;

ContextPtr CreateContext() {
  loomc_target_environment_t* environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment));
  EnvironmentPtr owner(environment);
  loomc_context_target_options_t target_options = {};
  target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target_options.structure_size = sizeof(target_options);
  target_options.target_environment = environment;
  loomc_context_options_t options = {.next = &target_options};
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(&options, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

std::string Text(const loomc_module_t* module) {
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_text_to_source(
      module, nullptr, loomc_allocator_system(), &source));
  SourcePtr owner(source);
  auto contents = loomc_source_contents(source);
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

struct HeaderProvider {
  // Immutable header source returned to one frontend invocation.
  SourcePtr header;

  static loomc_status_t Resolve(void* user_data, loomc_string_view_t path,
                                loomc_source_t** out_source) {
    auto& self = *static_cast<HeaderProvider*>(user_data);
    *out_source = nullptr;
    if (std::string(path.data, path.size) == "/headers/division.h") {
      loomc_source_retain(self.header.get());
      *out_source = self.header.get();
    }
    return loomc_ok_status();
  }
};

ResultPtr CompileDivisionForGfx1151(loomc_target_environment_t* environment,
                                    loomc_context_t* context,
                                    loomc_workspace_t* workspace,
                                    loomc_module_t* module) {
  const auto allocator = loomc_allocator_system();
  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(
      loomc_compiler_create(context, nullptr, allocator, &compiler));
  CompilerPtr compiler_owner(compiler);
  loomc_amdgpu_profile_options_t profile_options = {};
  profile_options.type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
  profile_options.structure_size = sizeof(profile_options);
  profile_options.identity.target = loomc_make_cstring_view("gfx1151");
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_OK(loomc_target_profile_create_amdgpu(
      environment, &profile_options, allocator, &profile));
  ProfilePtr profile_owner(profile);
  loomc_target_pipeline_options_t pipeline_options = {};
  pipeline_options.type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS;
  pipeline_options.structure_size = sizeof(pipeline_options);
  pipeline_options.kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
  pipeline_options.control_flow_lowering =
      LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
  pipeline_options.source_to_low_max_errors = 20;
  loomc_pass_program_t* program = nullptr;
  loomc_result_t* prepared = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &pipeline_options, allocator, &program, &prepared));
  ProgramPtr program_owner(program);
  ResultPtr prepared_owner(prepared);
  EXPECT_TRUE(loomc_result_succeeded(prepared));
  loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("divide"),
      .target_profile = profile};
  loomc_target_specialization_options_t target_options = {};
  target_options.type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS;
  target_options.structure_size = sizeof(target_options);
  target_options.specializations = &specialization;
  target_options.specialization_count = 1;
  loomc_compile_options_t compile_options = {.next = &target_options};
  loomc_result_t* compiled = nullptr;
  LOOMC_EXPECT_OK(loomc_compile_module(compiler, workspace, program, module,
                                       &compile_options, allocator, &compiled));
  return ResultPtr(compiled);
}

TEST(CxxAssemblyTest, SourceAndEnvironmentOwnershipSurvivesBothFormats) {
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    SCOPED_TRACE(format);
    auto context = CreateContext();
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    WorkspacePtr workspace_owner(workspace);
    const char text[] = R"cpp(
#include <loomcxx/low.h>
      struct [[loom::representation("amdgpu.gfx11.generic.core")]] Contract {};
      using Word = float __attribute__((ext_vector_type(1)));
      Word entry(Word input) {
        Word result = loom::low::assembly<Contract, Word>(R"(
          (%input: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) { return %input }
        )",
                                                          input);
        return result;
      }
    )cpp";
    loomc_source_options_t source_options = {
        .identifier = loomc_make_cstring_view("assembly.cxx"),
        .contents = loomc_make_byte_span(text, sizeof(text) - 1),
        .storage = LOOMC_SOURCE_STORAGE_COPY};
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source));
    SourcePtr source_owner(source);
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_import_cxx(context.get(), workspace, source,
                                            nullptr, loomc_allocator_system(),
                                            &module, &result));
    ModulePtr module_owner(module);
    ResultPtr result_owner(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    source_owner.reset();
    result_owner.reset();
    workspace_owner.reset();
    context.reset();

    std::string original = Text(module);
    EXPECT_NE(original.find("low.invoke inline"), std::string::npos);
    EXPECT_NE(original.find("reg<amdgpu.vgpr>"), std::string::npos);
    EXPECT_NE(original.find("vector<1xf32>"), std::string::npos);
    loomc_module_serialize_options_t serialize_options = {.format = format};
    loomc_source_t* serialized = nullptr;
    LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
        module, &serialize_options, loomc_allocator_system(), &serialized));
    SourcePtr serialized_owner(serialized);
    module_owner.reset();

    // A new context resolves durable representation keys after all importer
    // and original module state has been released.
    context = CreateContext();
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    workspace_owner.reset(workspace);
    module = nullptr;
    result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
        context.get(), workspace, serialized, nullptr, loomc_allocator_system(),
        &module, &result));
    module_owner.reset(module);
    result_owner.reset(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    serialized_owner.reset();
    result_owner.reset();
    context.reset();
    workspace_owner.reset();
    EXPECT_EQ(Text(module), original);
  }
}

TEST(CxxDiagnosticTest,
     BytecodeCompilationRetainsOriginalLocationAfterTeardown) {
  const auto allocator = loomc_allocator_system();
  SourcePtr bytecode;
  {
    auto context = CreateContext();
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, allocator, &workspace));
    WorkspacePtr workspace_owner(workspace);
    const char text[] =
        "[[loom::kernel, loom::workgroup_count(1,1,1), "
        "loom::workgroup_size(64,1,1)]]\n"
        "void divide(const unsigned* input, unsigned* output) {\n"
        "  output[0] = input[0] / input[1];\n"
        "}\n";
    loomc_source_options_t source_options = {
        .identifier = loomc_make_cstring_view("virtual/division.cxx"),
        .contents = loomc_make_byte_span(text, sizeof(text) - 1),
        .storage = LOOMC_SOURCE_STORAGE_COPY};
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options, allocator, &source));
    SourcePtr source_owner(source);
    loomc_module_t* module = nullptr;
    loomc_result_t* imported = nullptr;
    LOOMC_ASSERT_OK(loomc_module_import_cxx(context.get(), workspace, source,
                                            nullptr, allocator, &module,
                                            &imported));
    ModulePtr module_owner(module);
    ResultPtr imported_owner(imported);
    ASSERT_TRUE(loomc_result_succeeded(imported));
    loomc_source_t* serialized = nullptr;
    LOOMC_ASSERT_OK(loomc_module_serialize_bytecode_to_source(
        module, nullptr, allocator, &serialized));
    bytecode.reset(serialized);
  }

  // Only serialized bytes cross the importer/context ownership boundary.
  ResultPtr result;
  {
    loomc_target_environment_t* environment = nullptr;
    LOOMC_ASSERT_OK(
        loomc_target_environment_create_amdgpu(allocator, &environment));
    EnvironmentPtr environment_owner(environment);
    loomc_context_target_options_t target_context_options = {};
    target_context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_context_options.structure_size = sizeof(target_context_options);
    target_context_options.target_environment = environment;
    loomc_context_options_t context_options = {.next = &target_context_options};
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(
        loomc_context_create(&context_options, allocator, &context));
    ContextPtr context_owner(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, allocator, &workspace));
    WorkspacePtr workspace_owner(workspace);
    loomc_module_t* module = nullptr;
    loomc_result_t* admitted = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_bytecode_from_source(
        context, workspace, bytecode.get(), nullptr, allocator, &module,
        &admitted));
    ModulePtr module_owner(module);
    ResultPtr admitted_owner(admitted);
    ASSERT_TRUE(loomc_result_succeeded(admitted));
    bytecode.reset();

    result = CompileDivisionForGfx1151(environment, context, workspace, module);
  }

  // The result alone owns the original identity after every compiler owner
  // dies.
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  const loomc_diagnostic_t* failure = nullptr;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result.get());
       ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result.get(), i);
    if (std::string(diagnostic->code.data, diagnostic->code.size) ==
        "TARGET/003") {
      failure = diagnostic;
    }
  }
  ASSERT_NE(failure, nullptr);
  EXPECT_NE(std::string(failure->message.data, failure->message.size)
                .find("scalar.divui"),
            std::string::npos);
  const auto& range = failure->range;
  ASSERT_NE(range.source, nullptr);
  auto identifier = loomc_source_identifier(range.source);
  EXPECT_EQ(std::string(identifier.data, identifier.size),
            "virtual/division.cxx");
  EXPECT_EQ(range.start_line, 3u);
  EXPECT_EQ(range.start_column, 15u);
  EXPECT_EQ(range.end_line, 3u);
  EXPECT_EQ(range.end_column, 34u);
  EXPECT_EQ(loomc_source_contents(range.source).data_length, 0u);
}

TEST(CxxDiagnosticTest, CompilationSourceRetentionControlsHeaderSnapshots) {
  const auto allocator = loomc_allocator_system();
  const char header_text[] =
      "[[loom::kernel, loom::workgroup_count(1,1,1), "
      "loom::workgroup_size(64,1,1)]]\n"
      "void divide(const unsigned* input, unsigned* output) {\n"
      "  output[0] = input[0] / input[1];\n"
      "}\n";
  for (loomc_source_retention_t source_retention : {
           LOOMC_SOURCE_RETENTION_EXACT,
           LOOMC_SOURCE_RETENTION_METADATA_ONLY,
       }) {
    SCOPED_TRACE(source_retention);
    ResultPtr result;
    {
      loomc_target_environment_t* environment = nullptr;
      LOOMC_ASSERT_OK(
          loomc_target_environment_create_amdgpu(allocator, &environment));
      EnvironmentPtr environment_owner(environment);
      loomc_context_target_options_t target_context_options = {};
      target_context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
      target_context_options.structure_size = sizeof(target_context_options);
      target_context_options.target_environment = environment;
      loomc_context_options_t context_options = {
          .next = &target_context_options,
          .source_retention = source_retention};
      loomc_context_t* context = nullptr;
      LOOMC_ASSERT_OK(
          loomc_context_create(&context_options, allocator, &context));
      ContextPtr context_owner(context);
      loomc_workspace_t* workspace = nullptr;
      LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, allocator, &workspace));
      WorkspacePtr workspace_owner(workspace);

      loomc_source_options_t header_options = {
          .identifier = loomc_make_cstring_view("/headers/division.h"),
          .contents =
              loomc_make_byte_span(header_text, sizeof(header_text) - 1),
          .storage = LOOMC_SOURCE_STORAGE_COPY};
      loomc_source_t* header = nullptr;
      LOOMC_ASSERT_OK(loomc_source_create(&header_options, allocator, &header));
      HeaderProvider provider = {SourcePtr(header)};

      const char main_text[] = "#include <division.h>\n";
      loomc_source_options_t source_options = {
          .identifier = loomc_make_cstring_view("virtual/main.cxx"),
          .contents = loomc_make_byte_span(main_text, sizeof(main_text) - 1),
          .storage = LOOMC_SOURCE_STORAGE_COPY};
      loomc_source_t* source = nullptr;
      LOOMC_ASSERT_OK(loomc_source_create(&source_options, allocator, &source));
      SourcePtr source_owner(source);
      const loomc_string_view_t include_path =
          loomc_make_cstring_view("/headers");
      loomc_cxx_import_options_t import_options = {};
      import_options.source_provider = {HeaderProvider::Resolve, &provider};
      import_options.include_paths = &include_path;
      import_options.include_path_count = 1;
      import_options.flags = LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
      loomc_module_t* module = nullptr;
      loomc_result_t* imported = nullptr;
      LOOMC_ASSERT_OK(loomc_module_import_cxx(context, workspace, source,
                                              &import_options, allocator,
                                              &module, &imported));
      ModulePtr module_owner(module);
      ResultPtr imported_owner(imported);
      ASSERT_TRUE(loomc_result_succeeded(imported));
      source_owner.reset();
      provider.header.reset();
      imported_owner.reset();

      result =
          CompileDivisionForGfx1151(environment, context, workspace, module);
    }

    EXPECT_FALSE(loomc_result_succeeded(result.get()));
    const loomc_diagnostic_t* failure = nullptr;
    for (loomc_host_size_t i = 0;
         i < loomc_result_diagnostic_count(result.get()); ++i) {
      const auto* diagnostic = loomc_result_diagnostic_at(result.get(), i);
      if (std::string(diagnostic->code.data, diagnostic->code.size) ==
          "TARGET/003") {
        failure = diagnostic;
      }
    }
    ASSERT_NE(failure, nullptr);
    ASSERT_NE(failure->range.source, nullptr);
    const auto identifier = loomc_source_identifier(failure->range.source);
    EXPECT_EQ(std::string(identifier.data, identifier.size),
              "/headers/division.h");
    EXPECT_EQ(failure->range.start_line, 3u);
    EXPECT_EQ(failure->range.start_column, 15u);
    EXPECT_EQ(failure->range.end_line, 3u);
    EXPECT_EQ(failure->range.end_column, 34u);
    const auto contents = loomc_source_contents(failure->range.source);
    const std::string formatted(failure->formatted_text.data,
                                failure->formatted_text.size);
    if (source_retention == LOOMC_SOURCE_RETENTION_EXACT) {
      EXPECT_EQ(std::string(reinterpret_cast<const char*>(contents.data),
                            contents.data_length),
                header_text);
      EXPECT_NE(formatted.find("3 |   output[0] = input[0] / input[1];"),
                std::string::npos);
    } else {
      EXPECT_EQ(contents.data_length, 0u);
      EXPECT_EQ(failure->range.start, 0u);
      EXPECT_EQ(failure->range.end, 0u);
      EXPECT_EQ(formatted.find("3 |   output[0] = input[0] / input[1];"),
                std::string::npos);
    }
  }
}

}  // namespace
