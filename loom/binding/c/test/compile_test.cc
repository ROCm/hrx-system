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
#include "loom/binding/c/src/product.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/product.h"
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

using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;

using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;

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

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result = ToString(contents);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
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

SourcePtr SerializeModuleToBytecode(const loomc_module_t* module,
                                    const char* identifier) {
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
      /*.identifier=*/loomc_make_cstring_view(identifier),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

void DestroyAlternateProduct(loomc_product_t* base_product) {
  loomc_allocator_free(loomc_allocator_system(), base_product);
}

const loomc_product_descriptor_t kAlternateProductDescriptor = {
    /*.destroy=*/DestroyAlternateProduct,
};

RequestPtr CreateSingleRootRequest(
    SourcePtr source, const loomc_product_descriptor_t* product_descriptor =
                          loomc_compiled_module_product_descriptor()) {
  const loomc_request_root_t root = {
      /*.module_ordinal=*/0,
      /*.symbol_ordinal=*/0,
  };
  loomc_request_t* request = nullptr;
  loomc_status_t status = loomc_request_create(
      product_descriptor, source.get(), &root, 1, /*bindings=*/nullptr,
      /*binding_count=*/0, loomc_allocator_system(), &request);
  LOOMC_EXPECT_OK(status);
  return RequestPtr(request);
}

RequestPtr CreateRootedRequest(loomc_source_t* source,
                               loomc_request_root_t root) {
  loomc_request_t* request = nullptr;
  loomc_status_t status = loomc_request_create(
      loomc_compiled_module_product_descriptor(), source, &root, 1,
      /*bindings=*/nullptr, /*binding_count=*/0, loomc_allocator_system(),
      &request);
  LOOMC_EXPECT_OK(status);
  return RequestPtr(request);
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

ModulePtr CreateConfigConsumerModule(loomc_context_t* context,
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

ModulePtr CreateConfigProviderModule(loomc_context_t* context,
                                     loomc_workspace_t* workspace,
                                     const char* contents) {
  SourcePtr source = CreateTextSource("config-provider.loom", contents);
  return DeserializeModule(context, workspace, source.get());
}

ModulePtr RoundTripModuleThroughBytecode(loomc_context_t* context,
                                         loomc_workspace_t* workspace,
                                         ModulePtr module) {
  SourcePtr source = SerializeModuleToBytecode(module.get(), "config.loombc");
  module.reset();
  loomc_module_t* round_tripped_module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_bytecode_from_source(
      context, workspace, source.get(), /*options=*/nullptr,
      loomc_allocator_system(), &round_tripped_module, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(round_tripped_module);
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

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.artifact_flags=*/0,
      /*.config_flags=*/0,
      /*.config_module=*/nullptr,
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

TEST(CompileTest, CompileModuleAppliesBytecodeConfigModule) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigConsumerModule(context.get(), workspace.get());
  ModulePtr config_module = RoundTripModuleThroughBytecode(
      context.get(), workspace.get(),
      CreateConfigProviderModule(context.get(), workspace.get(), R"(
config.def @model36.model.hidden_size = 4096 : index
config.def @model36.unused = 1 : index
)"));
  const std::string config_text_before =
      SerializeModuleToText(config_module.get());
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      /*.config_module=*/config_module.get(),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  EXPECT_EQ(SerializeModuleToText(config_module.get()), config_text_before);

  config_module.reset();
  std::string text = SerializeModuleToText(module.get());
  EXPECT_NE(text.find("config.def @model36.model.hidden_size = 4096 : index"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
}

TEST(CompileTest, CompileModuleEmitsRequestedArtifacts) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigConsumerModule(context.get(), workspace.get());
  ModulePtr config_module = CreateConfigProviderModule(
      context.get(), workspace.get(),
      "config.def @model36.model.hidden_size = 4096 : index\n");
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT |
          LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
          LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
      /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      /*.config_module=*/config_module.get(),
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
  EXPECT_NE(loomc_byte_sequence_length(bytecode_artifact->contents), 0u);
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
  EXPECT_NE(report.find(R"("has_config_module":true)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_definition_count":1)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_materialized_count":1)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_ignored_count":0)"), std::string::npos);
}

TEST(CompileTest, CompileRequestReturnsOwnedProduct) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  RequestPtr request = CreateSingleRootRequest(
      SerializeModuleToBytecode(module.get(), "request.loombc"));
  module.reset();

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("request-product"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler.get(), workspace.get(), pass_program.get(), request.get(),
      &options, loomc_allocator_system(), &product, &result));
  ProductPtr product_ptr(product);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_NE(product_ptr.get(), nullptr);
  EXPECT_EQ(loomc_product_descriptor(product_ptr.get()),
            loomc_compiled_module_product_descriptor());
  EXPECT_EQ(loomc_product_artifact_count(product_ptr.get()), 1u);
  EXPECT_EQ(loomc_product_export_count(product_ptr.get()), 1u);
  EXPECT_EQ(loomc_product_requirement_count(product_ptr.get()), 0u);

  request.reset();
  result_ptr.reset();
  const loomc_artifact_t* artifact =
      loomc_product_artifact_at(product_ptr.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_MODULE);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  EXPECT_EQ(ToString(artifact->identifier), "request-product.loombc");
  EXPECT_NE(loomc_byte_sequence_length(artifact->contents), 0u);
}

TEST(CompileTest, CompileRequestAppliesSharedConfigModule) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigConsumerModule(context.get(), workspace.get());
  RequestPtr request = CreateSingleRootRequest(
      SerializeModuleToBytecode(module.get(), "configured-request.loombc"));
  module.reset();
  ModulePtr config_module = RoundTripModuleThroughBytecode(
      context.get(), workspace.get(),
      CreateConfigProviderModule(context.get(), workspace.get(), R"(
config.def @model36.model.hidden_size = 4096 : index
)"));

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("configured-request"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
      /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      /*.config_module=*/config_module.get(),
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_request(
      compiler.get(), workspace.get(), pass_program.get(), request.get(),
      &options, loomc_allocator_system(), &product, &result);
  LOOMC_EXPECT_OK(status);
  ProductPtr product_ptr(product);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_NE(product_ptr.get(), nullptr);
  ASSERT_EQ(loomc_product_artifact_count(product_ptr.get()), 1u);

  config_module.reset();
  const loomc_artifact_t* artifact =
      loomc_product_artifact_at(product_ptr.get(), 0);
  ASSERT_NE(artifact, nullptr);
  const std::string text = ToString(artifact->contents);
  EXPECT_NE(text.find("config.def @model36.model.hidden_size = 4096 : index"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
}

TEST(CompileTest, CompileRequestRejectsUnavailableRoots) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  SourcePtr source = SerializeModuleToBytecode(module.get(), "roots.loombc");
  module.reset();

  const loomc_request_root_t unavailable_roots[] = {
      {/*.module_ordinal=*/1, /*.symbol_ordinal=*/0},
      {/*.module_ordinal=*/0, /*.symbol_ordinal=*/UINT32_MAX},
  };
  for (const loomc_request_root_t root : unavailable_roots) {
    RequestPtr request = CreateRootedRequest(source.get(), root);
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_compile_request(
        compiler.get(), workspace.get(), pass_program.get(), request.get(),
        /*options=*/nullptr, loomc_allocator_system(), &product, &result);
    LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
    EXPECT_EQ(product, nullptr);
    EXPECT_EQ(result, nullptr);
  }
}

TEST(CompileTest, CompileRequestRejectsUnsupportedRootGoal) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  SourcePtr source = SerializeModuleToBytecode(module.get(), "goal.loombc");
  module.reset();

  const loomc_request_root_t requested_root = {
      /*.module_ordinal=*/0,
      /*.symbol_ordinal=*/0,
      /*.goal=*/17,
  };
  RequestPtr request = CreateRootedRequest(source.get(), requested_root);
  loomc_request_root_t retained_root = {};
  ASSERT_TRUE(loomc_request_root_at(request.get(), 0, &retained_root));
  EXPECT_EQ(retained_root.goal, requested_root.goal);

  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_compile_request(compiler.get(), workspace.get(), pass_program.get(),
                            request.get(), /*options=*/nullptr,
                            loomc_allocator_system(), &product, &result));
  EXPECT_EQ(product, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST(CompileTest, RequestCreationRejectsNonzeroReservedRootFields) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  SourcePtr source =
      SerializeModuleToBytecode(module.get(), "reserved-root.loombc");
  const loomc_request_root_t root = {
      /*.module_ordinal=*/0,
      /*.symbol_ordinal=*/0,
      /*.goal=*/LOOMC_REQUEST_ROOT_GOAL_DEFAULT,
      /*.reserved=*/1,
  };
  loomc_request_t* request = reinterpret_cast<loomc_request_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_request_create(loomc_compiled_module_product_descriptor(),
                           source.get(), &root, 1, /*bindings=*/nullptr,
                           /*binding_count=*/0, loomc_allocator_system(),
                           &request));
  EXPECT_EQ(request, nullptr);
}

TEST(CompileTest, CompileRequestRejectsAnotherProductContract) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  RequestPtr request = CreateSingleRootRequest(
      SerializeModuleToBytecode(module.get(), "wrong-contract.loombc"),
      &kAlternateProductDescriptor);

  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_request(
      compiler.get(), workspace.get(), pass_program.get(), request.get(),
      /*options=*/nullptr, loomc_allocator_system(), &product, &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(product, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST(CompileTest, CompileRequestReturnsFailedParseResultWithoutProduct) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  const char malformed_bytecode[] = "not Loom bytecode";
  loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(source_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
      /*.identifier=*/loomc_make_cstring_view("malformed.loombc"),
      /*.contents=*/
      loomc_make_byte_span(malformed_bytecode, sizeof(malformed_bytecode) - 1),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  RequestPtr request = CreateSingleRootRequest(SourcePtr(source));

  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler.get(), workspace.get(), pass_program.get(), request.get(),
      /*options=*/nullptr, loomc_allocator_system(), &product, &result));
  ProductPtr product_ptr(product);
  ResultPtr result_ptr(result);
  EXPECT_EQ(product_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_NE(loomc_result_diagnostic_count(result_ptr.get()), 0u);
}

TEST(CompileTest, CompileModuleIgnoresUnusedConfig) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  ModulePtr config_module = CreateConfigProviderModule(
      context.get(), workspace.get(), "config.def @tile_m = 128 : index\n");
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
      /*.config_flags=*/0,
      /*.config_module=*/config_module.get(),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_EQ(loomc_result_artifact_count(result_ptr.get()), 1u);
  const loomc_artifact_t* report_artifact = FindArtifact(
      result_ptr.get(), LOOMC_ARTIFACT_KIND_REPORT, LOOMC_ARTIFACT_FORMAT_JSON);
  ASSERT_NE(report_artifact, nullptr);
  std::string report = ToString(report_artifact->contents);
  EXPECT_NE(report.find(R"("state":"succeeded")"), std::string::npos);
  EXPECT_NE(report.find(R"("diagnostic_count":0)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_definition_count":1)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_materialized_count":0)"), std::string::npos);
  EXPECT_NE(report.find(R"("config_ignored_count":1)"), std::string::npos);
}

TEST(CompileTest, CompileModuleReportsUnresolvedConfigAsResultDiagnostic) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigConsumerModule(context.get(), workspace.get());

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      /*.config_module=*/nullptr,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectFailedResultCode(result_ptr.get(), "CONFIG/INVALID");
}

TEST(CompileTest, CompileModuleReportsNonConfigProviderAsDiagnostic) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  ModulePtr config_module = CreateValidModule(context.get(), workspace.get());

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config_flags=*/0,
      /*.config_module=*/config_module.get(),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectFailedResultCode(result_ptr.get(), "CONFIG/INVALID");
}

TEST(CompileTest, CompileModuleReportsConfigConstraintFailure) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateConfigConsumerModule(context.get(), workspace.get());
  ModulePtr config_module = CreateConfigProviderModule(
      context.get(), workspace.get(),
      "config.def @model36.model.hidden_size = 4095 : index\n");

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      /*.config_module=*/config_module.get(),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectFailedResultCode(result_ptr.get(), "CONFIG/INVALID");
}

TEST(CompileTest, CompileModuleRejectsProgramAsConfigModule) {
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
      /*.config_flags=*/0,
      /*.config_module=*/module.get(),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

TEST(CompileTest, CompileModuleRejectsConfigFromAnotherContext) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());
  ContextPtr other_context = CreateContext();
  WorkspacePtr other_workspace = CreateWorkspace();
  ModulePtr config_module =
      CreateConfigProviderModule(other_context.get(), other_workspace.get(),
                                 "config.def @unused = 1 : index\n");

  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.config_flags=*/0,
      /*.config_module=*/config_module.get(),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

TEST(CompileTest, CompileModuleRejectsUnknownConfigPolicy) {
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
      /*.config_flags=*/UINT32_MAX,
      /*.config_module=*/nullptr,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(result, nullptr);
}

TEST(CompileTest, CompileModuleRejectsArtifactIdentifierLengthOverflow) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  ModulePtr module = CreateValidModule(context.get(), workspace.get());

  const char module_name_storage = 'x';
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/
      loomc_make_string_view(&module_name_storage, LOOMC_HOST_SIZE_MAX),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_RESOURCE_EXHAUSTED, status);
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
