// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <initializer_list>
#include <string>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/artifact.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/product.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using LinkIndexBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_free>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

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

struct KernelRequestCapture {
  // Requests transferred by one command-product construction.
  std::vector<RequestPtr> requests;
};

loomc_status_t CaptureKernelRequest(void* user_data, loomc_request_t* request) {
  KernelRequestCapture* capture = static_cast<KernelRequestCapture*>(user_data);
  capture->requests.push_back(RequestPtr(request));
  return loomc_ok_status();
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
}

TargetEnvironmentPtr CreateAmdgpuTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateAmdgpuContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .target_environment = target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  loomc_status_t status = loomc_context_create(
      &context_options, loomc_allocator_system(), &context);
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
  loomc_compiler_t* compiler = nullptr;
  loomc_status_t status = loomc_compiler_create(
      context, nullptr, loomc_allocator_system(), &compiler);
  LOOMC_EXPECT_OK(status);
  return CompilerPtr(compiler);
}

SourcePtr CreateSource(loomc_source_format_t format, const char* identifier,
                       const void* contents, size_t contents_length) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = format,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(contents, contents_length),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  return CreateSource(LOOMC_SOURCE_FORMAT_TEXT, identifier, contents,
                      strlen(contents));
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        ToString(artifact->format) == format) {
      return artifact;
    }
  }
  return nullptr;
}

LinkIndexPtr CreateLinkIndex(
    loomc_context_t* context, std::initializer_list<loomc_source_t*> sources,
    loomc_link_provider_role_t role = LOOMC_LINK_PROVIDER_ROLE_INPUT) {
  loomc_link_index_builder_t* raw_builder = nullptr;
  LOOMC_EXPECT_OK(loomc_link_index_builder_allocate(
      context, nullptr, loomc_allocator_system(), &raw_builder));
  LinkIndexBuilderPtr builder(raw_builder);
  for (loomc_source_t* source : sources) {
    const loomc_link_index_source_options_t source_options = {
        .provider_name = loomc_source_identifier(source),
        .role = role,
    };
    LOOMC_EXPECT_OK(loomc_link_index_builder_add_source(
        builder.get(), source, &source_options, nullptr));
  }

  loomc_link_index_t* raw_index = nullptr;
  loomc_result_t* raw_index_result = nullptr;
  LOOMC_EXPECT_OK(loomc_link_index_builder_finish(builder.get(), &raw_index,
                                                  &raw_index_result));
  LinkIndexPtr index(raw_index);
  ResultPtr index_result(raw_index_result);
  ExpectSucceededResult(index_result.get());
  return index;
}

loomc_request_root_t FindRequestRoot(const loomc_link_index_t* index,
                                     const char* name) {
  for (loomc_host_size_t i = 0; i < loomc_link_index_symbol_count(index); ++i) {
    loomc_link_index_symbol_t symbol = {};
    EXPECT_TRUE(loomc_link_index_symbol_at(index, i, &symbol));
    if (ToString(symbol.name) == name) {
      return {
          .module_ordinal =
              static_cast<uint32_t>(symbol.provider_module_ordinal),
          .symbol_ordinal = static_cast<uint32_t>(symbol.module_symbol_ordinal),
      };
    }
  }
  ADD_FAILURE() << "Missing request root: " << name;
  return {};
}

TargetProfilePtr CreateTargetProfile(
    loomc_target_environment_t* target_environment, const char* target,
    loomc_amdgpu_amdhsa_feature_states_t amdhsa_features = {}) {
  loomc_amdgpu_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view(target),
      .identity =
          {
              .target = loomc_make_cstring_view(target),
              .amdhsa_features = amdhsa_features,
          },
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      target_environment, &profile_options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
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

SourcePtr SerializeModule(const loomc_module_t* module,
                          loomc_source_format_t format) {
  loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = format,
      .identifier = format == LOOMC_SOURCE_FORMAT_BYTECODE
                        ? loomc_make_cstring_view("compiled.loombc")
                        : loomc_make_cstring_view("compiled.loom"),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

PassProgramPtr CreatePreparedLowPassProgram(loomc_context_t* context) {
  loomc_target_pipeline_options_t pipeline_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(pipeline_options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("amdgpu-prepared-low-test"),
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      context, &pipeline_options, loomc_allocator_system(), &pass_program,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return PassProgramPtr(pass_program);
}

class CommandTargetOptionsTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(CommandTargetOptionsTest, PreservesProfilesInIndependentKernelRequests) {
  const auto [bind_declarations, build_request] = GetParam();
  TargetEnvironmentPtr environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr first_profile =
      CreateTargetProfile(environment.get(), "gfx1151");
  TargetProfilePtr second_profile =
      CreateTargetProfile(environment.get(), "gfx942");
  SourcePtr text_source = CreateTextSource("command_targets.loom", R"(
target.decl @first_device
target.decl @second_device

kernel.def target(@first_device) @first() {
  %c1 = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%size, %c1, %c1) workgroup_size(%size, %c1, %c1) : index
} launch() {
  kernel.return
}

kernel.def target(@second_device) @second() {
  %c1 = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%size, %c1, %c1) workgroup_size(%size, %c1, %c1) : index
} launch() {
  kernel.return
}

command.def public @dispatch() launch() {
  kernel.launch @first() : ()
  kernel.launch @second() : ()
  command.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), text_source.get());
  SourcePtr source =
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  LinkIndexPtr index = CreateLinkIndex(context.get(), {source.get()});
  const loomc_target_binding_t bindings[] = {
      {
          .target_symbol = loomc_make_cstring_view("first_device"),
          .target_profile = first_profile.get(),
      },
      {
          .target_symbol = loomc_make_cstring_view("second_device"),
          .target_profile = second_profile.get(),
      },
  };
  const loomc_target_specialization_t specializations[] = {
      {
          .function_symbol = loomc_make_cstring_view("first"),
          .target_profile = first_profile.get(),
      },
      {
          .function_symbol = loomc_make_cstring_view("second"),
          .target_profile = second_profile.get(),
      },
      {
          .function_symbol = loomc_make_cstring_view("dispatch"),
          .target_profile = second_profile.get(),
      },
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .specializations =
          bind_declarations ? &specializations[2] : specializations,
      .specialization_count = bind_declarations ? 1u : 3u,
      .target_bindings = bind_declarations ? bindings : nullptr,
      .target_binding_count = bind_declarations ? 2u : 0u,
  };
  KernelRequestCapture capture;
  const loomc_request_sink_t sink = {
      .publish = CaptureKernelRequest,
      .user_data = &capture,
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  if (build_request) {
    const loomc_request_root_t root = FindRequestRoot(index.get(), "dispatch");
    loomc_request_t* request = nullptr;
    LOOMC_ASSERT_OK(loomc_request_create(loomc_cmd_program_product_descriptor(),
                                         source.get(), &root, 1, nullptr, 0,
                                         loomc_allocator_system(), &request));
    RequestPtr request_owner(request);
    const loomc_cmd_program_request_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_REQUEST_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING,
        .request_sink = sink,
    };
    LOOMC_ASSERT_OK(loomc_cmd_program_product_build_request(
        context.get(), workspace.get(), request, &options,
        loomc_allocator_system(), &product, &result));
  } else {
    const loomc_cmd_program_product_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .link_index = index.get(),
        .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING,
        .request_sink = sink,
    };
    LOOMC_ASSERT_OK(loomc_cmd_program_product_build(workspace.get(), &options,
                                                    loomc_allocator_system(),
                                                    &product, &result));
  }
  ProductPtr product_owner(product);
  ResultPtr result_owner(result);
  ExpectSucceededResult(result);
  ASSERT_EQ(capture.requests.size(), 2u);
  ASSERT_EQ(loomc_result_artifact_count(result), 1u);
  const loomc_artifact_t* listing = loomc_result_artifact_at(result, 0);
  EXPECT_EQ(listing->kind, LOOMC_ARTIFACT_KIND_TEXT);
  EXPECT_EQ(ToString(listing->format),
            LOOMC_ARTIFACT_FORMAT_CMD_PROGRAM_LISTING);
  EXPECT_GT(loomc_byte_sequence_length(listing->contents), 0u);

  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pipeline = CreatePreparedLowPassProgram(context.get());
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };
  for (const RequestPtr& request : capture.requests) {
    loomc_request_binding_t binding = {};
    ASSERT_TRUE(loomc_request_binding_at(request.get(), 0, &binding));
    loomc_cmd_entry_requirement_t requirement = {};
    ASSERT_TRUE(loomc_cmd_program_product_entry_requirement_at(
        product, binding.requirement_ordinal, &requirement));
    const bool first = ToString(requirement.symbol) == "first";
    ASSERT_TRUE(first || ToString(requirement.symbol) == "second");
    loomc_product_t* native_product = nullptr;
    loomc_result_t* native_result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_request(
        compiler.get(), workspace.get(), pipeline.get(), request.get(),
        &compile_options, loomc_allocator_system(), &native_product,
        &native_result));
    ProductPtr native_product_owner(native_product);
    ResultPtr native_result_owner(native_result);
    ExpectSucceededResult(native_result);
    const loomc_artifact_t* artifact =
        loomc_product_artifact_at(native_product, 0);
    ASSERT_NE(artifact, nullptr);
    const std::string text = ToString(artifact->contents);
    EXPECT_NE(text.find(first ? "workgroup_size(32, 1, 1)"
                              : "workgroup_size(64, 1, 1)"),
              std::string::npos)
        << text;
    EXPECT_NE(text.find(first ? "target<amdgpu.rdna3_5.core>"
                              : "target<amdgpu.cdna3.core>"),
              std::string::npos)
        << text;
  }
}

INSTANTIATE_TEST_SUITE_P(EntryPaths, CommandTargetOptionsTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

TEST(TargetCmdSpecializationTest,
     CommandKernelRequestCompilesForIndependentTargetProfiles) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  TargetProfilePtr wave32_profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  TargetProfilePtr wave64_profile =
      CreateTargetProfile(target_environment.get(), "gfx942");
  SourcePtr source = CreateTextSource("command_request.loom", R"(
kernel.def @record_subgroup_size() {
  %c1 = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%size, %c1, %c1) : index
} launch() {
  kernel.return
}

command.def public @dispatch() launch() {
  kernel.launch @record_subgroup_size() : ()
  command.return
}
)");
  LinkIndexPtr index = CreateLinkIndex(context.get(), {source.get()});

  KernelRequestCapture request_capture;
  const loomc_cmd_program_product_options_t product_options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      .structure_size = sizeof(product_options),
      .next = nullptr,
      .link_index = index.get(),
      .root_symbol_ordinals = nullptr,
      .root_symbol_count = 0,
      .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      .config = {},
      .request_sink =
          {
              .publish = CaptureKernelRequest,
              .user_data = &request_capture,
          },
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* product_result = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_product_build(
      workspace.get(), &product_options, loomc_allocator_system(), &product,
      &product_result));
  ProductPtr product_ptr(product);
  ResultPtr product_result_ptr(product_result);
  ExpectSucceededResult(product_result_ptr.get());
  ASSERT_EQ(loomc_cmd_program_product_program_count(product_ptr.get()), 1u);
  ASSERT_EQ(request_capture.requests.size(), 1u);
  loomc_request_t* request = request_capture.requests[0].get();
  ASSERT_EQ(loomc_request_root_count(request), 1u);
  ASSERT_EQ(loomc_request_binding_count(request), 1u);
  loomc_request_binding_t request_binding = {};
  ASSERT_TRUE(loomc_request_binding_at(request, 0, &request_binding));
  EXPECT_EQ(request_binding.requirement_ordinal, 0u);
  EXPECT_EQ(request_binding.root_ordinal, 0u);
  loomc_source_t* request_source = loomc_request_source(request);
  ASSERT_NE(request_source, nullptr);
  EXPECT_EQ(loomc_source_format(request_source), LOOMC_SOURCE_FORMAT_BYTECODE);

  product_result_ptr.reset();
  product_ptr.reset();
  index.reset();
  source.reset();
  pass_program.reset();
  compiler.reset();
  workspace.reset();
  context.reset();

  ContextPtr request_context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr request_workspace = CreateWorkspace();
  CompilerPtr request_compiler = CreateCompiler(request_context.get());
  PassProgramPtr request_pass_program =
      CreatePreparedLowPassProgram(request_context.get());

  struct TargetCase {
    // Prepared target profile used for this independent compilation.
    loomc_target_profile_t* profile;
    // Target-specific Low descriptor set expected after compilation.
    const char* descriptor_set;
    // Target subgroup size expected in the specialized kernel geometry.
    const char* workgroup_size;
  };
  const TargetCase target_cases[] = {
      {
          wave32_profile.get(),
          "target<amdgpu.rdna3_5.core>",
          "workgroup_size(32, 1, 1)",
      },
      {
          wave64_profile.get(),
          "target<amdgpu.cdna3.core>",
          "workgroup_size(64, 1, 1)",
      },
  };
  for (const TargetCase& target_case : target_cases) {
    ModulePtr request_module = DeserializeModule(
        request_context.get(), request_workspace.get(), request_source);
    const loomc_target_specialization_t specialization = {
        .function_symbol = loomc_make_cstring_view("record_subgroup_size"),
        .target_profile = target_case.profile,
    };
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .specializations = &specialization,
        .specialization_count = 1,
    };
    const loomc_compile_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = &target_options,
        .module_name = loomc_make_cstring_view("command_request"),
        .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
    };
    loomc_result_t* compile_result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_module(
        request_compiler.get(), request_workspace.get(),
        request_pass_program.get(), request_module.get(), &compile_options,
        loomc_allocator_system(), &compile_result));
    ResultPtr compile_result_ptr(compile_result);
    ExpectSucceededResult(compile_result_ptr.get());
    const loomc_artifact_t* text_artifact =
        FindArtifact(compile_result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                     LOOMC_ARTIFACT_FORMAT_LOOM_TEXT);
    ASSERT_NE(text_artifact, nullptr);
    const std::string module_text = ToString(text_artifact->contents);
    EXPECT_NE(module_text.find(target_case.descriptor_set), std::string::npos)
        << module_text;
    EXPECT_NE(module_text.find(target_case.workgroup_size), std::string::npos)
        << module_text;
    EXPECT_EQ(module_text.find("target.subgroup.size"), std::string::npos)
        << module_text;
  }
}

class CommandTargetIdentityTest : public ::testing::TestWithParam<bool> {};

TEST_P(CommandTargetIdentityTest,
       PreservesPrivateIdentityAndGlobalDeclarationAliases) {
  TargetEnvironmentPtr environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr device_profile =
      CreateTargetProfile(environment.get(), "gfx1151");
  TargetProfilePtr host_profile =
      CreateTargetProfile(environment.get(), "gfx942");
  SourcePtr commands = CreateTextSource("commands.loom", R"(
target.decl @shared
kernel.decl @record(%groups: index) launch(%output: buffer)

func.def inline @read_width() -> (index) {
  %size = target.subgroup.size : index
  func.return %size : index
}

command.def public target(@shared) @dispatch() launch(%output: buffer) {
  %groups = func.call pure @read_width() : () -> (index)
  kernel.launch @record[%groups](%output) : [index](buffer)
  command.return
}
)");
  SourcePtr kernels = CreateTextSource("kernels.loom", R"(
target.decl @shared
target.decl @device

func.def inline target(@shared) @read_width() -> (i32) {
  %size = target.subgroup.size : index
  %value = index.cast %size : index to i32
  func.return %value : i32
}

kernel.def target(@device) @record(%groups: index) {
  %c1 = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%size, %c1, %c1) : index
} launch(%output: buffer) {
  %value = func.call @read_width() : () -> (i32)
  %offset = index.constant 0 : offset
  %index = index.constant 0 : index
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%offset] : buffer -> view<1xi32>
  view.store %value, %view[%index] : i32, view<1xi32>
  kernel.return
}
)");
  const loomc_target_binding_t bindings[] = {
      {
          .target_symbol = loomc_make_cstring_view("shared"),
          .target_profile = device_profile.get(),
      },
      {
          .target_symbol = loomc_make_cstring_view("device"),
          .target_profile = device_profile.get(),
      },
  };
  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("read_width"),
      .target_profile = host_profile.get(),
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .specializations = &specialization,
      .specialization_count = 1,
      .target_bindings = bindings,
      .target_binding_count = 2,
  };
  KernelRequestCapture capture;
  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  if (GetParam()) {
    ModulePtr command_module =
        DeserializeModule(context.get(), workspace.get(), commands.get());
    SourcePtr source =
        SerializeModule(command_module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
    LinkIndexPtr roots = CreateLinkIndex(context.get(), {source.get()});
    const loomc_request_root_t root = FindRequestRoot(roots.get(), "dispatch");
    loomc_request_t* request = nullptr;
    LOOMC_ASSERT_OK(loomc_request_create(loomc_cmd_program_product_descriptor(),
                                         source.get(), &root, 1, nullptr, 0,
                                         loomc_allocator_system(), &request));
    RequestPtr request_owner(request);
    LinkIndexPtr library = CreateLinkIndex(context.get(), {kernels.get()},
                                           LOOMC_LINK_PROVIDER_ROLE_LIBRARY);
    const loomc_cmd_program_request_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_REQUEST_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .library_index = library.get(),
        .request_sink =
            {
                .publish = CaptureKernelRequest,
                .user_data = &capture,
            },
    };
    LOOMC_ASSERT_OK(loomc_cmd_program_product_build_request(
        context.get(), workspace.get(), request, &options,
        loomc_allocator_system(), &product, &result));
  } else {
    LinkIndexPtr index =
        CreateLinkIndex(context.get(), {commands.get(), kernels.get()});
    const loomc_cmd_program_product_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .link_index = index.get(),
        .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
        .request_sink =
            {
                .publish = CaptureKernelRequest,
                .user_data = &capture,
            },
    };
    LOOMC_ASSERT_OK(loomc_cmd_program_product_build(workspace.get(), &options,
                                                    loomc_allocator_system(),
                                                    &product, &result));
  }
  ProductPtr product_owner(product);
  ResultPtr result_owner(result);
  ExpectSucceededResult(result);
  ASSERT_EQ(capture.requests.size(), 1u);
  ModulePtr request_module =
      DeserializeModule(context.get(), workspace.get(),
                        loomc_request_source(capture.requests[0].get()));
  SourcePtr request_text =
      SerializeModule(request_module.get(), LOOMC_SOURCE_FORMAT_TEXT);
  const std::string text = ToString(loomc_source_contents(request_text.get()));
  EXPECT_EQ(text.find("target(@shared)"), std::string::npos) << text;
  EXPECT_EQ(text.find("amdgpu.cdna3"), std::string::npos) << text;

  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pipeline = CreatePreparedLowPassProgram(context.get());
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };
  loomc_product_t* native_product = nullptr;
  loomc_result_t* native_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler.get(), workspace.get(), pipeline.get(),
      capture.requests[0].get(), &compile_options, loomc_allocator_system(),
      &native_product, &native_result));
  ProductPtr native_product_owner(native_product);
  ResultPtr native_result_owner(native_result);
  ExpectSucceededResult(native_result);
  const loomc_artifact_t* artifact =
      loomc_product_artifact_at(native_product, 0);
  ASSERT_NE(artifact, nullptr);
  const std::string native_text = ToString(artifact->contents);
  EXPECT_NE(native_text.find("workgroup_size(32, 1, 1)"), std::string::npos)
      << native_text;
}

INSTANTIATE_TEST_SUITE_P(EntryPaths, CommandTargetIdentityTest,
                         ::testing::Bool());

TEST(TargetCmdSpecializationTest, RejectsUnknownInputBeforePublishingRequests) {
  TargetEnvironmentPtr environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile = CreateTargetProfile(environment.get(), "gfx1151");
  SourcePtr source = CreateTextSource("unknown_target.loom", R"(
kernel.def @entry() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch() {
  kernel.return
}

command.def public @dispatch() launch() {
  kernel.launch @entry() : ()
  command.return
}
)");
  LinkIndexPtr index = CreateLinkIndex(context.get(), {source.get()});
  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("missing"),
      .target_profile = profile.get(),
  };
  const loomc_target_binding_t binding = {
      .target_symbol = loomc_make_cstring_view("missing"),
      .target_profile = profile.get(),
  };
  for (bool bind_declaration : {false, true}) {
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .specializations = bind_declaration ? nullptr : &specialization,
        .specialization_count = bind_declaration ? 0u : 1u,
        .target_bindings = bind_declaration ? &binding : nullptr,
        .target_binding_count = bind_declaration ? 1u : 0u,
    };
    KernelRequestCapture capture;
    const loomc_cmd_program_product_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .link_index = index.get(),
        .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
        .request_sink =
            {
                .publish = CaptureKernelRequest,
                .user_data = &capture,
            },
    };
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND,
                           loomc_cmd_program_product_build(
                               workspace.get(), &options,
                               loomc_allocator_system(), &product, &result));
    ProductPtr product_owner(product);
    ResultPtr result_owner(result);
    EXPECT_EQ(product, nullptr);
    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(capture.requests.empty());
  }
}

}  // namespace
