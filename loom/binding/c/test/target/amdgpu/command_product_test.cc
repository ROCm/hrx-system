// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/binding/c/test/target/amdgpu/command_profile_testdata.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/target/cmd.h"
#include "target_test_util.h"

namespace loomc::testing::amdgpu {
namespace {

struct KernelRequestCapture {
  // Requests transferred by one command-product construction.
  std::vector<RequestPtr> requests;
};

loomc_status_t CaptureKernelRequest(void* user_data, loomc_request_t* request) {
  KernelRequestCapture* capture = static_cast<KernelRequestCapture*>(user_data);
  capture->requests.push_back(RequestPtr(request));
  return loomc_ok_status();
}

class AmdgpuCommandProductTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(AmdgpuCommandProductTest,
       LateTargetBindingPreservesLaunchConfiguration) {
  const auto [bind_declaration, bytecode] = GetParam();
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  SourcePtr source = CreateTextSource("specialized_command_request.loom", R"(
target.decl @device

kernel.def target(@device) @record_subgroup_size() {
  %one = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%size, %one, %one) workgroup_size(%size, %one, %one) : index
} launch() {
  kernel.return
}

command.def public @dispatch() launch() {
  kernel.launch @record_subgroup_size() : ()
  command.return
}
)");
  if (bytecode) {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());

  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("record_subgroup_size"),
      .target_profile = profile.get(),
  };

  const loomc_target_binding_t target_binding = {
      .target_symbol = loomc_make_cstring_view("device"),
      .target_profile = profile.get(),
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .specializations = bind_declaration ? nullptr : &specialization,
      .specialization_count = bind_declaration ? 0u : 1u,
      .target_bindings = bind_declaration ? &target_binding : nullptr,
      .target_binding_count = bind_declaration ? 1u : 0u,
  };
  KernelRequestCapture request_capture;
  const loomc_cmd_program_product_options_t product_options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      .structure_size = sizeof(product_options),
      .next = &target_options,
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

  loomc_cmd_program_t command = {};
  ASSERT_TRUE(
      loomc_cmd_program_product_program_at(product_ptr.get(), 0, &command));
  const std::string command_bytes = ToString(command.artifact.contents);
  loom_cmd_program_t decoded = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(command_bytes.data(), command_bytes.size()),
      &decoded));
  ASSERT_EQ(decoded.commands.count, 1u);
  const auto dispatch = loom_cmd_program_command_at(&decoded, 0);
  EXPECT_EQ(dispatch.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_x, 32u);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_y, 1u);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_z, 1u);

  // The independent request owns its concrete target after planning ends.
  product_result_ptr.reset();
  product_ptr.reset();
  index.reset();
  source.reset();
  profile.reset();
  workspace.reset();
  context.reset();
  context = CreateAmdgpuContext(target_environment.get());
  workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = nullptr,
      .module_name = loomc_make_cstring_view("specialized_command_request"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };
  loomc_product_t* kernel_product = nullptr;
  loomc_result_t* compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler.get(), workspace.get(), pass_program.get(),
      request_capture.requests[0].get(), &compile_options,
      loomc_allocator_system(), &kernel_product, &compile_result));
  ProductPtr kernel_product_ptr(kernel_product);
  ResultPtr compile_result_ptr(compile_result);
  ExpectSucceededResult(compile_result_ptr.get());
  const loomc_artifact_t* text_artifact =
      loomc_product_artifact_at(kernel_product_ptr.get(), 0);
  ASSERT_NE(text_artifact, nullptr);
  const std::string module_text = ToString(text_artifact->contents);
  EXPECT_NE(module_text.find("target<amdgpu.rdna3_5.core>"), std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("workgroup_size(32, 1, 1)"), std::string::npos)
      << module_text;
}

INSTANTIATE_TEST_SUITE_P(SourceFormsAndBindings, AmdgpuCommandProductTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

SourcePtr CreateCommandProfileSource(const char* name) {
  const iree_file_toc_t* files = loomc_amdgpu_command_profile_testdata_create();
  for (size_t i = 0; i < loomc_amdgpu_command_profile_testdata_size(); ++i) {
    if (std::strcmp(files[i].name, name) == 0) {
      return CreateSource(LOOMC_SOURCE_FORMAT_TEXT, name, files[i].data,
                          files[i].size);
    }
  }
  ADD_FAILURE() << "Missing command profile source: " << name;
  return {};
}

enum class CommandSource { kText, kBytecode, kRequest, kModule };

// Exercises every command-product admission path with the same command root.
loomc_status_t BuildCommandProduct(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_source_t* source, const loomc_cmd_program_product_options_t& options,
    CommandSource source_kind, loomc_product_t** out_product,
    loomc_result_t** out_result) {
  if (source_kind == CommandSource::kText ||
      source_kind == CommandSource::kBytecode) {
    return loomc_cmd_program_product_build(
        workspace, &options, loomc_allocator_system(), out_product, out_result);
  }
  if (source_kind == CommandSource::kModule) {
    ModulePtr module = DeserializeModule(context, workspace, source);
    const loomc_string_view_t root = loomc_make_cstring_view("dispatch");
    const loomc_cmd_program_module_options_t module_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS,
        .structure_size = sizeof(module_options),
        .next = options.next,
        .provider_name = loomc_make_cstring_view("command-profile"),
        .root_symbols = &root,
        .root_symbol_count = 1,
        .flags = options.flags,
        .config = options.config,
        .target_profile = options.target_profile,
        .request_sink = options.request_sink,
    };
    return loomc_cmd_program_product_build_module(
        workspace, module.get(), &module_options, loomc_allocator_system(),
        out_product, out_result);
  }

  loomc_link_index_symbol_t symbol = {};
  EXPECT_TRUE(loomc_link_index_lookup_global(
      options.link_index, loomc_make_cstring_view("dispatch"), &symbol));
  const loomc_request_root_t root = {
      .module_ordinal = static_cast<uint32_t>(symbol.provider_module_ordinal),
      .symbol_ordinal = static_cast<uint32_t>(symbol.module_symbol_ordinal),
  };
  loomc_request_t* raw_request = nullptr;
  LOOMC_RETURN_IF_ERROR(loomc_request_create(
      loomc_cmd_program_product_descriptor(), source, &root, 1, nullptr, 0,
      loomc_allocator_system(), &raw_request));
  RequestPtr request(raw_request);
  const loomc_cmd_program_request_options_t request_options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_REQUEST_OPTIONS,
      .structure_size = sizeof(request_options),
      .next = options.next,
      .config = options.config,
      .target_profile = options.target_profile,
      .request_sink = options.request_sink,
  };
  return loomc_cmd_program_product_build_request(
      context, workspace, request.get(), &request_options,
      loomc_allocator_system(), out_product, out_result);
}

class AmdgpuCommandProfileTest
    : public ::testing::TestWithParam<std::tuple<CommandSource, bool, bool>> {};

TEST_P(AmdgpuCommandProfileTest, ConfiguresSelectedKernelsAndOwnsTargets) {
  const auto [source_kind, wave64, publish_requests] = GetParam();
  const uint32_t subgroup_size = wave64 ? 64 : 32;
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile = CreateTargetProfile(target_environment.get(),
                                                 wave64 ? "gfx942" : "gfx1151");
  SourcePtr source = CreateCommandProfileSource("kernel_profiles.loom");
  ASSERT_NE(source, nullptr);
  if (source_kind != CommandSource::kText) {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  const loomc_config_binding_t binding = {
      .key = loomc_make_cstring_view("test.count"),
      .value = loomc_make_cstring_view("127"),
  };
  const loomc_target_specialization_options_t empty_target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(empty_target_options),
  };
  KernelRequestCapture capture;
  const loomc_cmd_program_product_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &empty_target_options,
      .link_index = index.get(),
      .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      .config =
          {
              .bindings = &binding,
              .binding_count = 1,
              .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
          },
      .target_profile = profile.get(),
      .request_sink = publish_requests
                          ? loomc_request_sink_t{CaptureKernelRequest, &capture}
                          : loomc_request_sink_t{},
  };
  loomc_product_t* raw_product = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(BuildCommandProduct(context.get(), workspace.get(),
                                      source.get(), options, source_kind,
                                      &raw_product, &raw_result));
  ProductPtr product(raw_product);
  ResultPtr result(raw_result);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_cmd_program_product_program_count(product.get()), 1u);
  ASSERT_EQ(capture.requests.size(), publish_requests ? 2u : 0u);

  loomc_cmd_program_t command = {};
  ASSERT_TRUE(loomc_cmd_program_product_program_at(product.get(), 0, &command));
  const std::string command_bytes = ToString(command.artifact.contents);
  loom_cmd_program_t decoded = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(command_bytes.data(), command_bytes.size()),
      &decoded));
  ASSERT_EQ(decoded.commands.count, 2u);
  for (uint32_t i = 0; i < 2; ++i) {
    const auto dispatch = loom_cmd_program_command_at(&decoded, i);
    EXPECT_EQ(dispatch.kind,
              i == 0 ? LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT
                     : LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
    EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_x,
              128 / (subgroup_size * (i + 1)));
    EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_y, 1u);
    EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_z, 1u);
  }

  std::vector<uint32_t> workgroup_sizes;
  for (const auto& request : capture.requests) {
    ASSERT_EQ(loomc_request_binding_count(request.get()), 1u);
    loomc_request_binding_t request_binding = {};
    ASSERT_TRUE(loomc_request_binding_at(request.get(), 0, &request_binding));
    loomc_cmd_entry_requirement_t requirement = {};
    ASSERT_TRUE(loomc_cmd_program_product_entry_requirement_at(
        product.get(), request_binding.requirement_ordinal, &requirement));
    const std::string symbol = ToString(requirement.symbol);
    ASSERT_TRUE(symbol == "tiles" || symbol == "wide_tiles") << symbol;
    workgroup_sizes.push_back(subgroup_size * (symbol == "tiles" ? 1 : 2));
  }

  // All producer state can retire before the independent requests compile.
  result.reset();
  product.reset();
  index.reset();
  source.reset();
  profile.reset();
  workspace.reset();
  context.reset();
  target_environment.reset();
  if (!publish_requests) {
    return;
  }

  target_environment = CreateAmdgpuTargetEnvironment();
  context = CreateAmdgpuContext(target_environment.get());
  workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pipeline = CreatePreparedLowPassProgram(context.get());
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };
  for (size_t i = 0; i < capture.requests.size(); ++i) {
    raw_product = nullptr;
    raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_request(
        compiler.get(), workspace.get(), pipeline.get(),
        capture.requests[i].get(), &compile_options, loomc_allocator_system(),
        &raw_product, &raw_result));
    ProductPtr kernel_product(raw_product);
    ResultPtr compile_result(raw_result);
    ExpectSucceededResult(compile_result.get());
    const loomc_artifact_t* artifact =
        loomc_product_artifact_at(kernel_product.get(), 0);
    ASSERT_NE(artifact, nullptr);
    const std::string module_text = ToString(artifact->contents);
    EXPECT_NE(module_text.find(wave64 ? "target<amdgpu.cdna3.core>"
                                      : "target<amdgpu.rdna3_5.core>"),
              std::string::npos)
        << module_text;
    EXPECT_NE(module_text.find("workgroup_size(" +
                               std::to_string(workgroup_sizes[i]) + ", 1, 1)"),
              std::string::npos)
        << module_text;
    EXPECT_EQ(module_text.find("target.subgroup.size"), std::string::npos)
        << module_text;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SourceFormsAndProfiles, AmdgpuCommandProfileTest,
    ::testing::Combine(::testing::Values(CommandSource::kText,
                                         CommandSource::kBytecode,
                                         CommandSource::kRequest,
                                         CommandSource::kModule),
                       ::testing::Bool(), ::testing::Bool()));

class AmdgpuCommandProviderSelectionTest
    : public ::testing::TestWithParam<std::tuple<CommandSource, bool>> {};

TEST_P(AmdgpuCommandProviderSelectionTest,
       SelectsLaunchProviderThroughNestedProgram) {
  const auto [source_kind, wave64] = GetParam();
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile = CreateTargetProfile(target_environment.get(),
                                                 wave64 ? "gfx942" : "gfx1151");
  SourcePtr source =
      CreateCommandProfileSource("command_target_selection.loom");
  ASSERT_NE(source, nullptr);
  if (source_kind != CommandSource::kText) {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  KernelRequestCapture capture;
  const loomc_cmd_program_product_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      .structure_size = sizeof(options),
      .link_index = index.get(),
      .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      .target_profile = profile.get(),
      .request_sink = {CaptureKernelRequest, &capture},
  };
  loomc_product_t* raw_product = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(BuildCommandProduct(context.get(), workspace.get(),
                                      source.get(), options, source_kind,
                                      &raw_product, &raw_result));
  ProductPtr product(raw_product);
  ResultPtr result(raw_result);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_cmd_program_product_program_count(product.get()), 1u);
  ASSERT_EQ(capture.requests.size(), 1u);
  ASSERT_EQ(loomc_request_binding_count(capture.requests[0].get()), 1u);

  loomc_request_binding_t binding = {};
  ASSERT_TRUE(loomc_request_binding_at(capture.requests[0].get(), 0, &binding));
  loomc_cmd_entry_requirement_t requirement = {};
  ASSERT_TRUE(loomc_cmd_program_product_entry_requirement_at(
      product.get(), binding.requirement_ordinal, &requirement));
  EXPECT_EQ(ToString(requirement.symbol), wave64 ? "write_cdna" : "write_rdna");
}

INSTANTIATE_TEST_SUITE_P(
    SourceFormsAndProfiles, AmdgpuCommandProviderSelectionTest,
    ::testing::Combine(::testing::Values(CommandSource::kText,
                                         CommandSource::kBytecode,
                                         CommandSource::kRequest,
                                         CommandSource::kModule),
                       ::testing::Bool()));

class AmdgpuCommandTargetLivenessTest
    : public ::testing::TestWithParam<CommandSource> {
 protected:
  void SetUp() override {
    target_environment_ = CreateAmdgpuTargetEnvironment();
    context_ = CreateAmdgpuContext(target_environment_.get());
    workspace_ = CreateWorkspace();
    profile_ = CreateTargetProfile(target_environment_.get(), "gfx942");
    source_ = CreateCommandProfileSource("conditional_target.loom");
    ASSERT_NE(source_, nullptr);
    if (GetParam() != CommandSource::kText) {
      ModulePtr module =
          DeserializeModule(context_.get(), workspace_.get(), source_.get());
      source_ = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
    }
    index_ = CreateLinkIndex(context_.get(), source_.get());
  }

  void Build(const char* use_rdna, KernelRequestCapture* capture,
             ProductPtr* out_product, ResultPtr* out_result) {
    const loomc_config_binding_t config_binding = {
        .key = loomc_make_cstring_view("test.use_rdna"),
        .value = loomc_make_cstring_view(use_rdna),
    };
    const loomc_cmd_program_product_options_t options = {
        .link_index = index_.get(),
        .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
        .config =
            {
                .bindings = &config_binding,
                .binding_count = 1,
                .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
        .target_profile = profile_.get(),
        .request_sink = {CaptureKernelRequest, capture},
    };
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(BuildCommandProduct(context_.get(), workspace_.get(),
                                        source_.get(), options, GetParam(),
                                        &product, &result));
    out_product->reset(product);
    out_result->reset(result);
  }

  TargetEnvironmentPtr target_environment_;
  ContextPtr context_;
  WorkspacePtr workspace_;
  TargetProfilePtr profile_;
  SourcePtr source_;
  LinkIndexPtr index_;
};

TEST_P(AmdgpuCommandTargetLivenessTest,
       IgnoresIncompatibleKernelRemovedBySpecialization) {
  KernelRequestCapture capture;
  ProductPtr product;
  ResultPtr result;
  Build("0", &capture, &product, &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(product.get(), nullptr);
  ASSERT_EQ(capture.requests.size(), 1u);
  ASSERT_EQ(loomc_request_binding_count(capture.requests[0].get()), 1u);

  loomc_request_binding_t request_binding = {};
  ASSERT_TRUE(
      loomc_request_binding_at(capture.requests[0].get(), 0, &request_binding));
  loomc_cmd_entry_requirement_t requirement = {};
  ASSERT_TRUE(loomc_cmd_program_product_entry_requirement_at(
      product.get(), request_binding.requirement_ordinal, &requirement));
  EXPECT_EQ(ToString(requirement.symbol), "write_cdna");
}

TEST_P(AmdgpuCommandTargetLivenessTest,
       RejectsIncompatibleKernelRetainedBySpecialization) {
  KernelRequestCapture capture;
  ProductPtr product;
  ResultPtr result;
  Build("1", &capture, &product, &result);
  EXPECT_EQ(product, nullptr);
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_TRUE(capture.requests.empty());
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
            "TARGET/052");
}

INSTANTIATE_TEST_SUITE_P(ProductAdmissionPaths, AmdgpuCommandTargetLivenessTest,
                         ::testing::Values(CommandSource::kText,
                                           CommandSource::kBytecode,
                                           CommandSource::kRequest,
                                           CommandSource::kModule));

class AmdgpuCommandTargetPolicyTest
    : public ::testing::TestWithParam<CommandSource> {};

TEST_P(AmdgpuCommandTargetPolicyTest, RejectsMixedTargetPolicies) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  SourcePtr source = CreateCommandProfileSource("empty_command.loom");
  ASSERT_NE(source, nullptr);
  {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("dispatch"),
      .target_profile = profile.get(),
  };
  const loomc_target_binding_t target_binding = {
      .target_symbol = loomc_make_cstring_view("device"),
      .target_profile = profile.get(),
  };
  for (bool bind_declaration : {false, true}) {
    SCOPED_TRACE(bind_declaration);
    const loomc_target_specialization_options_t explicit_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(explicit_options),
        .specializations = bind_declaration ? nullptr : &specialization,
        .specialization_count = bind_declaration ? 0u : 1u,
        .target_bindings = bind_declaration ? &target_binding : nullptr,
        .target_binding_count = bind_declaration ? 1u : 0u,
    };
    const loomc_cmd_program_product_options_t options = {
        .next = &explicit_options,
        .link_index = index.get(),
        .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
        .target_profile = profile.get(),
    };
    loomc_product_t* raw_product = nullptr;
    loomc_result_t* raw_result = nullptr;
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_INVALID_ARGUMENT,
        BuildCommandProduct(context.get(), workspace.get(), source.get(),
                            options, GetParam(), &raw_product, &raw_result));
    ProductPtr product(raw_product);
    ResultPtr result(raw_result);
    EXPECT_EQ(product.get(), nullptr);
    EXPECT_EQ(result.get(), nullptr);
  }
}

TEST_P(AmdgpuCommandTargetPolicyTest, RejectsIncompatibleProfileEnvironment) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &raw_context));
  ContextPtr context(raw_context);
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateCommandProfileSource("empty_command.loom");
  ASSERT_NE(source, nullptr);
  {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  const loomc_cmd_program_product_options_t options = {
      .link_index = index.get(),
      .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      .target_profile = profile.get(),
  };
  loomc_product_t* raw_product = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      BuildCommandProduct(context.get(), workspace.get(), source.get(), options,
                          GetParam(), &raw_product, &raw_result));
  ProductPtr product(raw_product);
  ResultPtr result(raw_result);
  EXPECT_EQ(product.get(), nullptr);
  EXPECT_EQ(result.get(), nullptr);
}

TEST_P(AmdgpuCommandTargetPolicyTest, ReportsIncompatibleSourceTarget) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  SourcePtr source = CreateCommandProfileSource("source_target.loom");
  ASSERT_NE(source, nullptr);
  {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  const loomc_cmd_program_product_options_t options = {
      .link_index = index.get(),
      .flags = LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      .target_profile = profile.get(),
  };
  loomc_product_t* raw_product = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(BuildCommandProduct(context.get(), workspace.get(),
                                      source.get(), options, GetParam(),
                                      &raw_product, &raw_result));
  ProductPtr product(raw_product);
  ResultPtr result(raw_result);
  ASSERT_NE(result.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(product.get(), nullptr);
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
            "TARGET/052");
}

INSTANTIATE_TEST_SUITE_P(ProductAdmissionPaths, AmdgpuCommandTargetPolicyTest,
                         ::testing::Values(CommandSource::kBytecode,
                                           CommandSource::kRequest,
                                           CommandSource::kModule));

}  // namespace
}  // namespace loomc::testing::amdgpu
