// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/kernel.h"

#include <initializer_list>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loom/binding/c/test/target/kernel/kernel_product_testdata.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/vm.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LaunchProgramPtr = HandlePtr<loomc_vm_launch_config_program_t,
                                   loomc_vm_launch_config_program_release>;
using LinkIndexBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data != nullptr ? std::string(value.data, value.size)
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
}

struct RequestedRoot {
  const char* name;
  loomc_request_root_goal_t goal;
};

class KernelProductTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_target_environment_t* target_environment = nullptr;
    LOOMC_ASSERT_OK(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &target_environment));
    target_environment_.reset(target_environment);

    const loomc_amdgpu_profile_options_t profile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        /*.structure_size=*/sizeof(profile_options),
        /*.next=*/nullptr,
        /*.identifier=*/loomc_make_cstring_view("gfx1151"),
        /*.identity=*/
        {
            /*.target=*/loomc_make_cstring_view("gfx1151"),
        },
    };
    loomc_target_profile_t* target_profile = nullptr;
    LOOMC_ASSERT_OK(loomc_target_profile_create_amdgpu(
        target_environment_.get(), &profile_options, loomc_allocator_system(),
        &target_profile));
    target_profile_.reset(target_profile);

    const loomc_context_target_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_environment=*/target_environment_.get(),
    };
    const loomc_context_options_t context_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /*.structure_size=*/sizeof(context_options),
        /*.next=*/&target_options,
    };
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                         loomc_allocator_system(), &context));
    context_.reset(context);

    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(
        /*options=*/nullptr, loomc_allocator_system(), &workspace));
    workspace_.reset(workspace);

    loomc_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_compiler_create(context_.get(), /*options=*/nullptr,
                                          loomc_allocator_system(), &compiler));
    compiler_.reset(compiler);

    const loomc_target_pipeline_options_t pipeline_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        /*.structure_size=*/sizeof(pipeline_options),
        /*.next=*/nullptr,
        /*.identifier=*/loomc_make_cstring_view("kernel-product-test"),
        /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        /*.source_to_low_max_errors=*/20,
    };
    loomc_pass_program_t* pass_program = nullptr;
    loomc_result_t* pass_result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
        context_.get(), &pipeline_options, loomc_allocator_system(),
        &pass_program, &pass_result));
    pass_program_.reset(pass_program);
    ResultPtr pass_result_ptr(pass_result);
    ExpectSucceededResult(pass_result_ptr.get());

    const iree_file_toc_t* files = loomc_kernel_product_testdata_create();
    ASSERT_EQ(loomc_kernel_product_testdata_size(), 1u);
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view(files[0].name),
        /*.contents=*/loomc_make_byte_span(files[0].data, files[0].size),
        /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
    };
    loomc_source_t* text_source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(
        &source_options, loomc_allocator_system(), &text_source));
    SourcePtr text_source_ptr(text_source);

    loomc_module_t* module = nullptr;
    loomc_result_t* module_result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), text_source_ptr.get(),
        /*options=*/nullptr, loomc_allocator_system(), &module,
        &module_result));
    ModulePtr module_ptr(module);
    ResultPtr module_result_ptr(module_result);
    ExpectSucceededResult(module_result_ptr.get());

    const loomc_module_serialize_options_t serialize_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        /*.structure_size=*/sizeof(serialize_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
        /*.identifier=*/loomc_make_cstring_view("kernel-product.loombc"),
    };
    loomc_source_t* bytecode_source = nullptr;
    LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
        module_ptr.get(), &serialize_options, loomc_allocator_system(),
        &bytecode_source));
    bytecode_source_.reset(bytecode_source);

    loomc_link_index_builder_t* index_builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loomc_allocator_system(),
        &index_builder));
    LinkIndexBuilderPtr index_builder_ptr(index_builder);
    const loomc_link_index_source_options_t index_source_options = {
        /*.provider_name=*/loomc_make_cstring_view("kernel-product"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        index_builder_ptr.get(), bytecode_source_.get(), &index_source_options,
        /*out_slot=*/nullptr));
    loomc_link_index_t* index = nullptr;
    loomc_result_t* index_result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_finish(index_builder_ptr.get(),
                                                    &index, &index_result));
    link_index_.reset(index);
    ResultPtr index_result_ptr(index_result);
    ExpectSucceededResult(index_result_ptr.get());
  }

  RequestPtr CreateRequest(std::initializer_list<RequestedRoot> root_specs) {
    loomc_link_index_module_t module = {};
    if (loomc_link_index_module_count(link_index_.get()) != 1 ||
        !loomc_link_index_module_at(link_index_.get(), 0, &module)) {
      ADD_FAILURE() << "expected one indexed test module";
      return RequestPtr();
    }

    std::vector<loomc_request_root_t> roots;
    roots.reserve(root_specs.size());
    for (const RequestedRoot& root_spec : root_specs) {
      loomc_link_index_symbol_t symbol = {};
      if (!loomc_link_index_lookup_private(
              link_index_.get(), &module,
              loomc_make_cstring_view(root_spec.name), &symbol)) {
        ADD_FAILURE() << "missing test root " << root_spec.name;
        return RequestPtr();
      }
      if (symbol.provider_module_ordinal > UINT32_MAX ||
          symbol.module_symbol_ordinal > UINT32_MAX) {
        ADD_FAILURE() << "test root ordinal exceeds the request domain";
        return RequestPtr();
      }
      roots.push_back({
          /*.module_ordinal=*/
          static_cast<uint32_t>(symbol.provider_module_ordinal),
          /*.symbol_ordinal=*/
          static_cast<uint32_t>(symbol.module_symbol_ordinal),
          /*.goal=*/root_spec.goal,
          /*.reserved=*/0,
      });
    }

    loomc_request_t* request = nullptr;
    LOOMC_EXPECT_OK(loomc_request_create(
        loomc_kernel_product_descriptor(), bytecode_source_.get(), roots.data(),
        roots.size(), /*bindings=*/nullptr,
        /*binding_count=*/0, loomc_allocator_system(), &request));
    return RequestPtr(request);
  }

  ProductPtr Build(const loomc_request_t* request, ResultPtr* out_result) {
    const loomc_target_specialization_t specialization = {
        /*.function_symbol=*/loomc_make_cstring_view("gamma"),
        /*.target_profile=*/target_profile_.get(),
    };
    const loomc_target_specialization_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.specializations=*/&specialization,
        /*.specialization_count=*/1,
    };
    const loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/&target_options,
        /*.module_name=*/loomc_make_cstring_view("kernel-product"),
    };
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_kernel_product_build_request(
        compiler_.get(), workspace_.get(), pass_program_.get(), request,
        &compile_options, /*emit_options=*/nullptr, loomc_allocator_system(),
        &product, &result));
    out_result->reset(result);
    return ProductPtr(product);
  }

  TargetEnvironmentPtr target_environment_;
  TargetProfilePtr target_profile_;
  ContextPtr context_;
  WorkspacePtr workspace_;
  CompilerPtr compiler_;
  PassProgramPtr pass_program_;
  SourcePtr bytecode_source_;
  LinkIndexPtr link_index_;
};

TEST_F(KernelProductTest, ExecutableRootDoesNotProduceLaunchArtifact) {
  RequestPtr request = CreateRequest({
      {"beta", LOOMC_KERNEL_ROOT_GOAL_EXECUTABLE_ENTRY},
  });
  ASSERT_NE(request, nullptr);
  ResultPtr result;
  ProductPtr product = Build(request.get(), &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(product, nullptr);
  ASSERT_EQ(loomc_product_artifact_count(product.get()), 1u);
  ASSERT_EQ(loomc_product_export_count(product.get()), 1u);

  loomc_kernel_product_root_t root = {};
  ASSERT_TRUE(loomc_kernel_product_root_at(product.get(), 0, &root));
  ASSERT_LT(root.executable_artifact_ordinal,
            loomc_product_artifact_count(product.get()));
  const loomc_artifact_t* executable = loomc_product_artifact_at(
      product.get(), root.executable_artifact_ordinal);
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(executable->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(executable->format), LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  EXPECT_EQ(root.launch_config_artifact_ordinal,
            LOOMC_KERNEL_ARTIFACT_ORDINAL_INVALID);
  EXPECT_EQ(root.launch_config_function_ordinal,
            LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID);
}

TEST_F(KernelProductTest, MixedRootsShareArtifactsAndPreserveExactIdentity) {
  RequestPtr request = CreateRequest({
      {"beta", LOOMC_KERNEL_ROOT_GOAL_EXECUTABLE_ENTRY},
      {"alpha", LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE},
      {"beta", LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE},
      {"gamma", LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE},
  });
  ASSERT_NE(request, nullptr);
  ResultPtr result;
  ProductPtr product = Build(request.get(), &result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(product, nullptr);
  ASSERT_EQ(loomc_product_artifact_count(product.get()), 2u);
  ASSERT_EQ(loomc_product_export_count(product.get()), 4u);

  loomc_kernel_product_root_t roots[4] = {};
  for (loomc_host_size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
    ASSERT_TRUE(loomc_kernel_product_root_at(product.get(), i, &roots[i]));
    EXPECT_EQ(roots[i].executable_artifact_ordinal,
              roots[0].executable_artifact_ordinal);
  }
  EXPECT_EQ(roots[0].executable_function_ordinal,
            roots[2].executable_function_ordinal);
  EXPECT_EQ(roots[0].launch_config_artifact_ordinal,
            LOOMC_KERNEL_ARTIFACT_ORDINAL_INVALID);
  EXPECT_EQ(roots[0].launch_config_function_ordinal,
            LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID);
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(roots); ++i) {
    EXPECT_EQ(roots[i].launch_config_artifact_ordinal,
              roots[1].launch_config_artifact_ordinal);
    EXPECT_NE(roots[i].launch_config_function_ordinal,
              LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID);
  }

  const loomc_artifact_t* executable = loomc_product_artifact_at(
      product.get(), roots[0].executable_artifact_ordinal);
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(executable->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(executable->format), LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  const loomc_artifact_t* launch_artifact = loomc_product_artifact_at(
      product.get(), roots[1].launch_config_artifact_ordinal);
  ASSERT_NE(launch_artifact, nullptr);
  EXPECT_EQ(launch_artifact->kind, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG);
  EXPECT_EQ(ToString(launch_artifact->format),
            LOOMC_ARTIFACT_FORMAT_VM_BYTECODE);

  loomc_vm_launch_config_program_t* launch_program = nullptr;
  LOOMC_ASSERT_OK(loomc_vm_launch_config_program_load(
      launch_artifact, loomc_allocator_system(), &launch_program));
  LaunchProgramPtr launch_program_ptr(launch_program);
  const uint32_t expected_counts[] = {107, 207, 307};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_counts); ++i) {
    loomc_vm_launch_config_function_t function =
        loomc_vm_launch_config_function_invalid();
    ASSERT_TRUE(loomc_vm_launch_config_program_function_at(
        launch_program_ptr.get(), roots[i + 1].launch_config_function_ordinal,
        &function));
    const uint64_t arguments[] = {7};
    loomc_launch_config_t config = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        /*.structure_size=*/sizeof(config),
    };
    LOOMC_ASSERT_OK(loomc_vm_launch_config_program_invoke(
        launch_program_ptr.get(), function, arguments,
        IREE_ARRAYSIZE(arguments), &config));
    EXPECT_EQ(config.workgroup_count.x, expected_counts[i]);
    EXPECT_EQ(config.workgroup_size.x, 64u);
  }
}

TEST_F(KernelProductTest, InvalidGoalPublishesNoProductOrResult) {
  RequestPtr request = CreateRequest({
      {"alpha", 99},
  });
  ASSERT_NE(request, nullptr);
  loomc_product_t* product = reinterpret_cast<loomc_product_t*>(UINTPTR_MAX);
  loomc_result_t* result = reinterpret_cast<loomc_result_t*>(UINTPTR_MAX);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_kernel_product_build_request(
          compiler_.get(), workspace_.get(), pass_program_.get(), request.get(),
          /*compile_options=*/nullptr, /*emit_options=*/nullptr,
          loomc_allocator_system(), &product, &result));
  EXPECT_EQ(product, nullptr);
  EXPECT_EQ(result, nullptr);
}

}  // namespace
