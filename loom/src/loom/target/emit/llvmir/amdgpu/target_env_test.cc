// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/amdgpu/target_env.h"

#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/llvmir/builder.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string WriteText(const loom_llvmir_module_t* module) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_CHECK_OK(loom_llvmir_text_write_module(module, &stream));
  std::string text(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return text;
}

void ExpectSnapshotMatchesCommonEnv(
    const loom_target_snapshot_t* snapshot,
    const loom_llvmir_target_env_t* target_env) {
  ASSERT_NE(snapshot, nullptr);
  ASSERT_NE(target_env, nullptr);
  EXPECT_EQ(snapshot->codegen_format, LOOM_TARGET_CODEGEN_FORMAT_LLVMIR);
  EXPECT_EQ(snapshot->artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(snapshot->default_pointer_bitwidth,
            target_env->default_pointer_bitwidth);
  EXPECT_EQ(snapshot->index_bitwidth, target_env->index_bitwidth);
  EXPECT_EQ(snapshot->offset_bitwidth, target_env->offset_bitwidth);
  EXPECT_EQ(snapshot->memory_spaces.generic,
            target_env->address_spaces.generic);
  EXPECT_EQ(snapshot->memory_spaces.global, target_env->address_spaces.global);
  EXPECT_EQ(snapshot->memory_spaces.workgroup,
            target_env->address_spaces.local);
  EXPECT_EQ(snapshot->memory_spaces.constant,
            target_env->address_spaces.constant);
  EXPECT_EQ(snapshot->memory_spaces.private_memory,
            target_env->address_spaces.private_memory);
  EXPECT_EQ(snapshot->memory_spaces.descriptor,
            target_env->address_spaces.buffer_resource);
}

void ExpectDerivedProfileMatchesStatic(
    const loom_llvmir_target_profile_t* derived_profile,
    const loom_llvmir_target_profile_t* static_profile) {
  ASSERT_NE(derived_profile, nullptr);
  ASSERT_NE(static_profile, nullptr);
  ASSERT_NE(derived_profile->target_env, nullptr);
  ASSERT_NE(static_profile->target_env, nullptr);
  EXPECT_EQ(ToString(derived_profile->name), ToString(static_profile->name));
  EXPECT_EQ(derived_profile->kind, static_profile->kind);
  EXPECT_EQ(ToString(derived_profile->target_env->target_triple),
            ToString(static_profile->target_env->target_triple));
  EXPECT_EQ(ToString(derived_profile->target_env->data_layout),
            ToString(static_profile->target_env->data_layout));
  EXPECT_EQ(derived_profile->target_env->object_format,
            static_profile->target_env->object_format);
  EXPECT_EQ(derived_profile->target_env->default_pointer_bitwidth,
            static_profile->target_env->default_pointer_bitwidth);
  EXPECT_EQ(derived_profile->target_env->index_bitwidth,
            static_profile->target_env->index_bitwidth);
  EXPECT_EQ(derived_profile->target_env->offset_bitwidth,
            static_profile->target_env->offset_bitwidth);
  EXPECT_EQ(derived_profile->target_env->address_spaces.generic,
            static_profile->target_env->address_spaces.generic);
  EXPECT_EQ(derived_profile->target_env->address_spaces.global,
            static_profile->target_env->address_spaces.global);
  EXPECT_EQ(derived_profile->target_env->address_spaces.local,
            static_profile->target_env->address_spaces.local);
  EXPECT_EQ(derived_profile->target_env->address_spaces.constant,
            static_profile->target_env->address_spaces.constant);
  EXPECT_EQ(derived_profile->target_env->address_spaces.private_memory,
            static_profile->target_env->address_spaces.private_memory);
  EXPECT_EQ(derived_profile->target_env->address_spaces.buffer_resource,
            static_profile->target_env->address_spaces.buffer_resource);
  EXPECT_EQ(ToString(derived_profile->target_cpu),
            ToString(static_profile->target_cpu));
  EXPECT_EQ(ToString(derived_profile->target_features),
            ToString(static_profile->target_features));
  EXPECT_EQ(derived_profile->x86_packed_dot_feature_bits,
            static_profile->x86_packed_dot_feature_bits);
  EXPECT_EQ(derived_profile->exported_linkage,
            static_profile->exported_linkage);
  EXPECT_EQ(derived_profile->kernel.calling_convention,
            static_profile->kernel.calling_convention);
  EXPECT_EQ(
      ToString(derived_profile->kernel.required_workgroup_size_metadata_name),
      ToString(static_profile->kernel.required_workgroup_size_metadata_name));
  EXPECT_EQ(derived_profile->kernel.required_workgroup_size.x,
            static_profile->kernel.required_workgroup_size.x);
  EXPECT_EQ(derived_profile->kernel.required_workgroup_size.y,
            static_profile->kernel.required_workgroup_size.y);
  EXPECT_EQ(derived_profile->kernel.required_workgroup_size.z,
            static_profile->kernel.required_workgroup_size.z);
  EXPECT_EQ(derived_profile->kernel.flat_workgroup_size_min,
            static_profile->kernel.flat_workgroup_size_min);
  EXPECT_EQ(derived_profile->kernel.flat_workgroup_size_max,
            static_profile->kernel.flat_workgroup_size_max);
  EXPECT_EQ(derived_profile->kernel.binding_resource_flags,
            static_profile->kernel.binding_resource_flags);
}

TEST(LlvmIrAmdgpuTargetEnvTest, AmdgpuHalProfileNamesKernelAbi) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_amdgpu_hal();
  ASSERT_NE(profile, nullptr);
  ASSERT_NE(profile->target_env, nullptr);

  EXPECT_EQ(ToString(profile->name), "amdgpu-hal");
  EXPECT_EQ(profile->kind, LOOM_LLVMIR_TARGET_PROFILE_KERNEL);
  EXPECT_EQ(ToString(profile->target_env->target_triple), "amdgcn-amd-amdhsa");
  EXPECT_EQ(ToString(profile->target_env->data_layout),
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-"
            "p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:"
            "256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
            "v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-"
            "S32-A5-G1-ni:7:8:9");
  EXPECT_EQ(profile->target_env->address_spaces.generic, 0u);
  EXPECT_EQ(profile->target_env->address_spaces.global, 1u);
  EXPECT_EQ(profile->target_env->address_spaces.local, 3u);
  EXPECT_EQ(profile->target_env->address_spaces.constant, 4u);
  EXPECT_EQ(profile->target_env->address_spaces.private_memory, 5u);
  EXPECT_EQ(profile->target_env->address_spaces.buffer_resource, 7u);
  EXPECT_EQ(profile->kernel.calling_convention,
            LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL);
  EXPECT_EQ(profile->kernel.required_workgroup_size.x, 0u);
  EXPECT_EQ(profile->kernel.required_workgroup_size.y, 0u);
  EXPECT_EQ(profile->kernel.required_workgroup_size.z, 0u);
  EXPECT_EQ(profile->kernel.flat_workgroup_size_min, 1u);
  EXPECT_EQ(profile->kernel.flat_workgroup_size_max, 1024u);
  EXPECT_EQ(profile->kernel.binding_resource_flags, 0x31027000u);

  loom_llvmir_attr_t
      binding_attrs[LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
  iree_host_size_t binding_attr_count = 0;
  loom_llvmir_target_profile_kernel_binding_attrs(profile, binding_attrs,
                                                  &binding_attr_count);
  ASSERT_EQ(binding_attr_count, 2u);
  EXPECT_EQ(binding_attrs[0].kind, LOOM_LLVMIR_ATTR_INREG);
  EXPECT_EQ(binding_attrs[1].kind, LOOM_LLVMIR_ATTR_NOUNDEF);
}

TEST(LlvmIrAmdgpuTargetEnvTest, AmdgpuHalProfileHasGenericTargetBundle) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_amdgpu_hal();
  const loom_target_bundle_t* bundle = loom_llvmir_target_bundle_amdgpu_hal();
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(ToString(bundle->name), ToString(profile->name));
  ExpectSnapshotMatchesCommonEnv(bundle->snapshot, profile->target_env);
  EXPECT_EQ(bundle->snapshot->memory_spaces.host, UINT32_MAX);
  EXPECT_EQ(bundle->snapshot->max_workgroup_size.x, 1024u);
  EXPECT_EQ(bundle->snapshot->max_workgroup_size.y, 1024u);
  EXPECT_EQ(bundle->snapshot->max_workgroup_size.z, 1024u);
  EXPECT_EQ(bundle->snapshot->max_flat_workgroup_size, 1024u);
  EXPECT_EQ(bundle->snapshot->max_grid_size.x, 2147483647u);
  EXPECT_EQ(bundle->snapshot->max_grid_size.y, 65535u);
  EXPECT_EQ(bundle->snapshot->max_grid_size.z, 65535u);
  EXPECT_EQ(bundle->snapshot->max_flat_grid_size, 4294967295ull);
  ASSERT_NE(bundle->export_plan, nullptr);
  EXPECT_EQ(ToString(bundle->export_plan->name), ToString(profile->name));
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(bundle->export_plan->linkage, LOOM_TARGET_LINKAGE_DEFAULT);
  EXPECT_EQ(bundle->export_plan->hal_kernel.required_workgroup_size.x, 0u);
  EXPECT_EQ(bundle->export_plan->hal_kernel.required_workgroup_size.y, 0u);
  EXPECT_EQ(bundle->export_plan->hal_kernel.required_workgroup_size.z, 0u);
  EXPECT_EQ(bundle->export_plan->hal_kernel.flat_workgroup_size_min, 0u);
  EXPECT_EQ(bundle->export_plan->hal_kernel.flat_workgroup_size_max, 0u);
  EXPECT_EQ(profile->kernel.flat_workgroup_size_min, 1u);
  EXPECT_EQ(profile->kernel.flat_workgroup_size_max,
            bundle->snapshot->max_flat_workgroup_size);
  EXPECT_EQ(bundle->export_plan->hal_kernel.buffer_resource_flags,
            profile->kernel.binding_resource_flags);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(bundle->config->contract_set_key));
}

TEST(LlvmIrAmdgpuTargetEnvTest, DerivesAmdgpuHalProfileFromGenericBundle) {
  loom_llvmir_target_profile_storage_t storage = {};
  loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_amdgpu_hal(),
      loom_llvmir_target_profile_amdgpu_hal(), &storage);
  ExpectDerivedProfileMatchesStatic(&storage.profile,
                                    loom_llvmir_target_profile_amdgpu_hal());
}

TEST(LlvmIrAmdgpuTargetEnvTest,
     AmdgpuProjectionRequiresTripleForGenericKernel) {
  static const loom_target_snapshot_t kSnapshot = {
      /*.name=*/IREE_SVL("generic-llvmir-kernel"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.default_pointer_bitwidth=*/64,
      /*.index_bitwidth=*/64,
      /*.offset_bitwidth=*/64,
  };
  static const loom_target_export_plan_t kExportPlan = {
      /*.name=*/IREE_SVL("generic-llvmir-kernel"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
      /*.linkage=*/LOOM_TARGET_LINKAGE_DEFAULT,
  };
  static const loom_target_config_t kConfig = {
      /*.name=*/IREE_SVL("llvmir.generic.core"),
      /*.contract_set_key=*/IREE_SVL("llvmir.generic.core"),
  };
  static const loom_target_bundle_t kBundle = {
      /*.name=*/IREE_SVL("generic-llvmir-kernel"),
      /*.snapshot=*/&kSnapshot,
      /*.export_plan=*/&kExportPlan,
      /*.config=*/&kConfig,
  };

  const loom_llvmir_target_profile_provider_t* provider =
      loom_llvmir_amdgpu_target_profile_provider();
  const loom_llvmir_target_profile_t* profile = nullptr;
  loom_llvmir_target_profile_projection_request_t request = {
      /*.bundle=*/&kBundle,
      /*.target_triple=*/iree_string_view_empty(),
  };
  EXPECT_FALSE(provider->project_bundle(&request, &profile));
  EXPECT_EQ(profile, nullptr);

  request.target_triple = IREE_SV("amdgcn-amd-amdhsa");
  ASSERT_TRUE(provider->project_bundle(&request, &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_amdgpu_hal());
}

TEST(LlvmIrAmdgpuTargetEnvTest, AmdgpuHalProfileMaterializesKernelDecorations) {
  loom_llvmir_target_profile_storage_t storage = {};
  loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_amdgpu_hal(),
      loom_llvmir_target_profile_amdgpu_hal(), &storage);
  const loom_llvmir_target_profile_t* profile = &storage.profile;
  loom_llvmir_target_config_t config = {};
  loom_llvmir_target_profile_module_config(profile, IREE_SV("kernel-source"),
                                           &config);

  loom_llvmir_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(&config, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  loom_llvmir_attr_group_id_t attr_group = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_target_profile_add_kernel_attr_group(
      module_ptr.get(), profile, &attr_group));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("dispatch");
  function_desc.return_type = void_type;
  function_desc.linkage = profile->exported_linkage;
  function_desc.calling_convention = profile->kernel.calling_convention;
  function_desc.attr_group_id = attr_group;
  loom_llvmir_function_t* function = nullptr;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_attach_kernel_metadata(function, profile));

  loom_llvmir_block_t* entry = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  IREE_ASSERT_OK(loom_llvmir_build_ret_void(entry));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  std::string text = WriteText(module_ptr.get());
  EXPECT_NE(text.find("\"amdgpu-flat-work-group-size\"=\"1,1024\""),
            std::string::npos)
      << text;
  EXPECT_EQ(text.find("!reqd_work_group_size"), std::string::npos) << text;
}

TEST(LlvmIrAmdgpuTargetEnvTest, AmdgpuHalProfileCopyControlsKernelDecorations) {
  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_amdgpu_hal(&profile));
  profile.name = IREE_SV("amdgpu-hal-variant");
  profile.target_cpu = IREE_SV("gfx1100");
  profile.target_features = IREE_SV("+wavefrontsize64");
  profile.kernel.required_workgroup_size.x = 128;
  profile.kernel.required_workgroup_size.y = 2;
  profile.kernel.required_workgroup_size.z = 1;
  profile.kernel.flat_workgroup_size_min = 128;
  profile.kernel.flat_workgroup_size_max = 256;

  loom_llvmir_target_config_t config = {};
  loom_llvmir_target_profile_module_config(
      &profile, IREE_SV("kernel-variant-source"), &config);
  loom_llvmir_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(&config, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  loom_llvmir_attr_group_id_t attr_group = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_target_profile_add_kernel_attr_group(
      module_ptr.get(), &profile, &attr_group));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("dispatch");
  function_desc.return_type = void_type;
  function_desc.linkage = profile.exported_linkage;
  function_desc.calling_convention = profile.kernel.calling_convention;
  function_desc.attr_group_id = attr_group;
  loom_llvmir_function_t* function = nullptr;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));

  loom_llvmir_type_id_t global_pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_module_get_pointer_type(
      module_ptr.get(), profile.target_env->address_spaces.global,
      &global_pointer_type));
  loom_llvmir_attr_t
      binding_attrs[LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
  iree_host_size_t binding_attr_count = 0;
  loom_llvmir_target_profile_kernel_binding_attrs(&profile, binding_attrs,
                                                  &binding_attr_count);
  loom_llvmir_value_id_t parameter = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_parameter_desc_t parameter_desc = {};
  parameter_desc.type_id = global_pointer_type;
  parameter_desc.name = IREE_SV("input");
  parameter_desc.attrs = binding_attrs;
  parameter_desc.attr_count = binding_attr_count;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(function, &parameter_desc,
                                                    &parameter));
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_attach_kernel_metadata(function, &profile));

  loom_llvmir_block_t* entry = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  IREE_ASSERT_OK(loom_llvmir_build_ret_void(entry));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  std::string text = WriteText(module_ptr.get());
  EXPECT_NE(text.find("ptr addrspace(1) inreg noundef %input"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("\"amdgpu-flat-work-group-size\"=\"128,256\""),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("!0 = !{i32 128, i32 2, i32 1}\n"), std::string::npos)
      << text;
}

}  // namespace
}  // namespace loom
