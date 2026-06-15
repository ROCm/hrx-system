// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_env.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/emit/llvmir/x86/target_env.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

bool LookupRegisteredProfile(iree_string_view_t profile_name,
                             const loom_llvmir_target_profile_t** out_profile) {
  const loom_llvmir_target_profile_provider_t* providers[] = {
      loom_llvmir_x86_target_profile_provider(),
  };
  loom_llvmir_target_profile_registry_t registry = {};
  registry.default_profile = loom_llvmir_target_profile_x86_64_object();
  registry.providers = providers;
  registry.provider_count = IREE_ARRAYSIZE(providers);
  return loom_llvmir_target_profile_registry_lookup(&registry, profile_name,
                                                    out_profile);
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

TEST(LlvmIrTargetEnvTest, X86ObjectProfileNamesObjectAbi) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_object();
  ASSERT_NE(profile, nullptr);
  ASSERT_NE(profile->target_env, nullptr);

  EXPECT_EQ(ToString(profile->name), "x86_64-object");
  EXPECT_EQ(profile->kind, LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT);
  EXPECT_EQ(profile->target_env->object_format, LOOM_LLVMIR_OBJECT_FORMAT_ELF);
  EXPECT_EQ(ToString(profile->target_env->target_triple),
            "x86_64-unknown-linux-gnu");
  EXPECT_EQ(ToString(profile->target_env->data_layout),
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
            "f80:128-n8:16:32:64-S128");
  EXPECT_EQ(profile->target_env->address_spaces.generic, 0u);
  EXPECT_EQ(profile->target_env->address_spaces.global, 0u);
  EXPECT_EQ(profile->target_env->address_spaces.buffer_resource, UINT32_MAX);
  EXPECT_EQ(profile->exported_linkage, LOOM_LLVMIR_LINKAGE_DSO_LOCAL);

  loom_llvmir_target_config_t config = {};
  loom_llvmir_target_profile_module_config(profile, IREE_SV("x86-source"),
                                           &config);
  EXPECT_EQ(ToString(config.source_name), "x86-source");
  EXPECT_EQ(ToString(config.target_triple), "x86_64-unknown-linux-gnu");
  EXPECT_EQ(ToString(config.data_layout),
            ToString(profile->target_env->data_layout));
  EXPECT_EQ(config.default_pointer_bitwidth, 64u);
  EXPECT_EQ(config.index_bitwidth, 64u);
  EXPECT_EQ(config.offset_bitwidth, 64u);

  loom_llvmir_target_profile_t profile_copy = {};
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_initialize_x86_64_object(&profile_copy));
  EXPECT_EQ(ToString(profile_copy.name), ToString(profile->name));
  EXPECT_EQ(profile_copy.target_env, profile->target_env);
}

TEST(LlvmIrTargetEnvTest, X86ObjectProfileHasGenericTargetBundle) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_object();
  const loom_target_bundle_t* bundle =
      loom_llvmir_target_bundle_x86_64_object();
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(ToString(bundle->name), ToString(profile->name));
  ExpectSnapshotMatchesCommonEnv(bundle->snapshot, profile->target_env);
  EXPECT_EQ(bundle->snapshot->memory_spaces.host, 0u);
  ASSERT_NE(bundle->export_plan, nullptr);
  EXPECT_EQ(ToString(bundle->export_plan->name), ToString(profile->name));
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(bundle->export_plan->linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_EQ(ToString(bundle->config->contract_set_key), "x86.scalar.core");
}

TEST(LlvmIrTargetEnvTest, X86PackedDotProfileHasGenericTargetBundle) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_packed_dot_object();
  const loom_target_bundle_t* bundle =
      loom_llvmir_target_bundle_x86_64_packed_dot_object();
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(ToString(bundle->name), ToString(profile->name));
  ExpectSnapshotMatchesCommonEnv(bundle->snapshot, profile->target_env);
  ASSERT_NE(bundle->export_plan, nullptr);
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(bundle->export_plan->linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_EQ(ToString(bundle->config->contract_set_key), "x86.packed_dot.core");
  EXPECT_NE(
      ToString(bundle->config->contract_set_key),
      ToString(
          loom_llvmir_target_bundle_x86_64_object()->config->contract_set_key));
}

TEST(LlvmIrTargetEnvTest, DerivesX86ObjectProfileFromGenericBundle) {
  loom_llvmir_target_profile_storage_t storage = {};
  loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_x86_64_object(),
      loom_llvmir_target_profile_x86_64_object(), &storage);
  ExpectDerivedProfileMatchesStatic(&storage.profile,
                                    loom_llvmir_target_profile_x86_64_object());

  loom_llvmir_target_config_t config = {};
  loom_llvmir_target_profile_module_config(
      &storage.profile, IREE_SV("derived-x86-source"), &config);
  EXPECT_EQ(ToString(config.source_name), "derived-x86-source");
  EXPECT_EQ(ToString(config.target_triple), "x86_64-unknown-linux-gnu");
}

TEST(LlvmIrTargetEnvTest, DerivesX86PackedDotProfileFromGenericBundle) {
  loom_llvmir_target_profile_storage_t storage = {};
  loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_x86_64_packed_dot_object(),
      loom_llvmir_target_profile_x86_64_packed_dot_object(), &storage);
  ExpectDerivedProfileMatchesStatic(
      &storage.profile, loom_llvmir_target_profile_x86_64_packed_dot_object());
}

TEST(LlvmIrTargetEnvTest, LooksUpRegisteredProfilesByName) {
  const loom_llvmir_target_profile_t* profile = nullptr;
  ASSERT_TRUE(LookupRegisteredProfile(IREE_SV(""), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  ASSERT_TRUE(LookupRegisteredProfile(IREE_SV("x86_64-object"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  ASSERT_TRUE(LookupRegisteredProfile(IREE_SV("x86_64-avx2-object"), &profile));
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(ToString(profile->target_features), "+avx,+avx2,+fma");
}

TEST(LlvmIrTargetEnvTest, ProjectsX86ProfileByDescriptorSetKey) {
  static const loom_target_snapshot_t kSnapshot = {
      /*.name=*/IREE_SVL("native-x86-debug"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.default_pointer_bitwidth=*/64,
      /*.index_bitwidth=*/64,
      /*.offset_bitwidth=*/64,
  };
  static const loom_target_export_plan_t kExportPlan = {
      /*.name=*/IREE_SVL("native-x86-debug"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_OBJECT_FUNCTION,
      /*.linkage=*/LOOM_TARGET_LINKAGE_DSO_LOCAL,
  };
  static const loom_target_config_t kConfig = {
      /*.name=*/IREE_SVL("x86.avx2.core"),
      /*.contract_set_key=*/IREE_SVL("x86.avx2.core"),
  };
  static const loom_target_bundle_t kBundle = {
      /*.name=*/IREE_SVL("x86-avx2"),
      /*.snapshot=*/&kSnapshot,
      /*.export_plan=*/&kExportPlan,
      /*.config=*/&kConfig,
  };

  const loom_llvmir_target_profile_provider_t* provider =
      loom_llvmir_x86_target_profile_provider();
  const loom_llvmir_target_profile_t* profile = nullptr;
  const loom_llvmir_target_profile_projection_request_t request = {
      /*.bundle=*/&kBundle,
      /*.target_triple=*/IREE_SV("x86_64-unknown-linux-gnu"),
  };
  ASSERT_TRUE(provider->project_bundle(&request, &profile));
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(ToString(profile->name), "x86_64-avx2-object");
  EXPECT_EQ(ToString(profile->target_features), "+avx,+avx2,+fma");
}

TEST(LlvmIrTargetEnvTest, RejectsUnknownRegisteredProfileName) {
  const loom_llvmir_target_profile_t* profile = nullptr;
  EXPECT_FALSE(LookupRegisteredProfile(IREE_SV("spirv-vulkan"), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(LlvmIrTargetEnvTest, RegistryOnlySeesExplicitProviders) {
  const loom_llvmir_target_profile_provider_t* providers[] = {
      loom_llvmir_x86_target_profile_provider(),
  };
  loom_llvmir_target_profile_registry_t registry = {};
  registry.default_profile = loom_llvmir_target_profile_x86_64_object();
  registry.providers = providers;
  registry.provider_count = IREE_ARRAYSIZE(providers);

  const loom_llvmir_target_profile_t* profile = nullptr;
  ASSERT_TRUE(loom_llvmir_target_profile_registry_lookup(
      &registry, IREE_SV("x86_64-object"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  EXPECT_FALSE(loom_llvmir_target_profile_registry_lookup(
      &registry, IREE_SV("unlinked-kernel-target"), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(LlvmIrTargetEnvTest, BuildsLlcArgumentsFromTargetProfile) {
  loom_llvmir_target_profile_llc_arguments_t arguments = {};
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_object();
  IREE_ASSERT_OK(loom_llvmir_target_profile_llc_arguments(profile, &arguments));
  ASSERT_EQ(arguments.count, 1u);
  EXPECT_EQ(ToString(arguments.values[0]), "-mtriple=x86_64-unknown-linux-gnu");

  ASSERT_TRUE(
      LookupRegisteredProfile(IREE_SV("x86_64-packed-dot-object"), &profile));
  IREE_ASSERT_OK(loom_llvmir_target_profile_llc_arguments(profile, &arguments));
  ASSERT_EQ(arguments.count, 2u);
  EXPECT_EQ(ToString(arguments.values[0]), "-mtriple=x86_64-unknown-linux-gnu");
  EXPECT_EQ(ToString(arguments.values[1]),
            "-mattr=+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8");
}

}  // namespace
}  // namespace loom
