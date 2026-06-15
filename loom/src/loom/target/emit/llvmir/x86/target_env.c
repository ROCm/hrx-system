// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/target_env.h"

#include "loom/target/arch/x86/feature_bits.h"

#define LOOM_LLVMIR_X86_64_TARGET_TRIPLE IREE_SVL("x86_64-unknown-linux-gnu")
#define LOOM_LLVMIR_X86_64_DATA_LAYOUT              \
  IREE_SVL(                                         \
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:" \
      "64-i128:128-f80:128-n8:16:32:64-S128")

static const loom_llvmir_target_env_t kX86_64UnknownLinuxGnuTargetEnv = {
    .name = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .target_triple = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .data_layout = LOOM_LLVMIR_X86_64_DATA_LAYOUT,
    .object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .address_spaces =
        {
            .generic = 0,
            .global = 0,
            .local = 0,
            .constant = 0,
            .private_memory = 0,
            .buffer_resource = UINT32_MAX,
        },
};

static const loom_target_snapshot_t kX86_64ObjectSnapshot = {
    .name = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = 0,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_snapshot_t kX86_64PackedDotObjectSnapshot = {
    .name = IREE_SVL("x86_64-unknown-linux-gnu-packed-dot"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = 0,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kX86_64ObjectExportPlan = {
    .name = IREE_SVL("x86_64-object"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

#define LOOM_LLVMIR_X86_TARGET_FIXTURE(symbol_suffix, descriptor_set_key, \
                                       feature_bits)                      \
  static const loom_target_config_t kX86##symbol_suffix##ObjectConfig = { \
      .name = IREE_SVL(descriptor_set_key),                               \
      .contract_set_key = IREE_SVL(descriptor_set_key),                   \
      .contract_feature_bits = feature_bits,                              \
  };
#include "loom/target/emit/llvmir/x86/target_profiles.inl"
#undef LOOM_LLVMIR_X86_TARGET_FIXTURE

static const loom_target_bundle_t kX86_64ObjectBundle = {
    .name = IREE_SVL("x86_64-object"),
    .snapshot = &kX86_64ObjectSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86ScalarObjectConfig,
};

static const loom_target_bundle_t kX86_64PackedDotObjectBundle = {
    .name = IREE_SVL("x86_64-packed-dot-object"),
    .snapshot = &kX86_64PackedDotObjectSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86PackedDotObjectConfig,
};

// These built-in profiles are fixture/default provider conveniences. Production
// lowering should prefer derived profiles from the generic target bundles
// above.
#define LOOM_LLVMIR_X86_TARGET_PROFILE(                           \
    symbol_suffix, profile_descriptor_set_key, profile_debug_key, \
    profile_target_features, profile_feature_bits)                \
  static const loom_llvmir_target_profile_t                       \
      kX86##symbol_suffix##ObjectProfile = {                      \
          .name = IREE_SVL(profile_debug_key),                    \
          .target_env = &kX86_64UnknownLinuxGnuTargetEnv,         \
          .kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT,         \
          .target_features = IREE_SVL(profile_target_features),   \
          .exported_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,      \
          .x86_packed_dot_feature_bits = profile_feature_bits,    \
  };
#include "loom/target/emit/llvmir/x86/target_profiles.inl"
#undef LOOM_LLVMIR_X86_TARGET_PROFILE

static const loom_llvmir_target_profile_t* const kX86TargetProfiles[] = {
#define LOOM_LLVMIR_X86_TARGET_PROFILE(                           \
    symbol_suffix, profile_descriptor_set_key, profile_debug_key, \
    profile_target_features, profile_feature_bits)                \
  &kX86##symbol_suffix##ObjectProfile,
#include "loom/target/emit/llvmir/x86/target_profiles.inl"
#undef LOOM_LLVMIR_X86_TARGET_PROFILE
};

static bool loom_llvmir_x86_profile_by_name(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = NULL;
#define LOOM_LLVMIR_X86_TARGET_PROFILE(                                   \
    symbol_suffix, profile_descriptor_set_key, profile_debug_key,         \
    profile_target_features, profile_feature_bits)                        \
  if (iree_string_view_equal(profile_name, IREE_SV(profile_debug_key))) { \
    *out_profile = &kX86##symbol_suffix##ObjectProfile;                   \
    return true;                                                          \
  }
#include "loom/target/emit/llvmir/x86/target_profiles.inl"
#undef LOOM_LLVMIR_X86_TARGET_PROFILE
  return false;
}

static bool loom_llvmir_x86_profile_by_descriptor_set_key(
    iree_string_view_t requested_descriptor_set_key,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = NULL;
#define LOOM_LLVMIR_X86_TARGET_PROFILE(                              \
    symbol_suffix, profile_descriptor_set_key, profile_debug_key,    \
    profile_target_features, profile_feature_bits)                   \
  if (iree_string_view_equal(requested_descriptor_set_key,           \
                             IREE_SV(profile_descriptor_set_key))) { \
    *out_profile = &kX86##symbol_suffix##ObjectProfile;              \
    return true;                                                     \
  }
#include "loom/target/emit/llvmir/x86/target_profiles.inl"
#undef LOOM_LLVMIR_X86_TARGET_PROFILE
  return false;
}

static bool loom_llvmir_x86_project_bundle(
    const loom_llvmir_target_profile_projection_request_t* request,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = NULL;
  const loom_target_bundle_t* bundle = request->bundle;
  if (!iree_string_view_is_empty(request->target_triple) &&
      !iree_string_view_equal(request->target_triple,
                              kX86_64UnknownLinuxGnuTargetEnv.target_triple)) {
    return false;
  }
  if (loom_llvmir_x86_profile_by_name(bundle->name, out_profile)) {
    return true;
  }
  if (bundle->export_plan->abi_kind != LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    return false;
  }
  return loom_llvmir_x86_profile_by_descriptor_set_key(
      bundle->config->contract_set_key, out_profile);
}

static const loom_llvmir_target_profile_provider_t kX86TargetProfileProvider = {
    .name = IREE_SVL("x86"),
    .profiles = kX86TargetProfiles,
    .profile_count = IREE_ARRAYSIZE(kX86TargetProfiles),
    .llc_target_name = IREE_SVL("x86"),
    .project_bundle = loom_llvmir_x86_project_bundle,
};

const loom_target_bundle_t* loom_llvmir_target_bundle_x86_64_object(void) {
  return &kX86_64ObjectBundle;
}

const loom_target_bundle_t* loom_llvmir_target_bundle_x86_64_packed_dot_object(
    void) {
  return &kX86_64PackedDotObjectBundle;
}

const loom_llvmir_target_env_t* loom_llvmir_target_env_x86_64_unknown_linux_gnu(
    void) {
  return &kX86_64UnknownLinuxGnuTargetEnv;
}

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_x86_64_object(
    void) {
  return &kX86ScalarObjectProfile;
}

const loom_llvmir_target_profile_t*
loom_llvmir_target_profile_x86_64_packed_dot_object(void) {
  return &kX86PackedDotObjectProfile;
}

const loom_llvmir_target_profile_provider_t*
loom_llvmir_x86_target_profile_provider(void) {
  return &kX86TargetProfileProvider;
}

iree_status_t loom_llvmir_target_profile_initialize_x86_64_object(
    loom_llvmir_target_profile_t* out_profile) {
  *out_profile = kX86ScalarObjectProfile;
  return iree_ok_status();
}
