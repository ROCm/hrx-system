// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/records/target_records.h"

#include <stdint.h>

#include "loom/target/arch/x86/feature_bits.h"

#define LOOM_X86_LOW_SNAPSHOT(symbol, snapshot_name)           \
  static const loom_target_snapshot_t symbol = {               \
      .name = IREE_SVL(snapshot_name),                         \
      .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE, \
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,      \
      .default_pointer_bitwidth = 64,                          \
      .index_bitwidth = 64,                                    \
      .offset_bitwidth = 64,                                   \
      .memory_spaces =                                         \
          {                                                    \
              .generic = 0,                                    \
              .global = 0,                                     \
              .workgroup = 0,                                  \
              .constant = 0,                                   \
              .private_memory = 0,                             \
              .host = 0,                                       \
              .descriptor = UINT32_MAX,                        \
          },                                                   \
  }

#define LOOM_X86_NATIVE_TARGET_PROFILE(symbol_suffix, native_bundle_key,  \
                                       snapshot_name, descriptor_set_key, \
                                       feature_bits)                      \
  LOOM_X86_LOW_SNAPSHOT(kX86##symbol_suffix##Snapshot, snapshot_name);
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE
#undef LOOM_X86_LOW_SNAPSHOT

static const loom_target_export_plan_t kX86_64ObjectExportPlan = {
    .name = IREE_SVL("x86_64-object"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

#define LOOM_X86_NATIVE_TARGET_PROFILE(symbol_suffix, native_bundle_key,  \
                                       snapshot_name, descriptor_set_key, \
                                       feature_bits)                      \
  static const loom_target_config_t kX86##symbol_suffix##Config = {       \
      .name = IREE_SVL(descriptor_set_key),                               \
      .contract_set_key = IREE_SVL(descriptor_set_key),                   \
      .contract_feature_bits = feature_bits,                              \
  };
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

#define LOOM_X86_NATIVE_TARGET_PROFILE(symbol_suffix, native_bundle_key,   \
                                       snapshot_name, descriptor_set_key,  \
                                       feature_bits)                       \
  static const loom_target_bundle_t kX86LowTargetBundle##symbol_suffix = { \
      .name = IREE_SVL(native_bundle_key),                                 \
      .snapshot = &kX86##symbol_suffix##Snapshot,                          \
      .export_plan = &kX86_64ObjectExportPlan,                             \
      .config = &kX86##symbol_suffix##Config,                              \
  };
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

static const loom_target_bundle_t* const kX86TargetBundleValues[] = {
    NULL,
#define LOOM_X86_NATIVE_TARGET_PROFILE(symbol_suffix, native_bundle_key,  \
                                       snapshot_name, descriptor_set_key, \
                                       feature_bits)                      \
  &kX86LowTargetBundle##symbol_suffix,
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE
};

const loom_target_bundle_table_t loom_x86_target_bundles = {
    .values = kX86TargetBundleValues,
    .count = IREE_ARRAYSIZE(kX86TargetBundleValues),
};
