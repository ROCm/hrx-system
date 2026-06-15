// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/records/target_records.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/target_info.h"

// clang-format off
#define LOOM_AMDGPU_LOW_SNAPSHOT(symbol, snapshot_name, wavefront_size)      \
  static const loom_target_snapshot_t symbol = {                             \
      .name = IREE_SVL(snapshot_name),                                       \
      .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,               \
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,                    \
      .default_pointer_bitwidth = 64,                                        \
      .index_bitwidth = 32,                                                  \
      .offset_bitwidth = 64,                                                 \
      .max_workgroup_size = {.x = 1024, .y = 1024, .z = 1024},               \
      .max_flat_workgroup_size = 1024,                                       \
      .subgroup_size = wavefront_size,                                       \
      .max_grid_size = {.x = INT32_MAX, .y = UINT16_MAX, .z = UINT16_MAX},   \
      .max_flat_grid_size = UINT32_MAX,                                      \
      .max_workgroup_count = {.x = INT32_MAX, .y = UINT16_MAX, .z = UINT16_MAX}, \
      .memory_spaces = {                                                     \
          .generic = 0,                                                      \
          .global = 1,                                                       \
          .workgroup = 3,                                                    \
          .constant = 4,                                                     \
          .private_memory = 5,                                               \
          .host = UINT32_MAX,                                                \
          .descriptor = 7,                                                   \
      },                                                                     \
  }

#define LOOM_AMDGPU_TARGET_DESCRIPTOR_SET(symbol_suffix, bundle_name, \
                                          snapshot_name, key, wavefront_size) \
  LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpu##symbol_suffix##Snapshot, snapshot_name, \
                           wavefront_size);
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_DESCRIPTOR_SET

static const loom_target_export_plan_t kAmdgpuHalExportPlan = {
  .name = IREE_SVL("amdgpu-hal"),
  .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
  .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
  .hal_kernel = {
    .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
    .flat_workgroup_size_min = 0,
    .flat_workgroup_size_max = 0,
    .buffer_resource_flags = LOOM_AMDGPU_HAL_BUFFER_RESOURCE_FLAGS,
  },
};

#define LOOM_AMDGPU_LOW_CONFIG(symbol, key) \
  static const loom_target_config_t symbol = { \
      .name = IREE_SVL(key), \
      .contract_set_key = IREE_SVL(key), \
  }

#define LOOM_AMDGPU_TARGET_DESCRIPTOR_SET(symbol_suffix, bundle_name, \
                                          snapshot_name, key, wavefront_size) \
  LOOM_AMDGPU_LOW_CONFIG(kAmdgpu##symbol_suffix##Config, key);
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_DESCRIPTOR_SET

#define LOOM_AMDGPU_TARGET_DESCRIPTOR_SET(symbol_suffix, bundle_name, \
                                          snapshot_name, key, wavefront_size) \
  static const loom_target_bundle_t kAmdgpuLowTargetBundle##symbol_suffix##Core = { \
    .name = IREE_SVL(bundle_name), \
    .snapshot = &kAmdgpu##symbol_suffix##Snapshot, \
    .export_plan = &kAmdgpuHalExportPlan, \
    .config = &kAmdgpu##symbol_suffix##Config, \
  };
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_DESCRIPTOR_SET

#define LOOM_AMDGPU_TARGET_RECORD_INFO(record_suffix, target_kind_value, \
                                       processor_name, descriptor_set_ordinal_value, \
                                       bundle_suffix) \
  static const loom_amdgpu_target_record_info_t \
      kAmdgpuTargetRecordInfo##record_suffix = { \
          .target_kind = target_kind_value, \
          .default_processor_name = IREE_SVL(processor_name), \
          .descriptor_set_ordinal = descriptor_set_ordinal_value, \
          .bundle = &kAmdgpuLowTargetBundle##bundle_suffix##Core, \
      };
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_RECORD_INFO

static const loom_target_bundle_t* const kAmdgpuTargetBundleValues[] = {
  NULL,
#define LOOM_AMDGPU_TARGET_RECORD_INFO(record_suffix, target_kind_value, \
                                       processor_name, descriptor_set_ordinal_value, \
                                       bundle_suffix) \
  &kAmdgpuLowTargetBundle##bundle_suffix##Core,
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_RECORD_INFO
};

const loom_target_bundle_table_t loom_amdgpu_target_bundles = {
  .values = kAmdgpuTargetBundleValues,
  .count = IREE_ARRAYSIZE(kAmdgpuTargetBundleValues),
};

static const loom_amdgpu_target_record_info_t* const kAmdgpuTargetRecordInfos[] = {
  NULL,
#define LOOM_AMDGPU_TARGET_RECORD_INFO(record_suffix, target_kind_value, \
                                       processor_name, descriptor_set_ordinal_value, \
                                       bundle_suffix) \
  &kAmdgpuTargetRecordInfo##record_suffix,
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_RECORD_INFO
};

static const loom_amdgpu_target_record_info_t* const
    kAmdgpuDefaultTargetRecordInfosByDescriptorSet[] = {
#define LOOM_AMDGPU_TARGET_RECORD_DEFAULT(descriptor_set_ordinal, record_suffix) \
  &kAmdgpuTargetRecordInfo##record_suffix,
#define LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT(descriptor_set_ordinal) NULL,
#include "loom/target/arch/amdgpu/records/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_RECORD_DEFAULT
#undef LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT
};
// clang-format on

const loom_amdgpu_target_record_info_t* loom_amdgpu_target_record_info_for_kind(
    uint32_t target_kind) {
  if (target_kind >= IREE_ARRAYSIZE(kAmdgpuTargetRecordInfos)) {
    return NULL;
  }
  return kAmdgpuTargetRecordInfos[target_kind];
}

const loom_amdgpu_target_record_info_t*
loom_amdgpu_target_record_info_for_processor(
    iree_string_view_t processor_name) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(kAmdgpuTargetRecordInfos);
       ++i) {
    const loom_amdgpu_target_record_info_t* info = kAmdgpuTargetRecordInfos[i];
    if (info != NULL &&
        iree_string_view_equal(info->default_processor_name, processor_name)) {
      return info;
    }
  }
  return NULL;
}

const loom_amdgpu_target_record_info_t*
loom_amdgpu_target_record_default_info_for_descriptor_set(
    uint16_t descriptor_set_ordinal) {
  if (descriptor_set_ordinal >=
      IREE_ARRAYSIZE(kAmdgpuDefaultTargetRecordInfosByDescriptorSet)) {
    return NULL;
  }
  return kAmdgpuDefaultTargetRecordInfosByDescriptorSet[descriptor_set_ordinal];
}

const loom_target_bundle_t* loom_amdgpu_target_bundle_for_descriptor_set(
    uint16_t descriptor_set_ordinal) {
  const loom_amdgpu_target_record_info_t* info =
      loom_amdgpu_target_record_default_info_for_descriptor_set(
          descriptor_set_ordinal);
  return info != NULL ? info->bundle : NULL;
}
