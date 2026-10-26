// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/records/target_records.h"

#include <stdint.h>

#include "iree/schemas/cpu_data.h"
#include "loom/target/arch/x86/feature_bits.h"

// File-scope immutable tables require integer constant expressions. Derive bit
// positions from the canonical CPU schema because its public masks are C const
// objects.
enum {
#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_position, bit_name, \
                             llvm_name)                                 \
  LOOM_CPU_FEATURE_BIT_POSITION_##arch##_##field_index##_##bit_name =   \
      bit_position,
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT
};

#define LOOM_X86_CPU_FIELD0_BIT_POSITION(bit_name) \
  LOOM_CPU_FEATURE_BIT_POSITION_X86_64_0_##bit_name
#define LOOM_X86_CPU_FIELD0_BIT(bit_name) \
  (UINT64_C(1) << LOOM_X86_CPU_FIELD0_BIT_POSITION(bit_name))

static const loom_x86_target_tune_info_t kX86TargetTuneInfos[] = {
    [LOOM_X86_TARGET_TUNE_NONE] =
        {
            .schedule_model_stable_id =
                LOOM_X86_GENERIC_SCHEDULE_MODEL_STABLE_ID,
        },
#define LOOM_X86_TARGET_TUNE(enum_suffix, stable_value,         \
                             schedule_model_stable_id_arg)      \
  [LOOM_X86_TARGET_TUNE_##enum_suffix] = {                      \
      .schedule_model_stable_id = schedule_model_stable_id_arg, \
  },
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_TARGET_TUNE
};
static_assert(IREE_ARRAYSIZE(kX86TargetTuneInfos) ==
                  LOOM_X86_TARGET_TUNE_COUNT_,
              "x86 target tune metadata must cover every tune");

// Keep names as compact B-strings so named selection never scans a C string.
#define LOOM_X86_DEPLOYMENT_PROFILE(                                         \
    enum_suffix, ordinal, name_length, name_arg, cpu_field0_bits,            \
    executable_target_kinds, preferred_symbol_suffix, preferred_target_kind, \
    tune_suffix)                                                             \
  static const uint8_t kX86DeploymentProfileName##enum_suffix[] =            \
      LOOM_BSTRING_LITERAL(name_length, name_arg);
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_DEPLOYMENT_PROFILE

static const loom_x86_deployment_profile_t kX86DeploymentProfiles[] = {
#define LOOM_X86_DEPLOYMENT_PROFILE(                                       \
    enum_suffix, ordinal_arg, name_length, name_arg, cpu_bits_arg,         \
    executable_targets_arg, preferred_symbol_suffix, preferred_target_arg, \
    tune_suffix)                                                           \
  [ordinal_arg - 1] = {                                                    \
      .name = kX86DeploymentProfileName##enum_suffix,                      \
      .cpu_field0_bits = cpu_bits_arg,                                     \
      .executable_target_kinds = executable_targets_arg,                   \
      .ordinal = ordinal_arg,                                              \
      .preferred_target_kind = preferred_target_arg,                       \
      .tune = LOOM_X86_TARGET_TUNE_##tune_suffix,                          \
  },
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_DEPLOYMENT_PROFILE
};
static_assert(IREE_ARRAYSIZE(kX86DeploymentProfiles) + 1 ==
                  LOOM_X86_DEPLOYMENT_PROFILE_COUNT_,
              "x86 deployment metadata must cover every profile");

typedef struct loom_x86_processor_deployment_match_t {
  uint64_t vendor_low;
  uint32_t vendor_high;
  uint16_t family;
  uint8_t first_model;
  uint8_t last_model;
  loom_x86_deployment_profile_ordinal_t profile_ordinal;
} loom_x86_processor_deployment_match_t;

static const loom_x86_processor_deployment_match_t
    kX86ProcessorDeploymentMatches[] = {
#define LOOM_X86_PROCESSOR_DEPLOYMENT_MATCH(enum_suffix, vendor_low_arg,     \
                                            vendor_high_arg, family_arg,     \
                                            first_model_arg, last_model_arg) \
  {                                                                          \
      .vendor_low = vendor_low_arg,                                          \
      .vendor_high = vendor_high_arg,                                        \
      .family = family_arg,                                                  \
      .first_model = first_model_arg,                                        \
      .last_model = last_model_arg,                                          \
      .profile_ordinal = LOOM_X86_DEPLOYMENT_PROFILE_##enum_suffix,          \
  },
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_PROCESSOR_DEPLOYMENT_MATCH
};

const loom_x86_target_tune_info_t* loom_x86_target_tune_info_lookup(
    loom_x86_target_tune_t tune) {
  IREE_ASSERT(tune < LOOM_X86_TARGET_TUNE_COUNT_);
  return &kX86TargetTuneInfos[tune];
}

const loom_x86_deployment_profile_t* loom_x86_deployment_profile_at(
    loom_x86_deployment_profile_ordinal_t ordinal) {
  IREE_ASSERT(ordinal > LOOM_X86_DEPLOYMENT_PROFILE_NONE &&
              ordinal < LOOM_X86_DEPLOYMENT_PROFILE_COUNT_);
  return &kX86DeploymentProfiles[ordinal - 1];
}

const loom_x86_deployment_profile_t* loom_x86_deployment_profile_lookup(
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kX86DeploymentProfiles);
       ++i) {
    if (loom_bstring_equal(kX86DeploymentProfiles[i].name, name)) {
      return &kX86DeploymentProfiles[i];
    }
  }
  return NULL;
}

const loom_x86_deployment_profile_t* loom_x86_deployment_profile_select_cpu(
    const iree_cpu_data_t* cpu_data) {
  if (cpu_data->architecture != IREE_CPU_ARCHITECTURE_X86_64) {
    return NULL;
  }
  const uint64_t vendor_low =
      (cpu_data->fields[1] & IREE_CPU_DATA1_X86_64_VENDOR_ID_LOW_MASK) >>
      IREE_CPU_DATA1_X86_64_VENDOR_ID_LOW_SHIFT;
  const uint32_t vendor_high =
      (uint32_t)((cpu_data->fields[2] &
                  IREE_CPU_DATA2_X86_64_VENDOR_ID_HIGH_MASK) >>
                 IREE_CPU_DATA2_X86_64_VENDOR_ID_HIGH_SHIFT);
  const uint32_t signature =
      (uint32_t)((cpu_data->fields[2] &
                  IREE_CPU_DATA2_X86_64_PROCESSOR_SIGNATURE_MASK) >>
                 IREE_CPU_DATA2_X86_64_PROCESSOR_SIGNATURE_SHIFT);
  if (vendor_low == 0 || vendor_high == 0 || signature == 0) {
    return NULL;
  }

  const uint8_t base_family = (uint8_t)((signature >> 8) & 0x0F);
  const uint8_t extended_family = (uint8_t)((signature >> 20) & 0xFF);
  const uint16_t family =
      base_family == 0x0F ? base_family + extended_family : base_family;
  const uint8_t base_model = (uint8_t)((signature >> 4) & 0x0F);
  const uint8_t extended_model = (uint8_t)((signature >> 16) & 0x0F);
  const uint8_t model = base_family == 0x06 || base_family == 0x0F
                            ? base_model | (extended_model << 4)
                            : base_model;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kX86ProcessorDeploymentMatches); ++i) {
    const loom_x86_processor_deployment_match_t* match =
        &kX86ProcessorDeploymentMatches[i];
    if (vendor_low == match->vendor_low && vendor_high == match->vendor_high &&
        family == match->family && model >= match->first_model &&
        model <= match->last_model) {
      return loom_x86_deployment_profile_at(match->profile_ordinal);
    }
  }
  return NULL;
}

#define LOOM_X86_LOW_SNAPSHOT(symbol, snapshot_name)           \
  static const loom_target_snapshot_t symbol = {               \
      .name = IREE_SVL(snapshot_name),                         \
      .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE, \
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,      \
      .default_pointer_bitwidth = 64,                          \
      .index_bitwidth = 64,                                    \
      .offset_bitwidth = 64,                                   \
      .max_workgroup_size = {1, 1, 1},                         \
      .max_flat_workgroup_size = 1,                            \
      .subgroup_size = 1,                                      \
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

#define LOOM_X86_NATIVE_TARGET_PROFILE(                                     \
    symbol_suffix, target_kind, selector_name, native_bundle_key,           \
    snapshot_name, descriptor_set_key, descriptor_set_stable_id,            \
    feature_contract_set_key, feature_contract_set_stable_id,               \
    default_feature_bits, supported_feature_bits, required_cpu_field0_bits, \
    automatic_priority)                                                     \
  LOOM_X86_LOW_SNAPSHOT(kX86##symbol_suffix##Snapshot, snapshot_name);
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE
#undef LOOM_X86_LOW_SNAPSHOT

static const loom_target_export_plan_t kX86_64ObjectExportPlan = {
    .name = IREE_SVL("x86_64-object"),
    .calling_convention = IREE_SVL("sysv"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

#define LOOM_X86_NATIVE_TARGET_PROFILE(                                      \
    symbol_suffix, target_kind, selector_name, native_bundle_key,            \
    snapshot_name, descriptor_set_key, descriptor_set_stable_id,             \
    feature_contract_set_key, feature_contract_set_stable_id,                \
    default_feature_bits, supported_feature_bits, required_cpu_field0_bits,  \
    automatic_priority)                                                      \
  static const loom_target_config_t kX86##symbol_suffix##Config = {          \
      .name = IREE_SVL(descriptor_set_key),                                  \
      .schedule_model_stable_id = LOOM_X86_GENERIC_SCHEDULE_MODEL_STABLE_ID, \
      .contract_set =                                                        \
          {                                                                  \
              .stable_id = descriptor_set_stable_id,                         \
              .spelling = IREE_SVL(descriptor_set_key),                      \
          },                                                                 \
      .contract_feature_bits = default_feature_bits,                         \
  };
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

#define LOOM_X86_NATIVE_TARGET_PROFILE(                                     \
    symbol_suffix, target_kind, selector_name, native_bundle_key,           \
    snapshot_name, descriptor_set_key, descriptor_set_stable_id,            \
    feature_contract_set_key, feature_contract_set_stable_id,               \
    default_feature_bits, supported_feature_bits, required_cpu_field0_bits, \
    automatic_priority)                                                     \
  const loom_target_bundle_t kX86LowTargetBundle##symbol_suffix = {         \
      .name = IREE_SVL(native_bundle_key),                                  \
      .snapshot = &kX86##symbol_suffix##Snapshot,                           \
      .export_plan = &kX86_64ObjectExportPlan,                              \
      .config = &kX86##symbol_suffix##Config,                               \
  };
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

static const loom_target_bundle_t* const kX86TargetBundleValues[] = {
    NULL,
#define LOOM_X86_NATIVE_TARGET_PROFILE(                                     \
    symbol_suffix, target_kind, selector_name, native_bundle_key,           \
    snapshot_name, descriptor_set_key, descriptor_set_stable_id,            \
    feature_contract_set_key, feature_contract_set_stable_id,               \
    default_feature_bits, supported_feature_bits, required_cpu_field0_bits, \
    automatic_priority)                                                     \
  &kX86LowTargetBundle##symbol_suffix,
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE
};

const loom_target_bundle_table_t loom_x86_target_bundles = {
    .values = kX86TargetBundleValues,
    .count = IREE_ARRAYSIZE(kX86TargetBundleValues),
};

static const loom_x86_target_policy_t kX86TargetPolicies[] = {
    {.feature_contract_set = {0}},
#define LOOM_X86_NATIVE_TARGET_PROFILE(                                  \
    symbol_suffix, target_kind, selector_name, native_bundle_key,        \
    snapshot_name, descriptor_set_key, descriptor_set_stable_id,         \
    feature_contract_key_arg, feature_contract_stable_id_arg,            \
    default_feature_bits, supported_features_arg, required_cpu_bits_arg, \
    automatic_priority_arg)                                              \
  [target_kind] = {                                                      \
      .feature_contract_set =                                            \
          {                                                              \
              .stable_id = feature_contract_stable_id_arg,               \
              .spelling = IREE_SVL(feature_contract_key_arg),            \
          },                                                             \
      .supported_feature_bits = supported_features_arg,                  \
      .required_cpu_field0_bits = required_cpu_bits_arg,                 \
      .automatic_priority = automatic_priority_arg,                      \
  },
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE
};

// Runtime CPU capability bit position indexed by the corresponding dense
// target feature ordinal. UINT8_MAX identifies a target-only feature.
static const uint8_t kX86CpuFeatureBitPositionsByTargetOrdinal[] = {
#define LOOM_X86_CPU_FEATURE_MAPPING(cpu_field0_bit_position) \
  cpu_field0_bit_position,
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_CPU_FEATURE_MAPPING
};
static_assert(
    (1u << IREE_ARRAYSIZE(kX86CpuFeatureBitPositionsByTargetOrdinal)) - 1 ==
        LOOM_X86_FEATURE_KNOWN_BITS,
    "x86 CPU feature projection must cover every target feature ordinal");

const loom_x86_target_policy_t* loom_x86_target_policy_lookup(
    uint8_t selector) {
  IREE_ASSERT(selector > 0 && selector < IREE_ARRAYSIZE(kX86TargetPolicies));
  return &kX86TargetPolicies[selector];
}

loom_x86_feature_bits_t loom_x86_target_policy_project_cpu_features(
    const loom_x86_target_policy_t* policy, uint64_t cpu_field0_bits) {
  loom_x86_feature_bits_t feature_bits = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kX86CpuFeatureBitPositionsByTargetOrdinal); ++i) {
    const uint8_t bit_position = kX86CpuFeatureBitPositionsByTargetOrdinal[i];
    if (bit_position != UINT8_MAX &&
        iree_any_bit_set(cpu_field0_bits, UINT64_C(1) << bit_position)) {
      feature_bits |= (loom_x86_feature_bits_t)(1u << i);
    }
  }
  return feature_bits & policy->supported_feature_bits;
}

#undef LOOM_X86_CPU_FIELD0_BIT
#undef LOOM_X86_CPU_FIELD0_BIT_POSITION
