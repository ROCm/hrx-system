// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 target-family records.

#ifndef LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_

#include <stdint.h>

#include "iree/base/cpu_data.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/types.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable target-family identity for x86 low descriptor sets.
#define LOOM_X86_TARGET_STABLE_ID UINT64_C(0x3f1e78197f70e441)

// Explicit baseline model used without a calibrated processor tune.
#define LOOM_X86_GENERIC_SCHEDULE_MODEL_STABLE_ID UINT64_C(0x037ef8ad209f4f82)

// Stable processor scheduling-model identity.
typedef uint8_t loom_x86_target_tune_t;
enum loom_x86_target_tune_e {
  LOOM_X86_TARGET_TUNE_NONE = 0,
#define LOOM_X86_TARGET_TUNE(enum_suffix, stable_value, \
                             schedule_model_stable_id)  \
  LOOM_X86_TARGET_TUNE_##enum_suffix = stable_value,
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_TARGET_TUNE
  LOOM_X86_TARGET_TUNE_COUNT_,
};

// Schedule-model catalog identity for one processor tune.
typedef struct loom_x86_target_tune_info_t {
  // Stable descriptor schedule-model identity.
  uint64_t schedule_model_stable_id;
} loom_x86_target_tune_info_t;

// Returns the generated metadata for a verified |tune|, including NONE.
const loom_x86_target_tune_info_t* loom_x86_target_tune_info_lookup(
    loom_x86_target_tune_t tune);

// Dense identity of a generated processor deployment profile.
typedef uint8_t loom_x86_deployment_profile_ordinal_t;
enum loom_x86_deployment_profile_ordinal_e {
  LOOM_X86_DEPLOYMENT_PROFILE_NONE = 0,
#define LOOM_X86_DEPLOYMENT_PROFILE(enum_suffix, ordinal, name_length, name,  \
                                    cpu_field0_bits, executable_target_kinds, \
                                    preferred_symbol_suffix,                  \
                                    preferred_target_kind, tune_suffix)       \
  LOOM_X86_DEPLOYMENT_PROFILE_##enum_suffix = ordinal,
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_DEPLOYMENT_PROFILE
  LOOM_X86_DEPLOYMENT_PROFILE_COUNT_,
};

// Immutable guarantees and compiler policy for one named processor target.
typedef struct loom_x86_deployment_profile_t {
  // Canonical selector spelling without the `x86:` family prefix.
  loom_bstring_t name;

  // Canonical usable instruction features guaranteed by the deployment.
  uint64_t cpu_field0_bits;

  // Execution-row bits supported by |cpu_field0_bits|.
  uint32_t executable_target_kinds;

  // Dense generated profile identity.
  loom_x86_deployment_profile_ordinal_t ordinal;

  // Execution row used when no authored root narrows the profile.
  uint8_t preferred_target_kind;

  // Processor scheduling model used by the profile.
  loom_x86_target_tune_t tune;
} loom_x86_deployment_profile_t;

// Returns generated metadata for a verified nonzero |ordinal|.
const loom_x86_deployment_profile_t* loom_x86_deployment_profile_at(
    loom_x86_deployment_profile_ordinal_t ordinal);

// Looks up a deployment profile by its canonical selector spelling.
// Returns NULL when |name| is not a known processor profile.
const loom_x86_deployment_profile_t* loom_x86_deployment_profile_lookup(
    iree_string_view_t name);

// Selects a deployment profile from exact architectural CPU identity.
// Unknown vendors, families, and models return NULL.
const loom_x86_deployment_profile_t* loom_x86_deployment_profile_select_cpu(
    const iree_cpu_data_t* cpu_data);

// Native bundle constants shared by authored records and named profiles.
#define LOOM_X86_NATIVE_TARGET_PROFILE(                                     \
    symbol_suffix, target_kind, selector_name, native_bundle_key,           \
    snapshot_name, descriptor_set_key, descriptor_set_stable_id,            \
    feature_contract_set_key, feature_contract_set_stable_id,               \
    default_feature_bits, supported_feature_bits, required_cpu_field0_bits, \
    automatic_priority)                                                     \
  extern const loom_target_bundle_t kX86LowTargetBundle##symbol_suffix;
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

extern const loom_target_bundle_table_t loom_x86_target_bundles;

// Generated execution and optional-feature policy for one x86 selector.
typedef struct loom_x86_target_policy_t {
  // Descriptor contract selected when semantic features are present.
  loom_target_contract_set_t feature_contract_set;

  // Optional feature assertions this selector can enable.
  loom_x86_feature_bits_t supported_feature_bits;

  // Baseline runtime CPU features required for native execution.
  uint64_t required_cpu_field0_bits;

  // Automatic CPU-selection priority, or zero when execution is unavailable.
  uint8_t automatic_priority;
} loom_x86_target_policy_t;

// Returns the generated policy for a verified nonzero |selector|.
const loom_x86_target_policy_t* loom_x86_target_policy_lookup(uint8_t selector);

// Projects runtime CPU capabilities into the optional features supported by
// |policy|. Runtime features outside the selector's semantic domain are
// intentionally excluded.
loom_x86_feature_bits_t loom_x86_target_policy_project_cpu_features(
    const loom_x86_target_policy_t* policy, uint64_t cpu_field0_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_
