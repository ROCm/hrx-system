// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 target-family records.

#ifndef LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_

#include <stdint.h>

#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable target-family identity for x86 low descriptor sets.
#define LOOM_X86_TARGET_STABLE_ID UINT64_C(0x3f1e78197f70e441)

// Native bundle constants shared by authored records and named profiles.
#define LOOM_X86_NATIVE_TARGET_PROFILE(                                     \
    symbol_suffix, target_kind, selector_name, native_bundle_key,           \
    snapshot_name, descriptor_set_key, feature_contract_set_key,            \
    default_feature_bits, supported_feature_bits, required_cpu_field0_bits, \
    automatic_priority)                                                     \
  extern const loom_target_bundle_t kX86LowTargetBundle##symbol_suffix;
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

extern const loom_target_bundle_table_t loom_x86_target_bundles;

// Generated execution and optional-feature policy for one x86 selector.
typedef struct loom_x86_target_policy_t {
  // Descriptor contract selected when semantic features are present.
  iree_string_view_t feature_contract_set_key;

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
