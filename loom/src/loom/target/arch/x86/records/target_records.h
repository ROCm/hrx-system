// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 target-family records.

#ifndef LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_

#include <stdint.h>

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable target-family identity for x86 low descriptor sets.
#define LOOM_X86_TARGET_STABLE_ID UINT64_C(0x3f1e78197f70e441)

// Native bundle constants shared by authored records and named profiles.
#define LOOM_X86_NATIVE_TARGET_PROFILE(                           \
    symbol_suffix, target_kind, selector_name, native_bundle_key, \
    snapshot_name, descriptor_set_key, feature_bits)              \
  extern const loom_target_bundle_t kX86LowTargetBundle##symbol_suffix;
#include "loom/target/arch/x86/records/target_profiles.inl"
#undef LOOM_X86_NATIVE_TARGET_PROFILE

extern const loom_target_bundle_table_t loom_x86_target_bundles;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_RECORDS_TARGET_RECORDS_H_
