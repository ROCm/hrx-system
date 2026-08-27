// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target record rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_AMDGPU_RECORDS_TARGET_RECORDS_H_

#include <stdint.h>

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_target_record_info_t {
  // Numeric selector value used by amdgpu.target.
  uint32_t target_kind;
  // Canonical target selector named by this record.
  iree_string_view_t target_name;
  // Descriptor-set ordinal selected by the processor.
  uint16_t descriptor_set_ordinal;
  // Target bundle selected by the target record.
  const loom_target_bundle_t* bundle;
} loom_amdgpu_target_record_info_t;

extern const loom_target_bundle_table_t loom_amdgpu_target_bundles;

// Returns the target record selected by |target_kind|, or NULL.
const loom_amdgpu_target_record_info_t* loom_amdgpu_target_record_info_for_kind(
    uint32_t target_kind);

// Returns the target record whose selector is |target_name|.
const loom_amdgpu_target_record_info_t*
loom_amdgpu_target_record_info_for_target(iree_string_view_t target_name);

// Returns the default target record for |descriptor_set_ordinal|, or NULL.
const loom_amdgpu_target_record_info_t*
loom_amdgpu_target_record_default_info_for_descriptor_set(
    uint16_t descriptor_set_ordinal);

// Returns the target bundle selected by an AMDGPU descriptor-set ordinal, or
// NULL when no target record is supported for that descriptor set.
const loom_target_bundle_t* loom_amdgpu_target_bundle_for_descriptor_set(
    uint16_t descriptor_set_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_RECORDS_TARGET_RECORDS_H_
