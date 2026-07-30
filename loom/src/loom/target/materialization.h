// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Durable target-record projection and construction.

#ifndef LOOM_TARGET_MATERIALIZATION_H_
#define LOOM_TARGET_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// One family-owned target attribute appended beside the common projection.
typedef struct loom_target_record_extension_attr_t {
  // Attribute index in the target operation.
  uint8_t attr_index;

  // Structured attribute value, or absent when the profile uses the row
  // default.
  loom_attribute_t value;
} loom_target_record_extension_attr_t;

// Resolves the generated row and authored overrides of |target| into owned
// bundle storage.
//
// |record_name| supplies diagnostic-only bundle names. When provided,
// |out_authored_attrs| receives the projected attrs explicitly present on the
// target. Returns false when the target selector does not name a generated
// row. Verified internal target records always resolve.
bool loom_target_record_projection_resolve(
    const loom_module_t* module, loom_target_like_t target,
    iree_string_view_t record_name, loom_target_bundle_storage_t* out_storage,
    loom_target_authored_attr_set_t* out_authored_attrs);

// Returns whether the durable common projection of |target_op| equals the
// specialization of |selected_bundle| by |authored_target_op|.
//
// Symbol and bundle names are diagnostic labels and do not participate.
// Family-owned extension attributes remain the provider's responsibility.
// |authored_target_op| may be NULL when materializing an unqualified profile.
bool loom_target_record_projection_matches_bundle(
    const loom_module_t* module, const loom_op_t* target_op,
    const loom_target_bundle_t* selected_bundle,
    const loom_op_t* authored_target_op);

// Builds a target-like record by specializing a complete profile projection
// with preserved fields from |authored_target_op| and family-owned extension
// attributes.
//
// Fields equal to the generated selector row remain implicit. Differing
// projected fields become typed attributes. The operation kind must implement
// TargetLike as a metadata-only operation with no operands, results, or
// regions. |authored_target_op| may be NULL when materializing an unqualified
// profile.
iree_status_t loom_target_record_projection_build(
    loom_builder_t* builder, loom_op_kind_t op_kind, uint8_t selector,
    loom_symbol_ref_t symbol, const loom_target_bundle_t* selected_bundle,
    const loom_op_t* authored_target_op,
    const loom_target_record_extension_attr_t* extension_attrs,
    iree_host_size_t extension_attr_count, loom_location_id_t location,
    loom_op_t** out_target_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MATERIALIZATION_H_
