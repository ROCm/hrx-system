// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_CHECK_TARGET_LOW_REGISTRY_MANIFEST_H_
#define LOOM_TOOLS_LOOM_CHECK_TARGET_LOW_REGISTRY_MANIFEST_H_

#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Appends the loom-check target-low registry inventory as compact JSON.
//
// Schema:
//   {
//     "descriptor_set_count": <number>,
//     "descriptor_sets": [
//       {
//         "key": <string>,
//         "target": <string>,
//         "feature_namespace": <string>,
//         "abi_version": <number>,
//         "generator_version": <number>,
//         "table_counts": { ... }
//       }
//     ]
//   }
//
// This is a loom-check diagnostic/tooling surface for auditing which descriptor
// sets a test environment linked. It intentionally reports descriptor-set
// summaries instead of full instruction rows.
iree_status_t loom_check_target_low_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LOOM_TOOLS_LOOM_CHECK_TARGET_LOW_REGISTRY_MANIFEST_H_
