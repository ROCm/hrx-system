// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable products containing portable command programs.

#ifndef LOOM_TARGET_ARCH_CMD_PRODUCT_H_
#define LOOM_TARGET_ARCH_CMD_PRODUCT_H_

#include "iree/base/api.h"
#include "loom/product/registry.h"
#include "loom/target/arch/cmd/artifact_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public format name for portable Loom command-program products.
#define LOOM_CMD_PRODUCT_FORMAT_LOOM_COMMAND "loom-command"

// Process-local descriptor for portable command-program products.
extern const loom_product_descriptor_t loom_cmd_product_descriptor;

// Semantic command product operation and its durable root operation.
extern const loom_product_operation_t loom_cmd_product_operation;

// Portable command-program product format.
extern const loom_product_format_t loom_cmd_product_format;

// Creates an immutable command product by moving |*inout_artifact_set|.
//
// Ownership transfers only on success. The returned product preserves the
// program order as its export order and the entry table as its requirement
// order.
iree_status_t loom_cmd_product_create(
    loom_cmd_program_artifact_set_t* inout_artifact_set,
    iree_allocator_t allocator, loom_product_t** out_product);

// Returns the immutable command artifact set, or NULL for another product.
const loom_cmd_program_artifact_set_t* loom_cmd_product_artifact_set(
    const loom_product_t* product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_PRODUCT_H_
