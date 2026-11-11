// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact module-independent keys for closed staged literals.
// Keys and queries share the immutable compiler context owning their
// descriptors.

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_SPECIALIZATION_KEY_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_SPECIALIZATION_KEY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/transforms/kernel/kernel_specialization.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct loom_kernel_specialization_query_t
    loom_kernel_specialization_query_t;

typedef struct loom_kernel_specialization_key_t {
  // One owned packed stream; NULL when the size bound declines retention.
  const uint8_t* data;
  // Exact storage bytes, including semantic tags and lengths.
  iree_host_size_t length;
} loom_kernel_specialization_key_t;

// Projects exact closed tuples into an invocation-owned query. Payload spans
// borrow |module| through comparison or capture; no module-local identity is
// used as semantic identity. Traversal records reached type/encoding nodes once
// and keeps payload bytes borrowed until retention is admitted.
iree_status_t loom_kernel_specialization_query_allocate(
    const loom_module_t* module,
    const loom_kernel_specialization_value_t* values, uint16_t count,
    iree_arena_allocator_t* arena,
    loom_kernel_specialization_query_t** out_query);

// Bounded lookup accelerator; a match still requires exact key comparison.
uint32_t loom_kernel_specialization_query_hash(
    const loom_kernel_specialization_query_t* query);

// Complete key bytes required for independent retention.
iree_host_size_t loom_kernel_specialization_query_size(
    const loom_kernel_specialization_query_t* query);

// Copies a key only when its complete byte count fits the supplied limit.
iree_status_t loom_kernel_specialization_key_capture(
    const loom_kernel_specialization_query_t* query,
    iree_host_size_t byte_limit, iree_arena_allocator_t* arena,
    loom_kernel_specialization_key_t* out_key);

// Exact comparison consumes already projected query parts; no allocation.
bool loom_kernel_specialization_key_matches(
    const loom_kernel_specialization_query_t* query,
    loom_kernel_specialization_key_t key);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_SPECIALIZATION_KEY_H_
