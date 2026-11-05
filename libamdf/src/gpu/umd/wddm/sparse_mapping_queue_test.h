// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_SPARSE_MAPPING_QUEUE_TEST_H_
#define AMDF_SRC_GPU_UMD_WDDM_SPARSE_MAPPING_QUEUE_TEST_H_

#include <stdint.h>

// Test-only software-scheduled context.
typedef struct amdf_sparse_mapping_test_queue_t
    amdf_sparse_mapping_test_queue_t;

// Native engine selection, consumed by context creation without rediscovery.
typedef struct amdf_sparse_mapping_test_scheduler_t {
  // Driver scheduler identifier of an available software-scheduled SDMA engine.
  uint32_t scheduler;
  // Node ordinal corresponding to that scheduler.
  uint32_t node_ordinal;
  // Nonzero when firmware-managed graphics state is available.
  uint32_t supports_firmware_state_shadowing;
  // SDMA packet fields independent of the queue publication mechanism.
  uint64_t format_features;
} amdf_sparse_mapping_test_scheduler_t;

// Selects one software-scheduled SDMA engine from the shared adapter decoder.
int32_t amdf_sparse_mapping_test_queue_select(
    uint32_t adapter, amdf_sparse_mapping_test_scheduler_t* out_scheduler);

// Creates one context and monitored fence for a selected scheduler.
int32_t amdf_sparse_mapping_test_queue_create(
    uint32_t device, uint32_t physical_adapter_index,
    const amdf_sparse_mapping_test_scheduler_t* scheduler,
    amdf_sparse_mapping_test_queue_t** out_queue);

// Submits one immutable stream and waits for exact monitored progress.
int32_t amdf_sparse_mapping_test_queue_execute(
    amdf_sparse_mapping_test_queue_t* queue, uint64_t address,
    uint32_t byte_length, uint64_t progress);

// Releases a fully retired context and its fence.
int32_t amdf_sparse_mapping_test_queue_destroy(
    amdf_sparse_mapping_test_queue_t* queue);

#endif  // AMDF_SRC_GPU_UMD_WDDM_SPARSE_MAPPING_QUEUE_TEST_H_
