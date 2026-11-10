// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_NATIVE_SOURCE_CACHE_H_
#define LOOMC_TARGET_CMD_NATIVE_SOURCE_CACHE_H_

#include "loom/binding/c/src/link_materialization.h"
#include "loom/transforms/kernel/kernel_request_producer.h"

// Completed native source preparations over one exact immutable provider index.
// The owner keeps the index/context alive until all synchronous calls return.
typedef struct loomc_cmd_native_source_cache_t loomc_cmd_native_source_cache_t;

iree_status_t loomc_cmd_native_source_cache_allocate(
    const loom_link_module_index_t* index, iree_host_size_t retained_byte_limit,
    iree_allocator_t allocator, loomc_cmd_native_source_cache_t** out_cache);

void loomc_cmd_native_source_cache_free(loomc_cmd_native_source_cache_t* cache);

// Borrows normalized invocation state while preparing a miss, then leases the
// completed preparation through class collection and all synchronous callbacks.
// No cache lock is held while compiling, allocating, or calling the sink.
iree_status_t loomc_cmd_native_source_cache_publish(
    loomc_cmd_native_source_cache_t* cache,
    loomc_link_materialization_state_t* state,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes);

#endif  // LOOMC_TARGET_CMD_NATIVE_SOURCE_CACHE_H_
