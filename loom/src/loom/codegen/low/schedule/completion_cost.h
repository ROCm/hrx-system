// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Incremental completion-wait costs for resource-stall scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_COST_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_COST_H_

#include "iree/base/api.h"
#include "loom/codegen/low/schedule/dependency_incoming.h"
#include "loom/codegen/low/schedule/dependency_index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the strongest unresolved wait cost reaching |consumer_node|. The
// consumer's slot carries conservative cross-block cost; scheduled producer
// slots carry unresolved same-block costs read through |incoming_index|.
uint16_t loom_low_schedule_completion_cost_query(
    const uint16_t* node_cycles,
    const loom_low_schedule_dependency_index_t* dependency_index,
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t consumer_node);

// Records that |node_index| has scheduled. Its incoming same-block effect
// producers have now paid their modeled completion cost, and its slot begins
// carrying |opened_wait_cycles| for later effect consumers.
void loom_low_schedule_completion_cost_note_node_scheduled(
    uint16_t* node_cycles,
    const loom_low_schedule_dependency_index_t* dependency_index,
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t node_index, uint16_t opened_wait_cycles);

// Conservatively publishes a producer's wait cost to a consumer in another
// block, where path-sensitive retirement is unavailable during local list
// scheduling.
void loom_low_schedule_completion_cost_publish_cross_block(
    uint16_t* node_cycles, uint32_t consumer_node, uint16_t wait_cycles);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_COMPLETION_COST_H_
