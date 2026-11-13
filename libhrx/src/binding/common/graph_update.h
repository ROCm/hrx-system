// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_GRAPH_UPDATE_H_
#define LIBHRX_SRC_BINDING_COMMON_GRAPH_UPDATE_H_

#include "common/graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compares validated graph descriptions in visible node creation order. Hidden
// copy helpers do not participate in topology or node correspondence.
// Dependency insertion order may differ; node types and copy shapes must remain
// compatible. Borrows |arena| exclusively and restores its allocation state
// before returning. Neither graph is modified or retained. Incompatibility
// returns OK with its detailed result and the enclosing replacement node; only
// scratch allocation failure returns an error status. Callers serialize
// description mutation.
iree_status_t iree_hal_streaming_graph_check_update(
    const iree_hal_streaming_graph_t* old_graph,
    const iree_hal_streaming_graph_t* new_graph, iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_GRAPH_UPDATE_H_
