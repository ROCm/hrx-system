// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_GRAPH_ANALYSIS_H_
#define LIBHRX_GRAPH_ANALYSIS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum partition width, also bounding graph launch semaphore scratch.
#define HRX_GRAPH_MAX_WORKSTREAMS 8

// The low bits describe recording boundaries; the remaining bits carry the
// schedule's barrier decision and workstream. Payloads stay in the frontend.
enum hrx_graph_analysis_node_flags_e {
  HRX_GRAPH_ANALYSIS_NODE_RECORDABLE = 0,
  HRX_GRAPH_ANALYSIS_NODE_ISOLATED = 1,
  HRX_GRAPH_ANALYSIS_NODE_EMPTY = 2,
  HRX_GRAPH_ANALYSIS_NODE_KIND_MASK = 3,
  HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE = 1 << 2,
  HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT = 3,
};

// Compilation-only indexed graph. All arrays are arena-owned. Frontends
// linearize their source nodes, classify them, and count incoming edges in
// offsets[node + 1]. allocate_dependencies converts those counts into ranges
// and initializes order as write cursors. Frontends then append each source
// ordinal to dependencies[order[target]++]. analyze consumes the sealed ranges.
// Node identities and dependency counts fit uint32_t; no runtime graph state
// or reference ownership is created here.
typedef struct hrx_graph_analysis_t {
  // Number of dense source-node ordinals.
  uint32_t node_count;
  // Borrowed frontend payloads, never dereferenced by analysis.
  void** sources;
  // Incoming dependency offsets with a final sentinel (node_count + 1).
  uint32_t* offsets;
  // Contiguous incoming source-node ordinals.
  uint32_t* dependencies;
  // Write cursors during preparation, then topological source-node ordinals.
  uint32_t* order;
  // Packed node kind, barrier-before flag, and workstream by source ordinal.
  uint8_t* flags;
} hrx_graph_analysis_t;

// One bounded recordable range or one isolated/empty operation. The 24-bit
// count covers the recording bound; stream_count is at most eight.
typedef struct hrx_graph_partition_t {
  // First position in the schedule's order array.
  uint32_t start_index;
  // Number of ordered nodes in this partition.
  uint32_t count : 24;
  // Number of independent command recordings, or one for an isolated node.
  uint32_t stream_count : 8;
} hrx_graph_partition_t;

// Recording consumes order, flags, and the source table directly. No dependency
// walk or inverse lookup is needed after analysis. Storage is valid only until
// the compilation arena is reset; compiled blocks never borrow these arrays.
typedef struct hrx_graph_schedule_t {
  // Number of nodes in order and flags.
  uint32_t node_count;
  // Number of partition descriptors.
  uint32_t partition_count;
  // Total workstreams/isolated operations across all partitions.
  uint32_t block_count;
  // Borrowed frontend payloads indexed by source ordinal.
  void** sources;
  // Topological source-node ordinals.
  uint32_t* order;
  // Packed node kind, barrier-before flag, and workstream by source ordinal.
  uint8_t* flags;
  // Recording boundaries, sized to the actual partition count.
  hrx_graph_partition_t* partitions;
} hrx_graph_schedule_t;

// Allocates node tables with zero incoming counts. The public frontend owns
// node-identity validation; this boundary checks representable allocation
// sizes.
iree_status_t hrx_graph_analysis_initialize(iree_host_size_t node_count,
                                            iree_arena_allocator_t* arena,
                                            hrx_graph_analysis_t* out_analysis);

// Prefixes incoming counts and allocates one exact-sized dependency array.
iree_status_t hrx_graph_analysis_allocate_dependencies(
    hrx_graph_analysis_t* analysis, iree_arena_allocator_t* arena);

// Orders, partitions, and plans barriers for trusted frontend-prepared edges.
// Only allocation failure and an input cycle are fallible. No payload,
// resource, or queue API is involved; all state is local to this compilation.
iree_status_t hrx_graph_analyze(hrx_graph_analysis_t* analysis,
                                iree_arena_allocator_t* arena,
                                hrx_graph_schedule_t* out_schedule);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_GRAPH_ANALYSIS_H_
