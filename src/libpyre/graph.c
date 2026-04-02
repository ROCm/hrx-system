// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Graph stubs. Full implementation lives in the binding layer
// (hip_graph*.c) during the migration period. These stubs export the
// symbols so the DSO link succeeds.

#include "pyre_internal.h"

pyre_status_t pyre_graph_create(pyre_device_t device, uint32_t flags,
                                pyre_graph_t* out_graph) {
  (void)device;
  (void)flags;
  (void)out_graph;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_graph_create: use streaming layer");
}

void pyre_graph_retain(pyre_graph_t graph) { (void)graph; }
void pyre_graph_release(pyre_graph_t graph) { (void)graph; }

pyre_status_t pyre_graph_add_empty_node(pyre_graph_t graph,
                                        const pyre_graph_node_t* deps,
                                        size_t dep_count,
                                        pyre_graph_node_t* out_node) {
  (void)graph;
  (void)deps;
  (void)dep_count;
  (void)out_node;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_add_kernel_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_kernel_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  (void)graph;
  (void)deps;
  (void)dep_count;
  (void)attrs;
  (void)out_node;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_add_memcpy_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_memcpy_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  (void)graph;
  (void)deps;
  (void)dep_count;
  (void)attrs;
  (void)out_node;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_add_memset_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_memset_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  (void)graph;
  (void)deps;
  (void)dep_count;
  (void)attrs;
  (void)out_node;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_add_host_call_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_host_call_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  (void)graph;
  (void)deps;
  (void)dep_count;
  (void)attrs;
  (void)out_node;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_add_dependencies(pyre_graph_t graph,
                                          const pyre_graph_node_t* from_nodes,
                                          const pyre_graph_node_t* to_nodes,
                                          size_t count) {
  (void)graph;
  (void)from_nodes;
  (void)to_nodes;
  (void)count;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_size(pyre_graph_t graph, size_t* out_count) {
  (void)graph;
  (void)out_count;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_get_nodes(pyre_graph_t graph,
                                   pyre_graph_node_t* nodes,
                                   size_t* inout_count) {
  (void)graph;
  (void)nodes;
  (void)inout_count;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_instantiate(pyre_graph_t graph, uint32_t flags,
                                     pyre_graph_exec_t* out_exec) {
  (void)graph;
  (void)flags;
  (void)out_exec;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

void pyre_graph_exec_retain(pyre_graph_exec_t exec) { (void)exec; }
void pyre_graph_exec_release(pyre_graph_exec_t exec) { (void)exec; }

pyre_status_t pyre_graph_exec_launch(pyre_graph_exec_t exec,
                                     pyre_stream_t stream) {
  (void)exec;
  (void)stream;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_graph_exec_update(pyre_graph_exec_t exec,
                                     pyre_graph_t graph) {
  (void)exec;
  (void)graph;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_stream_begin_capture(pyre_stream_t stream,
                                        pyre_capture_mode_t mode) {
  (void)stream;
  (void)mode;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_status_t pyre_stream_end_capture(pyre_stream_t stream,
                                      pyre_graph_t* out_graph) {
  (void)stream;
  (void)out_graph;
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED, "graph stub");
}

pyre_capture_status_t pyre_stream_capture_status(pyre_stream_t stream) {
  (void)stream;
  return PYRE_CAPTURE_STATUS_NONE;
}

bool pyre_stream_is_capturing(pyre_stream_t stream) {
  (void)stream;
  return false;
}
