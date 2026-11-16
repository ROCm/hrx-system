// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/callable_inputs.h"

#include <string.h>

typedef struct loom_callable_input_union_t {
  // Completed predecessor rows and direct requirements for this component.
  const uint64_t* words;
  // Current component's union row, also used for its representative node.
  uint64_t* result;
  // Number of membership words per row.
  iree_host_size_t word_count;
} loom_callable_input_union_t;

static void loom_callable_input_union_node(loom_callable_input_union_t* state,
                                           iree_host_size_t node) {
  const uint64_t* words = state->words + node * state->word_count;
  for (iree_host_size_t i = 0; i < state->word_count; ++i) {
    state->result[i] |= words[i];
  }
}

static iree_status_t loom_callable_input_union_successor(
    void* user_data, iree_host_size_t node) {
  loom_callable_input_union_node(user_data, node);
  return iree_ok_status();
}

iree_status_t loom_callable_inputs_compute(
    const loom_scc_graph_t* graph, const loom_scc_list_t* components,
    iree_host_size_t input_count,
    const loom_callable_input_requirement_t* requirements,
    iree_host_size_t requirement_count, iree_arena_allocator_t* arena,
    loom_callable_inputs_t* out_inputs) {
  *out_inputs = (loom_callable_inputs_t){0};
  if (input_count == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t word_count =
      input_count / 64 + (input_count % 64 != 0);
  uint64_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->node_count, word_count * sizeof(*words), (void**)&words));
  memset(words, 0, graph->node_count * word_count * sizeof(*words));
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    const loom_callable_input_requirement_t requirement = requirements[i];
    words[requirement.node * word_count + requirement.input / 64] |=
        UINT64_C(1) << (requirement.input % 64);
  }
  for (iree_host_size_t i = 0; i < components->count; ++i) {
    const loom_scc_t* component = &components->values[i];
    loom_callable_input_union_t state = {
        .words = words,
        .result = words + component->nodes[0] * word_count,
        .word_count = word_count,
    };
    // A cycle shares all members' local demands and all outgoing demands.
    // Successor components are already complete; internal edges add only
    // subsets of the same component's union.
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      const iree_host_size_t node = component->nodes[j];
      loom_callable_input_union_node(&state, node);
      IREE_RETURN_IF_ERROR(graph->visit_successors.fn(
          graph->visit_successors.user_data, node,
          loom_scc_successor_callback_make(loom_callable_input_union_successor,
                                           &state)));
    }
    for (iree_host_size_t j = 1; j < component->node_count; ++j) {
      memcpy(words + component->nodes[j] * word_count, state.result,
             word_count * sizeof(*words));
    }
  }
  *out_inputs = (loom_callable_inputs_t){
      .words = words,
      .words_per_node = word_count,
  };
  return iree_ok_status();
}
