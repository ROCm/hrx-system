// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits.h>

#include "loom/target/reporting/report.h"
#include "loom/target/reporting/row_list.h"

static int loom_target_compile_report_call_edges_compare(
    const loom_target_compile_report_call_edge_t* lhs,
    const loom_target_compile_report_call_edge_t* rhs) {
  int order = 0;
  if (lhs->function.module_id != rhs->function.module_id) {
    order = lhs->function.module_id < rhs->function.module_id ? -1 : 1;
  }
  if (order == 0 && lhs->function.symbol_id != rhs->function.symbol_id) {
    order = lhs->function.symbol_id < rhs->function.symbol_id ? -1 : 1;
  }
  if (order == 0 && lhs->callee.module_id != rhs->callee.module_id) {
    order = lhs->callee.module_id < rhs->callee.module_id ? -1 : 1;
  }
  if (order == 0 && lhs->callee.symbol_id != rhs->callee.symbol_id) {
    order = lhs->callee.symbol_id < rhs->callee.symbol_id ? -1 : 1;
  }
  return order;
}

static uint8_t loom_target_compile_report_call_edge_height(
    const loom_target_compile_report_call_edge_t* edge) {
  return edge != NULL ? edge->index.height : 0;
}

static void loom_target_compile_report_call_edge_update_height(
    loom_target_compile_report_call_edge_t* edge) {
  edge->index.height =
      1 +
      iree_max(
          loom_target_compile_report_call_edge_height(edge->index.children[0]),
          loom_target_compile_report_call_edge_height(edge->index.children[1]));
}

static loom_target_compile_report_call_edge_t*
loom_target_compile_report_call_edge_rotate(
    loom_target_compile_report_call_edge_t* edge, uint32_t direction) {
  loom_target_compile_report_call_edge_t* pivot =
      edge->index.children[direction];
  edge->index.children[direction] = pivot->index.children[1 - direction];
  pivot->index.children[1 - direction] = edge;
  loom_target_compile_report_call_edge_update_height(edge);
  loom_target_compile_report_call_edge_update_height(pivot);
  return pivot;
}

static loom_target_compile_report_call_edge_t*
loom_target_compile_report_call_edge_balance(
    loom_target_compile_report_call_edge_t* edge) {
  loom_target_compile_report_call_edge_update_height(edge);
  const int balance =
      (int)loom_target_compile_report_call_edge_height(
          edge->index.children[0]) -
      (int)loom_target_compile_report_call_edge_height(edge->index.children[1]);
  if (balance < -1 || balance > 1) {
    const uint32_t direction = balance < 0;
    loom_target_compile_report_call_edge_t* child =
        edge->index.children[direction];
    if (loom_target_compile_report_call_edge_height(
            child->index.children[direction]) <
        loom_target_compile_report_call_edge_height(
            child->index.children[1 - direction])) {
      edge->index.children[direction] =
          loom_target_compile_report_call_edge_rotate(child, 1 - direction);
    }
    edge = loom_target_compile_report_call_edge_rotate(edge, direction);
  }
  return edge;
}

iree_status_t loom_target_compile_report_record_call_edge(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_call_edge_t* edge) {
  if (iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  loom_target_compile_report_call_edge_t**
      path[2 * sizeof(iree_host_size_t) * CHAR_BIT];
  iree_host_size_t depth = 0;
  loom_target_compile_report_call_edge_t** link = &report->call_edge_root;
  while (*link != NULL) {
    const int order =
        loom_target_compile_report_call_edges_compare(edge, *link);
    if (order == 0) {
      break;
    }
    path[depth++] = link;
    link = &(*link)->index.children[order > 0];
  }
  loom_target_compile_report_call_edge_t* summary = *link;
  if (summary != NULL) {
    summary->callsite_count += edge->callsite_count;
    if (iree_all_bits_set(summary->flags,
                          LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT) &&
        iree_all_bits_set(edge->flags,
                          LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT)) {
      if (!iree_checked_add_u64(summary->dynamic_call_count,
                                edge->dynamic_call_count,
                                &summary->dynamic_call_count)) {
        summary->flags &= ~LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT;
        summary->dynamic_call_count = 0;
      }
    } else {
      summary->flags &= ~LOOM_TARGET_COMPILE_REPORT_CALL_EDGE_DYNAMIC_COUNT;
      summary->dynamic_call_count = 0;
    }
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
      &report->call_edge_rows, sizeof(*edge), report->allocator, edge));
  loom_target_compile_report_vec_t* tail = report->call_edge_rows.tail;
  summary = (loom_target_compile_report_call_edge_t*)
                loom_target_compile_report_vec_rows(tail) +
            (tail->count - 1);
  summary->index.children[0] = NULL;
  summary->index.children[1] = NULL;
  summary->index.height = 1;
  *link = summary;
  while (depth != 0) {
    --depth;
    link = path[depth];
    const uint8_t old_height = (*link)->index.height;
    *link = loom_target_compile_report_call_edge_balance(*link);
    if ((*link)->index.height == old_height) {
      break;
    }
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_CALL_EDGES;
  return iree_ok_status();
}

const loom_target_compile_report_call_edge_t*
loom_target_compile_report_find_call_edge(
    const loom_target_compile_report_t* report, loom_symbol_ref_t function,
    loom_symbol_ref_t callee) {
  const loom_target_compile_report_call_edge_t key = {
      .function = function,
      .callee = callee,
  };
  const loom_target_compile_report_call_edge_t* edge = report->call_edge_root;
  while (edge != NULL) {
    const int order = loom_target_compile_report_call_edges_compare(&key, edge);
    if (order == 0) {
      return edge;
    }
    edge = edge->index.children[order > 0];
  }
  return NULL;
}
