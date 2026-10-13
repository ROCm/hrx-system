// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only bridge to the production packer. Each input is a complete readiness
// table and shape catalog; no allocation or selection policy is duplicated
// here.

#include <iostream>
#include <vector>

#include "loom/serve/service/text/packing.h"

int main() {
  iree_host_size_t row_count = 0;
  while (std::cin >> row_count) {
    iree_host_size_t shape_count = 0;
    iree_host_size_t chunk_size = 0;
    iree_host_size_t cursor = 0;
    if (!(std::cin >> shape_count >> chunk_size >> cursor) || !row_count ||
        !shape_count || !chunk_size || cursor >= row_count) {
      return 1;
    }
    std::vector<loom_serve_ready_span_t> ready(row_count);
    for (auto& row : ready) {
      if (!(std::cin >> row.token_count >> row.minimum_count) ||
          (row.token_count &&
           (!row.minimum_count || row.minimum_count > row.token_count))) {
        return 1;
      }
    }
    std::vector<loom_serve_packing_shape_t> shapes(shape_count);
    for (auto& shape : shapes) {
      if (!(std::cin >> shape.token_capacity >> shape.span_capacity) ||
          !shape.token_capacity || !shape.span_capacity) {
        return 1;
      }
    }
    std::vector<loom_serve_packed_span_t> spans(row_count);
    std::vector<loom_serve_packed_span_t> scratch(row_count);
    iree_host_size_t shape = 0;
    const iree_host_size_t count = loom_serve_pack_shapes(
        row_count, ready.data(), shape_count, shapes.data(), chunk_size,
        &cursor, spans.data(), scratch.data(), &shape);
    std::cout << shape << ' ' << cursor << ' ' << count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      std::cout << ' ' << spans[i].row_index << ' ' << spans[i].token_count;
    }
    std::cout << '\n';
  }
  std::cout.flush();
  return std::cin.eof() && std::cout.good() ? 0 : 1;
}
