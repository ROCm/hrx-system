// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/input_ring.h"

iree_host_size_t iree_benchmark_loom_input_ring_select_count(
    iree_host_size_t requested_count, uint64_t requested_min_bytes,
    uint64_t binding_set_bytes, iree_host_size_t dispatches_per_batch) {
  if (requested_count != 0) {
    return requested_count;
  }
  if (requested_min_bytes == 0 || binding_set_bytes == 0) {
    return 1;
  }
  const uint64_t byte_sized_count =
      requested_min_bytes / binding_set_bytes +
      (requested_min_bytes % binding_set_bytes == 0 ? 0 : 1);
  return (iree_host_size_t)iree_max(
      UINT64_C(1), iree_min(byte_sized_count, (uint64_t)dispatches_per_batch));
}
