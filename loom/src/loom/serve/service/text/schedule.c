// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/schedule.h"

iree_host_size_t loom_serve_text_request_reservation(
    iree_host_size_t context_capacity, iree_host_size_t input_count,
    iree_host_size_t output_count, iree_host_size_t proposal_depth,
    iree_host_size_t block_size) {
  iree_host_size_t extent = input_count + output_count - 1;
  if (proposal_depth && output_count >= 3 &&
      context_capacity - input_count >= proposal_depth + 1) {
    // The last two-output verifier writes one anchor and a full draft, while
    // only two inputs can commit. Near the context limit only legal launches
    // contribute; an ordinary single-output tail requires no extra credit.
    extent = iree_min(context_capacity, extent + proposal_depth - 1);
  }
  return iree_host_size_ceil_div(extent, block_size) * block_size;
}
