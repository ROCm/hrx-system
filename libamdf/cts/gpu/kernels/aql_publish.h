// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_AQL_PUBLISH_H_
#define AMDF_CTS_GPU_KERNELS_AQL_PUBLISH_H_

#include <cstddef>
#include <cstdint>

namespace kernels::aql_publish {

struct Arguments {
  // GPU address of one reserved, available INVALID native packet slot.
  uint64_t slot;
  // GPU address of the immutable 64-byte dispatch template.
  uint64_t packet_template;
  // GPU-produced X/Y/Z workgroup counts.
  uint64_t counts;
  // Producer-local address of the consumer queue's reservation index.
  uint64_t write_index;
  // Producer-local address of the consumer queue's write-only doorbell.
  uint64_t notification;
  // Monotonic packet index associated with slot.
  uint64_t packet_index;
  // Nonzero transfers reservation and notification to this dispatch.
  uint32_t notify;
};
static_assert(offsetof(Arguments, notify) == 48);
static_assert(sizeof(Arguments) == 56);

}  // namespace kernels::aql_publish

#endif  // AMDF_CTS_GPU_KERNELS_AQL_PUBLISH_H_
