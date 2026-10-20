// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"

namespace {

// CPU component cost only. Native mapped-memory and MMIO costs require the
// complete queue benchmark on hardware; this row claims neither.
void BM_HostMemoryPublication(benchmark::State& state) {
  std::array<iree_hal_amd_gpu_aql_packet_t, 1024> packets = {};
  iree_atomic_int64_t read_index = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_int64_t write_index = IREE_ATOMIC_VAR_INIT(0);
  uint64_t doorbell = 0;
  iree_hal_amd_gpu_aql_ring_t ring;
  iree_hal_amd_gpu_aql_ring_initialize(packets.data(), 0x1000, packets.size(),
                                       &read_index, &write_index, &doorbell,
                                       &ring);
  const auto packet = iree_hal_amd_gpu_aql_packet_barrier(
      IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND, 0, nullptr, 0x2000);
  const uint32_t count = state.range(0);
  for (auto _ : state) {
    uint64_t position;
    if (!iree_hal_amd_gpu_aql_ring_try_reserve(&ring, count, &position)) {
      state.SkipWithError("joined packet storage was not reusable");
      break;
    }
    for (uint32_t i = 0; i < count; ++i) {
      iree_hal_amd_gpu_aql_ring_write_body(&ring, position + i, &packet);
      iree_hal_amd_gpu_aql_ring_publish_header(&ring, position + i,
                                               packet.words[0]);
    }
    iree_hal_amd_gpu_aql_ring_notify(&ring);
    benchmark::ClobberMemory();
    iree_atomic_store(&read_index, ring.reserved, iree_memory_order_release);
  }
  state.SetItemsProcessed(state.iterations() * count);
  state.counters["publisher_bytes"] = sizeof(ring);
}
BENCHMARK(BM_HostMemoryPublication)->Arg(1)->Arg(16)->Arg(64);

}  // namespace
