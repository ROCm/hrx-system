// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_AQL_PUBLICATION_H_
#define AMDF_CTS_GPU_AQL_PUBLICATION_H_

#include <cstring>

#include "libamdf/cts/gpu/aql/encoding/packets.h"
#include "libamdf/cts/gpu/util/user_queue.h"

namespace aql {

// The caller reserves a packet index before publishing. Read-index progress
// permits slot reuse, not signal or workload-memory reuse.
inline void Publish(const GpuUserQueue& queue, uint64_t index,
                    const Packet& packet) {
  const uint64_t capacity = queue.host.ring_byte_length / sizeof(Packet);
  while (index - GpuLoadAcquire<uint64_t>(queue.host.read_index_address) >=
         capacity) {
    std::this_thread::yield();
  }
  auto* slot = reinterpret_cast<uint32_t*>(queue.host.ring_address) +
               (index & (capacity - 1)) * 16;
  std::memcpy(slot + 1, packet.data() + 1, sizeof(Packet) - sizeof(uint32_t));
  GpuStoreRelease(reinterpret_cast<uintptr_t>(slot), packet[0]);
  GpuStoreRelease(queue.host.doorbell_address, index);
}

}  // namespace aql

#endif  // AMDF_CTS_GPU_AQL_PUBLICATION_H_
