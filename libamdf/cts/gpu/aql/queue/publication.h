// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_AQL_QUEUE_PUBLICATION_H_
#define AMDF_CTS_GPU_AQL_QUEUE_PUBLICATION_H_

#include <cstring>
#include <span>

#include "gtest/gtest.h"
#include "libamdf/cts/gpu/aql/encoding/packets.h"
#include "libamdf/cts/gpu/util/user_queue.h"

namespace aql {

// The caller reserves a packet index before publishing. Read-index progress
// permits slot reuse, not signal or workload-memory reuse.
[[nodiscard]] inline ::testing::AssertionResult Publish(
    const GpuUserQueue& queue, uint64_t index, const Packet& packet) {
  const uint64_t capacity = queue.host.ring_byte_length / sizeof(Packet);
  uint64_t consumed = 0;
  const auto result = amdf::cts::WaitUntil("AQL ring capacity", [&] {
    consumed = GpuLoadAcquire<uint64_t>(queue.host.read_index_address);
    return index - consumed < capacity;
  });
  if (!result) {
    return ::testing::AssertionFailure()
           << result.message() << ": packet " << index << ", consumed "
           << consumed << ", capacity " << capacity;
  }
  auto* slot = reinterpret_cast<uint32_t*>(queue.host.ring_address) +
               (index & (capacity - 1)) * 16;
  std::memcpy(slot + 1, packet.data() + 1, sizeof(Packet) - sizeof(uint32_t));
  const amdf_status_t status = queue.Flush();
  if (!amdf_status_is_ok(status)) {
    return ::testing::AssertionFailure()
           << "native host publication: " << status;
  }
  GpuStoreRelease(reinterpret_cast<uintptr_t>(slot), packet[0]);
  GpuRingDoorbell(queue.host.doorbell_address, index);
  return ::testing::AssertionSuccess();
}

// Prepare a complete finite batch while its first header remains INVALID.
// The caller publishes that header only after every dependent batch is ready.
inline void PrepareBatch(const GpuUserQueue& queue, uint64_t begin,
                         std::span<const Packet> packets) {
  const uint64_t capacity = queue.host.ring_byte_length / sizeof(Packet);
  ASSERT_LT(packets.size(), capacity);
  ASSERT_EQ(GpuLoadAcquire<uint64_t>(queue.host.read_index_address), begin);
  auto* ring = reinterpret_cast<Packet*>(queue.host.ring_address);
  auto& first = ring[begin & (capacity - 1)];
  std::memcpy(first.data() + 1, packets.front().data() + 1,
              sizeof(Packet) - sizeof(uint32_t));
  for (size_t i = 1; i < packets.size(); ++i) {
    ring[(begin + i) & (capacity - 1)] = packets[i];
  }
}

// Publishes the same nonempty batch prepared above, reserving its whole extent
// before making its first header valid. No packet may be edited afterward.
inline void PublishBatch(const GpuUserQueue& queue, uint64_t begin,
                         std::span<const Packet> packets) {
  const uint64_t capacity = queue.host.ring_byte_length / sizeof(Packet);
  auto* ring = reinterpret_cast<Packet*>(queue.host.ring_address);
  GpuStoreRelease(queue.host.write_index_address, begin + packets.size());
  ASSERT_EQ(queue.Flush(), AMDF_STATUS_OK);
  GpuStoreRelease(
      reinterpret_cast<uintptr_t>(ring[begin & (capacity - 1)].data()),
      packets.front()[0]);
  GpuRingDoorbell(queue.host.doorbell_address, begin + packets.size() - 1);
}

}  // namespace aql

#endif  // AMDF_CTS_GPU_AQL_QUEUE_PUBLICATION_H_
