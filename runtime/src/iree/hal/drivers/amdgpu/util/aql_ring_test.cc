// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/aql_ring.h"

#include <cstdint>
#include <cstring>

#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Fills the queue fields aql_ring_initialize reads: the packet ring base/size
// and the doorbell signal handle. The dispatch-id fields are read by address
// and left zero.
static void ConfigureQueue(iree_hal_amdgpu_aql_packet_t* base, uint32_t size,
                           const iree_amd_signal_t* doorbell_signal,
                           iree_amd_queue_t* out_queue) {
  out_queue->hsa_queue.base_address = base;
  out_queue->hsa_queue.size = size;
  out_queue->hsa_queue.doorbell_signal.handle = (uint64_t)doorbell_signal;
}

TEST(AqlRingTest, CommitsExtendedDispatchFormatAndSetupAtomically) {
  iree_hal_amdgpu_aql_packet_t packet;
  std::memset(&packet, 0xCC, sizeof(packet));

  iree_hal_amdgpu_aql_dispatch_params_t params = {};
  params.kernel_object = 0x1234;
  params.workgroup_size[0] = 64;
  params.workgroup_size[1] = 1;
  params.workgroup_size[2] = 1;
  params.workgroup_count[0] = 2;
  params.workgroup_count[1] = 1;
  params.workgroup_count[2] = 1;
  params.workgroup_cluster_size[0] = 2;
  params.workgroup_cluster_size[1] = 1;
  params.workgroup_cluster_size[2] = 1;
  params.packet_control = iree_hal_amdgpu_aql_packet_control_barrier_system();

  uint16_t header = 0;
  uint16_t setup = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_emit_extended_dispatch(
      &packet.extended_dispatch, &params, &header, &setup));

  uint32_t first_dword = 0;
  std::memcpy(&first_dword, &packet, sizeof(first_dword));
  EXPECT_EQ(first_dword, 0xCCCCCCCCu);

  iree_hal_amdgpu_aql_ring_commit(&packet, header, setup);

  std::memcpy(&first_dword, &packet, sizeof(first_dword));
  EXPECT_EQ(first_dword, static_cast<uint32_t>(header) |
                             (static_cast<uint32_t>(setup) << 16));
  EXPECT_EQ(packet.extended_dispatch.header, header);
  EXPECT_EQ(packet.extended_dispatch.amd_format,
            IREE_HSA_AMD_AQL_FORMAT_EXT_KERNEL_DISPATCH);
  EXPECT_EQ(packet.extended_dispatch.setup, 3u);
}

// A DOORBELL-kind signal uses the platform's correct notification path while
// initialization records the remaining hot pointers.
TEST(AqlRingTest, InitializeResolvesPathForDoorbellKind) {
  volatile int64_t doorbell_mmio = 0;
  iree_amd_signal_t signal = {};
  signal.kind = IREE_AMD_SIGNAL_KIND_DOORBELL;
  signal.hardware_doorbell_ptr = (volatile uint64_t*)&doorbell_mmio;

  iree_hal_amdgpu_aql_packet_t packets[4] = {};
  iree_amd_queue_t queue = {};
  ConfigureQueue(packets, IREE_ARRAYSIZE(packets), &signal, &queue);

  iree_hal_amdgpu_libhsa_t libhsa = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  iree_hal_amdgpu_aql_ring_initialize(&libhsa, &queue, &ring);

// Windows ROCr must notify its DXG queue worker through the HSA signal API;
// other platforms can write a DOORBELL-kind signal's MMIO pointer directly.
#if defined(IREE_PLATFORM_WINDOWS)
  EXPECT_EQ((void*)ring.doorbell.ptr, nullptr);
#else
  EXPECT_EQ((void*)ring.doorbell.ptr, (void*)&doorbell_mmio);
#endif  // IREE_PLATFORM_WINDOWS
  // The signal handle and libhsa are cached regardless of kind.
  EXPECT_EQ(ring.doorbell.signal.handle, (uint64_t)&signal);
  EXPECT_EQ(ring.libhsa, &libhsa);
  // Remaining hot pointers come straight from the queue descriptor.
  EXPECT_EQ(ring.base, packets);
  EXPECT_EQ(ring.mask, IREE_ARRAYSIZE(packets) - 1);
  EXPECT_EQ((void*)ring.write_dispatch_id, (void*)&queue.write_dispatch_id);
  EXPECT_EQ((void*)ring.read_dispatch_id, (void*)&queue.read_dispatch_id);
}

// A USER-kind signal stores a value (not a pointer) in the same union slot, so
// the fast-path pointer must stay NULL even when that value is non-zero.
TEST(AqlRingTest, InitializeLeavesPointerNullForUserKind) {
  iree_amd_signal_t signal = {};
  signal.kind = IREE_AMD_SIGNAL_KIND_USER;
  signal.value = 0x1234;  // a blind pointer read would yield a non-NULL ptr

  iree_hal_amdgpu_aql_packet_t packets[4] = {};
  iree_amd_queue_t queue = {};
  ConfigureQueue(packets, IREE_ARRAYSIZE(packets), &signal, &queue);

  iree_hal_amdgpu_libhsa_t libhsa = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  iree_hal_amdgpu_aql_ring_initialize(&libhsa, &queue, &ring);

  EXPECT_EQ((void*)ring.doorbell.ptr, nullptr);
  EXPECT_EQ(ring.doorbell.signal.handle, (uint64_t)&signal);
  EXPECT_EQ(ring.libhsa, &libhsa);
}

// A null doorbell signal handle must not be dereferenced; the pointer stays
// NULL and the (null) handle is still cached for the fallback path.
TEST(AqlRingTest, InitializeLeavesPointerNullForNullSignal) {
  iree_hal_amdgpu_aql_packet_t packets[4] = {};
  iree_amd_queue_t queue = {};
  ConfigureQueue(packets, IREE_ARRAYSIZE(packets), nullptr, &queue);

  iree_hal_amdgpu_libhsa_t libhsa = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  iree_hal_amdgpu_aql_ring_initialize(&libhsa, &queue, &ring);

  EXPECT_EQ((void*)ring.doorbell.ptr, nullptr);
  EXPECT_EQ(ring.doorbell.signal.handle, 0u);
}

// The fallback is observed via a recording thunk in the libhsa function-pointer
// table, which only exists in the default dynamic build; these two tests are
// therefore dynamic-only.
#if !IREE_HAL_AMDGPU_LIBHSA_STATIC

// Captures the arguments of the most recent hsa_signal_store_screlease call.
struct RecordedSignalStore {
  hsa_signal_t signal;
  hsa_signal_value_t value;
  int count;
};
RecordedSignalStore g_recorded_signal_store;

static void RecordSignalStore(hsa_signal_t signal, hsa_signal_value_t value) {
  g_recorded_signal_store.signal = signal;
  g_recorded_signal_store.value = value;
  ++g_recorded_signal_store.count;
}

// A libhsa whose only populated thunk is the signal store used by the doorbell
// fallback; every other entry stays NULL.
static iree_hal_amdgpu_libhsa_t MakeRecordingLibhsa() {
  iree_hal_amdgpu_libhsa_t libhsa = {};
  libhsa.hsa_signal_store_screlease = RecordSignalStore;
  return libhsa;
}

// With a resolved MMIO pointer the doorbell is a direct atomic store and never
// routes through libhsa.
TEST(AqlRingTest, DoorbellFastPathWritesMmioRegister) {
  volatile int64_t doorbell_mmio = 0;
  iree_hal_amdgpu_libhsa_t libhsa = MakeRecordingLibhsa();
  g_recorded_signal_store = {};

  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.libhsa = &libhsa;
  ring.doorbell.ptr = &doorbell_mmio;

  iree_hal_amdgpu_aql_ring_doorbell(&ring, 0x1234);

  EXPECT_EQ(doorbell_mmio, int64_t{0x1234});
  EXPECT_EQ(g_recorded_signal_store.count, 0);
}

// Without an MMIO pointer the doorbell falls back to the HSA signal store,
// passing the cached signal handle and the packet ID as the value.
TEST(AqlRingTest, DoorbellFallbackStoresSignalViaLibhsa) {
  iree_hal_amdgpu_libhsa_t libhsa = MakeRecordingLibhsa();
  g_recorded_signal_store = {};

  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.libhsa = &libhsa;
  ring.doorbell.ptr = nullptr;  // forces the fallback path
  ring.doorbell.signal.handle = 0xABCD;

  iree_hal_amdgpu_aql_ring_doorbell(&ring, 0x5678);

  EXPECT_EQ(g_recorded_signal_store.count, 1);
  EXPECT_EQ(g_recorded_signal_store.signal.handle, 0xABCDu);
  EXPECT_EQ(g_recorded_signal_store.value, int64_t{0x5678});
}

#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC

}  // namespace
