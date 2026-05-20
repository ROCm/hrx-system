// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA-specific one-sided operation tests.

#include <cstring>

#include "iree/async/slab.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

static constexpr iree_host_size_t kTransferLength = 2048;

struct RdmaRegisteredTestRegion {
  // Backend-created carrier-local registered region.
  RegisteredRegion registered_region = {};

  RdmaRegisteredTestRegion() = default;

  RdmaRegisteredTestRegion(const RdmaRegisteredTestRegion&) = delete;
  RdmaRegisteredTestRegion& operator=(const RdmaRegisteredTestRegion&) = delete;

  ~RdmaRegisteredTestRegion() { Reset(); }

  iree_async_region_t* region() const { return registered_region.region; }

  iree_async_span_t span(iree_host_size_t offset,
                         iree_host_size_t length) const {
    return iree_async_span_make(registered_region.region, offset, length);
  }

  uint8_t* data() const {
    return static_cast<uint8_t*>(
        iree_async_slab_base_ptr(registered_region.slab));
  }

  void Reset() {
    iree_async_region_release(registered_region.region);
    iree_async_slab_release(registered_region.slab);
    registered_region = {};
  }
};

void FillPattern(iree_async_span_t span, uint32_t seed) {
  uint8_t* data = iree_async_span_ptr(span);
  for (iree_host_size_t i = 0; i < span.length; ++i) {
    data[i] = (uint8_t)((seed + i * 17u) & 0xFFu);
  }
}

bool SpanEquals(iree_async_span_t lhs, iree_async_span_t rhs) {
  if (lhs.length != rhs.length) return false;
  return memcmp(iree_async_span_ptr(lhs), iree_async_span_ptr(rhs),
                lhs.length) == 0;
}

bool SpanHasByte(iree_async_span_t span, uint8_t value) {
  const uint8_t* data = iree_async_span_ptr(span);
  for (iree_host_size_t i = 0; i < span.length; ++i) {
    if (data[i] != value) return false;
  }
  return true;
}

class RdmaDirectTest : public CarrierTestBase<> {
 protected:
  void SetUp() override {
    CarrierTestBase::SetUp();
    backend_ = this->GetParam();
    if (!backend_.create_registered_region) {
      GTEST_SKIP() << "Backend lacks registered-region CTS support";
    }
  }

  void CreateRegisteredRegion(iree_net_carrier_t* carrier,
                              iree_host_size_t byte_length,
                              iree_async_buffer_access_flags_t access_flags,
                              RdmaRegisteredTestRegion* out_region) {
    IREE_ASSERT_OK(backend_.create_registered_region(
        carrier, byte_length, access_flags, iree_allocator_system(),
        &out_region->registered_region));
    ASSERT_NE(out_region->registered_region.slab, nullptr);
    ASSERT_NE(out_region->registered_region.region, nullptr);
  }

  BackendInfo backend_;
};

TEST_P(RdmaDirectTest, UnregisterInvalidatesRemoteHandle) {
  RdmaRegisteredTestRegion source;
  CreateRegisteredRegion(client_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &source);
  RdmaRegisteredTestRegion target;
  CreateRegisteredRegion(server_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
                         &target);

  iree_async_span_t source_span = source.span(0, kTransferLength);
  iree_async_span_t target_span = target.span(0, kTransferLength);
  FillPattern(source_span, 0x3Du);
  memset(target.data(), 0, kTransferLength);

  iree_net_remote_handle_t target_handle = iree_net_remote_handle_null();
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(server_, target.region(),
                                                  &target_handle));

  ActivateBothWithNullHandlers();
  int32_t client_pending_before = iree_net_carrier_pending_count(client_);

  iree_net_direct_write_params_t params = {};
  params.local = source_span;
  params.remote = target_handle;
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_NONE;
  IREE_ASSERT_OK(iree_net_carrier_direct_write(client_, &params));
  ASSERT_TRUE(PollUntil([&] {
    return SpanEquals(source_span, target_span) &&
           iree_net_carrier_pending_count(client_) == client_pending_before;
  }));

  iree_net_carrier_unregister_buffer(server_, target_handle);
  memset(target.data(), 0, kTransferLength);

  iree_status_t stale_write_status =
      iree_net_carrier_direct_write(client_, &params);
  if (iree_status_is_ok(stale_write_status)) {
    ASSERT_TRUE(PollUntil([&] {
      iree_net_carrier_send_budget_t budget =
          iree_net_carrier_query_direct_write_budget(
              client_, IREE_NET_DIRECT_WRITE_FLAG_NONE);
      return budget.slots == 0 &&
             iree_net_carrier_pending_count(client_) == client_pending_before;
    }));
  } else {
    IREE_EXPECT_NOT_OK(stale_write_status);
  }
  EXPECT_TRUE(SpanHasByte(target_span, 0));
}

CTS_REGISTER_TEST_SUITE(RdmaDirectTest);

}  // namespace
}  // namespace iree::net::carrier::cts
