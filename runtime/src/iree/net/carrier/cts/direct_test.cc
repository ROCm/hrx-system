// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One-sided carrier operation tests.
//
// These tests exercise the generic direct_write/direct_read/register_buffer
// carrier APIs using backend-provided registered regions. RDMA is the primary
// target, but the tests intentionally avoid RDMA-specific headers so other
// one-sided carriers can participate by providing the same CTS hook.

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

#include "iree/async/slab.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

static constexpr iree_host_size_t kTransferLength = 2048;
static constexpr iree_host_size_t kGuardLength = 64;
static constexpr uint32_t kImmediateValue = 0xCAFE1023u;

struct RegisteredTestRegion {
  // Backend-created carrier-local registered region.
  RegisteredRegion registered_region = {};

  RegisteredTestRegion() = default;

  RegisteredTestRegion(const RegisteredTestRegion&) = delete;
  RegisteredTestRegion& operator=(const RegisteredTestRegion&) = delete;

  ~RegisteredTestRegion() { Reset(); }

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

struct DirectSignalCapture {
  // Number of signal callbacks observed.
  std::atomic<int> call_count{0};

  // Last immediate value delivered by the carrier.
  std::atomic<uint32_t> last_immediate{0};

  static iree_status_t Handler(void* user_data, uint32_t immediate) {
    auto* capture = static_cast<DirectSignalCapture*>(user_data);
    capture->last_immediate.store(immediate, std::memory_order_relaxed);
    capture->call_count.fetch_add(1, std::memory_order_release);
    return iree_ok_status();
  }

  iree_net_carrier_signal_handler_t AsHandler() { return {Handler, this}; }
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

class DirectTestBase : public CarrierTestBase<> {
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
                              RegisteredTestRegion* out_region) {
    IREE_ASSERT_OK(backend_.create_registered_region(
        carrier, byte_length, access_flags, iree_allocator_system(),
        &out_region->registered_region));
    ASSERT_NE(out_region->registered_region.slab, nullptr);
    ASSERT_NE(out_region->registered_region.region, nullptr);
  }

  BackendInfo backend_;
};

class DirectWriteTest : public DirectTestBase {};
class DirectReadTest : public DirectTestBase {};

TEST_P(DirectWriteTest, BufferRegistration) {
  RegisteredTestRegion target;
  CreateRegisteredRegion(server_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
                         &target);

  iree_net_remote_handle_t handle = iree_net_remote_handle_null();
  IREE_ASSERT_OK(
      iree_net_carrier_register_buffer(server_, target.region(), &handle));
  EXPECT_FALSE(iree_net_remote_handle_is_null(handle));

  iree_net_remote_handle_t offset_handle =
      iree_net_remote_handle_offset(handle, kGuardLength);
  EXPECT_EQ(offset_handle.opaque[0], handle.opaque[0]);
  EXPECT_EQ(offset_handle.opaque[1], handle.opaque[1] + kGuardLength);

  iree_net_carrier_unregister_buffer(server_, handle);
}

TEST_P(DirectWriteTest, RegisterBufferRejectsMissingRemoteAccess) {
  RegisteredTestRegion target;
  CreateRegisteredRegion(
      server_, kTransferLength,
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
      &target);

  iree_net_remote_handle_t handle = iree_net_remote_handle_null();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_net_carrier_register_buffer(server_, target.region(), &handle));
  EXPECT_TRUE(iree_net_remote_handle_is_null(handle));
}

TEST_P(DirectWriteTest, DirectWriteBudgetAfterActivate) {
  ActivateBothWithNullHandlers();

  iree_net_carrier_send_budget_t plain_budget =
      iree_net_carrier_query_direct_write_budget(
          client_, IREE_NET_DIRECT_WRITE_FLAG_NONE);
  EXPECT_GT(plain_budget.bytes, 0u);
  EXPECT_GT(plain_budget.slots, 0u);

  iree_net_carrier_send_budget_t signal_budget =
      iree_net_carrier_query_direct_write_budget(
          client_, IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER);
  EXPECT_GT(signal_budget.bytes, 0u);
  EXPECT_GT(signal_budget.slots, 0u);
}

TEST_P(DirectWriteTest, DirectWriteToRemote) {
  RegisteredTestRegion source;
  CreateRegisteredRegion(client_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &source);
  RegisteredTestRegion target;
  CreateRegisteredRegion(server_, kGuardLength + kTransferLength + kGuardLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
                         &target);

  iree_async_span_t source_span = source.span(0, kTransferLength);
  iree_async_span_t target_prefix = target.span(0, kGuardLength);
  iree_async_span_t target_span = target.span(kGuardLength, kTransferLength);
  iree_async_span_t target_suffix =
      target.span(kGuardLength + kTransferLength, kGuardLength);
  FillPattern(source_span, 0x31u);
  memset(target.data(), 0, kGuardLength + kTransferLength + kGuardLength);

  iree_net_remote_handle_t target_handle = iree_net_remote_handle_null();
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(server_, target.region(),
                                                  &target_handle));

  ActivateBothWithNullHandlers();
  int32_t client_pending_before = iree_net_carrier_pending_count(client_);

  iree_net_direct_write_params_t params = {};
  params.local = source_span;
  params.remote = iree_net_remote_handle_offset(target_handle, kGuardLength);
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_NONE;
  IREE_ASSERT_OK(iree_net_carrier_direct_write(client_, &params));

  ASSERT_TRUE(PollUntil([&] {
    return SpanEquals(source_span, target_span) &&
           iree_net_carrier_pending_count(client_) == client_pending_before;
  }));
  EXPECT_TRUE(SpanHasByte(target_prefix, 0));
  EXPECT_TRUE(SpanHasByte(target_suffix, 0));

  iree_net_carrier_unregister_buffer(server_, target_handle);
}

TEST_P(DirectWriteTest, DirectWriteWithImmediate) {
  RegisteredTestRegion source;
  CreateRegisteredRegion(client_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &source);
  RegisteredTestRegion target;
  CreateRegisteredRegion(server_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
                         &target);

  iree_async_span_t source_span = source.span(0, kTransferLength);
  iree_async_span_t target_span = target.span(0, kTransferLength);
  FillPattern(source_span, 0x49u);
  memset(target.data(), 0, kTransferLength);

  iree_net_remote_handle_t target_handle = iree_net_remote_handle_null();
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(server_, target.region(),
                                                  &target_handle));

  DirectSignalCapture signal_capture;
  iree_net_carrier_set_signal_handler(server_, signal_capture.AsHandler());
  ActivateBothWithNullHandlers();
  int32_t client_pending_before = iree_net_carrier_pending_count(client_);

  iree_net_direct_write_params_t params = {};
  params.local = source_span;
  params.remote = target_handle;
  params.flags = IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  params.immediate = kImmediateValue;
  IREE_ASSERT_OK(iree_net_carrier_direct_write(client_, &params));

  ASSERT_TRUE(PollUntil([&] {
    return signal_capture.call_count.load(std::memory_order_acquire) >= 1 &&
           SpanEquals(source_span, target_span) &&
           iree_net_carrier_pending_count(client_) == client_pending_before;
  }));
  EXPECT_EQ(signal_capture.last_immediate.load(std::memory_order_relaxed),
            kImmediateValue);

  iree_net_carrier_unregister_buffer(server_, target_handle);
}

TEST_P(DirectReadTest, DirectReadFromRemote) {
  RegisteredTestRegion source;
  CreateRegisteredRegion(server_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ,
                         &source);
  RegisteredTestRegion target;
  CreateRegisteredRegion(client_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &target);

  iree_async_span_t source_span = source.span(0, kTransferLength);
  iree_async_span_t target_span = target.span(0, kTransferLength);
  FillPattern(source_span, 0x77u);
  memset(target.data(), 0, kTransferLength);

  iree_net_remote_handle_t source_handle = iree_net_remote_handle_null();
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(server_, source.region(),
                                                  &source_handle));

  ActivateBothWithNullHandlers();
  int32_t client_pending_before = iree_net_carrier_pending_count(client_);

  iree_net_direct_read_params_t params = {};
  params.local = target_span;
  params.remote = source_handle;
  IREE_ASSERT_OK(iree_net_carrier_direct_read(client_, &params));

  ASSERT_TRUE(PollUntil([&] {
    return SpanEquals(source_span, target_span) &&
           iree_net_carrier_pending_count(client_) == client_pending_before;
  }));

  iree_net_carrier_unregister_buffer(server_, source_handle);
}

TEST_P(DirectReadTest, DirectReadRequiresWritableLocalRegion) {
  RegisteredTestRegion source;
  CreateRegisteredRegion(server_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
                             IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ,
                         &source);
  RegisteredTestRegion target;
  CreateRegisteredRegion(client_, kTransferLength,
                         IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &target);

  iree_net_remote_handle_t source_handle = iree_net_remote_handle_null();
  IREE_ASSERT_OK(iree_net_carrier_register_buffer(server_, source.region(),
                                                  &source_handle));

  ActivateBothWithNullHandlers();

  iree_net_direct_read_params_t params = {};
  params.local = target.span(0, kTransferLength);
  params.remote = source_handle;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_net_carrier_direct_read(client_, &params));

  iree_net_carrier_unregister_buffer(server_, source_handle);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(DirectWriteTest,
                                  (std::vector<std::string>{
                                      "registered_regions", "direct_write"}),
                                  {});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(DirectReadTest,
                                  (std::vector<std::string>{
                                      "registered_regions", "direct_read"}),
                                  {});
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DirectWriteTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DirectReadTest);

}  // namespace
}  // namespace iree::net::carrier::cts
