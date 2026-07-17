// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for transport factory CTS tests.
//
// Provides the fixture and helpers shared across all transport factory tests.
// Parameterized on BackendInfo — the create_factory, make_bind_address,
// resolve_connect_address, and make_unreachable_address fields in BackendInfo
// drive transport-specific behavior.
//
// SetUp creates a proactor, slab/region/recv_pool (for transports that need
// registered buffers), and the transport factory. TearDown releases everything
// in reverse order.
//
// This base class is also usable by non-CTS tests that need a working factory
// for setup — any test that creates connections or carriers via the factory
// interface can derive from this and set up BackendInfo appropriately.

#ifndef IREE_NET_CARRIER_CTS_UTIL_FACTORY_TEST_BASE_H_
#define IREE_NET_CARRIER_CTS_UTIL_FACTORY_TEST_BASE_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"
#include "iree/net/transport_factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Callback tracking utilities
//===----------------------------------------------------------------------===//

// Tracks the result of an async connect callback.
struct ConnectResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_connection_t* connection = nullptr;

  static void Callback(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto* result = static_cast<ConnectResult*>(user_data);
    result->fired = true;
    result->status_code = iree_status_code(status);
    result->connection = connection;
    iree_status_free(status);
  }
};

// Tracks the result of an async open_endpoint callback.
struct EndpointReadyResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_message_endpoint_t endpoint = {nullptr, nullptr};

  static void Callback(void* user_data, iree_status_t status,
                       iree_net_message_endpoint_t endpoint) {
    auto* result = static_cast<EndpointReadyResult*>(user_data);
    result->fired = true;
    result->status_code = iree_status_code(status);
    result->endpoint = endpoint;
    iree_status_free(status);
  }
};

//===----------------------------------------------------------------------===//
// Factory test base fixture
//===----------------------------------------------------------------------===//

// Base class for all transport factory CTS tests. Parameterized on BackendInfo.
// The factory-level fields in GetParam() drive transport-specific behavior
// (factory creation, address generation, unreachable address creation).
class FactoryTestBase : public ::testing::TestWithParam<BackendInfo> {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));

    // Create slab/region/recv_pool. Transports that need registered buffers
    // (e.g., TCP with io_uring) use the pool; transports that don't (e.g.,
    // loopback) pass NULL and the pool is harmlessly unused.
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        region_, iree_allocator_system(), &recv_pool_));

    iree_status_t factory_status =
        GetParam().create_factory(iree_allocator_system(), &factory_);
    if (!iree_status_is_ok(factory_status) &&
        iree_status_code(factory_status) == IREE_STATUS_UNAVAILABLE) {
      iree::Status consumed_status(std::move(factory_status));
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' factory unavailable on this system: "
                   << consumed_status.ToString();
    }
    IREE_ASSERT_OK(factory_status);
  }

  void TearDown() override {
    iree_net_transport_factory_release(factory_);
    factory_ = nullptr;
    iree_async_buffer_pool_release(recv_pool_);
    recv_pool_ = nullptr;
    iree_async_region_release(region_);
    region_ = nullptr;
    iree_async_slab_release(slab_);
    slab_ = nullptr;
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Address delegates
  //===--------------------------------------------------------------------===//

  iree::StatusOr<std::string> MakeBindAddress() {
    return GetParam().make_bind_address();
  }

  iree::StatusOr<std::string> ResolveConnectAddress(
      const std::string& bind_address, iree_net_listener_t* listener) {
    return GetParam().resolve_connect_address(bind_address, listener);
  }

  iree::StatusOr<std::string> MakeUnreachableAddress() {
    return GetParam().make_unreachable_address(proactor_);
  }

  //===--------------------------------------------------------------------===//
  // Polling helpers
  //===--------------------------------------------------------------------===//

  // Polls the proactor until |condition| returns true. The outer test harness
  // owns hang detection for valid asynchronous work.
  bool PollUntil(std::function<bool()> condition) {
    while (!condition()) {
      iree_host_size_t completed = 0;
      iree::Status status(PollProactorOnce(proactor_, &completed));
      if (!status.ok()) {
        ADD_FAILURE() << status.ToString();
        return false;
      }
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Endpoint lifecycle helpers
  //===--------------------------------------------------------------------===//

  // Deactivates a message endpoint and polls until the callback fires.
  void DeactivateEndpointAndWait(iree_net_message_endpoint_t endpoint) {
    bool deactivated = false;
    iree_status_t status = iree_net_message_endpoint_deactivate(
        endpoint,
        [](void* user_data) { *static_cast<bool*>(user_data) = true; },
        &deactivated);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    if (!PollUntil([&]() { return deactivated; })) {
      iree_status_abort(iree_make_status(
          IREE_STATUS_INTERNAL,
          "proactor failed while joining endpoint deactivation"));
    }
  }

  //===--------------------------------------------------------------------===//
  // Listener lifecycle helpers
  //===--------------------------------------------------------------------===//

  // Stops a listener and polls until the stopped callback fires.
  iree_status_t StopAndWaitStatus(iree_net_listener_t* listener) {
    if (!listener) return iree_ok_status();
    bool stopped = false;
    iree_status_t status = iree_net_listener_stop(
        listener,
        {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
         &stopped});
    if (iree_status_is_ok(status)) {
      PollUntil([&]() { return stopped; });
    }
    return status;
  }

  void StopAndWait(iree_net_listener_t* listener) {
    IREE_ASSERT_OK(StopAndWaitStatus(listener));
  }

  //===--------------------------------------------------------------------===//
  // Connection establishment helpers
  //===--------------------------------------------------------------------===//

  // A connected client/server pair with the listener that produced them.
  struct ConnectPair {
    iree_net_connection_t* client = nullptr;
    iree_net_connection_t* server = nullptr;
    iree_net_listener_t* listener = nullptr;
    std::string connect_address;
  };

  // Creates a listener, connects to it, and returns the resulting pair.
  // Polls until both accept and connect callbacks fire.
  iree::StatusOr<ConnectPair> EstablishConnection() {
    ConnectPair pair;
    IREE_ASSIGN_OR_RETURN(std::string bind_str, MakeBindAddress());
    iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

    struct AcceptCtx {
      iree_net_connection_t* connection = nullptr;
      iree::Status status;
      bool fired = false;
    } accept_ctx;

    iree_status_t status = iree_net_transport_factory_create_listener(
        factory_, bind_addr, proactor_, recv_pool_,
        [](void* user_data, iree_status_t status,
           iree_net_connection_t* connection) {
          auto* ctx = static_cast<AcceptCtx*>(user_data);
          ctx->status = iree::Status(std::move(status));
          if (ctx->status.ok()) ctx->connection = connection;
          ctx->fired = true;
        },
        &accept_ctx, iree_allocator_system(), &pair.listener);

    if (iree_status_is_ok(status)) {
      iree::StatusOr<std::string> connect_address =
          ResolveConnectAddress(bind_str, pair.listener);
      if (connect_address.ok()) {
        pair.connect_address = std::move(connect_address).value();
      } else {
        status = std::move(connect_address).status();
      }
    }
    iree_string_view_t connect_addr =
        iree_make_cstring_view(pair.connect_address.c_str());

    struct ConnectCtx {
      iree_net_connection_t* connection = nullptr;
      iree::Status status;
      bool fired = false;
    } connect_ctx;

    if (iree_status_is_ok(status)) {
      status = iree_net_transport_factory_connect(
          factory_, connect_addr, proactor_, recv_pool_,
          [](void* user_data, iree_status_t status,
             iree_net_connection_t* connection) {
            auto* ctx = static_cast<ConnectCtx*>(user_data);
            ctx->status = iree::Status(std::move(status));
            if (ctx->status.ok()) ctx->connection = connection;
            ctx->fired = true;
          },
          &connect_ctx);
    }

    if (iree_status_is_ok(status)) {
      bool ok = PollUntil([&]() {
        return (accept_ctx.fired && !accept_ctx.status.ok()) ||
               (connect_ctx.fired && !connect_ctx.status.ok()) ||
               (accept_ctx.fired && connect_ctx.fired);
      });
      if (!ok) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "proactor polling failed during connection establishment");
      }
    }
    if (iree_status_is_ok(status) && !accept_ctx.status.ok()) {
      status = std::move(accept_ctx.status);
    }
    if (iree_status_is_ok(status) && !connect_ctx.status.ok()) {
      status = std::move(connect_ctx.status);
    }

    if (!iree_status_is_ok(status)) {
      iree_net_connection_release(connect_ctx.connection);
      iree_net_connection_release(accept_ctx.connection);
      if (pair.listener) {
        status = iree_status_join(status, StopAndWaitStatus(pair.listener));
        iree_net_listener_free(pair.listener);
        pair.listener = nullptr;
      }
      return iree::Status(std::move(status));
    }

    pair.client = connect_ctx.connection;
    pair.server = accept_ctx.connection;
    return pair;
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
};

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_FACTORY_TEST_BASE_H_
