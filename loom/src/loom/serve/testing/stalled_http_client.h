// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_STALLED_HTTP_CLIENT_H_
#define LOOM_SERVE_TESTING_STALLED_HTTP_CLIENT_H_

#include <string>
#include <thread>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/loopback/carrier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/transport/http/server.h"

namespace loom_serve::testing {

// A real carrier with one response credit. The caller controls only this
// client's progress; HTTP and service work use the server's normal workers.
class StalledHttpClient {
 public:
  ~StalledHttpClient() { Close(); }

  void Send(loom_serve_http_server_t* server, const char* target,
            const std::string& body, const char* method = "POST",
            const std::string& headers = {}) {
    const auto host_allocator = iree_allocator_system();
    auto proactor_options = iree_async_proactor_options_default();
    proactor_options.threading_mode =
        IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        proactor_options, host_allocator, &proactor_));
    iree_async_proactor_t* server_proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_get(
        loom_serve_http_server_proactor_pool(server), 0, &server_proactor));
    auto options = iree_net_loopback_carrier_options_default();
    options.max_send_operations = 1;
    IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
        proactor_, server_proactor, &options, host_allocator, &carrier_,
        &peer_));
    iree_net_carrier_retain(peer_);
    uint64_t wait_value = 0;
    const bool consumed =
        loom_serve_http_server_try_accept(server, peer_, {}, &wait_value);
    if (!consumed) {
      iree_net_carrier_release(peer_);
    }
    ASSERT_TRUE(consumed);
    IREE_ASSERT_OK(iree_net_carrier_set_handlers(
        carrier_, {[](void* self, iree_async_span_t bytes,
                      iree_async_buffer_lease_t* lease) -> iree_status_t {
                     auto* client = static_cast<StalledHttpClient*>(self);
                     client->response_.append(reinterpret_cast<const char*>(
                                                  iree_async_span_ptr(bytes)),
                                              bytes.length);
                     return iree_ok_status();
                   },
                   [](void* self, iree_status_t status) {
                     // Loopback reports peer deactivation as terminal
                     // UNAVAILABLE.
                     IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
                     static_cast<StalledHttpClient*>(self)->departed_ = true;
                   },
                   this}));
    IREE_ASSERT_OK(iree_net_carrier_activate(carrier_));
    const std::string request =
        std::string(method) + " " + target +
        " HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\n" +
        headers + "Content-Length: " + std::to_string(body.size()) +
        "\r\n\r\n" + body;
    const iree_net_send_params_t params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(request.data(), request.size())),
        .completion_callback = {
            [](void* user_data, iree_status_t status,
               iree_host_size_t bytes_sent) { IREE_EXPECT_OK(status); },
            nullptr}};
    IREE_ASSERT_OK(iree_net_carrier_send(carrier_, &params));
  }

  std::string Read() {
    while (!departed_) {
      Poll();
    }
    return response_;
  }

  void WaitUntilSendCreditExhausted() {
    while (iree_net_carrier_state(peer_) != IREE_NET_CARRIER_STATE_ACTIVE ||
           iree_net_carrier_query_send_budget(peer_).slots != 0) {
      std::this_thread::yield();
    }
  }

  void Close() {
    if (carrier_) {
      bool drained = false;
      iree_net_carrier_deactivate(
          carrier_, [](void* self) { *static_cast<bool*>(self) = true; },
          &drained);
      while (!drained) {
        Poll();
      }
      iree_net_carrier_release(carrier_);
      carrier_ = nullptr;
    }
    iree_net_carrier_release(peer_);
    peer_ = nullptr;
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

 private:
  void Poll() {
    IREE_CHECK_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }

  // Independent client owner; one thread performs every poll and Close.
  iree_async_proactor_t* proactor_ = nullptr;
  // Owned client half; the HTTP server owns the other half.
  iree_net_carrier_t* carrier_ = nullptr;
  // Retained server half for querying actual carrier credit, not model state.
  iree_net_carrier_t* peer_ = nullptr;
  // Bytes delivered while this client is explicitly polled.
  std::string response_;
  // Published only by this client's poll owner after server deactivation.
  bool departed_ = false;
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_STALLED_HTTP_CLIENT_H_
