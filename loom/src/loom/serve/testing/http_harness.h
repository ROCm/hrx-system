// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_HTTP_HARNESS_H_
#define LOOM_SERVE_TESTING_HTTP_HARNESS_H_

#include <atomic>
#include <functional>
#include <string>
#include <thread>

#include "loom/serve/transport/http/listener.h"

namespace loom_serve::testing {

// Owns a client descriptor across assertion exits. release() transfers it to a
// consuming operation such as HttpHarness::Read; otherwise destruction closes
// the client without waiting for its response.
class HttpClient {
 public:
  explicit HttpClient(int descriptor = -1) : descriptor_(descriptor) {}
  ~HttpClient();
  HttpClient(const HttpClient&) = delete;
  HttpClient& operator=(const HttpClient&) = delete;

  int get() const { return descriptor_; }
  // Receives a new descriptor only after default construction or release.
  int* put() {
    IREE_ASSERT(descriptor_ == -1);
    return &descriptor_;
  }
  int release();

 private:
  // Owned nonblocking client socket, or -1 after transfer.
  int descriptor_;
};

// Linux integration-test client with explicit service and socket readiness.
// The fixture owns the server/listener until after Join and all client calls.
// Service work is optional: a fixture may instead drive admitted requests
// itself with TakeRequest. No timer substitutes for either ownership boundary.
class HttpHarness {
 public:
  ~HttpHarness();

  iree_status_t Initialize(loom_serve_http_server_t* server,
                           loom_serve_http_listener_t* listener);

  // Starts one outer service owner, retaining its terminal status until Join.
  // Start follows Initialize and any previous owner has already been joined.
  void Start(std::function<iree_status_t()> run);
  bool joinable() const { return thread_.joinable(); }
  // The caller requests shutdown before joining an active service.
  iree_status_t Join();

  // A transport failure ends this client's observation lifetime. Polling
  // callers stop even when an expected-failure test intercepts its diagnostic.
  bool failed() const { return failed_; }

  // Sends one complete request, transferring the connected nonblocking
  // descriptor to the caller. Read consumes it; intentional disconnects close
  // it.
  iree_status_t Send(const std::string& request, int* out_client);
  // Waits for the first response byte without consuming it. The caller still
  // owns the descriptor on every path, including terminal failure.
  iree_status_t AwaitResponse(int client);
  // Reads a close-delimited response and closes the descriptor on every path.
  // Cancellation tests may explicitly permit an empty orderly response.
  iree_status_t Read(int client, bool allow_empty, std::string* out_response);

  // Checks that a pending request has not responded, disconnected, or lost its
  // service owner. A negative client checks only the owner, for waits following
  // an intentional disconnect or shutdown. Does not consume response bytes.
  iree_status_t CheckPending(int client);

  // A dependency callback publishes its state before Notify. Await observes
  // that state as well as socket/owner failure, without taking the router's
  // observer. Only the outer client thread waits; callbacks never block.
  void Notify();
  iree_status_t Await(int client, const std::function<bool()>& ready);

  // Claims an HTTP request while observing the corresponding client's socket.
  // Only a manually driven fixture calls this: it temporarily owns the server
  // observer, and no router or other request consumer is running concurrently.
  iree_status_t TakeRequest(int client,
                            loom_serve_http_connection_t** out_connection,
                            const loom_serve_http_request_t** out_request);

 private:
  iree_status_t WaitSocket(int client, short events, int notification = -1);
  iree_status_t Diagnose(iree_status_t status);

  // Borrowed transport and listener, owned by the enclosing fixture.
  loom_serve_http_server_t* server_ = nullptr;
  // Borrowed listener supplies the bound ephemeral address.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Readable until Join once the service owner has returned.
  int completion_ = -1;
  // Coalesced request/diagnostic wake while TakeRequest owns the observer.
  int notification_ = -1;
  // Publishes terminal status before the completion descriptor is signaled.
  std::atomic<bool> completed_{false};
  // Written only by the owner; borrowed after acquire and consumed by Join.
  iree_status_t status_ = iree_ok_status();
  // Optional service owner, joined before any fixture dependency is destroyed.
  std::thread thread_;
  // Client-thread-owned terminal transport observation, independent of GTest.
  bool failed_ = false;
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_HTTP_HARNESS_H_
