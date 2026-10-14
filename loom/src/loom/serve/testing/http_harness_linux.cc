// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

#include "loom/serve/telemetry/stream.h"
#include "loom/serve/testing/http_harness.h"

namespace loom_serve::testing {
namespace {

iree_status_t SocketError(const char* operation, int error_number) {
  return iree_make_status(iree_status_code_from_errno(error_number),
                          "%s: %s (%d)", operation, strerror(error_number),
                          error_number);
}

void Signal(int descriptor) {
  const uint64_t value = 1;
  ssize_t count;
  do {
    count = write(descriptor, &value, sizeof(value));
  } while (count < 0 && errno == EINTR);
  if (count < 0 && errno != EAGAIN) {
    iree_status_abort(SocketError("signal HTTP test readiness", errno));
  }
}

void Consume(int descriptor) {
  uint64_t value;
  ssize_t count;
  do {
    count = read(descriptor, &value, sizeof(value));
  } while (count < 0 && errno == EINTR);
  if (count < 0 && errno != EAGAIN) {
    iree_status_abort(SocketError("consume HTTP test readiness", errno));
  }
}

}  // namespace

HttpClient::~HttpClient() {
  if (descriptor_ >= 0 && close(descriptor_) < 0) {
    iree_status_abort(SocketError("close HTTP client", errno));
  }
}

int HttpClient::release() { return std::exchange(descriptor_, -1); }

HttpHarness::~HttpHarness() {
  IREE_ASSERT(!thread_.joinable());
  if (completion_ >= 0 && close(completion_) < 0) {
    iree_status_abort(SocketError("close service completion", errno));
  }
  if (notification_ >= 0 && close(notification_) < 0) {
    iree_status_abort(SocketError("close HTTP notification", errno));
  }
}

iree_status_t HttpHarness::Initialize(loom_serve_http_server_t* server,
                                      loom_serve_http_listener_t* listener) {
  server_ = server;
  listener_ = listener;
  completion_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (completion_ < 0) {
    return SocketError("service completion eventfd", errno);
  }
  notification_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (notification_ < 0) {
    return SocketError("HTTP notification eventfd", errno);
  }
  return iree_ok_status();
}

void HttpHarness::Start(std::function<iree_status_t()> run) {
  completed_.store(false, std::memory_order_relaxed);
  thread_ = std::thread([this, run = std::move(run)] {
    status_ = run();
    completed_.store(true, std::memory_order_release);
    Signal(completion_);
  });
}

iree_status_t HttpHarness::Join() {
  if (!thread_.joinable()) {
    return iree_ok_status();
  }
  thread_.join();
  Consume(completion_);
  completed_.store(false, std::memory_order_relaxed);
  return std::exchange(status_, iree_ok_status());
}

iree_status_t HttpHarness::Diagnose(iree_status_t status) {
  failed_ = true;
  if (completed_.load(std::memory_order_acquire)) {
    status = iree_status_join(status, iree_status_clone(status_));
  }
  const auto* stream = loom_serve_http_server_diagnostics(server_);
  const uint64_t head = loom_serve_telemetry_stream_head(stream);
  uint64_t cursor = 0;
  while (cursor < head) {
    loom_serve_telemetry_record_t record;
    const auto result =
        loom_serve_telemetry_stream_read(stream, &cursor, &record);
    if (result == LOOM_SERVE_TELEMETRY_READ_BUSY ||
        result == LOOM_SERVE_TELEMETRY_READ_EMPTY) {
      break;
    }
    if (result == LOOM_SERVE_TELEMETRY_READ_RECORD &&
        record.event == &LOOM_SERVE_TELEMETRY_ERROR) {
      status = iree_status_annotate_f(
          status, "HTTP transport: %s: %.*s%s",
          iree_status_code_string((iree_status_code_t)record.data.error.code),
          (int)record.data.error.length, record.data.error.message,
          record.data.error.truncated ? " [truncated]" : "");
    }
  }
  return status;
}

iree_status_t HttpHarness::WaitSocket(int client, short events,
                                      int notification) {
  pollfd descriptors[] = {
      {client, events, 0}, {completion_, POLLIN, 0}, {notification, POLLIN, 0}};
  int result;
  do {
    result = poll(descriptors, IREE_ARRAYSIZE(descriptors), -1);
  } while (result < 0 && errno == EINTR);
  if (result < 0) {
    return SocketError("poll HTTP readiness", errno);
  }
  if (descriptors[0].revents) {
    return iree_ok_status();
  }
  if (descriptors[1].revents) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "service owner exited while waiting for HTTP");
  }
  Consume(notification);
  return iree_ok_status();
}

iree_status_t HttpHarness::Send(const std::string& request, int* out_client) {
  *out_client = -1;
  const auto* address = loom_serve_http_listener_address(listener_);
  const auto* native = reinterpret_cast<const sockaddr*>(address->storage);
  const int client =
      socket(native->sa_family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (client < 0) {
    return Diagnose(SocketError("socket", errno));
  }
  iree_status_t status = iree_ok_status();
  if (connect(client, native, address->length) < 0) {
    const int error_number = errno;
    if (error_number == EINPROGRESS || error_number == EINTR) {
      status = WaitSocket(client, POLLOUT);
      if (iree_status_is_ok(status)) {
        int connection_error = 0;
        socklen_t length = sizeof(connection_error);
        if (getsockopt(client, SOL_SOCKET, SO_ERROR, &connection_error,
                       &length) < 0) {
          status = SocketError("getsockopt SO_ERROR", errno);
        } else if (connection_error) {
          status = SocketError("connect", connection_error);
        }
      }
    } else {
      status = SocketError("connect", error_number);
    }
  }
  size_t offset = 0;
  while (iree_status_is_ok(status) && offset < request.size()) {
    const ssize_t count = send(client, request.data() + offset,
                               request.size() - offset, MSG_NOSIGNAL);
    if (count > 0) {
      offset += count;
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      status = WaitSocket(client, POLLOUT);
    } else {
      status = count < 0
                   ? SocketError("send HTTP request", errno)
                   : iree_make_status(IREE_STATUS_UNAVAILABLE,
                                      "send HTTP request made no progress");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_client = client;
  } else {
    if (close(client) < 0) {
      status =
          iree_status_join(status, SocketError("close HTTP client", errno));
    }
    status = Diagnose(status);
  }
  return status;
}

iree_status_t HttpHarness::AwaitResponse(int client) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    char byte;
    const ssize_t count = recv(client, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (count > 0) {
      break;
    } else if (count == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "HTTP peer closed before sending a response");
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status = WaitSocket(client, POLLIN);
    } else {
      status = SocketError("await HTTP response", errno);
    }
  }
  return iree_status_is_ok(status) ? status : Diagnose(status);
}

iree_status_t HttpHarness::CheckPending(int client) {
  if (client >= 0) {
    char byte;
    ssize_t count;
    do {
      count = recv(client, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    } while (count < 0 && errno == EINTR);
    if (count >= 0) {
      return Diagnose(iree_make_status(
          IREE_STATUS_UNAVAILABLE, "HTTP peer %s before expected model state",
          count == 0 ? "closed" : "responded"));
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return Diagnose(SocketError("await model state", errno));
    }
  }
  if (completed_.load(std::memory_order_acquire)) {
    return Diagnose(iree_make_status(
        IREE_STATUS_UNAVAILABLE, "service owner exited before model state"));
  }
  return iree_ok_status();
}

void HttpHarness::Notify() { Signal(notification_); }

iree_status_t HttpHarness::Await(int client,
                                 const std::function<bool()>& ready) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !ready()) {
    pollfd descriptors[] = {{client, POLLIN, 0},
                            {completion_, POLLIN, 0},
                            {notification_, POLLIN, 0}};
    int result;
    do {
      result = poll(descriptors, IREE_ARRAYSIZE(descriptors), -1);
    } while (result < 0 && errno == EINTR);
    if (result < 0) {
      status = Diagnose(SocketError("poll model readiness", errno));
    } else {
      if (descriptors[2].revents) {
        Consume(notification_);
      }
      // Published model state wins over a subsequent response or owner exit.
      // In particular, admission denial may immediately send and close a 503.
      if (!ready() && (descriptors[0].revents || descriptors[1].revents)) {
        status = CheckPending(client);
      }
    }
  }
  return status;
}

iree_status_t HttpHarness::Read(int client, bool allow_empty,
                                std::string* out_response) {
  out_response->clear();
  iree_status_t status = iree_ok_status();
  bool complete = false;
  while (iree_status_is_ok(status) && !complete) {
    char bytes[4096];
    const ssize_t count = recv(client, bytes, sizeof(bytes), MSG_DONTWAIT);
    if (count > 0) {
      out_response->append(bytes, count);
    } else if (count == 0) {
      complete = true;
      if (!allow_empty && out_response->empty()) {
        status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "HTTP peer closed before sending a response");
      }
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status = WaitSocket(client, POLLIN);
    } else {
      status = SocketError("receive HTTP response", errno);
    }
  }
  if (close(client) < 0) {
    status = iree_status_join(status, SocketError("close HTTP client", errno));
  }
  return iree_status_is_ok(status) ? status : Diagnose(status);
}

iree_status_t HttpHarness::TakeRequest(
    int client, loom_serve_http_connection_t** out_connection,
    const loom_serve_http_request_t** out_request) {
  *out_connection = nullptr;
  *out_request = nullptr;
  loom_serve_http_server_set_observer(
      server_,
      {[](void* self) { static_cast<HttpHarness*>(self)->Notify(); }, this});
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !*out_connection) {
    *out_connection = loom_serve_http_server_take_request(server_, out_request);
    if (*out_connection) {
      break;
    }
    char byte;
    const ssize_t count = recv(client, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (count >= 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "HTTP peer %s before request admission",
                                count == 0 ? "closed" : "responded");
    } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      status = SocketError("await HTTP request admission", errno);
    } else {
      status = WaitSocket(client, POLLIN, notification_);
    }
  }
  loom_serve_http_server_set_observer(server_, {});
  return iree_status_is_ok(status) ? status : Diagnose(status);
}

}  // namespace loom_serve::testing
